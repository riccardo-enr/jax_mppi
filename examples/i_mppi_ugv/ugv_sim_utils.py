"""Simulation constants and utility functions for UGV I-MPPI.

Mirrors examples/i_mppi/sim_utils.py but adapted for 5D diff-drive UGV.
Reuses generic utilities (fov_grid_update, interp2d) from sim_utils.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp

from jax_mppi.i_mppi.fsmi import compute_info_field
from jax_mppi.i_mppi.ugv_environment import (
    GOAL_POS_2D,
    step_ugv,
    ugv_informative_running_cost,
)

# Import generic utilities from the UAV sim_utils
import os
import sys

_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "i_mppi")
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from sim_utils import fov_grid_update, interp2d  # noqa: E402

# --- Simulation Constants ---
DT = 0.05
NX = 5  # UGV state: [x, y, θ, v, ω]
NU = 2  # UGV control: [a, α]
CONTROL_HZ = 50.0
FSMI_HZ = 5.0
FSMI_STEPS = int(round(CONTROL_HZ / FSMI_HZ))

U_MIN = jnp.array([-1.0, -2.0])  # [max decel, max angular decel]
U_MAX = jnp.array([1.0, 2.0])  # [max accel, max angular accel]
U_INIT = jnp.array([0.0, 0.0])  # stationary
NOISE_SIGMA = jnp.diag(jnp.array([0.5, 1.0]) ** 2)

# Early termination
GOAL_DONE_THRESHOLD = 1.5  # distance to goal below which goal is "reached"


# ---------------------------------------------------------------------------
# Field-gradient reference trajectory (2D, no altitude)
# ---------------------------------------------------------------------------


def field_gradient_trajectory(
    field, field_origin, field_res, start_xy, horizon, ref_speed, dt
):
    """Generate 2D reference trajectory via gradient ascent on the info field.

    Same algorithm as the UAV version but returns (horizon, 2) — no altitude.

    Args:
        field: (Nx, Ny) FSMI information field
        field_origin: (2,) world coordinates of field[0, 0]
        field_res: scalar, meters per field cell
        start_xy: (2,) starting position
        horizon: number of trajectory steps
        ref_speed: trajectory speed [m/s]
        dt: time step [s]

    Returns:
        (horizon, 2) reference trajectory in world coordinates
    """
    grad_x = jnp.gradient(field, field_res, axis=0)
    grad_y = jnp.gradient(field, field_res, axis=1)

    step_dist = ref_speed * dt

    def step_fn(carry, _):
        xy = carry
        gx = interp2d(grad_x, field_origin, field_res, xy)
        gy = interp2d(grad_y, field_origin, field_res, xy)
        grad = jnp.array([gx, gy])
        grad_norm = jnp.linalg.norm(grad)
        direction = grad / jnp.maximum(grad_norm, 1e-6)
        next_xy = xy + step_dist * direction
        return next_xy, next_xy

    _, trajectory_xy = jax.lax.scan(step_fn, start_xy, None, length=horizon)
    return trajectory_xy  # (horizon, 2)


# ---------------------------------------------------------------------------
# Smoothness metrics
# ---------------------------------------------------------------------------


def compute_smoothness(
    actions: jax.Array,
    positions: jax.Array,
    dt_val: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute action jerk and trajectory jerk metrics."""
    action_jerk = jnp.diff(actions, n=2, axis=0) / (dt_val**2)
    action_jerk_mean = jnp.mean(jnp.linalg.norm(action_jerk, axis=1))

    pos = positions[:, :2]
    vel = jnp.diff(pos, axis=0) / dt_val
    acc = jnp.diff(vel, axis=0) / dt_val
    jerk = jnp.diff(acc, axis=0) / dt_val
    traj_jerk_mean = jnp.mean(jnp.linalg.norm(jerk, axis=1))

    return action_jerk_mean, traj_jerk_mean


# ---------------------------------------------------------------------------
# Parallel I-MPPI simulation builder (UGV)
# ---------------------------------------------------------------------------


def build_ugv_parallel_sim_fn(
    mppi_config: Any,
    fsmi_module: Any,
    uniform_fsmi: Any,
    info_field_config: Any,
    grid_map_obj: Any,
    sim_steps: int,
    progress_callback: Callable[[int], None] | None = None,
) -> Any:
    """Build a JIT-compiled Parallel I-MPPI simulation function for UGV.

    Same two-layer architecture as the UAV version:
    - Layer 2 (5 Hz): FSMI info field + gradient reference trajectory
    - Layer 3 (50 Hz): biased MPPI + Uniform-FSMI + FOV grid update

    Args:
        mppi_config: MPPI configuration.
        fsmi_module: FSMIModule for field computation.
        uniform_fsmi: UniformFSMI instance for local reactivity.
        info_field_config: InfoFieldConfig with field and ref trajectory params.
        grid_map_obj: GridMap object for environment.
        sim_steps: Number of simulation steps.
        progress_callback: Optional callable(step) for progress.

    Returns:
        Simulation function: (initial_state_5d, initial_ctrl_state) ->
            (final_state, history_x, actions, done_step,
             history_field, history_field_origin, final_grid, history_ref_traj)
    """
    from jax_mppi import mppi

    initial_grid = grid_map_obj.grid
    grid_origin = grid_map_obj.origin
    grid_resolution = grid_map_obj.resolution

    def step_fn(carry, t):
        (
            current_state,
            current_ctrl_state,
            info_field,
            field_origin,
            grid,
            ref_traj,
            done_step,
        ) = carry

        if progress_callback is not None:
            jax.debug.callback(progress_callback, t)

        # Check termination: goal reached AND map explored
        goal_dist = jnp.linalg.norm(current_state[:2] - GOAL_POS_2D)
        goal_reached = goal_dist < GOAL_DONE_THRESHOLD
        uncertainty = 4.0 * grid * (1.0 - grid)
        unknown_frac = jnp.mean(uncertainty > 0.5)
        map_explored = unknown_frac < 0.05
        done_step = jnp.where(
            (done_step == 0) & goal_reached & map_explored, t, done_step
        )
        is_done = done_step > 0

        # --- FOV grid update ---
        pos_xy = current_state[:2]
        yaw = current_state[2]
        updated_grid = fov_grid_update(
            grid,
            pos_xy,
            yaw,
            grid_origin,
            grid_resolution,
        )

        # --- Low-rate: recompute field + ref trajectory (every N steps) ---
        do_update = jnp.equal(
            jnp.mod(t, info_field_config.field_update_interval), 0
        )
        new_field, new_origin = jax.lax.cond(
            do_update,
            lambda: compute_info_field(
                fsmi_module, updated_grid, pos_xy, info_field_config
            ),
            lambda: (info_field, field_origin),
        )

        # Generate gradient-ascent reference trajectory (2D)
        new_ref_traj = jax.lax.cond(
            do_update,
            lambda: field_gradient_trajectory(
                new_field,
                new_origin,
                info_field_config.field_res,
                pos_xy,
                info_field_config.ref_horizon,
                info_field_config.ref_speed,
                DT,
            ),
            lambda: ref_traj,
        )

        # --- Cost function ---
        def cost_fn(x, u, t_step):
            cost = ugv_informative_running_cost(
                x,
                u,
                t_step,
                target=new_ref_traj,
                grid_map=updated_grid,
                uniform_fsmi_fn=uniform_fsmi.compute,
                info_weight=info_field_config.lambda_local,
                grid_origin=grid_origin,
                grid_resolution=grid_resolution,
                target_weight=info_field_config.target_weight,
            )
            # Field-based guidance term
            field_value = interp2d(
                new_field, new_origin, info_field_config.field_res, x[:2]
            )
            field_cost = -info_field_config.lambda_info * field_value
            # Constant goal attraction
            goal_cost = info_field_config.goal_weight * jnp.linalg.norm(
                x[:2] - GOAL_POS_2D
            )
            return cost + field_cost + goal_cost

        # --- Dynamics: UGV (5D) ---
        def dynamics_fn(state, action, _t=None):
            return step_ugv(state, action, DT)

        # --- MPPI ---
        action, next_ctrl_state = mppi.command(
            mppi_config,
            current_ctrl_state,
            current_state,
            dynamics_fn,
            cost_fn,
        )

        # Stop when done (zero action instead of hover)
        action = jnp.where(is_done, U_INIT, action)

        next_state = step_ugv(current_state, action, DT)
        next_state = jnp.where(is_done, current_state, next_state)

        return (
            next_state,
            next_ctrl_state,
            new_field,
            new_origin,
            updated_grid,
            new_ref_traj,
            done_step,
        ), (
            next_state,
            new_field,
            new_origin,
            action,
            new_ref_traj,
        )

    def sim_fn(initial_state, initial_ctrl_state):
        # Initialize field + ref trajectory
        init_field, init_origin = compute_info_field(
            fsmi_module, initial_grid, initial_state[:2], info_field_config
        )
        init_ref_traj = field_gradient_trajectory(
            init_field,
            init_origin,
            info_field_config.field_res,
            initial_state[:2],
            info_field_config.ref_horizon,
            info_field_config.ref_speed,
            DT,
        )
        init_done_step = jnp.array(0, dtype=jnp.int32)

        (
            (
                final_state,
                _,
                final_field,
                final_origin,
                final_grid,
                _,
                done_step,
            ),
            (
                history_x,
                history_field,
                history_field_origin,
                actions,
                history_ref_traj,
            ),
        ) = jax.lax.scan(
            step_fn,
            (
                initial_state,
                initial_ctrl_state,
                init_field,
                init_origin,
                initial_grid,
                init_ref_traj,
                init_done_step,
            ),
            jnp.arange(sim_steps),
        )
        return (
            final_state,
            history_x,
            actions,
            done_step,
            history_field,
            history_field_origin,
            final_grid,
            history_ref_traj,
        )

    return jax.jit(sim_fn)
