"""Simulation constants and utility functions for the I-MPPI notebook."""

from dataclasses import replace
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from jax_mppi.i_mppi.environment import (
    GOAL_POS,
    augmented_dynamics_with_grid,
    informative_running_cost,
)
from jax_mppi.i_mppi.fsmi import compute_info_field
from jax_mppi.i_mppi.planner import biased_mppi_command

# --- Simulation Constants ---
DT = 0.05
NX = 16  # 13 (quad) + 3 (info zones)
NU = 4
CONTROL_HZ = 50.0
FSMI_HZ = 5.0
FSMI_STEPS = int(round(CONTROL_HZ / FSMI_HZ))

U_MIN = jnp.array([0.0, -10.0, -10.0, -10.0])
U_MAX = jnp.array([4.0 * 9.81, 10.0, 10.0, 10.0])
U_INIT = jnp.array([9.81, 0.0, 0.0, 0.0])
NOISE_SIGMA = jnp.diag(jnp.array([2.0, 0.5, 0.5, 0.5]) ** 2)

# Early termination thresholds
INFO_DONE_THRESHOLD = 5.0  # info level below which a zone is "visited"
GOAL_DONE_THRESHOLD = 1.5  # distance to goal below which goal is "reached"

# Value that depleted zone cells are set to (known free space)
_FREE_PROB = 0.2


def _build_zone_masks(info_zones, origin, resolution, width, height):
    """Precompute per-zone boolean masks over the grid.

    Returns:
        (N, H, W) boolean array where N = number of info zones.
    """
    y_range = jnp.arange(height)
    x_range = jnp.arange(width)
    Y, X = jnp.meshgrid(y_range, x_range, indexing="ij")

    world_x = origin[0] + (X + 0.5) * resolution
    world_y = origin[1] + (Y + 0.5) * resolution

    def zone_mask(zone):
        cx, cy, w, h = zone[0], zone[1], zone[2], zone[3]
        in_x = (world_x >= cx - w / 2) & (world_x <= cx + w / 2)
        in_y = (world_y >= cy - h / 2) & (world_y <= cy + h / 2)
        return in_x & in_y

    return jax.vmap(zone_mask)(info_zones)  # (N, H, W)


def _update_grid_from_info(initial_grid, zone_masks, info_levels):
    """Update grid probabilities based on current info levels.

    For each info zone, cells inside the zone are interpolated between
    their initial value (when info = 100) and free-space (when info = 0).
    All other cells are preserved exactly as in the initial grid.

    Args:
        initial_grid: (H, W) the ORIGINAL occupancy grid.
        zone_masks: (N, H, W) precomputed boolean masks for each zone.
        info_levels: (N,) current info level per zone [0, 100].

    Returns:
        (H, W) updated grid.
    """
    grid = initial_grid
    n_zones = zone_masks.shape[0]
    # alpha = 1.0 when info=100 (keep original), 0.0 when info=0 (free)
    alphas = jnp.clip(info_levels / 100.0, 0.0, 1.0)

    # Obstacle cells (>= 0.7) are preserved even inside zones
    is_obstacle = initial_grid >= 0.7

    for i in range(n_zones):
        mask = zone_masks[i] & ~is_obstacle
        alpha = alphas[i]
        # Blend: alpha * original + (1 - alpha) * free
        blended = alpha * initial_grid + (1.0 - alpha) * _FREE_PROB
        grid = jnp.where(mask, blended, grid)

    return grid


def make_u_ref_from_traj(current_state, ref_traj):
    """Convert position reference trajectory to control reference."""
    pos = current_state[:3]
    err = ref_traj - pos[None, :]

    k_thrust = 3.0
    thrust = U_INIT[0] + k_thrust * err[:, 2]

    k_omega = 0.6
    omega_x = k_omega * err[:, 1]
    omega_y = -k_omega * err[:, 0]
    omega_z = jnp.zeros_like(omega_x)

    u_ref = jnp.stack([thrust, omega_x, omega_y, omega_z], axis=1)
    u_ref = jnp.clip(u_ref, U_MIN, U_MAX)
    return u_ref


def compute_smoothness(
    actions: jax.Array,
    positions: jax.Array,
    dt_val: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute action jerk and trajectory jerk metrics."""
    action_jerk = jnp.diff(actions, n=2, axis=0) / (dt_val**2)
    action_jerk_mean = jnp.mean(jnp.linalg.norm(action_jerk, axis=1))

    pos = positions[:, :3]
    vel = jnp.diff(pos, axis=0) / dt_val
    acc = jnp.diff(vel, axis=0) / dt_val
    jerk = jnp.diff(acc, axis=0) / dt_val
    traj_jerk_mean = jnp.mean(jnp.linalg.norm(jerk, axis=1))

    return action_jerk_mean, traj_jerk_mean


def build_sim_fn(
    config,
    fsmi_planner,
    uniform_fsmi,
    uniform_fsmi_config,
    grid_map_obj,
    horizon,
    sim_steps,
    progress_callback=None,
):
    """Build a JIT-compiled simulation function.

    Args:
        config: MPPI configuration.
        fsmi_planner: Layer 2 FSMI trajectory generator.
        uniform_fsmi: Layer 3 Uniform-FSMI instance.
        uniform_fsmi_config: Uniform-FSMI configuration.
        grid_map_obj: GridMap object for the environment.
        horizon: Planning horizon.
        sim_steps: Number of simulation steps.
        progress_callback: Optional callable(step) invoked each step
            via ``jax.debug.callback`` for progress reporting.
    """
    info_zones = fsmi_planner.info_zones
    initial_grid = grid_map_obj.grid

    # Precompute zone masks (done once at build time, not inside JIT)
    zone_masks = _build_zone_masks(
        info_zones,
        grid_map_obj.origin,
        grid_map_obj.resolution,
        grid_map_obj.width,
        grid_map_obj.height,
    )

    def step_fn(carry, t):
        current_state, current_ctrl_state, ref_traj, grid, done_step = carry

        if progress_callback is not None:
            jax.debug.callback(progress_callback, t)

        # Check if simulation is already done
        current_info = current_state[13:]
        all_zones_visited = jnp.all(current_info < INFO_DONE_THRESHOLD)
        goal_dist = jnp.linalg.norm(current_state[:3] - GOAL_POS)
        goal_reached = goal_dist < GOAL_DONE_THRESHOLD
        newly_done = all_zones_visited & goal_reached
        # Record the step at which we first finish (0 means not done yet)
        done_step = jnp.where((done_step == 0) & newly_done, t, done_step)
        is_done = done_step > 0

        # --- Active step (when not done) ---

        # Update grid: blend zone cells based on depleted info levels
        updated_grid = _update_grid_from_info(
            initial_grid, zone_masks, current_info
        )

        # Layer 2: Full FSMI reference trajectory (slow, 5 Hz)
        do_update = jnp.equal(jnp.mod(t, FSMI_STEPS), 0)
        info_data = (updated_grid, current_info)

        # Update the planner's grid map for FSMI gain computation
        updated_planner = replace(
            fsmi_planner,
            grid_map=replace(fsmi_planner.grid_map, grid=updated_grid),
        )

        new_ref_traj = jax.lax.cond(
            do_update & ~is_done,
            lambda _: updated_planner.get_reference_trajectory(
                current_state, info_data, horizon, DT
            )[0],
            lambda _: ref_traj,
            operand=None,
        )

        # Layer 3: Biased I-MPPI with Uniform-FSMI (fast, 50 Hz)
        cost_fn = partial(
            informative_running_cost,
            target=new_ref_traj,
            grid_map=updated_grid,
            uniform_fsmi_fn=uniform_fsmi.compute,
            info_weight=uniform_fsmi_config.info_weight,
            grid_origin=grid_map_obj.origin,
            grid_resolution=grid_map_obj.resolution,
        )

        U_ref_local = make_u_ref_from_traj(current_state, new_ref_traj)

        # Dynamics with grid-based line-of-sight for info depletion
        dynamics_fn = partial(
            augmented_dynamics_with_grid,
            dt=DT,
            grid=updated_grid,
            grid_origin=grid_map_obj.origin,
            grid_resolution=grid_map_obj.resolution,
        )

        action, next_ctrl_state = biased_mppi_command(
            config,
            current_ctrl_state,
            current_state,
            dynamics_fn,
            cost_fn,
            U_ref_local,
            bias_alpha=0.2,
        )

        # When done, use hover action instead
        hover_action = U_INIT
        action = jnp.where(is_done, hover_action, action)

        next_state = dynamics_fn(current_state, action)
        # When done, keep current state
        next_state = jnp.where(is_done, current_state, next_state)

        return (
            next_state,
            next_ctrl_state,
            new_ref_traj,
            updated_grid,
            done_step,
        ), (
            next_state,
            current_info,
            new_ref_traj[0],
            action,
        )

    def sim_fn(initial_state, initial_ctrl_state):
        info_data = (grid_map_obj.grid, initial_state[13:])
        init_ref_traj, _ = fsmi_planner.get_reference_trajectory(
            initial_state, info_data, horizon, DT
        )
        init_grid = grid_map_obj.grid
        init_done_step = jnp.array(0, dtype=jnp.int32)

        (
            (final_state, final_ctrl_state, _, _, done_step),
            (history_x, history_info, targets, actions),
        ) = jax.lax.scan(
            step_fn,
            (
                initial_state,
                initial_ctrl_state,
                init_ref_traj,
                init_grid,
                init_done_step,
            ),
            jnp.arange(sim_steps),
        )
        return final_state, history_x, history_info, targets, actions, done_step

    return jax.jit(sim_fn)


def interp2d(field, field_origin, field_res, positions):
    """Bilinear interpolation of a 2D field at query positions.

    Args:
        field: (Nx, Ny) 2D array
        field_origin: (2,) world coordinates of field[0, 0]
        field_res: scalar, meters per cell
        positions: (K, 2) or (2,) query positions in world coordinates

    Returns:
        (K,) or scalar interpolated values. Out-of-bounds returns 0.0.
    """
    # Handle both single position and batch
    positions_2d = jnp.atleast_2d(positions)
    single_pos = positions.ndim == 1

    # Convert world coordinates to continuous grid indices
    gx = (positions_2d[:, 0] - field_origin[0]) / field_res
    gy = (positions_2d[:, 1] - field_origin[1]) / field_res

    # Floor indices
    ix = jnp.floor(gx).astype(jnp.int32)
    iy = jnp.floor(gy).astype(jnp.int32)

    # Fractional parts
    fx = gx - jnp.floor(gx)
    fy = gy - jnp.floor(gy)

    # Clamp to valid range
    Nx, Ny = field.shape
    ix0 = jnp.clip(ix, 0, Nx - 2)
    iy0 = jnp.clip(iy, 0, Ny - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    # Check if out of bounds and use 0.0 for out-of-bounds
    out_of_bounds = (ix < 0) | (ix >= Nx - 1) | (iy < 0) | (iy >= Ny - 1)

    # Bilinear interpolation
    v00 = field[ix0, iy0]
    v10 = field[ix1, iy0]
    v01 = field[ix0, iy1]
    v11 = field[ix1, iy1]

    interp = (
        v00 * (1 - fx) * (1 - fy)
        + v10 * fx * (1 - fy)
        + v01 * (1 - fx) * fy
        + v11 * fx * fy
    )

    # Return 0.0 for out-of-bounds
    interp = jnp.where(out_of_bounds, 0.0, interp)

    if single_pos:
        return interp[0]
    return interp


def build_parallel_sim_fn(
    mppi_config: Any,
    fsmi_module: Any,
    uniform_fsmi: Any,
    info_field_config: Any,
    grid_map_obj: Any,
    info_zones: jax.Array,
    sim_steps: int,
    progress_callback: Callable[[int], None] | None = None,
) -> Any:

    """Build a JIT-compiled Parallel I-MPPI simulation function.

    Single MPPI controller at 50 Hz guided by information potential field
    updated at 5-10 Hz. No reference trajectory or biased sampling.

    Args:
        mppi_config: MPPI configuration.
        fsmi_module: FSMIModule for field computation.
        uniform_fsmi: UniformFSMI instance for local reactivity.
        info_field_config: InfoFieldConfig with field parameters.
        grid_map_obj: GridMap object for environment.
        info_zones: (N, 4) array of info zones [cx, cy, w, h].
        sim_steps: Number of simulation steps.
        progress_callback: Optional callable(step) for progress.

    Returns:
        Simulation function that takes (initial_state, initial_ctrl_state)
        and returns (final_state, history_x, history_info, actions,
                     done_step, history_field, history_field_origin).
    """
    from jax_mppi import mppi

    initial_grid = grid_map_obj.grid

    # Precompute zone masks
    zone_masks = _build_zone_masks(
        info_zones,
        grid_map_obj.origin,
        grid_map_obj.resolution,
        grid_map_obj.width,
        grid_map_obj.height,
    )

    def step_fn(carry, t):
        (
            current_state,
            current_ctrl_state,
            info_field,
            field_origin,
            grid,
            done_step,
        ) = carry

        if progress_callback is not None:
            jax.debug.callback(progress_callback, t)

        # Check if simulation is done
        current_info = current_state[13:]
        all_zones_visited = jnp.all(current_info < INFO_DONE_THRESHOLD)
        goal_dist = jnp.linalg.norm(current_state[:3] - GOAL_POS)
        goal_reached = goal_dist < GOAL_DONE_THRESHOLD
        newly_done = all_zones_visited & goal_reached
        done_step = jnp.where((done_step == 0) & newly_done, t, done_step)
        is_done = done_step > 0

        # Update grid based on current info levels
        updated_grid = _update_grid_from_info(
            initial_grid, zone_masks, current_info
        )

        # Low-rate: recompute field every field_update_interval steps
        do_update = jnp.equal(
            jnp.mod(t, info_field_config.field_update_interval), 0
        )
        uav_pos = current_state[:3]
        new_field, new_origin = jax.lax.cond(
            do_update,
            lambda: compute_info_field(fsmi_module, updated_grid, uav_pos, info_field_config),
            lambda: (info_field, field_origin),
        )

        # High-rate: unbiased MPPI with field + local FSMI cost
        # Use a dummy target = goal position for informative_running_cost
        dummy_target = GOAL_POS

        def cost_fn(x, u, t_step):
            # Base cost from informative_running_cost but with field modifications
            # Since we don't have a reference trajectory, we use GOAL_POS as target
            cost = informative_running_cost(
                x, u, t_step,
                target=dummy_target,
                grid_map=updated_grid,
                uniform_fsmi_fn=uniform_fsmi.compute,
                info_weight=info_field_config.lambda_local,
                grid_origin=grid_map_obj.origin,
                grid_resolution=grid_map_obj.resolution,
            )

            # Add field-based guidance term
            pos_xy = x[:2]
            field_value = interp2d(
                new_field, new_origin, info_field_config.field_res, pos_xy
            )
            # Negative because MPPI minimizes cost, we want to maximize field value
            field_cost = -info_field_config.lambda_info * field_value

            # Adaptive goal attraction: increase when field is depleted
            max_field_val = jnp.max(new_field)
            field_depleted = max_field_val < info_field_config.min_field_threshold
            goal_attraction = jnp.where(
                field_depleted,
                info_field_config.goal_fallback_weight,
                info_field_config.goal_weight,
            )
            goal_dist = jnp.linalg.norm(pos_xy - GOAL_POS[:2])
            goal_cost = goal_attraction * goal_dist

            return cost + field_cost + goal_cost

        # Dynamics with grid-based line-of-sight
        dynamics_fn = partial(
            augmented_dynamics_with_grid,
            dt=DT,
            grid=updated_grid,
            grid_origin=grid_map_obj.origin,
            grid_resolution=grid_map_obj.resolution,
        )

        # Standard MPPI (unbiased)
        action, next_ctrl_state = mppi.command(
            mppi_config,
            current_ctrl_state,
            current_state,
            dynamics_fn,
            cost_fn,
        )

        # Hover when done
        hover_action = U_INIT
        action = jnp.where(is_done, hover_action, action)

        next_state = dynamics_fn(current_state, action)
        next_state = jnp.where(is_done, current_state, next_state)

        return (
            next_state,
            next_ctrl_state,
            new_field,
            new_origin,
            updated_grid,
            done_step,
        ), (
            next_state,
            current_info,
            new_field,
            new_origin,
            action,
        )

    def sim_fn(initial_state, initial_ctrl_state):
        # Initialize field
        init_field, init_origin = compute_info_field(
            fsmi_module, initial_grid, initial_state[:3], info_field_config
        )
        init_done_step = jnp.array(0, dtype=jnp.int32)

        (
            (final_state, final_ctrl_state, final_field, final_origin, _, done_step),
            (history_x, history_info, history_field, history_field_origin, actions),
        ) = jax.lax.scan(
            step_fn,
            (
                initial_state,
                initial_ctrl_state,
                init_field,
                init_origin,
                initial_grid,
                init_done_step,
            ),
            jnp.arange(sim_steps),
        )
        return (
            final_state,
            history_x,
            history_info,
            actions,
            done_step,
            history_field,
            history_field_origin,
        )

    return jax.jit(sim_fn)
