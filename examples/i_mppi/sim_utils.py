"""Simulation constants and utility functions for the I-MPPI notebook."""

from typing import Any, Callable

import jax
import jax.numpy as jnp

from jax_mppi.i_mppi.environment import (
    GOAL_POS,
    informative_running_cost,
    quat_to_yaw,
    step_quadrotor,
)
from jax_mppi.i_mppi.fsmi import compute_info_field

# --- Simulation Constants ---
DT = 0.05
NX = 13  # quadrotor state only
NU = 4
CONTROL_HZ = 50.0
FSMI_HZ = 5.0
FSMI_STEPS = int(round(CONTROL_HZ / FSMI_HZ))

U_MIN = jnp.array([0.0, -10.0, -10.0, -10.0])
U_MAX = jnp.array([4.0 * 9.81, 10.0, 10.0, 10.0])
U_INIT = jnp.array([9.81, 0.0, 0.0, 0.0])
NOISE_SIGMA = jnp.diag(jnp.array([2.0, 0.5, 0.5, 0.5]) ** 2)

# Early termination
GOAL_DONE_THRESHOLD = 1.5  # distance to goal below which goal is "reached"

# FOV grid update parameters
_FOV_RAD = 1.57  # 90 deg sensor FOV
_SENSOR_RANGE = 4.0  # meters
_N_RAYS = 24
_RAY_STEP = 0.15  # meters
_FREE_PROB = 0.01  # known-free: measured cell is ~certain (odds ≈ 0.01)
_OCC_PROB = 0.99  # known-occupied: measured wall is ~certain


# ---------------------------------------------------------------------------
# FOV-based grid update
# ---------------------------------------------------------------------------


def fov_grid_update(
    grid,
    pos_xy,
    yaw,
    grid_origin,
    grid_resolution,
    fov_rad=_FOV_RAD,
    max_range=_SENSOR_RANGE,
    n_rays=_N_RAYS,
    ray_step=_RAY_STEP,
):
    """Update grid cells visible to UAV to known-free.

    Casts rays from the UAV across its FOV. Any non-obstacle cell that is
    visible (in range, in FOV, not blocked by LOS) is immediately set to
    known-free — a measurement tells you the state of the cell.

    Args:
        grid: (H, W) occupancy grid
        pos_xy: (2,) UAV position in world frame
        yaw: scalar heading in radians
        grid_origin: (2,) world coords of grid[0, 0]
        grid_resolution: scalar meters/cell
        fov_rad: sensor FOV in radians
        max_range: maximum sensor range in meters
        n_rays: number of rays to cast
        ray_step: step size along each ray

    Returns:
        (H, W) updated grid
    """
    H, W = grid.shape

    # Cast rays uniformly across FOV
    angles = jnp.linspace(yaw - fov_rad / 2, yaw + fov_rad / 2, n_rays)
    steps = jnp.arange(0, max_range, ray_step)

    # Ray sample world coordinates: (n_rays, n_steps)
    ray_x = pos_xy[0] + steps[None, :] * jnp.cos(angles[:, None])
    ray_y = pos_xy[1] + steps[None, :] * jnp.sin(angles[:, None])

    # Convert to grid indices
    gx = ((ray_x - grid_origin[0]) / grid_resolution).astype(jnp.int32)
    gy = ((ray_y - grid_origin[1]) / grid_resolution).astype(jnp.int32)

    # Boundary check
    valid = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    safe_gx = jnp.clip(gx, 0, W - 1)
    safe_gy = jnp.clip(gy, 0, H - 1)

    # Look up cell probabilities
    probs = grid[safe_gy, safe_gx]  # (n_rays, n_steps)
    probs = jnp.where(valid, probs, 0.5)

    # LOS blocking: cumulative max of obstacle indicator along ray
    # Once a cell >= 0.7 is hit, all subsequent cells are blocked
    is_obstacle = probs >= 0.7

    # Use cumulative maximum via associative scan for LOS
    def _cummax_fn(a, b):
        return jnp.maximum(a, b)

    blocked = jax.vmap(lambda row: jax.lax.associative_scan(_cummax_fn, row))(
        is_obstacle.astype(jnp.float32)
    )
    blocked = blocked >= 0.5

    # Visible = in bounds, not an obstacle, not blocked by prior obstacle
    # Shift blocked one step: first cell on each ray is never blocked by prior
    blocked_shifted = jnp.concatenate(
        [jnp.zeros((n_rays, 1), dtype=bool), blocked[:, :-1]], axis=1
    )
    visible_free = valid & ~is_obstacle & ~blocked_shifted

    # First obstacle on each ray is also observed (the wall you see)
    visible_occ = valid & is_obstacle & ~blocked_shifted

    # Immediate measurement: observed free → _FREE_PROB, observed wall → _OCC_PROB
    flat_gy = safe_gy.ravel()
    flat_gx = safe_gx.ravel()
    flat_vis_free = visible_free.ravel()
    flat_vis_occ = visible_occ.ravel()

    # Free cells: use .min (only decreases, safe for overlapping rays)
    update_free = jnp.where(flat_vis_free, _FREE_PROB, grid[flat_gy, flat_gx])
    updated_grid = grid.at[flat_gy, flat_gx].min(update_free)

    # Obstacle cells: use .max (only increases toward certainty)
    update_occ = jnp.where(
        flat_vis_occ, _OCC_PROB, updated_grid[flat_gy, flat_gx]
    )
    updated_grid = updated_grid.at[flat_gy, flat_gx].max(update_occ)

    return updated_grid


# ---------------------------------------------------------------------------
# Field-gradient reference trajectory
# ---------------------------------------------------------------------------


def field_gradient_trajectory(
    field, field_origin, field_res, start_xy, horizon, ref_speed, dt, altitude
):
    """Generate reference trajectory via gradient ascent on the info field.

    Follows the steepest ascent of the FSMI field to generate a position
    reference trajectory that MPPI can track.

    Args:
        field: (Nx, Ny) FSMI information field
        field_origin: (2,) world coordinates of field[0, 0]
        field_res: scalar, meters per field cell
        start_xy: (2,) starting position
        horizon: number of trajectory steps
        ref_speed: trajectory speed [m/s]
        dt: time step [s]
        altitude: constant z-coordinate

    Returns:
        (horizon, 3) reference trajectory in world coordinates
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
    z_col = jnp.full((horizon, 1), altitude)
    return jnp.concatenate([trajectory_xy, z_col], axis=1)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parallel I-MPPI simulation builder
# ---------------------------------------------------------------------------


def build_parallel_sim_fn(
    mppi_config: Any,
    fsmi_module: Any,
    uniform_fsmi: Any,
    info_field_config: Any,
    grid_map_obj: Any,
    sim_steps: int,
    progress_callback: Callable[[int], None] | None = None,
) -> Any:
    """Build a JIT-compiled Parallel I-MPPI simulation function.

    Single MPPI controller at 50 Hz guided by:
    - Field-gradient reference trajectory (updated at 5 Hz)
    - Uniform-FSMI local reactivity (50 Hz)
    - FSMI field lookup in cost function

    The occupancy grid is updated per step via FOV ray-casting: cells
    visible to the UAV are blended toward known-free.

    Args:
        mppi_config: MPPI configuration.
        fsmi_module: FSMIModule for field computation.
        uniform_fsmi: UniformFSMI instance for local reactivity.
        info_field_config: InfoFieldConfig with field and ref trajectory params.
        grid_map_obj: GridMap object for environment.
        sim_steps: Number of simulation steps.
        progress_callback: Optional callable(step) for progress.

    Returns:
        Simulation function: (initial_state_13, initial_ctrl_state) ->
            (final_state, history_x, actions, done_step,
             history_field, history_field_origin, history_grid, history_ref_traj)
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
        goal_dist = jnp.linalg.norm(current_state[:3] - GOAL_POS)
        goal_reached = goal_dist < GOAL_DONE_THRESHOLD
        # Map explored: fraction of high-uncertainty cells (entropy proxy > 0.5)
        uncertainty = 4.0 * grid * (1.0 - grid)  # 1.0 at p=0.5, ~0 at known
        unknown_frac = jnp.mean(uncertainty > 0.5)
        map_explored = unknown_frac < 0.05  # <5% unknown cells remain
        done_step = jnp.where(
            (done_step == 0) & goal_reached & map_explored, t, done_step
        )
        is_done = done_step > 0

        # --- FOV grid update ---
        pos_xy = current_state[:2]
        yaw = quat_to_yaw(current_state[6:10])
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
        uav_pos = current_state[:3]
        new_field, new_origin = jax.lax.cond(
            do_update,
            lambda: compute_info_field(
                fsmi_module, updated_grid, uav_pos, info_field_config
            ),
            lambda: (info_field, field_origin),
        )

        # Generate gradient-ascent reference trajectory
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
                uav_pos[2],
            ),
            lambda: ref_traj,
        )

        # --- Cost function ---
        def cost_fn(x, u, t_step):
            cost = informative_running_cost(
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
                x[:3] - GOAL_POS
            )
            return cost + field_cost + goal_cost

        # --- Dynamics: pure quadrotor (13D) ---
        # MPPI calls dynamics(state, action, t) with step_dependent_dynamics=True
        def dynamics_fn(state, action, _t=None):
            return step_quadrotor(state, action, DT)

        # --- MPPI ---
        action, next_ctrl_state = mppi.command(
            mppi_config,
            current_ctrl_state,
            current_state,
            dynamics_fn,
            cost_fn,
        )

        # Hover when done
        action = jnp.where(is_done, U_INIT, action)

        next_state = step_quadrotor(current_state, action, DT)
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
            fsmi_module, initial_grid, initial_state[:3], info_field_config
        )
        init_ref_traj = field_gradient_trajectory(
            init_field,
            init_origin,
            info_field_config.field_res,
            initial_state[:2],
            info_field_config.ref_horizon,
            info_field_config.ref_speed,
            DT,
            initial_state[2],
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
