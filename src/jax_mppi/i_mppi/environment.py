from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from jax_mppi.dynamics.quadrotor import normalize_quaternion, rk4_step

# --- Environment Configuration ---
WALLS = jnp.array([
    # [x1, y1, x2, y2]
    [0.0, 2.0, 4.0, 2.0],  # Bottom wall of first segment (Horizontal)
    [0.0, 8.0, 4.0, 8.0],  # Top wall of first segment (Horizontal)
    [4.0, 2.0, 4.0, 0.0],  # Corner down (Vertical)
    [4.0, 8.0, 4.0, 10.0],  # Corner up (Vertical)
    [4.0, 0.0, 12.0, 0.0],  # Bottom long wall (Horizontal)
    [4.0, 10.0, 12.0, 10.0],  # Top long wall (Horizontal)
    [12.0, 0.0, 12.0, 10.0],  # End wall (Vertical)
])

# Info sources: [cx, cy, width, height, initial_value]
# These should match the unknown regions in the occupancy grid
# The grid has unknown rooms at:
#   - Bottom-left room: x=1-4m, y=4-8m (center ~2.5, 6)
#   - Bottom-right room: x=10-13m, y=4-8m (center ~11.5, 6)
#   - Top-right room: x=10-13m, y=1-3m (center ~11.5, 2)
INFO_ZONES = jnp.array([
    [2.5, 6.0, 3.0, 4.0, 100.0],   # Bottom-left room (high info)
    [11.5, 6.0, 3.0, 4.0, 100.0],  # Bottom-right room (high info)
    [11.5, 2.0, 3.0, 2.0, 100.0],  # Top-right room (high info)
])

GOAL_POS = jnp.array([9.0, 5.0, -2.0])  # x, y, z (z is neg altitude)

# Sensor parameters for FOV-aware depletion
SENSOR_FOV_RAD = 1.57  # 90 degrees, matches FSMI configs
SENSOR_MAX_RANGE = 2.5  # metres, matches Layer 3 uniform FSMI

_LOS_RAY_STEPS = 16  # fixed sample count for line-of-sight check
_LOS_RAY_STEP = 0.2  # metres between LOS samples
_COV_SAMPLES = 5  # NxN grid for FOV coverage estimation


def dist_rect(p, center, size):
    """Calculate distance to a rectangle (0 if inside)."""
    # p: [x, y], center: [cx, cy], size: [w, h]
    half_size = size / 2.0
    d = jnp.abs(p - center) - half_size
    # exterior distance
    return jnp.linalg.norm(jnp.maximum(d, 0.0))


def quat_to_yaw(q):
    """Extract yaw from quaternion [qw, qx, qy, qz]."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return jnp.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _fov_coverage(pos_xy, yaw, zone_center, zone_size):
    """Fraction of zone area within the sensor FOV cone and range.

    Samples a grid of points inside the zone and checks each against
    the FOV half-angle and max range.  Returns a scalar in [0, 1].
    """
    half = zone_size / 2.0
    n = _COV_SAMPLES
    # Sample points uniformly across the zone
    ts = jnp.linspace(-1.0, 1.0, n)
    gx, gy = jnp.meshgrid(
        zone_center[0] + ts * half[0],
        zone_center[1] + ts * half[1],
    )
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)  # (n*n, 2)

    half_fov = SENSOR_FOV_RAD / 2.0

    def _visible(pt):
        vec = pt - pos_xy
        d = jnp.linalg.norm(vec)
        bearing = jnp.arctan2(vec[1], vec[0])
        ang = jnp.arctan2(jnp.sin(bearing - yaw), jnp.cos(bearing - yaw))
        in_fov = jnp.abs(ang) <= half_fov
        in_range = d <= SENSOR_MAX_RANGE
        return (in_fov & in_range).astype(jnp.float32)

    return jnp.mean(jax.vmap(_visible)(pts))


def _fov_coverage_with_los(pos_xy, yaw, zone_center, zone_size,
                           grid, grid_origin, grid_resolution):
    """Like ``_fov_coverage`` but each sample also requires line-of-sight."""
    half = zone_size / 2.0
    n = _COV_SAMPLES
    ts = jnp.linspace(-1.0, 1.0, n)
    gx, gy = jnp.meshgrid(
        zone_center[0] + ts * half[0],
        zone_center[1] + ts * half[1],
    )
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)

    half_fov = SENSOR_FOV_RAD / 2.0

    def _visible(pt):
        vec = pt - pos_xy
        d = jnp.linalg.norm(vec)
        bearing = jnp.arctan2(vec[1], vec[0])
        ang = jnp.arctan2(jnp.sin(bearing - yaw), jnp.cos(bearing - yaw))
        in_fov = jnp.abs(ang) <= half_fov
        in_range = d <= SENSOR_MAX_RANGE
        los = _line_of_sight_grid(pos_xy, pt, grid, grid_origin, grid_resolution)
        return (in_fov & in_range).astype(jnp.float32) * los

    return jnp.mean(jax.vmap(_visible)(pts))


def _line_of_sight_grid(p, q, grid, origin, resolution):
    """Check line of sight from *p* to *q* on the occupancy grid.

    Samples the grid along a straight ray. Returns 1.0 if clear,
    0.0 if any sample hits an obstacle cell (occupancy >= 0.7).
    """
    direction = q - p
    dist = jnp.linalg.norm(direction)
    safe_dist = jnp.maximum(dist, 1e-6)
    dir_norm = direction / safe_dist

    # Fixed-size sample array; mask out samples past the target
    step_dists = jnp.arange(_LOS_RAY_STEPS) * _LOS_RAY_STEP
    valid = step_dists < dist

    xs = p[0] + dir_norm[0] * step_dists
    ys = p[1] + dir_norm[1] * step_dists

    cols = jnp.int32(jnp.floor((xs - origin[0]) / resolution))
    rows = jnp.int32(jnp.floor((ys - origin[1]) / resolution))
    rows = jnp.clip(rows, 0, grid.shape[0] - 1)
    cols = jnp.clip(cols, 0, grid.shape[1] - 1)

    occupancy = grid[rows, cols]
    blocked = (occupancy >= 0.7) & valid

    return jnp.where(jnp.any(blocked), 0.0, 1.0)


@partial(jax.jit, static_argnames=["dt"])
def augmented_dynamics(
    state: jax.Array,
    action: jax.Array,
    t: Optional[jax.Array] = None,
    dt: float = 0.05,
) -> jax.Array:
    """Dynamics for Quadrotor + Info levels."""
    # Split state
    quad_state = state[:13]
    info_levels = state[13:]

    # Quadrotor dynamics
    mass = 1.0
    gravity = 9.81
    tau_omega = 0.05
    u_min = jnp.array([0.0, -10.0, -10.0, -10.0])
    u_max = jnp.array([4.0 * 9.81, 10.0, 10.0, 10.0])

    action_clipped = jnp.clip(action, u_min, u_max)
    next_quad_state = rk4_step(
        quad_state, action_clipped, dt, mass, gravity, tau_omega
    )
    next_quat = normalize_quaternion(next_quad_state[6:10])
    next_quad_state = next_quad_state.at[6:10].set(next_quat)

    # Info Dynamics — deplete proportionally to FOV coverage of the zone
    pos = quad_state[:3]
    yaw = quat_to_yaw(quad_state[6:10])

    def update_info(info_val, zone_idx):
        zone_center = INFO_ZONES[zone_idx, :2]
        zone_size = INFO_ZONES[zone_idx, 2:4]

        coverage = _fov_coverage(pos[:2], yaw, zone_center, zone_size)
        return info_val * (1.0 - coverage)

    next_info_levels = jax.vmap(update_info)(
        info_levels, jnp.arange(len(INFO_ZONES))
    )

    return jnp.concatenate([next_quad_state, next_info_levels])


def augmented_dynamics_with_grid(
    state: jax.Array,
    action: jax.Array,
    t: Optional[jax.Array] = None,
    *,
    dt: float = 0.05,
    grid: jax.Array,
    grid_origin: jax.Array,
    grid_resolution: float,
) -> jax.Array:
    """Dynamics for Quadrotor + Info levels with grid-based LOS check.

    Same as ``augmented_dynamics`` but rays that hit an obstacle cell
    (occupancy >= 0.7) in *grid* block information depletion.
    """
    quad_state = state[:13]
    info_levels = state[13:]

    mass = 1.0
    gravity = 9.81
    tau_omega = 0.05
    u_min = jnp.array([0.0, -10.0, -10.0, -10.0])
    u_max = jnp.array([4.0 * 9.81, 10.0, 10.0, 10.0])

    action_clipped = jnp.clip(action, u_min, u_max)
    next_quad_state = rk4_step(
        quad_state, action_clipped, dt, mass, gravity, tau_omega
    )
    next_quat = normalize_quaternion(next_quad_state[6:10])
    next_quad_state = next_quad_state.at[6:10].set(next_quat)

    pos = quad_state[:3]
    yaw = quat_to_yaw(quad_state[6:10])

    def update_info(info_val, zone_idx):
        zone_center = INFO_ZONES[zone_idx, :2]
        zone_size = INFO_ZONES[zone_idx, 2:4]

        coverage = _fov_coverage_with_los(
            pos[:2], yaw, zone_center, zone_size,
            grid, grid_origin, grid_resolution,
        )
        return info_val * (1.0 - coverage)

    next_info_levels = jax.vmap(update_info)(
        info_levels, jnp.arange(len(INFO_ZONES))
    )

    return jnp.concatenate([next_quad_state, next_info_levels])


@jax.jit
def running_cost(
    state: jax.Array,
    action: jax.Array,
    t: int,
    target: jax.Array,
) -> jax.Array:
    # State: [px, py, pz, vx, vy, vz, ..., info1, info2]
    pos = state[:3]
    info = state[13:]

    # Obstacle Cost
    def wall_cost_fn(p):
        x, y = p[0], p[1]
        cost = 0.0
        # Iterate over walls
        for w in WALLS:
            # w: [x1, y1, x2, y2]
            min_x, max_x = jnp.minimum(w[0], w[2]), jnp.maximum(w[0], w[2])
            min_y, max_y = jnp.minimum(w[1], w[3]), jnp.maximum(w[1], w[3])

            # Expanded by robot radius (e.g. 0.3)
            margin = 0.3
            in_x = jnp.logical_and(x >= min_x - margin, x <= max_x + margin)
            in_y = jnp.logical_and(y >= min_y - margin, y <= max_y + margin)
            in_wall = jnp.logical_and(in_x, in_y)

            cost += jnp.where(in_wall, 1000.0, 0.0)
        return cost

    coll_cost = wall_cost_fn(pos)

    # Info Cost (Reward) — FOV-aware (matches dynamics)
    yaw = quat_to_yaw(state[6:10])

    def get_info_reward(p, inf):
        reward = 0.0
        for i in range(len(INFO_ZONES)):
            zone_center = INFO_ZONES[i, :2]
            zone_size = INFO_ZONES[i, 2:4]

            coverage = _fov_coverage(p[:2], yaw, zone_center, zone_size)
            has_info = jnp.tanh(inf[i])
            reward += has_info * coverage

        return reward

    info_gain = get_info_reward(pos, info)
    info_cost = -50.0 * info_gain

    # Target Attraction
    if target.ndim == 2:
        target_pos = target[t]
    else:
        target_pos = target
    dist_target = jnp.linalg.norm(pos - target_pos)
    target_cost = 1.0 * dist_target

    # Stay within bounds
    bounds_cost = 0.0
    bounds_cost += jnp.where(pos[0] < -1.0, 1000.0, 0.0)
    bounds_cost += jnp.where(pos[0] > 14.0, 1000.0, 0.0)
    bounds_cost += jnp.where(pos[1] < -1.0, 1000.0, 0.0)
    bounds_cost += jnp.where(pos[1] > 11.0, 1000.0, 0.0)

    # Height cost (hold -2.0)
    height_cost = 10.0 * (pos[2] - (-2.0)) ** 2

    return (
        coll_cost
        + info_cost
        + bounds_cost
        + height_cost
        + target_cost
        + 0.01 * jnp.sum(action**2)
    )


def informative_running_cost(
    state: jax.Array,
    action: jax.Array,
    t: int,
    target: jax.Array,
    grid_map: jax.Array,
    uniform_fsmi_fn,
    info_weight: float = 5.0,
    grid_origin: Optional[jax.Array] = None,
    grid_resolution: Optional[float] = None,
) -> jax.Array:
    """
    Running cost with Layer 3 Uniform-FSMI informative term.

    This implements the I-MPPI cost function:
        J = tracking_cost + obstacles - λ * Uniform_FSMI(local)

    The Uniform-FSMI term ensures the controller maintains informative
    viewpoints even when tracking the Layer 2 reference trajectory.

    Args:
        state: Full state [pos(3), vel(3), quat(4), omega(3), info(N)]
        action: Control input [thrust, omega_x, omega_y, omega_z]
        t: Time step index
        target: Reference trajectory from Layer 2 (horizon, 3) or (3,)
        grid_map: (H, W) occupancy probability grid
        uniform_fsmi_fn: Callable computing local information gain
        info_weight: Weight for informative term (default 5.0)

    Returns:
        Scalar cost value
    """
    pos = state[:3]
    vel = state[3:6]

    # Obstacle Cost (same as running_cost)
    def wall_cost_fn(p):
        x, y = p[0], p[1]
        cost = 0.0
        for w in WALLS:
            min_x, max_x = jnp.minimum(w[0], w[2]), jnp.maximum(w[0], w[2])
            min_y, max_y = jnp.minimum(w[1], w[3]), jnp.maximum(w[1], w[3])
            margin = 0.3
            in_x = jnp.logical_and(x >= min_x - margin, x <= max_x + margin)
            in_y = jnp.logical_and(y >= min_y - margin, y <= max_y + margin)
            in_wall = jnp.logical_and(in_x, in_y)
            cost += jnp.where(in_wall, 1000.0, 0.0)
        return cost

    coll_cost = wall_cost_fn(pos)

    # Target Attraction (tracking Layer 2 reference)
    if target.ndim == 2:
        target_pos = target[t]
    else:
        target_pos = target
    dist_target = jnp.linalg.norm(pos - target_pos)
    target_cost = 1.0 * dist_target

    # Bounds cost
    bounds_cost = 0.0
    bounds_cost += jnp.where(pos[0] < -1.0, 1000.0, 0.0)
    bounds_cost += jnp.where(pos[0] > 14.0, 1000.0, 0.0)
    bounds_cost += jnp.where(pos[1] < -1.0, 1000.0, 0.0)
    bounds_cost += jnp.where(pos[1] > 11.0, 1000.0, 0.0)

    # Height cost
    height_cost = 10.0 * (pos[2] - (-2.0)) ** 2

    # === Uniform-FSMI Information Term (Layer 3) ===
    # Compute yaw from velocity direction or target direction
    vel_xy = vel[:2]
    vel_norm = jnp.linalg.norm(vel_xy)
    dir_xy = target_pos[:2] - pos[:2]
    dir_norm = jnp.linalg.norm(dir_xy)

    # Use velocity direction if moving, else target direction
    yaw = jnp.where(
        vel_norm > 0.1,
        jnp.arctan2(vel_xy[1], vel_xy[0]),
        jnp.arctan2(dir_xy[1], dir_xy[0]),
    )

    # Compute local information gain using Uniform-FSMI
    info_gain = uniform_fsmi_fn(grid_map, pos[:2], yaw)

    # Cost = tracking + obstacles - λ * info_gain
    info_cost = -info_weight * info_gain

    # Grid-based obstacle cost (supplements wall_cost_fn with full grid)
    if grid_origin is not None and grid_resolution is not None:
        col_f = (pos[0] - grid_origin[0]) / grid_resolution
        row_f = (pos[1] - grid_origin[1]) / grid_resolution
        row_i = jnp.clip(jnp.int32(jnp.floor(row_f)), 0, grid_map.shape[0] - 1)
        col_i = jnp.clip(jnp.int32(jnp.floor(col_f)), 0, grid_map.shape[1] - 1)
        grid_val = grid_map[row_i, col_i]
        grid_obstacle_cost = 1000.0 * jnp.where(grid_val >= 0.7, 1.0, 0.0)
    else:
        grid_obstacle_cost = 0.0

    return (
        coll_cost
        + grid_obstacle_cost
        + target_cost
        + bounds_cost
        + height_cost
        + info_cost
        + 0.01 * jnp.sum(action**2)
    )
