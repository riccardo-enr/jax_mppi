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
    [2.5, 6.0, 3.0, 4.0, 100.0],  # Bottom-left room (high info)
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

# Per-step depletion scaling factor.  info *= (1 - DEPLETION_ALPHA * coverage)
# With alpha=0.02 and full coverage, ~50 steps (1 s at 50 Hz) to halve the info.
DEPLETION_ALPHA = 0.02

# Cost function constants
_COLLISION_PENALTY = 1000.0
_ROBOT_RADIUS = 0.3
_ACTION_REG = 0.01
_HEIGHT_WEIGHT = 10.0
_TARGET_ALTITUDE = -2.0
_OCC_THRESHOLD = 0.7

# Quadrotor physics constants
_MASS = 1.0
_GRAVITY = 9.81
_TAU_OMEGA = 0.05
_U_MIN = jnp.array([0.0, -10.0, -10.0, -10.0])
_U_MAX = jnp.array([4.0 * 9.81, 10.0, 10.0, 10.0])


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


def _fov_coverage_with_los(
    pos_xy, yaw, zone_center, zone_size, grid, grid_origin, grid_resolution
):
    """Like ``_fov_coverage`` but each sample also requires line-of-sight.

    Each sample point is weighted by the uncertainty (entropy proxy) of the
    corresponding grid cell so that already-known cells (occupied or free)
    do not contribute to information depletion.
    """
    half = zone_size / 2.0
    n = _COV_SAMPLES
    ts = jnp.linspace(-1.0, 1.0, n)
    gx, gy = jnp.meshgrid(
        zone_center[0] + ts * half[0],
        zone_center[1] + ts * half[1],
    )
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)

    half_fov = SENSOR_FOV_RAD / 2.0
    H, W = grid.shape

    def _visible(pt):
        vec = pt - pos_xy
        d = jnp.linalg.norm(vec)
        bearing = jnp.arctan2(vec[1], vec[0])
        ang = jnp.arctan2(jnp.sin(bearing - yaw), jnp.cos(bearing - yaw))
        in_fov = jnp.abs(ang) <= half_fov
        in_range = d <= SENSOR_MAX_RANGE
        los = _line_of_sight_grid(
            pos_xy, pt, grid, grid_origin, grid_resolution
        )

        # Weight by cell uncertainty: only unknown cells (p≈0.5) contribute.
        # Known occupied (p≈0.9) and known free (p≈0.2) contribute little.
        col = jnp.int32(jnp.floor((pt[0] - grid_origin[0]) / grid_resolution))
        row = jnp.int32(jnp.floor((pt[1] - grid_origin[1]) / grid_resolution))
        row = jnp.clip(row, 0, H - 1)
        col = jnp.clip(col, 0, W - 1)
        p = grid[row, col]
        # Entropy proxy: 1.0 at p=0.5, 0.0 at p=0 or p=1
        uncertainty = 4.0 * p * (1.0 - p)

        return (in_fov & in_range).astype(jnp.float32) * los * uncertainty

    return jnp.mean(jax.vmap(_visible)(pts))


def _step_quadrotor(quad_state, action, dt):
    """Advance quadrotor state by one RK4 step with clamped controls."""
    action_clipped = jnp.clip(action, _U_MIN, _U_MAX)
    next_quad_state = rk4_step(
        quad_state, action_clipped, dt, _MASS, _GRAVITY, _TAU_OMEGA
    )
    next_quat = normalize_quaternion(next_quad_state[6:10])
    return next_quad_state.at[6:10].set(next_quat)


def _wall_cost(pos):
    """Collision cost for segment-based walls."""
    x, y = pos[0], pos[1]
    cost = 0.0
    for w in WALLS:
        min_x, max_x = jnp.minimum(w[0], w[2]), jnp.maximum(w[0], w[2])
        min_y, max_y = jnp.minimum(w[1], w[3]), jnp.maximum(w[1], w[3])
        in_x = (x >= min_x - _ROBOT_RADIUS) & (x <= max_x + _ROBOT_RADIUS)
        in_y = (y >= min_y - _ROBOT_RADIUS) & (y <= max_y + _ROBOT_RADIUS)
        cost += jnp.where(in_x & in_y, _COLLISION_PENALTY, 0.0)
    return cost


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
    blocked = (occupancy >= _OCC_THRESHOLD) & valid

    return jnp.where(jnp.any(blocked), 0.0, 1.0)


@partial(jax.jit, static_argnames=["dt"])
def augmented_dynamics(
    state: jax.Array,
    action: jax.Array,
    t: Optional[jax.Array] = None,
    dt: float = 0.05,
) -> jax.Array:
    """Dynamics for Quadrotor + Info levels."""
    quad_state = state[:13]
    info_levels = state[13:]

    next_quad_state = _step_quadrotor(quad_state, action, dt)

    # Info Dynamics — deplete proportionally to FOV coverage of the zone
    pos = quad_state[:3]
    yaw = quat_to_yaw(quad_state[6:10])

    def update_info(info_val, zone_idx):
        zone_center = INFO_ZONES[zone_idx, :2]
        zone_size = INFO_ZONES[zone_idx, 2:4]

        coverage = _fov_coverage(pos[:2], yaw, zone_center, zone_size)
        return info_val * (1.0 - DEPLETION_ALPHA * coverage)

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

    next_quad_state = _step_quadrotor(quad_state, action, dt)

    pos = quad_state[:3]
    yaw = quat_to_yaw(quad_state[6:10])

    def update_info(info_val, zone_idx):
        zone_center = INFO_ZONES[zone_idx, :2]
        zone_size = INFO_ZONES[zone_idx, 2:4]

        coverage = _fov_coverage_with_los(
            pos[:2],
            yaw,
            zone_center,
            zone_size,
            grid,
            grid_origin,
            grid_resolution,
        )
        return info_val * (1.0 - DEPLETION_ALPHA * coverage)

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
    pos = state[:3]
    info = state[13:]

    coll_cost = _wall_cost(pos)

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
    bounds_cost += jnp.where(pos[0] < -1.0, _COLLISION_PENALTY, 0.0)
    bounds_cost += jnp.where(pos[0] > 14.0, _COLLISION_PENALTY, 0.0)
    bounds_cost += jnp.where(pos[1] < -1.0, _COLLISION_PENALTY, 0.0)
    bounds_cost += jnp.where(pos[1] > 11.0, _COLLISION_PENALTY, 0.0)

    # Height cost
    height_cost = _HEIGHT_WEIGHT * (pos[2] - _TARGET_ALTITUDE) ** 2

    return (
        coll_cost
        + info_cost
        + bounds_cost
        + height_cost
        + target_cost
        + _ACTION_REG * jnp.sum(action**2)
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

    coll_cost = _wall_cost(pos)

    # Target Attraction (tracking Layer 2 reference)
    if target.ndim == 2:
        target_pos = target[t]
    else:
        target_pos = target
    dist_target = jnp.linalg.norm(pos - target_pos)
    target_cost = 1.0 * dist_target

    # Bounds cost
    bounds_cost = 0.0
    bounds_cost += jnp.where(pos[0] < -1.0, _COLLISION_PENALTY, 0.0)
    bounds_cost += jnp.where(pos[0] > 14.0, _COLLISION_PENALTY, 0.0)
    bounds_cost += jnp.where(pos[1] < -1.0, _COLLISION_PENALTY, 0.0)
    bounds_cost += jnp.where(pos[1] > 11.0, _COLLISION_PENALTY, 0.0)

    # Height cost
    height_cost = _HEIGHT_WEIGHT * (pos[2] - _TARGET_ALTITUDE) ** 2

    # === Uniform-FSMI Information Term (Layer 3) ===
    yaw = quat_to_yaw(state[6:10])

    # Compute local information gain using Uniform-FSMI
    info_gain = uniform_fsmi_fn(grid_map, pos[:2], yaw)

    # Cost = tracking + obstacles - lambda * info_gain
    info_cost = -info_weight * info_gain

    # Grid-based obstacle cost (supplements _wall_cost with full grid)
    if grid_origin is not None and grid_resolution is not None:
        col_f = (pos[0] - grid_origin[0]) / grid_resolution
        row_f = (pos[1] - grid_origin[1]) / grid_resolution
        row_i = jnp.clip(jnp.int32(jnp.floor(row_f)), 0, grid_map.shape[0] - 1)
        col_i = jnp.clip(jnp.int32(jnp.floor(col_f)), 0, grid_map.shape[1] - 1)
        grid_val = grid_map[row_i, col_i]
        grid_obstacle_cost = _COLLISION_PENALTY * jnp.where(
            grid_val >= _OCC_THRESHOLD, 1.0, 0.0
        )
    else:
        grid_obstacle_cost = 0.0

    return (
        coll_cost
        + grid_obstacle_cost
        + target_cost
        + bounds_cost
        + height_cost
        + info_cost
        + _ACTION_REG * jnp.sum(action**2)
    )
