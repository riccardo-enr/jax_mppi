"""UGV-specific environment: cost functions and dynamics wrappers.

Adapts the UAV I-MPPI environment for a differential-drive UGV.
State (5D): [x, y, θ, v, ω]  —  no altitude, no quaternions.
Reuses the same map, walls, info zones, and sensor model.
"""

from typing import Optional

import jax
import jax.numpy as jnp

from jax_mppi.dynamics.ugv import rk4_step
from jax_mppi.i_mppi.environment import (
    DEPLETION_ALPHA,
    GOAL_POS,
    INFO_ZONES,
    SENSOR_FOV_RAD,
    SENSOR_MAX_RANGE,
    WALLS,
    _COLLISION_PENALTY,
    _OCC_THRESHOLD,
    _ROBOT_RADIUS,
    _fov_coverage,
    _fov_coverage_with_los,
    _wall_cost,
)

# 2D goal (no altitude)
GOAL_POS_2D = GOAL_POS[:2]

# UGV physics defaults
_MAX_V = 2.0
_MAX_OMEGA = 2.0
_DRAG = 0.1
_U_MIN = jnp.array([-1.0, -2.0])  # [max_accel, max_alpha]
_U_MAX = jnp.array([1.0, 2.0])
_ACTION_REG = 0.01


def step_ugv(state, action, dt, max_v=_MAX_V, max_omega=_MAX_OMEGA, drag=_DRAG):
    """Advance UGV state by one RK4 step with clamped controls."""
    action_clipped = jnp.clip(action, _U_MIN, _U_MAX)
    return rk4_step(state, action_clipped, dt, max_v, max_omega, drag)


def ugv_informative_running_cost(
    state: jax.Array,
    action: jax.Array,
    t: int,
    target: jax.Array,
    grid_map: jax.Array,
    uniform_fsmi_fn,
    info_weight: float = 5.0,
    grid_origin: Optional[jax.Array] = None,
    grid_resolution: Optional[float] = None,
    target_weight: float = 1.0,
) -> jax.Array:
    """Running cost for UGV with Uniform-FSMI informative term.

    Same structure as the UAV ``informative_running_cost`` but adapted for
    the 5D UGV state: direct yaw access, no altitude cost.

    Args:
        state: [x, y, θ, v, ω] (5D)
        action: [a, α] (2D)
        t: Time step index
        target: Reference trajectory (horizon, 2) or (2,)
        grid_map: (H, W) occupancy probability grid
        uniform_fsmi_fn: Callable computing local information gain
        info_weight: Weight for informative term
        grid_origin: (2,) world coords of grid[0,0]
        grid_resolution: meters/cell
        target_weight: Reference tracking weight

    Returns:
        Scalar cost value
    """
    pos_xy = state[:2]
    yaw = state[2]

    # Wall collision cost
    coll_cost = _wall_cost(jnp.array([pos_xy[0], pos_xy[1], 0.0]))

    # Target tracking (2D)
    if target.ndim == 2:
        target_pos = target[t]
    else:
        target_pos = target
    dist_target = jnp.linalg.norm(pos_xy - target_pos[:2])
    target_cost = target_weight * dist_target

    # Bounds cost
    bounds_cost = jnp.float32(0.0)
    bounds_cost += jnp.where(pos_xy[0] < -1.0, _COLLISION_PENALTY, 0.0)
    bounds_cost += jnp.where(pos_xy[0] > 14.0, _COLLISION_PENALTY, 0.0)
    bounds_cost += jnp.where(pos_xy[1] < -1.0, _COLLISION_PENALTY, 0.0)
    bounds_cost += jnp.where(pos_xy[1] > 11.0, _COLLISION_PENALTY, 0.0)

    # Uniform-FSMI local information gain
    info_gain = uniform_fsmi_fn(grid_map, pos_xy, yaw)
    info_cost = -info_weight * info_gain

    # Grid-based obstacle cost
    if grid_origin is not None and grid_resolution is not None:
        col_f = (pos_xy[0] - grid_origin[0]) / grid_resolution
        row_f = (pos_xy[1] - grid_origin[1]) / grid_resolution
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
        + info_cost
        + _ACTION_REG * jnp.sum(action**2)
    )
