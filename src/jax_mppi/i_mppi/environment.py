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

# Info sources: [x, y, radius, initial_value]
INFO_ZONES = jnp.array([
    [6.0, 2.0, 1.5, 100.0],  # Bottom info zone
    [6.0, 8.0, 1.5, 100.0],  # Top info zone
])

GOAL_POS = jnp.array([9.0, 5.0, -2.0])  # x, y, z (z is neg altitude)


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
    # Quadrotor params
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

    # Info Dynamics
    # Info depletes if robot is close
    pos = quad_state[:3]

    def update_info(info_val, zone_idx):
        zone_pos = INFO_ZONES[zone_idx, :2]  # x, y
        radius = INFO_ZONES[zone_idx, 2]
        dist = jnp.linalg.norm(pos[:2] - zone_pos)
        # Simple depletion: if inside radius, reduce by rate * dt
        # Smooth depletion: rate * exp(-dist^2 / radius^2)
        rate = 20.0  # info per second
        depletion = rate * dt * jnp.exp(-(dist**2) / (0.5 * radius) ** 2)
        return jnp.maximum(0.0, info_val - depletion)

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

    # Info Cost (Reward)
    def get_info_rate(p, inf):
        rate_sum = 0.0
        for i in range(len(INFO_ZONES)):
            zone_pos = INFO_ZONES[i, :2]
            radius = INFO_ZONES[i, 2]
            dist = jnp.linalg.norm(p[:2] - zone_pos)
            # Depletion rate (matches dynamics)
            dr = 20.0 * jnp.exp(-(dist**2) / (0.5 * radius) ** 2)
            # Soft check if info exists
            has_info = jnp.tanh(inf[i])
            rate_sum += has_info * dr

        return rate_sum

    info_gain = get_info_rate(pos, info)
    info_cost = -10.0 * info_gain

    # Target Attraction
    dist_target = jnp.linalg.norm(pos - target)
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
