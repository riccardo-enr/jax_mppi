"""I-MPPI Simulation: Informative Model Predictive Path Integral.

This script implements the simulation described in docs/i_mppi.qmd.
It features:
- Layer 2: FSMI-driven trajectory generation (simulated by a state machine).
- Layer 3: Biased-MPPI control with mixture sampling.
- Environment: 2D corridor with walls and depletable information sources.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable
from dataclasses import dataclass, replace
from functools import partial
import matplotlib.pyplot as plt
import os

from jax_mppi.mppi import MPPIConfig, MPPIState, _shift_nominal, _bound_action, _scaled_bounds, _compute_noise_cost, _compute_weights, _state_for_cost, create, _compute_rollout_costs
from jax_mppi.dynamics.quadrotor import create_quadrotor_dynamics, rk4_step, normalize_quaternion

# --- Environment Configuration ---
WALLS = jnp.array([
    # [x1, y1, x2, y2]
    [0.0, 2.0, 4.0, 2.0],    # Bottom wall of first segment (Horizontal)
    [0.0, 8.0, 4.0, 8.0],    # Top wall of first segment (Horizontal)
    [4.0, 2.0, 4.0, 0.0],    # Corner down (Vertical)
    [4.0, 8.0, 4.0, 10.0],   # Corner up (Vertical)
    [4.0, 0.0, 12.0, 0.0],   # Bottom long wall (Horizontal)
    [4.0, 10.0, 12.0, 10.0], # Top long wall (Horizontal)
    [12.0, 0.0, 12.0, 10.0], # End wall (Vertical)
])

# Info sources: [x, y, radius, initial_value]
INFO_ZONES = jnp.array([
    [6.0, 2.0, 1.5, 100.0],  # Bottom info zone
    [6.0, 8.0, 1.5, 100.0],  # Top info zone
])

GOAL_POS = jnp.array([9.0, 5.0, -2.0]) # x, y, z (z is neg altitude)

# --- Augmented Dynamics ---
# State: [13 quadrotor, 2 info_levels] = 15 dims

@partial(jax.jit, static_argnames=["dt"])
def augmented_dynamics(
    state: jax.Array,
    action: jax.Array,
    t: Optional[jax.Array] = None,
    dt: float = 0.05
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
    next_quad_state = rk4_step(quad_state, action_clipped, dt, mass, gravity, tau_omega)
    next_quat = normalize_quaternion(next_quad_state[6:10])
    next_quad_state = next_quad_state.at[6:10].set(next_quat)

    # Info Dynamics
    # Info depletes if robot is close
    pos = quad_state[:3]

    def update_info(info_val, zone_idx):
        zone_pos = INFO_ZONES[zone_idx, :2] # x, y
        radius = INFO_ZONES[zone_idx, 2]
        dist = jnp.linalg.norm(pos[:2] - zone_pos)
        # Simple depletion: if inside radius, reduce by rate * dt
        # Smooth depletion: rate * exp(-dist^2 / radius^2)
        rate = 20.0 # info per second
        depletion = rate * dt * jnp.exp(-dist**2 / (0.5 * radius)**2)
        return jnp.maximum(0.0, info_val - depletion)

    next_info_levels = jax.vmap(update_info)(info_levels, jnp.arange(len(INFO_ZONES)))

    return jnp.concatenate([next_quad_state, next_info_levels])


# --- Biased MPPI Logic ---

def biased_mppi_command(
    config: MPPIConfig,
    mppi_state: MPPIState,
    current_obs: jax.Array,
    dynamics: Callable,
    running_cost: Callable,
    U_ref: jax.Array, # The reference trajectory to bias towards (T, nu)
    bias_alpha: float = 0.5, # Mixture weight for biased samples
    terminal_cost: Optional[Callable] = None,
    shift: bool = True,
) -> Tuple[jax.Array, MPPIState]:
    """Biased MPPI command with mixture sampling."""

    key, subkey1, subkey2 = jax.random.split(mppi_state.key, 3)

    K = config.num_samples
    K_biased = int(K * bias_alpha)
    K_nominal = K - K_biased

    # Sample nominal noise
    noise_nominal = jax.random.multivariate_normal(
        subkey1,
        mean=mppi_state.noise_mu,
        cov=mppi_state.noise_sigma,
        shape=(K_nominal, config.horizon),
    )

    # Sample biased noise
    noise_biased_base = jax.random.multivariate_normal(
        subkey2,
        mean=mppi_state.noise_mu,
        cov=mppi_state.noise_sigma,
        shape=(K_biased, config.horizon),
    )

    # Difference between reference and current nominal
    delta_ref = U_ref - mppi_state.U
    noise_biased = noise_biased_base + delta_ref[None, :, :]

    # Combine noise
    noise = jnp.concatenate([noise_nominal, noise_biased], axis=0)

    perturbed_actions = mppi_state.U[None, :, :] + noise
    scaled_actions = perturbed_actions * config.u_scale
    scaled_actions = _bound_action(
        scaled_actions, mppi_state.u_min, mppi_state.u_max
    )

    # Compute Costs
    rollout_costs = _compute_rollout_costs(
        config,
        current_obs,
        scaled_actions,
        dynamics,
        running_cost,
        terminal_cost,
    )

    noise_costs = _compute_noise_cost(
        noise, mppi_state.noise_sigma_inv, config.noise_abs_cost
    )

    total_costs = rollout_costs + noise_costs

    weights = _compute_weights(total_costs, config.lambda_)

    # Update U
    delta_U = jnp.tensordot(weights, noise, axes=1)
    U_new = mppi_state.U + delta_U

    u_min_scaled, u_max_scaled = _scaled_bounds(
        mppi_state.u_min, mppi_state.u_max, config.u_scale
    )
    U_new = _bound_action(U_new, u_min_scaled, u_max_scaled)

    action_seq = U_new[: config.u_per_command]
    scaled_action_seq = _bound_action(
        action_seq * config.u_scale, mppi_state.u_min, mppi_state.u_max
    )
    action = (
        scaled_action_seq[0] if config.u_per_command == 1 else scaled_action_seq
    )

    new_state = replace(mppi_state, U=U_new, key=key)
    if shift:
        new_state = _shift_nominal(new_state, config.u_per_command)

    return action, new_state


# --- Cost Function ---

@jax.jit
def running_cost(state: jax.Array, action: jax.Array, t: int, target: jax.Array) -> jax.Array:
    # State: [px, py, pz, vx, vy, vz, ..., info1, info2]
    pos = state[:3]
    vel = state[3:6]
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
            dr = 20.0 * jnp.exp(-dist**2 / (0.5 * radius)**2)
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
    height_cost = 10.0 * (pos[2] - (-2.0))**2

    return coll_cost + info_cost + bounds_cost + height_cost + target_cost + 0.01 * jnp.sum(action**2)


# --- Simulation Loop ---

def main():
    print("Setting up I-MPPI simulation...")

    # Config
    dt = 0.05
    nx = 15
    nu = 4
    horizon = 40 # 2 seconds

    # Initial State
    # Quad: [0, 5, -2, ...] (Start at left middle)
    start_pos = jnp.array([1.0, 5.0, -2.0])
    # info levels = max
    info_init = jnp.array([100.0, 100.0])

    x0 = jnp.zeros(13)
    x0 = x0.at[:3].set(start_pos)
    x0 = x0.at[6].set(1.0) # qw=1

    state = jnp.concatenate([x0, info_init])

    # MPPI Setup
    # Noise: Thrust ~ 5.0, Omega ~ 1.0
    noise_sigma = jnp.diag(jnp.array([2.0, 0.5, 0.5, 0.5])**2)

    config, mppi_state = create(
        nx=nx,
        nu=nu,
        noise_sigma=noise_sigma,
        num_samples=200, # reduced for speed
        horizon=horizon,
        lambda_=0.1,
        u_min=jnp.array([0.0, -5.0, -5.0, -5.0]),
        u_max=jnp.array([40.0, 5.0, 5.0, 5.0]),
        u_init=jnp.array([9.81, 0.0, 0.0, 0.0]),
        step_dependent_dynamics=True
    )

    # Run loop
    sim_steps = 300 # 15 seconds
    history_x = []
    history_info = []
    targets = []

    print("Starting simulation loop...")

    for t in range(sim_steps):
        # --- Layer 2: FSMI Logic ---
        # Determine target
        current_info = state[13:]
        info_threshold = 20.0

        target_pos = GOAL_POS # Default
        target_mode = "GOAL"

        if current_info[0] > info_threshold:
            # Go to Info 1 (Bottom)
            target_pos = jnp.array([INFO_ZONES[0,0], INFO_ZONES[0,1], -2.0])
            target_mode = "INFO 1"
        elif current_info[1] > info_threshold:
            # Go to Info 2 (Top)
            target_pos = jnp.array([INFO_ZONES[1,0], INFO_ZONES[1,1], -2.0])
            target_mode = "INFO 2"

        targets.append(target_pos)

        # U_ref = Hover
        U_ref = jnp.tile(mppi_state.u_init, (horizon, 1))

        # --- Layer 3: Biased MPPI ---
        # Bind target to cost function
        cost_fn = partial(running_cost, target=target_pos)

        action, mppi_state = biased_mppi_command(
            config,
            mppi_state,
            state,
            augmented_dynamics,
            cost_fn,
            U_ref,
            bias_alpha=0.2
        )

        # Step Dynamics
        state = augmented_dynamics(state, action, dt)

        history_x.append(state)
        history_info.append(state[13:])

        if t % 20 == 0:
            print(f"Step {t}: Pos={state[:2]}, Info={state[13:]}, Mode={target_mode}")

    # --- Visualization ---
    history_x = jnp.stack(history_x)
    targets = jnp.stack(targets)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot Walls
    for w in WALLS:
        rect = plt.Rectangle((w[0], w[1]), w[2]-w[0], w[3]-w[1], color='gray', alpha=0.5)
        ax.add_patch(rect)

    # Plot Info Zones
    for i in range(len(INFO_ZONES)):
        circle = plt.Circle((INFO_ZONES[i,0], INFO_ZONES[i,1]), INFO_ZONES[i,2], color='yellow', alpha=0.3, label='Info Zone' if i==0 else "")
        ax.add_patch(circle)

    # Plot Goal
    ax.plot(GOAL_POS[0], GOAL_POS[1], 'r*', markersize=15, label='Goal')

    # Plot Start
    ax.plot(start_pos[0], start_pos[1], 'go', label='Start')

    # Plot Trajectory
    ax.plot(history_x[:, 0], history_x[:, 1], 'b-', linewidth=2, label='Trajectory')

    ax.set_xlim(-1, 14)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(f"I-MPPI Simulation\nFinal Info: {history_info[-1]}")

    output_path = "i_mppi_simulation.png"
    plt.savefig(output_path)
    print(f"Simulation complete. Plot saved to {output_path}")

if __name__ == "__main__":
    main()
