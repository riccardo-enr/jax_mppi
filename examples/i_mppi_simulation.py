"""I-MPPI Simulation: Informative Model Predictive Path Integral.

This script implements the simulation described in docs/i_mppi.qmd.
It features:
- Layer 2: FSMI-driven trajectory generation (simulated by a state machine).
- Layer 3: Biased-MPPI control with mixture sampling.
- Environment: 2D corridor with walls and depletable information sources.
"""

import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import os

from jax_mppi.mppi import create
from examples.i_mppi_modules.fsmi import FSMITrajectoryGenerator, FSMIConfig
from examples.i_mppi_modules.planner import biased_mppi_command
from examples.i_mppi_modules.environment import augmented_dynamics, running_cost, INFO_ZONES, WALLS, GOAL_POS

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

    # Layer 2 Setup
    fsmi_config = FSMIConfig(info_threshold=20.0, goal_pos=GOAL_POS)
    fsmi_planner = FSMITrajectoryGenerator(fsmi_config, INFO_ZONES)

    # Run loop
    sim_steps = 300 # 15 seconds
    history_x = []
    history_info = []
    targets = []

    print("Starting simulation loop...")

    for t in range(sim_steps):
        # --- Layer 2: FSMI Logic ---
        current_info = state[13:]
        target_pos, target_mode = fsmi_planner.get_target(current_info)
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
        state = augmented_dynamics(state, action, dt=dt)

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
