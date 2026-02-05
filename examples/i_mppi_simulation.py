"""I-MPPI Simulation: Informative Model Predictive Path Integral.

This script implements the simulation described in docs/i_mppi.qmd.
It features:
- Layer 2: FSMI-driven trajectory generation (simulated by a state machine).
- Layer 3: Biased-MPPI control with mixture sampling.
- Environment: 2D corridor with walls and depletable information sources.
"""

from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax_mppi.i_mppi.environment import (
    GOAL_POS,
    INFO_ZONES,
    WALLS,
    augmented_dynamics,
    running_cost,
)
from jax_mppi.i_mppi.fsmi import FSMIConfig, FSMITrajectoryGenerator
from jax_mppi.i_mppi.planner import biased_mppi_command
from jax_mppi.mppi import create

# --- Simulation Loop ---


def main():
    print("Setting up I-MPPI simulation...")

    # Config
    dt = 0.05
    nx = 15
    nu = 4
    horizon = 40  # 2 seconds

    # Initial State
    # Quad: [0, 5, -2, ...] (Start at left middle)
    start_pos = jnp.array([1.0, 5.0, -2.0])
    # info levels = max
    info_init = jnp.array([100.0, 100.0])

    x0 = jnp.zeros(13)
    x0 = x0.at[:3].set(start_pos)
    x0 = x0.at[6].set(1.0)  # qw=1

    state = jnp.concatenate([x0, info_init])

    # MPPI Setup
    # Noise: Thrust ~ 5.0, Omega ~ 1.0
    noise_sigma = jnp.diag(jnp.array([2.0, 0.5, 0.5, 0.5]) ** 2)

    config, mppi_state = create(
        nx=nx,
        nu=nu,
        noise_sigma=noise_sigma,
        num_samples=1000,  # reduced for speed
        horizon=horizon,
        lambda_=0.1,
        u_min=jnp.array([0.0, -5.0, -5.0, -5.0]),
        u_max=jnp.array([40.0, 5.0, 5.0, 5.0]),
        u_init=jnp.array([9.81, 0.0, 0.0, 0.0]),
        step_dependent_dynamics=True,
    )

    # Layer 2 Setup
    fsmi_config = FSMIConfig(info_threshold=20.0, goal_pos=GOAL_POS)
    fsmi_planner = FSMITrajectoryGenerator(fsmi_config, INFO_ZONES)

    # Run loop
    sim_steps = 300  # 15 seconds
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
            bias_alpha=0.2,
        )

        # Step Dynamics
        state = augmented_dynamics(state, action, dt=dt)

        history_x.append(state)
        history_info.append(state[13:])

        if t % 20 == 0:
            pos_xy = state[:2]
            info_levels = state[13:]
            print(
                f"Step {t}: Pos={pos_xy}, Info={info_levels}, "
                f"Mode={target_mode}"
            )

    # --- Visualization ---
    history_x = jnp.stack(history_x)
    targets = jnp.stack(targets)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    # Plot Walls
    for w in WALLS:
        rect = plt.Rectangle(
            (w[0], w[1]), w[2] - w[0], w[3] - w[1], color="gray", alpha=0.5
        )
        ax.add_patch(rect)

    # Plot Info Zones
    for i in range(len(INFO_ZONES)):
        circle = plt.Circle(
            (INFO_ZONES[i, 0], INFO_ZONES[i, 1]),
            INFO_ZONES[i, 2],
            color="yellow",
            alpha=0.3,
            label="Info Zone" if i == 0 else "",
        )
        ax.add_patch(circle)

    # Plot Goal
    ax.plot(GOAL_POS[0], GOAL_POS[1], "r*", markersize=15, label="Goal")

    # Plot Start
    ax.plot(start_pos[0], start_pos[1], "go", label="Start")

    # Plot Trajectory
    ax.plot(
        history_x[:, 0], history_x[:, 1], "b-", linewidth=2, label="Trajectory"
    )

    ax.set_xlim(-1, 14)
    ax.set_ylim(-1, 12)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"I-MPPI Simulation (2D)\nFinal Info: {history_info[-1]}")

    # --- 3D View ---
    ax3d.plot(
        history_x[:, 0],
        history_x[:, 1],
        history_x[:, 2],
        "b-",
        linewidth=2,
        label="Trajectory",
    )
    ax3d.plot(
        targets[:, 0],
        targets[:, 1],
        targets[:, 2],
        "k--",
        linewidth=1.5,
        alpha=0.7,
        label="Targets",
    )
    ax3d.scatter(
        start_pos[0],
        start_pos[1],
        start_pos[2],
        c="g",
        marker="o",
        s=60,
        label="Start",
    )
    ax3d.scatter(
        GOAL_POS[0],
        GOAL_POS[1],
        GOAL_POS[2],
        c="r",
        marker="*",
        s=160,
        label="Goal",
    )
    ax3d.set_xlim(-1, 14)
    ax3d.set_ylim(-1, 12)
    ax3d.set_zlim(-4, 1)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("I-MPPI Simulation (3D)")
    ax3d.legend()

    output_path = "i_mppi_simulation.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Simulation complete. Plot saved to {output_path}")


if __name__ == "__main__":
    main()
