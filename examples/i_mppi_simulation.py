"""I-MPPI Simulation: Informative Model Predictive Path Integral.

This script implements the simulation described in docs/i_mppi.qmd.
It features:
- Layer 2: FSMI-driven trajectory generation (simulated by a state machine).
- Layer 3: Biased-MPPI control with mixture sampling.
- Environment: 2D corridor with walls and depletable information sources.
"""

from functools import partial

import jax
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
from jax_mppi.i_mppi.map import rasterize_environment
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
    # info levels = max (one per INFO_ZONE)
    info_init = jnp.array([100.0, 100.0, 100.0])

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
    # Create Map
    map_origin = jnp.array([-2.0, -2.0])
    map_width, map_height = 40, 40
    map_res = 0.5  # 20m x 20m cover

    grid_map = rasterize_environment(
        WALLS, INFO_ZONES, map_origin, map_width, map_height, map_res
    )

    fsmi_config = FSMIConfig(
        info_threshold=20.0,
        goal_pos=GOAL_POS,
        gain_weight=2.0,  # Stronger exploration bias
        dist_weight=0.5,
    )

    fsmi_planner = FSMITrajectoryGenerator(fsmi_config, INFO_ZONES, grid_map)

    # Run loop
    sim_steps = 300  # 15 seconds

    print("Starting simulation loop (JIT-compiled scan)...")

    # U_ref = Hover
    U_ref = jnp.tile(mppi_state.u_init, (horizon, 1))

    # Initial target
    current_target = GOAL_POS  # Will be updated immediately
    last_update_t = -100  # Force update

    # 5Hz if dt=0.05. Total 20Hz MPPI.
    # The doc says MPPI 50Hz. Let's assume dt=0.02? No, code says dt=0.05.
    # So MPPI runs at 20Hz.
    # FSMI should run at 5Hz -> Every 4 steps.
    update_period_steps = 4

    def step_fn(carry, t):
        current_state, current_mppi_state, planner, target, last_upd = carry

        current_info = current_state[13:]
        current_pos = current_state[:3]

        # --- Layer 2: FSMI Logic ---
        # 1. Update Map Belief
        updated_planner = planner.update_map(current_info)

        # 2. Check Schedule
        should_update = (t - last_upd) >= update_period_steps

        # 3. Select Target (if needed)
        def update_target_fn(_):
            new_tgt, mode = updated_planner.select_target(current_pos)
            return new_tgt, mode, t

        def keep_target_fn(_):
            # We don't have mode in carry, just assume we keep target
            return target, 0, last_upd  # mode 0 dummy

        new_target, _, new_last_upd = jax.lax.cond(
            should_update, update_target_fn, keep_target_fn, None
        )

        # --- Layer 3: Biased MPPI ---
        # Bind target to cost function
        cost_fn = partial(running_cost, target=new_target)

        action, next_mppi_state = biased_mppi_command(
            config,
            current_mppi_state,
            current_state,
            augmented_dynamics,
            cost_fn,
            U_ref,
            bias_alpha=0.2,
        )

        # Step Dynamics
        next_state = augmented_dynamics(current_state, action, dt=dt)

        # Output
        # We output new_target to visualize

        next_carry = (
            next_state,
            next_mppi_state,
            updated_planner,
            new_target,
            new_last_upd,
        )

        return next_carry, (
            next_state,
            current_info,
            new_target,
        )

    init_carry = (
        state,
        mppi_state,
        fsmi_planner,
        current_target,
        last_update_t,
    )

    (
        (final_state, _, final_planner, _, _),
        (history_x, history_info, targets),
    ) = jax.lax.scan(step_fn, init_carry, jnp.arange(sim_steps))

    # Print final status
    print(f"Final Step: Pos={final_state[:2]}, Info={final_state[13:]}")

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
        cx, cy = INFO_ZONES[i, 0], INFO_ZONES[i, 1]
        w, h = INFO_ZONES[i, 2], INFO_ZONES[i, 3]
        rect = plt.Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            color="yellow",
            alpha=0.3,
            label="Info Zone" if i == 0 else "",
        )
        ax.add_patch(rect)

    # Plot Goal
    ax.plot(GOAL_POS[0], GOAL_POS[1], "r*", markersize=15, label="Goal")

    # Plot Start
    ax.plot(start_pos[0], start_pos[1], "go", label="Start")

    # Plot Trajectory
    ax.plot(
        history_x[:, 0], history_x[:, 1], "b-", linewidth=2, label="Trajectory"
    )

    # Plot Targets (decimated)
    ax.plot(
        targets[::10, 0], targets[::10, 1], "rx", markersize=5, label="Targets"
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
        targets[::10, 0],
        targets[::10, 1],
        targets[::10, 2],
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

    output_path = "docs/_media/i_mppi_simulation.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Simulation complete. Plot saved to {output_path}")


if __name__ == "__main__":
    main()
