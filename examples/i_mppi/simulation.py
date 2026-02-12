"""I-MPPI Simulation with Two-Layer Architecture.

This script implements the full I-MPPI architecture as described in Zhang et al. (2020):

Layer 2 (FSMI Analyzer, ~5 Hz):
    - Uses full FSMI with O(n^2) computation on occupancy grid
    - Generates optimal reference trajectory maximizing global information
    - Output: tau_ref (reference trajectory for Layer 3)

Layer 3 (I-MPPI Controller, ~50 Hz):
    - Uses Uniform-FSMI with O(n) computation for local reactivity
    - Cost: J = Tracking(tau_ref) + Obstacles - lambda * Uniform_FSMI(local)
    - Maintains informative viewpoints during trajectory execution

Key insight (from the architecture description):
    If you remove the informative term from Layer 3, you're doing standard
    trajectory tracking, NOT informative control. The Uniform-FSMI term ensures:
    - Reactive viewpoint maintenance during disturbances
    - Local occlusion handling between Layer 2 updates
    - True I-MPPI behavior

Comparison: Runs MPPI, SMPPI, and KMPPI variants for runtime and smoothness analysis.
"""

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from jax_mppi import kmppi, mppi, smppi
from jax_mppi.i_mppi.environment import (
    GOAL_POS,
    INFO_ZONES,
    WALLS,
    augmented_dynamics,
    informative_running_cost,
    running_cost,
)
from jax_mppi.i_mppi.fsmi import (
    FSMIConfig,
    FSMITrajectoryGenerator,
    UniformFSMI,
    UniformFSMIConfig,
)
from jax_mppi.i_mppi.map import GridMap


def create_occupancy_grid():
    """
    Create an office-like occupancy grid with rooms, doorways, and corridors.

    Office layout:
    - Multiple rooms with narrow doorways
    - Central corridor connecting rooms
    - Cluttered areas (desks, furniture)
    - Unknown regions behind closed areas (high information gain)
    - Narrow passages that require careful navigation

    The grid represents:
    - Free space: p=0.2 (known clear)
    - Obstacles (walls, furniture): p=0.9 (known occupied)
    - Unknown regions: p=0.5 (unexplored rooms - high information gain)
    - Partial observations: p=0.3-0.4 (doorway views into rooms)

    Returns:
        grid_map: (H, W) occupancy probability grid [0, 1]
        map_origin: (x, y) world origin
        resolution: Meters per cell
    """
    world_width = 14.0
    world_height = 12.0
    resolution = 0.1

    width = int(world_width / resolution)
    height = int(world_height / resolution)

    # Start with unknown (unexplored office)
    grid = 0.5 * jnp.ones((height, width))

    # === KNOWN FREE SPACE ===
    grid = grid.at[35:85, 5:135].set(0.2)

    # === OUTER WALLS ===
    grid = grid.at[0:5, :].set(0.9)
    grid = grid.at[115:120, :].set(0.9)
    grid = grid.at[:, 0:5].set(0.9)
    grid = grid.at[:, 135:140].set(0.9)

    # === OFFICE ROOMS ===
    # Room 1: Bottom-left office
    grid = grid.at[85:115, 5:45].set(0.9)
    grid = grid.at[35:115, 5:10].set(0.9)
    grid = grid.at[35:85, 40:45].set(0.9)
    grid = grid.at[35:45, 40:45].set(0.2)
    grid = grid.at[40:80, 10:40].set(0.5)
    grid = grid.at[40:50, 30:40].set(0.35)

    # Room 2: Top-left office
    grid = grid.at[5:35, 5:45].set(0.9)
    grid = grid.at[5:35, 40:45].set(0.9)
    grid = grid.at[28:36, 40:45].set(0.2)
    grid = grid.at[10:30, 10:40].set(0.5)

    # Room 3: Bottom-right office
    grid = grid.at[85:115, 95:135].set(0.9)
    grid = grid.at[85:115, 130:135].set(0.9)
    grid = grid.at[35:85, 95:100].set(0.9)
    grid = grid.at[40:50, 95:100].set(0.2)
    grid = grid.at[40:80, 100:130].set(0.5)
    grid = grid.at[50:60, 105:115].set(0.8)
    grid = grid.at[65:75, 120:125].set(0.8)

    # Room 4: Top-right office
    grid = grid.at[5:35, 95:135].set(0.9)
    grid = grid.at[5:35, 95:100].set(0.9)
    grid = grid.at[28:36, 95:100].set(0.2)
    grid = grid.at[10:30, 100:130].set(0.5)
    grid = grid.at[25:32, 100:110].set(0.35)

    # === CENTRAL OBSTACLES ===
    grid = grid.at[45:55, 50:60].set(0.85)
    grid = grid.at[65:75, 70:80].set(0.85)
    grid = grid.at[40:45, 85:90].set(0.8)
    grid = grid.at[75:80, 20:25].set(0.8)

    # === NARROW PASSAGES ===
    grid = grid.at[35:85, 45:52].set(0.2)
    grid = grid.at[55:65, 60:70].set(0.2)

    # === INFO ZONES (High uncertainty regions) ===
    grid = grid.at[50:75, 12:35].set(0.5)
    grid = grid.at[55:70, 15:30].set(0.55)
    grid = grid.at[12:28, 102:128].set(0.5)
    grid = grid.at[15:25, 105:125].set(0.55)

    # === ADDITIONAL COMPLEXITY ===
    grid = grid.at[70:82, 48:52].set(0.9)
    grid = grid.at[72:80, 48:52].set(0.2)
    grid = grid.at[72:80, 45:48].set(0.52)
    grid = grid.at[55:70, 90:95].set(0.9)
    grid = grid.at[35:36, 40:45].set(0.75)
    grid = grid.at[84:85, 95:100].set(0.75)

    map_origin = jnp.array([0.0, 0.0])
    return grid, map_origin, resolution


def main():
    print("=" * 70)
    print("I-MPPI Simulation with Two-Layer Architecture")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  Layer 2 (FSMI Analyzer, 5 Hz): Full FSMI -> Reference Trajectory")
    print("  Layer 3 (I-MPPI, 50 Hz): Tracking + Uniform-FSMI -> Control")
    print()
    print("Setting up simulation...")

    # Config
    dt = 0.05
    nx = 16  # 13 (quad state) + 3 (info zones)
    nu = 4
    horizon = 40

    # Create occupancy grid
    print("Creating occupancy grid...")
    grid_map, map_origin, map_resolution = create_occupancy_grid()
    print(
        f"  Grid size: {grid_map.shape} "
        f"({grid_map.shape[1] * map_resolution:.1f}m x "
        f"{grid_map.shape[0] * map_resolution:.1f}m)"
    )
    print(f"  Resolution: {map_resolution}m/cell")

    # Initial State
    start_pos = jnp.array([1.0, 5.0, -2.0])
    info_init = jnp.array([100.0, 100.0, 100.0])  # 3 info zones

    x0 = jnp.zeros(13)
    x0 = x0.at[:3].set(start_pos)
    x0 = x0.at[6].set(1.0)  # qw=1

    state = jnp.concatenate([x0, info_init])

    # Common MPPI Setup
    noise_sigma = jnp.diag(jnp.array([2.0, 0.5, 0.5, 0.5]) ** 2)
    u_min = jnp.array([0.0, -10.0, -10.0, -10.0])
    u_max = jnp.array([4.0 * 9.81, 10.0, 10.0, 10.0])
    rate_time = 0.5
    action_range = u_max - u_min
    u_vel_max = action_range / rate_time
    u_vel_min = -u_vel_max
    u_init = jnp.array([9.81, 0.0, 0.0, 0.0])

    # === Layer 2 Setup: Full FSMI (5 Hz) ===
    print("\nSetting up Layer 2 (FSMI Analyzer, ~5 Hz)...")
    fsmi_config = FSMIConfig(
        use_grid_fsmi=True,
        goal_pos=GOAL_POS,
        fov_rad=1.57,  # 90 degrees
        num_beams=12,  # Full scan
        max_range=5.0,  # Long range for planning
        ray_step=0.15,
        sigma_range=0.15,
        gaussian_truncation_sigma=3.0,
        trajectory_subsample_rate=8,
        info_weight=25.0,  # Higher weight to prioritize exploration
        motion_weight=0.5,  # Lower weight to allow detours
    )

    grid_map_obj = GridMap(
        grid=grid_map,
        origin=map_origin,
        resolution=map_resolution,
        width=grid_map.shape[1],
        height=grid_map.shape[0],
    )
    fsmi_planner = FSMITrajectoryGenerator(
        fsmi_config,
        INFO_ZONES,
        grid_map_obj,
    )
    print(f"  Beams: {fsmi_config.num_beams}")
    print(f"  Range: {fsmi_config.max_range}m")
    print(f"  Subsampling: every {fsmi_config.trajectory_subsample_rate} steps")

    # === Layer 3 Setup: Uniform-FSMI (50 Hz) ===
    print("\nSetting up Layer 3 (Uniform-FSMI, ~50 Hz)...")
    uniform_fsmi_config = UniformFSMIConfig(
        fov_rad=1.57,  # 90 degrees
        num_beams=6,  # Reduced for speed
        max_range=2.5,  # Local only (2.5m vs 5m)
        ray_step=0.2,  # Coarser resolution
        info_weight=5.0,  # Local reactivity weight
    )

    uniform_fsmi = UniformFSMI(
        uniform_fsmi_config,
        map_origin,
        map_resolution,
    )
    print(f"  Beams: {uniform_fsmi_config.num_beams}")
    print(f"  Range: {uniform_fsmi_config.max_range}m (local)")
    print(f"  Info weight: {uniform_fsmi_config.info_weight}")

    # Run loop
    sim_duration = 60.0
    control_hz = 50.0
    sim_steps = int(round(sim_duration * control_hz))
    fsmi_hz = 5.0
    fsmi_steps = int(round(control_hz / fsmi_hz))

    print("\nSimulation parameters:")
    print(f"  Duration: {sim_duration}s")
    print(f"  Control rate: {control_hz} Hz (Layer 3)")
    print(f"  FSMI update rate: {fsmi_hz} Hz (Layer 2)")
    print(f"  Total steps: {sim_steps}")

    U_ref = jnp.tile(u_init, (horizon, 1))

    def block_until_ready(tree):
        for leaf in jax.tree_util.tree_leaves(tree):
            leaf.block_until_ready()

    def make_u_ref_from_traj(current_state, ref_traj):
        pos = current_state[:3]
        err = ref_traj - pos[None, :]

        k_thrust = 3.0
        thrust = u_init[0] + k_thrust * err[:, 2]

        k_omega = 0.6
        omega_x = k_omega * err[:, 1]
        omega_y = -k_omega * err[:, 0]
        omega_z = jnp.zeros_like(omega_x)

        u_ref = jnp.stack([thrust, omega_x, omega_y, omega_z], axis=1)
        u_ref = jnp.clip(u_ref, u_min, u_max)
        return u_ref

    # Import biased MPPI commands
    from jax_mppi.i_mppi.planner import (
        biased_kmppi_command,
        biased_mppi_command,
        biased_smppi_command,
    )

    def make_sim_fn(
        controller_name,
        config,
        ctrl_state,
        use_informative_cost=True,
        kernel_fn=None,
    ):
        def step_fn(carry, t):
            current_state, current_ctrl_state, ref_traj = carry

            # --- Layer 2: Grid-Based FSMI (slow path, 5 Hz) ---
            current_info = current_state[13:]
            do_update = jnp.equal(jnp.mod(t, fsmi_steps), 0)

            info_data = (grid_map, current_info)

            ref_traj = jax.lax.cond(
                do_update,
                lambda _: fsmi_planner.get_reference_trajectory(
                    current_state, info_data, horizon, dt
                )[0],
                lambda _: ref_traj,
                operand=None,
            )

            # --- Layer 3: Biased I-MPPI with Uniform-FSMI (50 Hz) ---
            if use_informative_cost:
                # Full I-MPPI: Tracking + Local Information
                # J = tracking + obstacles - lambda * Uniform_FSMI(local)
                cost_fn = partial(
                    informative_running_cost,
                    target=ref_traj,
                    grid_map=grid_map,
                    uniform_fsmi_fn=uniform_fsmi.compute,
                    info_weight=uniform_fsmi_config.info_weight,
                )
            else:
                # Baseline: Tracking only (standard biased MPPI)
                cost_fn = partial(running_cost, target=ref_traj)

            U_ref_local = make_u_ref_from_traj(current_state, ref_traj)

            if controller_name == "mppi":
                action, next_ctrl_state = biased_mppi_command(
                    config,
                    current_ctrl_state,
                    current_state,
                    augmented_dynamics,
                    cost_fn,
                    U_ref_local,
                    bias_alpha=0.2,
                )
            elif controller_name == "smppi":
                action, next_ctrl_state = biased_smppi_command(
                    config,
                    current_ctrl_state,
                    current_state,
                    augmented_dynamics,
                    cost_fn,
                    U_ref_local,
                    bias_alpha=0.2,
                )
            elif controller_name == "kmppi":
                assert kernel_fn is not None
                action, next_ctrl_state = biased_kmppi_command(
                    config,
                    current_ctrl_state,
                    current_state,
                    augmented_dynamics,
                    cost_fn,
                    kernel_fn,
                    U_ref_local,
                    bias_alpha=0.2,
                )
            else:
                raise ValueError(f"Unknown controller: {controller_name}")

            next_state = augmented_dynamics(current_state, action, dt=dt)

            return (next_state, next_ctrl_state, ref_traj), (
                next_state,
                current_info,
                ref_traj[0],
                action,
            )

        def sim_fn(initial_state, initial_ctrl_state):
            info_data = (grid_map, initial_state[13:])
            init_ref_traj, _ = fsmi_planner.get_reference_trajectory(
                initial_state, info_data, horizon, dt
            )
            (
                (final_state, final_ctrl_state, _),
                (history_x, history_info, targets, actions),
            ) = jax.lax.scan(
                step_fn,
                (initial_state, initial_ctrl_state, init_ref_traj),
                jnp.arange(sim_steps),
            )
            return (
                final_state,
                final_ctrl_state,
                history_x,
                history_info,
                targets,
                actions,
            )

        return jax.jit(sim_fn)

    def compute_smoothness(actions, positions, dt_local):
        action_jerk = jnp.diff(actions, n=2, axis=0) / (dt_local**2)
        action_jerk_mean = jnp.mean(jnp.linalg.norm(action_jerk, axis=1))

        pos = positions[:, :3]
        vel = jnp.diff(pos, axis=0) / dt_local
        acc = jnp.diff(vel, axis=0) / dt_local
        jerk = jnp.diff(acc, axis=0) / dt_local
        traj_jerk_mean = jnp.mean(jnp.linalg.norm(jerk, axis=1))

        return action_jerk_mean, traj_jerk_mean

    def run_controller(controller_name, use_informative_cost=True):
        mode_str = "I-MPPI" if use_informative_cost else "Tracking-Only"
        print(f"\n--- Running {controller_name.upper()} ({mode_str}) ---")

        if controller_name == "mppi":
            config, ctrl_state = mppi.create(
                nx=nx,
                nu=nu,
                noise_sigma=noise_sigma,
                num_samples=1000,
                horizon=horizon,
                lambda_=0.1,
                u_min=u_min,
                u_max=u_max,
                u_init=u_init,
                step_dependent_dynamics=True,
            )
            sim_fn = make_sim_fn(
                controller_name, config, ctrl_state, use_informative_cost
            )
        elif controller_name == "smppi":
            config, ctrl_state = smppi.create(
                nx=nx,
                nu=nu,
                noise_sigma=noise_sigma,
                num_samples=1000,
                horizon=horizon,
                lambda_=0.1,
                u_min=u_vel_min,
                u_max=u_vel_max,
                u_init=jnp.zeros_like(u_init),
                U_init=U_ref,
                action_min=u_min,
                action_max=u_max,
                step_dependent_dynamics=True,
                w_action_seq_cost=0.5,
                delta_t=dt,
            )
            sim_fn = make_sim_fn(
                controller_name, config, ctrl_state, use_informative_cost
            )
        elif controller_name == "kmppi":
            config, ctrl_state, kernel_fn = kmppi.create(
                nx=nx,
                nu=nu,
                noise_sigma=noise_sigma,
                num_samples=1000,
                horizon=horizon,
                lambda_=0.1,
                u_min=u_min,
                u_max=u_max,
                u_init=u_init,
                U_init=U_ref,
                step_dependent_dynamics=True,
            )
            sim_fn = make_sim_fn(
                controller_name,
                config,
                ctrl_state,
                use_informative_cost,
                kernel_fn=kernel_fn,
            )
        else:
            raise ValueError(f"Unknown controller: {controller_name}")

        # Warm-up
        print("  JIT compiling...")
        warm = sim_fn(state, ctrl_state)
        block_until_ready(warm)
        print("  Compilation done!")

        # Timed run
        print("  Running simulation...")
        start = time.perf_counter()
        (
            final_state,
            final_ctrl_state,
            history_x,
            history_info,
            targets,
            actions,
        ) = sim_fn(state, ctrl_state)
        block_until_ready(final_state)
        elapsed = time.perf_counter() - start
        print(
            f"  Runtime: {elapsed:.2f}s ({elapsed / sim_duration:.2f}x realtime)"
        )

        action_jerk, traj_jerk = compute_smoothness(actions, history_x, dt)

        return {
            "config": config,
            "ctrl_state": final_ctrl_state,
            "history_x": history_x,
            "history_info": history_info,
            "targets": targets,
            "actions": actions,
            "runtime_s": elapsed,
            "action_jerk": action_jerk,
            "traj_jerk": traj_jerk,
            "final_state": final_state,
        }

    # Run all controllers with full I-MPPI (Uniform-FSMI in Layer 3)
    print("\n" + "=" * 70)
    print("Running Controllers with Full I-MPPI Architecture")
    print("(Layer 2: Full FSMI + Layer 3: Uniform-FSMI)")
    print("=" * 70)

    controllers = ["mppi", "smppi", "kmppi"]
    results = {
        name: run_controller(name, use_informative_cost=True)
        for name in controllers
    }

    # Print results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for name in controllers:
        final_state = results[name]["final_state"]
        print(f"\n{name.upper()}:")
        print(f"  Final pos: [{final_state[0]:.2f}, {final_state[1]:.2f}]")
        print(f"  Info levels: {final_state[13:]}")
        print(f"  Runtime: {results[name]['runtime_s']:.2f}s")

    print("\n" + "-" * 70)
    print(
        f"{'Controller':<12} {'Runtime (ms)':>15} {'Action Jerk':>15} {'Traj Jerk':>15}"
    )
    print("-" * 70)
    for name in controllers:
        runtime_ms = 1000.0 * results[name]["runtime_s"]
        action_jerk = float(results[name]["action_jerk"])
        traj_jerk = float(results[name]["traj_jerk"])
        print(
            f"{name:<12} {runtime_ms:>15.1f} {action_jerk:>15.4f} {traj_jerk:>15.4f}"
        )
    print("-" * 70)

    # Visualization
    print("\nGenerating visualization...")

    import numpy as np
    from matplotlib.patches import Rectangle

    with plt.style.context(["science", "no-latex"]):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[1, 1], hspace=0.25, wspace=0.15
        )

        # Row 1, Col 1: 2D Trajectories
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("I-MPPI with Two-Layer Architecture (2D)")

        for w in WALLS:
            rect = Rectangle(
                (w[0], w[1]),
                w[2] - w[0],
                w[3] - w[1],
                facecolor="gray",
                alpha=0.5,
                edgecolor="none",
            )
            ax1.add_patch(rect)

        for i in range(len(INFO_ZONES)):
            cx, cy = INFO_ZONES[i, 0], INFO_ZONES[i, 1]
            w, h = INFO_ZONES[i, 2], INFO_ZONES[i, 3]
            rect = Rectangle(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                facecolor="yellow",
                alpha=0.3,
                edgecolor="none",
            )
            ax1.add_patch(rect)

        ax1.plot([], [], "s", color="yellow", label="Info Zone")
        ax1.plot(
            GOAL_POS[0],
            GOAL_POS[1],
            "*",
            color="red",
            markersize=14,
            label="Goal",
        )
        ax1.plot(
            start_pos[0],
            start_pos[1],
            "o",
            color="green",
            markersize=10,
            label="Start",
        )

        colors = {"mppi": "blue", "smppi": "green", "kmppi": "orange"}
        for name in controllers:
            history_x = results[name]["history_x"]
            ax1.plot(
                history_x[:, 0],
                history_x[:, 1],
                color=colors[name],
                linewidth=2,
                label=name.upper(),
            )

        ax1.set_xlim(-1, 14)
        ax1.set_ylim(-1, 12)
        ax1.set_aspect("equal")
        ax1.legend(loc="upper right", fontsize=7)

        # Row 1, Col 2: 3D View
        ax2 = fig.add_subplot(gs[0, 1], projection="3d")
        ax2.set_title("3D Trajectory View")

        for i in range(len(INFO_ZONES)):
            cx, cy = INFO_ZONES[i, 0], INFO_ZONES[i, 1]
            w, h = INFO_ZONES[i, 2], INFO_ZONES[i, 3]
            zone_z = float(GOAL_POS[2])
            x0, x1 = float(cx - w / 2), float(cx + w / 2)
            y0, y1 = float(cy - h / 2), float(cy + h / 2)
            xs = [x0, x1, x1, x0, x0]
            ys = [y0, y0, y1, y1, y0]
            zs = [zone_z] * 5
            ax2.plot(xs, ys, zs, color="yellow", alpha=0.5)

        for name in controllers:
            history_x = results[name]["history_x"]
            ax2.plot(
                history_x[:, 0],
                history_x[:, 1],
                history_x[:, 2],
                color=colors[name],
                linewidth=2,
            )

        ax2.scatter(*start_pos, color="green", s=40)
        ax2.scatter(*GOAL_POS, color="red", s=60, marker="D")
        ax2.set_xlim(-1, 14)
        ax2.set_ylim(-1, 12)
        ax2.set_zlim(-4, 1)
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")

        # Row 2: Occupancy Grid (spanning both columns)
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_title("Occupancy Grid with Trajectories")

        extent = [
            0,
            grid_map.shape[1] * map_resolution,
            0,
            grid_map.shape[0] * map_resolution,
        ]
        im = ax3.imshow(
            np.array(grid_map),
            cmap="Greys",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=1,
            aspect="equal",
        )
        plt.colorbar(im, ax=ax3, label="Occupancy", shrink=0.6)

        history_x = results["mppi"]["history_x"]
        ax3.plot(history_x[:, 0], history_x[:, 1], color="cyan", linewidth=2)
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")

        fig.suptitle(
            "I-MPPI: Layer 2 (Full FSMI, 5Hz) + Layer 3 (Uniform-FSMI, 50Hz)",
            fontsize=14,
        )

    output_path = "docs/_media/i_mppi/i_mppi_simulation.png"
    os.makedirs("docs/_media/i_mppi", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {output_path}")

    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
