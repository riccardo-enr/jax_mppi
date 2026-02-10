#!/usr/bin/env python3
"""I-MPPI simulation script.

Standalone version of the interactive notebook.  Run from the repo root::

    python examples/i_mppi/i_mppi_simulation.py

or from the examples/i_mppi/ directory::

    python i_mppi_simulation.py
"""

import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup – make helper modules and the library importable regardless of
# whether the script is launched from the repo root or examples/i_mppi/.
# ---------------------------------------------------------------------------
_candidates = [
    os.path.dirname(os.path.abspath(__file__)),  # script dir
    os.path.join(os.getcwd(), "examples", "i_mppi"),  # repo root
]
for _d in _candidates:
    if os.path.isfile(os.path.join(_d, "env_setup.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from env_setup import create_grid_map  # noqa: E402
from sim_utils import (  # noqa: E402
    CONTROL_HZ,
    DT,
    NOISE_SIGMA,
    NU,
    NX,
    U_INIT,
    U_MAX,
    U_MIN,
    build_sim_fn,
    compute_smoothness,
)
from tqdm import tqdm  # noqa: E402
from viz_utils import (  # noqa: E402
    create_trajectory_gif,
    plot_environment,
    plot_info_levels,
)

from jax_mppi import mppi  # noqa: E402
from jax_mppi.i_mppi.environment import GOAL_POS, INFO_ZONES  # noqa: E402
from jax_mppi.i_mppi.fsmi import (  # noqa: E402
    FSMIConfig,
    FSMITrajectoryGenerator,
    UniformFSMI,
    UniformFSMIConfig,
)

# ---------------------------------------------------------------------------
# Parameters – edit these or override via CLI later
# ---------------------------------------------------------------------------
START_X = 1.0
START_Y = 5.0
SIM_DURATION = 30.0  # seconds

NUM_SAMPLES = 1000
HORIZON = 40
LAMBDA = 0.1
INFO_WEIGHT = 5.0

FSMI_BEAMS = 12
FSMI_RANGE = 5.0

MEDIA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "docs",
    "_media",
    "i_mppi",
)
GIF_PATH = os.path.join(MEDIA_DIR, "i_mppi_trajectory.gif")
SUMMARY_PATH = os.path.join(MEDIA_DIR, "i_mppi_summary.png")


def main() -> None:
    # --- JAX device info ---
    print(f"JAX version : {jax.__version__}")
    print(f"Devices     : {jax.devices()}")
    print()

    # --- Environment ---
    grid_map_obj, grid_array, map_origin, map_resolution = create_grid_map()

    sim_steps = int(round(SIM_DURATION * CONTROL_HZ))

    print("=" * 60)
    print("I-MPPI Simulation")
    print("=" * 60)
    print(f"  Start      : ({START_X}, {START_Y})")
    print(f"  Duration   : {SIM_DURATION}s ({sim_steps} steps)")
    print(
        f"  Samples    : {NUM_SAMPLES},  Horizon: {HORIZON},  Lambda: {LAMBDA}"
    )
    print(f"  Info Weight: {INFO_WEIGHT}")
    print(f"  FSMI Beams : {FSMI_BEAMS},  Range: {FSMI_RANGE}m")
    print()

    # --- Initial state ---
    start_pos = jnp.array([START_X, START_Y, -2.0])
    info_init = jnp.array([100.0, 100.0, 100.0])
    x0 = jnp.zeros(13)
    x0 = x0.at[:3].set(start_pos)
    x0 = x0.at[6].set(1.0)  # qw = 1
    state = jnp.concatenate([x0, info_init])

    # --- Layer 2: FSMI planner ---
    fsmi_config = FSMIConfig(
        use_grid_fsmi=True,
        goal_pos=GOAL_POS,
        fov_rad=1.57,
        num_beams=FSMI_BEAMS,
        max_range=FSMI_RANGE,
        ray_step=0.15,
        sigma_range=0.15,
        gaussian_truncation_sigma=3.0,
        trajectory_subsample_rate=8,
        info_weight=25.0,
        motion_weight=0.5,
    )
    fsmi_planner = FSMITrajectoryGenerator(
        config=fsmi_config,
        info_zones=INFO_ZONES,
        grid_map=grid_map_obj,
    )

    # --- Layer 3: Uniform-FSMI ---
    uniform_fsmi_config = UniformFSMIConfig(
        fov_rad=1.57,
        num_beams=6,
        max_range=2.5,
        ray_step=0.2,
        info_weight=INFO_WEIGHT,
    )
    uniform_fsmi = UniformFSMI(
        uniform_fsmi_config,
        map_origin,
        map_resolution,
    )

    # --- MPPI config ---
    config, ctrl_state = mppi.create(
        nx=NX,
        nu=NU,
        noise_sigma=NOISE_SIGMA,
        num_samples=NUM_SAMPLES,
        horizon=HORIZON,
        lambda_=LAMBDA,
        u_min=U_MIN,
        u_max=U_MAX,
        u_init=U_INIT,
        step_dependent_dynamics=True,
    )

    # --- Build & run simulation ---
    pbar = tqdm(
        total=sim_steps,
        desc="Simulation",
        unit="step",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} steps"
            " [{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )

    def _progress(t):
        pbar.update(int(t) - pbar.n)

    sim_fn = build_sim_fn(
        config,
        fsmi_planner,
        uniform_fsmi,
        uniform_fsmi_config,
        grid_map_obj,
        HORIZON,
        sim_steps,
        progress_callback=_progress,
    )

    print("JIT compiling + running ...")
    t0 = time.perf_counter()
    final_state, history_x, history_info, targets, actions, done_step = sim_fn(
        state, ctrl_state
    )
    final_state.block_until_ready()
    runtime = time.perf_counter() - t0
    pbar.update(sim_steps - pbar.n)
    pbar.close()

    # --- Truncate to active steps ---
    done_step_int = int(done_step)
    if done_step_int > 0:
        n_active = done_step_int
        actual_duration = n_active * DT
        print(f"  Task completed at step {n_active} ({actual_duration:.1f}s)")
    else:
        n_active = sim_steps
        actual_duration = SIM_DURATION
        print(f"  Timeout reached ({SIM_DURATION}s)")

    history_x = history_x[:n_active]
    history_info = history_info[:n_active]
    actions = actions[:n_active]

    # --- Metrics ---
    action_jerk, traj_jerk = compute_smoothness(actions, history_x, DT)
    final_pos = final_state[:3]
    goal_dist = float(jnp.linalg.norm(final_pos - GOAL_POS))

    status = "Completed" if done_step_int > 0 else "Timeout"

    print(f"Runtime : {runtime:.2f}s ({runtime / SIM_DURATION:.2f}x realtime)")
    print(f"  Goal dist  : {goal_dist:.2f}m")
    print(f"  Info levels: {final_state[13:]}")
    print()
    print("=" * 60)
    print(f"{'Metric':<25} {'Value':>15}")
    print("-" * 60)
    print(f"{'Status':<25} {status:>15}")
    print(f"{'Sim Duration (s)':<25} {actual_duration:>15.1f}")
    print(f"{'Runtime (ms)':<25} {runtime * 1000:>15.1f}")
    print(f"{'Goal Distance (m)':<25} {goal_dist:>15.2f}")
    print(f"{'Action Jerk':<25} {float(action_jerk):>15.4f}")
    print(f"{'Trajectory Jerk':<25} {float(traj_jerk):>15.4f}")
    print("=" * 60)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 2D trajectory
    ax = axes[0]
    plot_environment(ax, grid_array, map_resolution, show_labels=False)
    positions = np.array(history_x[:, :2])
    n_pts = len(positions)
    colors_traj = plt.colormaps["viridis"](np.linspace(0, 1, n_pts))
    for i in range(n_pts - 1):
        ax.plot(
            positions[i : i + 2, 0],
            positions[i : i + 2, 1],
            color=colors_traj[i],
            linewidth=2,
        )
    ax.set_title("I-MPPI Trajectory")

    # Info zone depletion
    plot_info_levels(axes[1], history_info, DT)
    axes[1].set_title("Information Zone Depletion")

    # Control inputs
    acts = np.array(actions)
    t_arr = np.arange(len(acts)) * DT
    ax3 = axes[2]
    ax3.plot(t_arr, acts[:, 0], color="#1f77b4", linewidth=1, label="Thrust")
    ax3.plot(t_arr, acts[:, 1], color="#ff7f0e", linewidth=1, label="wx")
    ax3.plot(t_arr, acts[:, 2], color="#2ca02c", linewidth=1, label="wy")
    ax3.plot(t_arr, acts[:, 3], color="#d62728", linewidth=1, label="wz")
    ax3.set_ylabel("Control Input")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Control Inputs")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(
        f"I-MPPI: Full FSMI (Layer 2) + "
        f"Biased MPPI with Uniform-FSMI (Layer 3) [{status}]",
        fontsize=14,
    )
    plt.tight_layout()
    os.makedirs(MEDIA_DIR, exist_ok=True)
    plt.savefig(SUMMARY_PATH, dpi=150)
    print(f"\nSaved summary plot to {SUMMARY_PATH}")
    plt.show()

    # --- GIF ---
    print("Generating trajectory GIF ...")
    gif_path = create_trajectory_gif(
        history_x,
        history_info,
        grid_array,
        map_resolution,
        DT,
        save_path=GIF_PATH,
        fps=10,
        step_skip=2,
    )
    print(f"Saved to: {gif_path}")


if __name__ == "__main__":
    main()
