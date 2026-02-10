#!/usr/bin/env python3
"""Parallel I-MPPI simulation script.

Single MPPI controller guided by a precomputed information potential field.
No biased sampling or reference trajectory — the field provides directional
guidance while Uniform-FSMI handles local reactivity.

Run from the repo root::

    just run-parallel-imppi

or directly::

    python examples/i_mppi/i_mppi_parallel_simulation.py
"""

import argparse
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Path setup – make helper modules importable regardless of working dir.
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
import matplotlib.animation as animation  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from .env_setup import create_grid_map  # noqa: E402
from .sim_utils import (  # noqa: E402
    CONTROL_HZ,
    DT,
    NOISE_SIGMA,
    NU,
    NX,
    U_INIT,
    U_MAX,
    U_MIN,
    build_parallel_sim_fn,
    compute_smoothness,
)
from tqdm import tqdm  # noqa: E402
from .viz_utils import (  # noqa: E402
    plot_environment,
)

from jax_mppi import mppi  # noqa: E402
from jax_mppi.i_mppi.environment import GOAL_POS, INFO_ZONES  # noqa: E402
from jax_mppi.i_mppi.fsmi import (  # noqa: E402
    FSMIConfig,
    FSMIModule,
    InfoFieldConfig,
    UniformFSMI,
    UniformFSMIConfig,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
START_X = 1.0
START_Y = 5.0
SIM_DURATION = 60.0  # seconds

NUM_SAMPLES = 1000
HORIZON = 40
LAMBDA = 0.1

# FSMI module (for info field computation)
FSMI_BEAMS = 12
FSMI_RANGE = 5.0

# Info field parameters
FIELD_RES = 0.5  # meters per field cell
FIELD_EXTENT = 5.0  # half-width of local workspace [m]
FIELD_N_YAW = 8  # candidate yaw angles
FIELD_UPDATE_INTERVAL = 10  # MPPI steps between field updates
LAMBDA_INFO = 5.0  # field lookup cost weight
LAMBDA_LOCAL = 10.0  # Uniform-FSMI cost weight
GOAL_WEIGHT = 0.3  # constant pull toward goal
MIN_FIELD_THRESHOLD = 0.1  # below this, activate strong goal pull
GOAL_FALLBACK_WEIGHT = 2.0  # goal weight when field is depleted

MEDIA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "docs",
    "_media",
    "i_mppi",
)
GIF_PATH = os.path.join(MEDIA_DIR, "parallel_imppi_trajectory.gif")
SUMMARY_PATH = os.path.join(MEDIA_DIR, "parallel_imppi_summary.png")
DATA_PATH = os.path.join(MEDIA_DIR, "parallel_imppi_flight_data.npz")


# ---------------------------------------------------------------------------
# GIF with info field overlay
# ---------------------------------------------------------------------------


def create_parallel_trajectory_gif(
    history_x: jax.Array,
    history_info: jax.Array,
    history_field: jax.Array,
    history_field_origin: jax.Array,
    grid: jax.Array,
    resolution: float,
    field_res: float,
    dt: float,
    save_path: str,
    fps: int = 10,
    step_skip: int = 5,
) -> str:
    """Create animated GIF showing trajectory + info field heatmap.

    The info field is rendered as a semi-transparent heatmap centered on
    the UAV, updated whenever the simulation recomputes it.
    """
    positions = np.array(history_x[:, :2])
    info = np.array(history_info)
    fields = np.array(history_field)  # (N, Nx, Ny)
    field_origins = np.array(history_field_origin)  # (N, 2)
    n_steps = len(positions)

    frame_indices = list(range(0, n_steps, step_skip))
    if frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)

    fig, (ax_map, ax_info) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    # Static environment
    extent = [0, grid.shape[1] * resolution, 0, grid.shape[0] * resolution]
    ax_map.imshow(
        np.array(grid),
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=1,
        alpha=0.8,
    )

    # Info zone patches
    zone_patches = []
    for i in range(len(INFO_ZONES)):
        cx, cy = float(INFO_ZONES[i, 0]), float(INFO_ZONES[i, 1])
        w, h = float(INFO_ZONES[i, 2]), float(INFO_ZONES[i, 3])
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.05",
            facecolor="yellow",
            alpha=0.4,
            edgecolor="orange",
            linewidth=1.5,
        )
        ax_map.add_patch(rect)
        zone_patches.append(rect)

    ax_map.plot(START_X, START_Y, "go", markersize=10, zorder=5, label="Start")
    ax_map.plot(
        float(GOAL_POS[0]),
        float(GOAL_POS[1]),
        "r*",
        markersize=15,
        zorder=5,
        label="Goal",
    )
    ax_map.set_xlim(-0.5, 14.5)
    ax_map.set_ylim(-0.5, 12.5)
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_aspect("equal")

    # Info field heatmap (initially empty, updated per frame)
    field_vmax = (
        float(np.percentile(fields[fields > 0], 99))
        if np.any(fields > 0)
        else 1.0
    )
    field_img = ax_map.imshow(
        np.zeros((2, 2)),
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="hot",
        vmin=0,
        vmax=field_vmax,
        alpha=0.5,
        zorder=2,
        interpolation="bilinear",
    )

    # Animated elements
    (trail_line,) = ax_map.plot([], [], "c-", linewidth=1.5, alpha=0.6)
    (uav_dot,) = ax_map.plot([], [], "co", markersize=8, zorder=10)
    title = ax_map.set_title("")

    # Colorbar for info field
    cbar = fig.colorbar(field_img, ax=ax_map, shrink=0.6, pad=0.02)
    cbar.set_label("FSMI (info gain)", fontsize=8)

    # Info level plot
    t_all = np.arange(n_steps) * dt
    zone_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    info_lines = []
    for i in range(info.shape[1]):
        (line,) = ax_info.plot(
            [],
            [],
            linewidth=2,
            color=zone_colors[i % len(zone_colors)],
            label=f"Zone {i + 1}",
        )
        info_lines.append(line)
    ax_info.set_xlim(0, t_all[-1])
    ax_info.set_ylim(-5, 105)
    ax_info.set_xlabel("Time (s)")
    ax_info.set_ylabel("Information Level")
    ax_info.set_title("Info Zone Depletion")
    ax_info.legend(loc="upper right")
    ax_info.grid(True, alpha=0.3)

    plt.tight_layout()

    def update(frame_idx: int) -> list[Any]:
        k = frame_indices[frame_idx]

        # Trail
        trail_line.set_data(positions[: k + 1, 0], positions[: k + 1, 1])
        # UAV position
        uav_dot.set_data([positions[k, 0]], [positions[k, 1]])

        # Info field heatmap
        field = fields[k]  # (Nx, Ny)
        fo = field_origins[k]  # (2,)
        Nx, Ny = field.shape
        fx0 = fo[0]
        fy0 = fo[1]
        fx1 = fx0 + Nx * field_res
        fy1 = fy0 + Ny * field_res
        # imshow with origin="lower" expects (rows=Y, cols=X), field is (Nx, Ny)
        # so transpose to (Ny, Nx)
        field_img.set_data(field.T)
        field_img.set_extent([fx0, fx1, fy0, fy1])

        # Title
        title.set_text(
            f"Parallel I-MPPI  t = {k * dt:.1f}s  |  field max = {field.max():.3f}"
        )

        # Zone patch opacity
        for i, patch in enumerate(zone_patches):
            alpha = max(0.05, info[min(k, len(info) - 1), i] / 100.0 * 0.4)
            patch.set_alpha(alpha)

        # Info level lines
        for i, line in enumerate(info_lines):
            line.set_data(t_all[: k + 1], info[: k + 1, i])

        return (
            [trail_line, uav_dot, field_img, title] + zone_patches + info_lines
        )

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 // fps, blit=True
    )
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel I-MPPI simulation")
    parser.add_argument(
        "--gif", action="store_true", help="Generate trajectory GIF"
    )  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help="Generate GIF from saved flight data (skip simulation)",
    )  # pyright: ignore[reportUnusedCallResult]
    return parser.parse_args()


def gif_from_data() -> None:
    """Load saved flight data and generate GIF only."""
    if not os.path.isfile(DATA_PATH):
        print(f"No flight data found at {DATA_PATH}")
        print("Run the simulation first: just run-parallel-imppi")
        sys.exit(1)

    print(f"Loading flight data from {DATA_PATH} ...")
    data = np.load(DATA_PATH, allow_pickle=True)

    os.makedirs(MEDIA_DIR, exist_ok=True)
    print("Generating trajectory GIF with info field ...")
    gif_path = create_parallel_trajectory_gif(
        data["history_x"],
        data["history_info"],
        data["history_field"],
        data["history_field_origin"],
        data["grid"],
        float(data["map_resolution"]),
        FIELD_RES,
        float(data["dt"]),
        save_path=GIF_PATH,
        fps=10,
        step_skip=5,
    )
    print(f"Saved to: {gif_path}")


def main() -> None:
    args = parse_args()

    if args.gif_only:
        gif_from_data()
        return

    # --- JAX device info ---
    print(f"JAX version : {jax.__version__}")
    print(f"Devices     : {jax.devices()}")
    print()

    # --- Environment ---
    grid_map_obj, grid_array, map_origin, map_resolution = create_grid_map()

    sim_steps = int(round(SIM_DURATION * CONTROL_HZ))

    print("=" * 60)
    print("Parallel I-MPPI Simulation")
    print("=" * 60)
    print(f"  Start      : ({START_X}, {START_Y})")
    print(f"  Duration   : {SIM_DURATION}s ({sim_steps} steps)")
    print(
        f"  Samples    : {NUM_SAMPLES},  Horizon: {HORIZON},  Lambda: {LAMBDA}"
    )
    print(
        f"  Field      : res={FIELD_RES}m, extent={FIELD_EXTENT}m, n_yaw={FIELD_N_YAW}"
    )
    print(
        f"  Weights    : info={LAMBDA_INFO}, local={LAMBDA_LOCAL}, goal={GOAL_WEIGHT}"
    )
    print(f"  FSMI Beams : {FSMI_BEAMS},  Range: {FSMI_RANGE}m")
    print()

    # --- Initial state ---
    start_pos = jnp.array([START_X, START_Y, -2.0])
    info_init = jnp.array([100.0, 100.0, 100.0])
    x0 = jnp.zeros(13)
    x0 = x0.at[:3].set(start_pos)
    x0 = x0.at[6].set(1.0)  # qw = 1
    state = jnp.concatenate([x0, info_init])

    # --- FSMIModule (for info field computation) ---
    fsmi_config = FSMIConfig(
        num_beams=FSMI_BEAMS,
        max_range=FSMI_RANGE,
        ray_step=0.15,
        fov_rad=1.57,
    )
    fsmi_module = FSMIModule(
        fsmi_config,
        map_origin,
        map_resolution,
    )

    # --- Info field configuration ---
    info_field_config = InfoFieldConfig(
        field_res=FIELD_RES,
        field_extent=FIELD_EXTENT,
        n_yaw=FIELD_N_YAW,
        field_update_interval=FIELD_UPDATE_INTERVAL,
        lambda_info=LAMBDA_INFO,
        lambda_local=LAMBDA_LOCAL,
        goal_weight=GOAL_WEIGHT,
        min_field_threshold=MIN_FIELD_THRESHOLD,
        goal_fallback_weight=GOAL_FALLBACK_WEIGHT,
    )

    # --- Uniform-FSMI (local reactivity at 50 Hz) ---
    uniform_fsmi_config = UniformFSMIConfig(
        fov_rad=1.57,
        num_beams=6,
        max_range=2.5,
        ray_step=0.2,
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

    def _progress(t: int) -> None:
        _ = pbar.update(int(t) - pbar.n)

    sim_fn = build_parallel_sim_fn(
        mppi_config=config,
        fsmi_module=fsmi_module,
        uniform_fsmi=uniform_fsmi,
        info_field_config=info_field_config,
        grid_map_obj=grid_map_obj,
        info_zones=INFO_ZONES,
        sim_steps=sim_steps,
        progress_callback=_progress,
    )

    print("JIT compiling + running ...")
    t0 = time.perf_counter()
    (
        final_state,
        history_x,
        history_info,
        actions,
        done_step,
        history_field,
        history_field_origin,
    ) = sim_fn(state, ctrl_state)
    final_state.block_until_ready()
    runtime = time.perf_counter() - t0
    _ = pbar.update(sim_steps - pbar.n)
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
    history_field = history_field[:n_active]
    history_field_origin = history_field_origin[:n_active]

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

    # --- Save flight data ---
    os.makedirs(MEDIA_DIR, exist_ok=True)
    np.savez_compressed(
        DATA_PATH,
        history_x=np.array(history_x),
        history_info=np.array(history_info),
        actions=np.array(actions),
        history_field=np.array(history_field),
        history_field_origin=np.array(history_field_origin),
        grid=np.array(grid_array),
        map_resolution=map_resolution,
        dt=DT,
        done_step=done_step_int,
        runtime=runtime,
        goal_dist=goal_dist,
        status=status,
    )
    print(f"\nSaved flight data to {DATA_PATH}")

    # --- 2D trajectory figure ---
    _fig, ax = plt.subplots(figsize=(8, 7))
    plot_environment(ax, grid_array, map_resolution, show_labels=False)
    positions = np.array(history_x[:, :2])
    n_pts = len(positions)
    colors_traj = plt.colormaps["viridis"](np.linspace(0, 1, n_pts))
    for i in range(n_pts - 1):
        _ = ax.plot(
            positions[i : i + 2, 0],
            positions[i : i + 2, 1],
            color=colors_traj[i],
            linewidth=2,
        )
    _ = ax.set_title(f"Parallel I-MPPI Trajectory [{status}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(SUMMARY_PATH, dpi=150)
    print(f"Saved summary plot to {SUMMARY_PATH}")
    plt.show()

    # --- GIF with info field (optional) ---
    if args.gif:
        print("Generating trajectory GIF with info field ...")
        gif_path = create_parallel_trajectory_gif(
            history_x,
            history_info,
            history_field,
            history_field_origin,
            grid_array,
            map_resolution,
            FIELD_RES,
            DT,
            save_path=GIF_PATH,
            fps=10,
            step_skip=5,
        )
        print(f"Saved to: {gif_path}")


if __name__ == "__main__":
    main()
