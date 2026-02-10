#!/usr/bin/env python3
"""Parallel I-MPPI simulation script.

Single MPPI controller guided by a field-gradient reference trajectory and
information potential field. FOV-based grid update replaces zone tracking.

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
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
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
    build_parallel_sim_fn,
    compute_smoothness,
)
from tqdm import tqdm  # noqa: E402
from viz_utils import plot_trajectory_2d  # noqa: E402

from jax_mppi import mppi  # noqa: E402
from jax_mppi.i_mppi.environment import GOAL_POS  # noqa: E402
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
FSMI_RANGE = 10.0

# Info field parameters
FIELD_RES = 0.5  # meters per field cell
FIELD_EXTENT = 5.0  # half-width of local workspace [m]
FIELD_N_YAW = 8  # candidate yaw angles
FIELD_UPDATE_INTERVAL = 10  # MPPI steps between field updates
LAMBDA_INFO = 20.0  # field lookup cost weight
LAMBDA_LOCAL = 10.0  # Uniform-FSMI cost weight

# Reference trajectory parameters
REF_SPEED = 2.0  # gradient trajectory speed [m/s]
REF_HORIZON = 40  # gradient trajectory steps
TARGET_WEIGHT = 1.0  # MPPI target tracking weight
GOAL_WEIGHT = 0.2  # constant goal attraction weight

MEDIA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "docs",
    "_media",
    "i_mppi",
)
HTML_PATH = os.path.join(MEDIA_DIR, "parallel_imppi_trajectory.html")
SUMMARY_PATH = os.path.join(MEDIA_DIR, "parallel_imppi_summary.html")
DATA_PATH = os.path.join(MEDIA_DIR, "parallel_imppi_flight_data.npz")


# ---------------------------------------------------------------------------
# HTML visualization with info field overlay
# ---------------------------------------------------------------------------


def create_parallel_trajectory_html(
    history_x: jax.Array,
    history_field: jax.Array,
    history_field_origin: jax.Array,
    history_ref_traj: jax.Array,
    grid: jax.Array,
    resolution: float,
    field_res: float,
    dt: float,
    save_path: str,
    fps: int = 10,
    step_skip: int = 5,
) -> str:
    """Create animated HTML showing trajectory + reference trajectory + info field heatmap."""
    from viz_utils import _INFO_GAIN_COLORSCALE

    states = np.array(history_x)  # (N, 13)
    positions = states[:, :2]
    # Extract yaw from quaternion
    quats = np.array(states[:, 6:10])
    yaws = np.arctan2(
        2 * (quats[:, 0] * quats[:, 3] + quats[:, 1] * quats[:, 2]),
        1 - 2 * (quats[:, 2] ** 2 + quats[:, 3] ** 2),
    )
    fields = np.array(history_field)  # (N, Nx, Ny)
    field_origins = np.array(history_field_origin)  # (N, 2)
    ref_trajs = np.array(history_ref_traj)  # (N, horizon, 3)
    n_steps = len(positions)

    frame_indices = list(range(0, n_steps, step_skip))
    if frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)

    grid_np = np.array(grid)
    extent = [0, grid.shape[1] * resolution, 0, grid.shape[0] * resolution]

    fig = go.Figure()

    # Occupancy grid
    fig.add_trace(
        go.Heatmap(
            z=grid_np,
            x=np.linspace(0, extent[1], grid.shape[1]),
            y=np.linspace(0, extent[3], grid.shape[0]),
            colorscale="Greys",
            reversescale=True,
            zmin=0,
            zmax=1,
            opacity=0.8,
            showscale=False,
            hoverinfo="skip",
        )
    )

    # Info field overlay (animated)
    field_vmax = (
        float(np.percentile(fields[fields > 0], 99))
        if np.any(fields > 0)
        else 1.0
    )
    k0 = frame_indices[0]
    field_0 = fields[k0]
    fo_0 = field_origins[k0]
    Nx, Ny = field_0.shape
    fx0_0 = fo_0[0]
    fy0_0 = fo_0[1]
    fx1_0 = fx0_0 + Nx * field_res
    fy1_0 = fy0_0 + Ny * field_res
    field_masked = np.where(field_0 > 0, field_0, np.nan)

    fig.add_trace(
        go.Heatmap(
            z=field_masked,
            x=np.linspace(fx0_0, fx1_0, Nx),
            y=np.linspace(fy0_0, fy1_0, Ny),
            colorscale=_INFO_GAIN_COLORSCALE,
            opacity=0.6,
            showscale=True,
            colorbar=dict(title="FSMI", x=1.02),
            hoverinfo="z",
            zmin=0,
            zmax=field_vmax,
        )
    )

    # Start and goal
    fig.add_trace(
        go.Scatter(
            x=[START_X],
            y=[START_Y],
            mode="markers",
            marker=dict(size=10, color="green", symbol="circle"),
            name="Start",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[float(GOAL_POS[0])],
            y=[float(GOAL_POS[1])],
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name="Goal",
            showlegend=False,
        )
    )

    # Reference trajectory (animated)
    ref_traj_0 = ref_trajs[k0]
    fig.add_trace(
        go.Scatter(
            x=ref_traj_0[:, 0],
            y=ref_traj_0[:, 1],
            mode="lines",
            line=dict(color="magenta", width=2, dash="dash"),
            name="Reference Traj",
        )
    )

    # Trajectory trail (animated)
    fig.add_trace(
        go.Scatter(
            x=positions[:1, 0],
            y=positions[:1, 1],
            mode="lines",
            line=dict(color="cyan", width=2),
            name="Executed Traj",
        )
    )

    # UAV position (animated)
    fig.add_trace(
        go.Scatter(
            x=[positions[k0, 0]],
            y=[positions[k0, 1]],
            mode="markers",
            marker=dict(size=10, color="cyan", symbol="circle"),
            name="UAV",
            showlegend=False,
        )
    )

    # Create animation frames
    frames = []
    for frame_idx in range(len(frame_indices)):
        k = frame_indices[frame_idx]
        x, y = positions[k, 0], positions[k, 1]

        # Info field for this frame
        field = fields[k]
        fo = field_origins[k]
        fx0 = fo[0]
        fy0 = fo[1]
        fx1 = fx0 + Nx * field_res
        fy1 = fy0 + Ny * field_res
        field_masked = np.where(field > 0, field, np.nan)

        # Reference trajectory
        ref_traj = ref_trajs[k]

        frame_data = [
            go.Heatmap(z=grid_np),  # Grid (static)
            go.Heatmap(
                z=field_masked,
                x=np.linspace(fx0, fx1, Nx),
                y=np.linspace(fy0, fy1, Ny),
            ),  # Info field
            go.Scatter(x=[START_X], y=[START_Y]),  # Start
            go.Scatter(x=[float(GOAL_POS[0])], y=[float(GOAL_POS[1])]),  # Goal
            go.Scatter(x=ref_traj[:, 0], y=ref_traj[:, 1]),  # Reference traj
            go.Scatter(x=positions[: k + 1, 0], y=positions[: k + 1, 1]),  # Trail
            go.Scatter(x=[x], y=[y]),  # UAV
        ]

        frames.append(
            go.Frame(
                data=frame_data,
                name=f"frame_{frame_idx}",
                layout=go.Layout(
                    title_text=f"Parallel I-MPPI  t = {k * dt:.1f}s  |  field max = {field.max():.3f}"
                ),
            )
        )

    fig.frames = frames

    # Add play/pause buttons
    fig.update_layout(
        title=f"Parallel I-MPPI  t = 0.0s  |  field max = {field_0.max():.3f}",
        xaxis=dict(title="X (m)", range=[-0.5, 14.5]),
        yaxis=dict(title="Y (m)", range=[-0.5, 12.5], scaleanchor="x", scaleratio=1),
        width=1000,
        height=800,
        showlegend=True,
        hovermode="closest",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 1000 // fps, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
                x=0.1,
                y=0,
                xanchor="left",
                yanchor="top",
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        args=[
                            [f"frame_{i}"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        label=f"{frame_indices[i] * dt:.1f}s",
                        method="animate",
                    )
                    for i in range(len(frame_indices))
                ],
                x=0.1,
                y=0,
                len=0.8,
                xanchor="left",
                yanchor="top",
            )
        ],
    )

    fig.write_html(save_path)
    fig.show()
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel I-MPPI simulation")
    parser.add_argument(
        "--animation", action="store_true", help="Generate trajectory animation"
    )  # pyright: ignore[reportUnusedCallResult]
    parser.add_argument(
        "--animation-only",
        action="store_true",
        help="Generate animation from saved flight data (skip simulation)",
    )  # pyright: ignore[reportUnusedCallResult]
    return parser.parse_args()


def animation_from_data() -> None:
    """Load saved flight data and generate animation only."""
    if not os.path.isfile(DATA_PATH):
        print(f"No flight data found at {DATA_PATH}")
        print("Run the simulation first: just run-parallel-imppi")
        sys.exit(1)

    print(f"Loading flight data from {DATA_PATH} ...")
    data = np.load(DATA_PATH, allow_pickle=True)

    os.makedirs(MEDIA_DIR, exist_ok=True)
    print("Generating trajectory animation with info field ...")
    dt = float(data["dt"])
    step_skip = 2
    fps = int(1.0 / (step_skip * dt))  # 1:1 real-time (25 fps)
    anim_path = create_parallel_trajectory_html(
        data["history_x"],
        data["history_field"],
        data["history_field_origin"],
        data["history_ref_traj"],
        data["grid"],
        float(data["map_resolution"]),
        FIELD_RES,
        dt,
        save_path=HTML_PATH,
        step_skip=step_skip,
        fps=fps,
    )
    print(f"Saved to: {anim_path}")


def main() -> None:
    args = parse_args()

    if args.animation_only:
        animation_from_data()
        return

    # --- JAX device info ---
    print(f"JAX version : {jax.__version__}")
    print(f"Devices     : {jax.devices()}")
    print()

    # --- Environment ---
    grid_map_obj, grid_array, map_origin, map_resolution = create_grid_map()

    sim_steps = int(round(SIM_DURATION * CONTROL_HZ))

    print("=" * 60)
    print("Parallel I-MPPI Simulation (FOV Grid Update)")
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
        f"  Weights    : info={LAMBDA_INFO}, local={LAMBDA_LOCAL}, target={TARGET_WEIGHT}"
    )
    print(f"  Ref traj   : speed={REF_SPEED}m/s, horizon={REF_HORIZON}")
    print(f"  FSMI Beams : {FSMI_BEAMS},  Range: {FSMI_RANGE}m")
    print()

    # --- Initial state (13D quadrotor only) ---
    x0 = jnp.zeros(NX)
    x0 = x0.at[:3].set(jnp.array([START_X, START_Y, -2.0]))
    x0 = x0.at[6].set(1.0)  # qw = 1

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
        ref_speed=REF_SPEED,
        ref_horizon=REF_HORIZON,
        target_weight=TARGET_WEIGHT,
        goal_weight=GOAL_WEIGHT,
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
        sim_steps=sim_steps,
        progress_callback=_progress,
    )

    print("JIT compiling + running ...")
    t0 = time.perf_counter()
    (
        final_state,
        history_x,
        actions,
        done_step,
        history_field,
        history_field_origin,
        final_grid,
        history_ref_traj,
    ) = sim_fn(x0, ctrl_state)
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
    actions = actions[:n_active]
    history_field = history_field[:n_active]
    history_field_origin = history_field_origin[:n_active]
    history_ref_traj = history_ref_traj[:n_active]

    # --- Metrics ---
    action_jerk, traj_jerk = compute_smoothness(actions, history_x, DT)
    final_pos = final_state[:3]
    goal_dist = float(jnp.linalg.norm(final_pos - GOAL_POS))

    status = "Completed" if done_step_int > 0 else "Timeout"

    print(f"Runtime : {runtime:.2f}s ({runtime / SIM_DURATION:.2f}x realtime)")
    print(f"  Goal dist  : {goal_dist:.2f}m")
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
        actions=np.array(actions),
        history_field=np.array(history_field),
        history_field_origin=np.array(history_field_origin),
        history_ref_traj=np.array(history_ref_traj),
        grid=np.array(grid_array),
        final_grid=np.array(final_grid),
        map_resolution=map_resolution,
        dt=DT,
        done_step=done_step_int,
        runtime=runtime,
        goal_dist=goal_dist,
        status=status,
    )
    print(f"\nSaved flight data to {DATA_PATH}")

    # --- 2D trajectory figure ---
    fig = plot_trajectory_2d(
        history_x,
        grid_array,
        map_resolution,
        title=f"Parallel I-MPPI Trajectory [{status}]",
    )
    fig.write_html(SUMMARY_PATH)
    print(f"Saved summary plot to {SUMMARY_PATH}")
    fig.show()

    # --- Animation with info field (optional) ---
    if args.animation:
        print("Generating trajectory animation with info field ...")
        anim_path = create_parallel_trajectory_html(
            history_x,
            history_field,
            history_field_origin,
            history_ref_traj,
            grid_array,
            map_resolution,
            FIELD_RES,
            DT,
            save_path=HTML_PATH,
            step_skip=2,
            fps=int(1.0 / (2 * DT)),  # 1:1 real-time (25 fps)
        )
        print(f"Saved to: {anim_path}")


if __name__ == "__main__":
    main()
