#!/usr/bin/env python3
"""I-MPPI UGV simulation script.

Two-layer hierarchical I-MPPI for a differential-drive UGV:
- Layer 2 (5 Hz): FSMI info field + gradient reference trajectory
- Layer 3 (50 Hz): biased MPPI + Uniform-FSMI + FOV grid update

Same environment as the UAV version (L-shaped corridor with info zones).

Run from the repo root::

    pixi run python examples/i_mppi_ugv/i_mppi_ugv_simulation.py
"""

import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    _script_dir,
    os.path.join(os.getcwd(), "examples", "i_mppi_ugv"),
]
for _d in _candidates:
    if os.path.isfile(os.path.join(_d, "ugv_sim_utils.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break

# Also add i_mppi dir for env_setup and viz_utils
_imppi_dir = os.path.join(_script_dir, "..", "i_mppi")
if _imppi_dir not in sys.path:
    sys.path.insert(0, _imppi_dir)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from env_setup import create_grid_map  # noqa: E402
from ugv_sim_utils import (  # noqa: E402
    CONTROL_HZ,
    DT,
    NOISE_SIGMA,
    NU,
    NX,
    U_INIT,
    U_MAX,
    U_MIN,
    build_ugv_parallel_sim_fn,
    compute_smoothness,
)
from tqdm import tqdm  # noqa: E402
from viz_utils import plot_trajectory_2d  # noqa: E402

from jax_mppi import mppi  # noqa: E402
from jax_mppi.i_mppi.ugv_environment import GOAL_POS_2D  # noqa: E402
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

MEDIA_DIR = os.path.join(_script_dir, "..", "..", "docs", "_media", "i_mppi_ugv")
SUMMARY_PATH = os.path.join(MEDIA_DIR, "ugv_imppi_summary.png")
DATA_PATH = os.path.join(MEDIA_DIR, "ugv_imppi_flight_data.npz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # --- JAX device info ---
    print(f"JAX version : {jax.__version__}")
    print(f"Devices     : {jax.devices()}")
    print()

    # --- Environment (same map as UAV) ---
    grid_map_obj, grid_array, map_origin, map_resolution = create_grid_map()

    sim_steps = int(round(SIM_DURATION * CONTROL_HZ))

    print("=" * 60)
    print("I-MPPI UGV Simulation (Differential Drive)")
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
    print(f"  State dim  : {NX} (UGV),  Control dim: {NU}")
    print()

    # --- Initial state (5D UGV) ---
    x0 = jnp.array([START_X, START_Y, 0.0, 0.0, 0.0])

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

    sim_fn = build_ugv_parallel_sim_fn(
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

    # --- Metrics ---
    action_jerk, traj_jerk = compute_smoothness(actions, history_x, DT)
    final_pos = final_state[:2]
    goal_dist = float(jnp.linalg.norm(final_pos - GOAL_POS_2D))

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
        history_field=np.array(history_field[:n_active]),
        history_field_origin=np.array(history_field_origin[:n_active]),
        history_ref_traj=np.array(history_ref_traj[:n_active]),
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
        title=f"I-MPPI UGV Trajectory [{status}]",
    )
    fig.savefig(SUMMARY_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved summary plot to {SUMMARY_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
