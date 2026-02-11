#!/usr/bin/env python3
"""CUDA I-MPPI simulation with proper quadrotor dynamics.

This script runs the CUDA I-MPPI controller using proper RK4 quadrotor dynamics
(NED-FRD convention) with controls [T, wx_cmd, wy_cmd, wz_cmd].

Run from repo root:
    pixi run -e dev python examples/i_mppi/i_mppi_cuda_simulation.py
"""

import os
import sys
import time

# IMPORTANT: Import CUDA module FIRST, before JAX!
# This prevents CUDA context conflicts
sys.path.insert(0, 'third_party/cuda-mppi/build')
import cuda_mppi

# Path setup
_candidates = [
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(os.getcwd(), "examples", "i_mppi"),
]
for _d in _candidates:
    if os.path.isfile(os.path.join(_d, "env_setup.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break

import numpy as np
import matplotlib.pyplot as plt
from env_setup import create_grid_map
from sim_utils import DT, CONTROL_HZ
from tqdm import tqdm

from jax_mppi.i_mppi.environment import GOAL_POS, INFO_ZONES

# ---------------------------------------------------------------------------
# Quadrotor dynamics (numpy, matching CUDA/JAX RK4)
# ---------------------------------------------------------------------------
MASS = 1.0
GRAVITY = 9.81
TAU_OMEGA = 0.05
U_MIN = np.array([0.0, -10.0, -10.0, -10.0], dtype=np.float32)
U_MAX = np.array([4.0 * GRAVITY, 10.0, 10.0, 10.0], dtype=np.float32)
HOVER_THRUST = MASS * GRAVITY  # 9.81 N


def quat_to_rot(q):
    """Quaternion [qw,qx,qy,qz] to rotation matrix (body FRD -> world NED)."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2),  2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),      1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def quat_deriv(q, omega):
    """Quaternion derivative: q_dot = 0.5 * q x [0, omega]."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    wx, wy, wz = omega[0], omega[1], omega[2]
    return 0.5 * np.array([
        -wx*qx - wy*qy - wz*qz,
         wx*qw + wz*qy - wy*qz,
         wy*qw - wz*qx + wx*qz,
         wz*qw + wy*qx - wx*qy,
    ], dtype=np.float64)


def quadrotor_deriv(state, action):
    """State derivative for 13D quadrotor. Control = [T, wx_cmd, wy_cmd, wz_cmd]."""
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]

    thrust = action[0]
    omega_cmd = action[1:4]

    R = quat_to_rot(quat)
    f_gravity = np.array([0.0, 0.0, MASS * GRAVITY], dtype=np.float64)
    f_thrust = R @ np.array([0.0, 0.0, -thrust], dtype=np.float64)
    accel = (f_gravity + f_thrust) / MASS

    omega_dot = (omega_cmd - omega) / TAU_OMEGA
    q_dot = quat_deriv(quat, omega)

    return np.concatenate([vel, accel, q_dot, omega_dot])


def rk4_step(state, action, dt):
    """RK4 integration step with quaternion normalization."""
    s = state.astype(np.float64)
    u = action.astype(np.float64)

    k1 = quadrotor_deriv(s, u)
    k2 = quadrotor_deriv(s + 0.5 * dt * k1, u)
    k3 = quadrotor_deriv(s + 0.5 * dt * k2, u)
    k4 = quadrotor_deriv(s + dt * k3, u)

    s_next = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Normalize quaternion
    qnorm = np.linalg.norm(s_next[6:10]) + 1e-8
    s_next[6:10] /= qnorm

    return s_next.astype(np.float32)


def quat_to_yaw(q):
    """Extract yaw from quaternion [qw, qx, qy, qz]."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2)))


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
START_X = 1.0
START_Y = 5.0
START_Z = -2.0  # NED: negative Z is up
SIM_DURATION = 60.0

# MPPI parameters
NUM_SAMPLES = 1000
HORIZON = 40
LAMBDA = 0.1

# FSMI parameters
FSMI_BEAMS = 12
FSMI_RANGE = 10.0
RAY_STEP = 0.15

# Info field parameters
FIELD_RES = 0.5
FIELD_EXTENT = 5.0
FIELD_N_YAW = 8
FIELD_UPDATE_INTERVAL = 10  # steps between field recomputation

# Cost weights
LAMBDA_INFO = 20.0
LAMBDA_LOCAL = 10.0
TARGET_WEIGHT = 1.0
GOAL_WEIGHT = 0.2

# Reference trajectory
REF_SPEED = 2.0
REF_HORIZON = HORIZON

# FOV sensor parameters
FOV_RAD = 1.57
SENSOR_RANGE = 4.0
FOV_N_RAYS = 64
FOV_RAY_STEP = 0.15

# Goal threshold
GOAL_DONE_THRESHOLD = 1.5

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("=" * 60)
print("CUDA I-MPPI Simulation")
print("=" * 60)

grid_map_obj, grid_array, map_origin, map_resolution = create_grid_map()
grid_flat = np.array(grid_array).flatten().astype(np.float32)
height, width = grid_array.shape

print(f"Grid: {width}x{height}, res={map_resolution}")
print(f"Goal: {GOAL_POS}")

# CUDA grid
cuda_grid = cuda_mppi.OccupancyGrid2D(
    width, height, map_resolution,
    float(map_origin[0]), float(map_origin[1])
)
cuda_grid.upload(grid_flat)

# CUDA info field
info_field = cuda_mppi.InfoField()

# Dynamics
dynamics = cuda_mppi.QuadrotorDynamics()
print(f"Dynamics: mass={dynamics.mass}, g={dynamics.gravity}")

# Cost
cost = cuda_mppi.InformativeCost()
cost.lambda_info = LAMBDA_INFO
cost.lambda_local = LAMBDA_LOCAL
cost.target_weight = TARGET_WEIGHT
cost.goal_weight = GOAL_WEIGHT
cost.collision_penalty = 100.0
cost.height_weight = 1.0
cost.target_altitude = START_Z
cost.action_reg = 0.01

# MPPI config
config = cuda_mppi.MPPIConfig(
    num_samples=NUM_SAMPLES,
    horizon=HORIZON,
    nx=13,
    nu=4,
    lambda_=LAMBDA,
    dt=DT,
    u_scale=1.0,
    w_action_seq_cost=0.0,
    lambda_info=LAMBDA_INFO,
    alpha=0.0,
)
print(f"MPPI config: K={NUM_SAMPLES}, T={HORIZON}, dt={DT}")

# Controller
controller = cuda_mppi.QuadrotorIMPPI(config, dynamics, cost)
controller.update_cost_grid(cuda_grid)

# Initialize control reference to hover (T=mg, zero angular rates)
hover_ref = np.tile(
    np.array([HOVER_THRUST, 0.0, 0.0, 0.0], dtype=np.float32), HORIZON
)
controller.set_reference_trajectory(hover_ref)

# Trajectory generator for reference trajectory
info_zones_np = np.array(INFO_ZONES)
zones_list = [
    [float(z[0]), float(z[1]), float(z[2]), float(z[3]), float(z[4])]
    for z in info_zones_np
]
tg_config = cuda_mppi.TrajectoryGeneratorConfig()
tg_config.ref_speed = REF_SPEED
tg_config.goal_x = float(GOAL_POS[0])
tg_config.goal_y = float(GOAL_POS[1])
tg_config.goal_z = float(GOAL_POS[2])
traj_gen = cuda_mppi.TrajectoryGenerator(tg_config, zones_list)

# Compute initial info field and reference trajectory
info_field.compute(
    cuda_grid,
    uav_x=START_X, uav_y=START_Y,
    field_res=FIELD_RES, field_extent=FIELD_EXTENT, n_yaw=FIELD_N_YAW,
    num_beams=FSMI_BEAMS, max_range=FSMI_RANGE, ray_step=RAY_STEP,
)
controller.update_cost_info_field(info_field)

# Generate initial reference trajectory via field gradient
field_data = info_field.download()
field_Nx, field_Ny = info_field.Nx, info_field.Ny
ref_traj_flat = traj_gen.field_gradient_trajectory(
    field_data, field_Nx, field_Ny,
    info_field.origin_x, info_field.origin_y, info_field.res,
    START_X, START_Y, REF_HORIZON, REF_SPEED, DT, START_Z,
)
controller.set_position_reference(ref_traj_flat, REF_HORIZON)
print(f"Initial ref trajectory: {len(ref_traj_flat)//3} waypoints")

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
state = np.zeros(13, dtype=np.float32)
state[0] = START_X
state[1] = START_Y
state[2] = START_Z
state[6] = 1.0  # qw = 1 (identity quaternion)

# Info levels for zone tracking (not part of CUDA state, tracked externally)
info_levels = [float(z[4]) for z in info_zones_np]

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
sim_steps = int(round(SIM_DURATION * CONTROL_HZ))
history_x = np.zeros((sim_steps, 13), dtype=np.float32)
history_actions = np.zeros((sim_steps, 4), dtype=np.float32)
history_field = []
history_field_origin = []

print(f"\nRunning: {sim_steps} steps ({SIM_DURATION}s at {CONTROL_HZ} Hz)")
print("=" * 60)

pbar = tqdm(total=sim_steps, desc="Simulation", unit="step")
t0 = time.perf_counter()
step_count = sim_steps
done = False

for step in range(sim_steps):
    pos = state[0:3]
    dist_to_goal = np.linalg.norm(pos - np.array(GOAL_POS))
    if dist_to_goal < GOAL_DONE_THRESHOLD:
        done = True
        step_count = step
        break

    # --- Low-rate: recompute info field + reference trajectory ---
    if step % FIELD_UPDATE_INTERVAL == 0:
        info_field.compute(
            cuda_grid,
            uav_x=float(state[0]), uav_y=float(state[1]),
            field_res=FIELD_RES, field_extent=FIELD_EXTENT, n_yaw=FIELD_N_YAW,
            num_beams=FSMI_BEAMS, max_range=FSMI_RANGE, ray_step=RAY_STEP,
        )
        controller.update_cost_info_field(info_field)

        # Generate reference trajectory via field gradient ascent
        field_data = info_field.download()
        field_Nx, field_Ny = info_field.Nx, info_field.Ny
        ref_traj_flat = traj_gen.field_gradient_trajectory(
            field_data, field_Nx, field_Ny,
            info_field.origin_x, info_field.origin_y, info_field.res,
            float(state[0]), float(state[1]),
            REF_HORIZON, REF_SPEED, DT, float(state[2]),
        )
        controller.set_position_reference(ref_traj_flat, REF_HORIZON)

        # Store for visualization
        history_field.append(field_data.reshape(field_Nx, field_Ny).copy())
        history_field_origin.append([info_field.origin_x, info_field.origin_y])

    # --- FOV grid update ---
    yaw = quat_to_yaw(state[6:10])
    cuda_grid.update_fov(
        uav_x=float(state[0]), uav_y=float(state[1]),
        yaw=float(yaw),
        fov_rad=FOV_RAD, max_range=SENSOR_RANGE,
        n_rays=FOV_N_RAYS, ray_step=FOV_RAY_STEP,
    )
    controller.update_cost_grid(cuda_grid)

    # --- MPPI control ---
    controller.compute(state)
    action = controller.get_action()

    # Clamp control to physical limits
    action = np.clip(action, U_MIN, U_MAX)

    # --- Proper RK4 quadrotor dynamics ---
    state = rk4_step(state, action, DT)

    # Store
    history_x[step] = state
    history_actions[step] = action

    controller.shift()
    pbar.update(1)

runtime = time.perf_counter() - t0
pbar.close()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if done:
    n_active = step_count
    actual_duration = n_active * DT
    status = "Completed"
else:
    n_active = sim_steps
    actual_duration = SIM_DURATION
    status = "Timeout"

history_x = history_x[:n_active]
history_actions = history_actions[:n_active]

final_pos = history_x[-1, 0:3]
goal_dist = np.linalg.norm(final_pos - np.array(GOAL_POS))

print("\n" + "=" * 60)
print(f"{'Metric':<25} {'Value':>15}")
print("-" * 60)
print(f"{'Status':<25} {status:>15}")
print(f"{'Sim Duration (s)':<25} {actual_duration:>15.1f}")
print(f"{'Runtime (s)':<25} {runtime:>15.2f}")
print(f"{'Realtime Factor':<25} {actual_duration/runtime:>15.2f}x")
print(f"{'Goal Distance (m)':<25} {goal_dist:>15.2f}")
print(f"{'Hz (effective)':<25} {n_active/runtime:>15.1f}")
print(f"{'Avg Thrust (N)':<25} {np.mean(history_actions[:, 0]):>15.2f}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: trajectory on map
ax = axes[0]
extent = [0, width * map_resolution, 0, height * map_resolution]
ax.imshow(grid_array, cmap='Greys', origin='lower', extent=extent, alpha=0.8)
ax.plot(history_x[:, 0], history_x[:, 1], 'cyan', linewidth=2, label='Trajectory')
ax.plot(START_X, START_Y, 'go', markersize=10, label='Start')
ax.plot(float(GOAL_POS[0]), float(GOAL_POS[1]), 'r*', markersize=15, label='Goal')
for z in info_zones_np:
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (z[0] - z[2]/2, z[1] - z[3]/2), z[2], z[3],
        linewidth=2, edgecolor='orange', facecolor='yellow', alpha=0.2,
    )
    ax.add_patch(rect)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'CUDA I-MPPI Trajectory [{status}]')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Right: control inputs
ax2 = axes[1]
t = np.arange(n_active) * DT
ax2.plot(t, history_actions[:, 0], label='Thrust (N)')
ax2.plot(t, history_actions[:, 1], label='wx_cmd')
ax2.plot(t, history_actions[:, 2], label='wy_cmd')
ax2.plot(t, history_actions[:, 3], label='wz_cmd')
ax2.axhline(HOVER_THRUST, color='gray', linestyle='--', alpha=0.5, label='Hover')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Control')
ax2.set_title('Control Inputs [T, wx, wy, wz]')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/_media/i_mppi/cuda_imppi_test.png', dpi=150)
print("\nSaved plot to docs/_media/i_mppi/cuda_imppi_test.png")
plt.show()

print("\nCUDA I-MPPI simulation completed!")
