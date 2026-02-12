import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from env_setup import create_grid_map
from sim_utils import DT
from tqdm import tqdm
from viz_utils import create_trajectory_gif

from jax_mppi import mppi
from jax_mppi.i_mppi.environment import GOAL_POS, INFO_ZONES

# ---------------------------------------------------------------------------
# Setup and Configuration
# ---------------------------------------------------------------------------

try:
    import google.colab  # type: ignore # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Running on JAX backend: {jax.default_backend()}")

# Output directory for media
MEDIA_DIR = "docs/_media/i_mppi"
os.makedirs(MEDIA_DIR, exist_ok=True)


def run_simulation(
    max_steps: int = 600,
    seed: int = 42,
    lambda_: float = 0.05,
    num_samples: int = 2000,
    save_gif: bool = True,
):
    """Run the I-MPPI simulation and return history for visualization."""
    from sim_utils import (
        NOISE_SIGMA,
        NU,
        NX,
        U_INIT,
        U_MAX,
        U_MIN,
        build_parallel_sim_fn,
    )

    # 1. Environment Setup
    grid_map = create_grid_map()
    print(f"Grid Map created: {grid_map.width}x{grid_map.height}")

    # 2. Controller Setup (Parallel I-MPPI)
    config, mppi_state = mppi.create(
        nx=NX,
        nu=NU,
        horizon=40,
        num_samples=num_samples,
        noise_sigma=NOISE_SIGMA,
        lambda_=lambda_,
        u_min=U_MIN,
        u_max=U_MAX,
        u_init=U_INIT,
        step_dependent_dynamics=True,
    )

    # JIT-compile the simulation step
    sim_step_fn = build_parallel_sim_fn(config, grid_map)

    # 3. Initial State
    # [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, info1, info2, info3]
    start_pos = jnp.array([1.0, 6.0, -2.0])
    # Facing +x (yaw=0) -> quaternion [1, 0, 0, 0]
    start_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    start_vel = jnp.zeros(3)
    start_omega = jnp.zeros(3)
    start_info = INFO_ZONES[:, 4]  # Initial values [100, 100, 100]

    state = jnp.concatenate(
        [start_pos, start_vel, start_quat, start_omega, start_info]
    )

    # 4. Simulation Loop
    history_x = []
    history_info = []
    actions = []
    trajectory_data = []

    print("Starting simulation...")
    for t in tqdm(range(max_steps)):
        state, action, mppi_state, info = sim_step_fn(t, state, mppi_state)

        # Store data
        history_x.append(np.array(state[:13]))
        history_info.append(np.array(state[13:]))
        actions.append(np.array(action))

        # Store trajectory for visualization (every 5 steps)
        if t % 5 == 0:
            # Extract rollouts from the controller state for viz
            # This is a bit tricky with the functional API; we'd need to re-rollout
            # For now, we'll just store the robot state
            trajectory_data.append(
                {
                    "time": t * DT,
                    "state": np.array(state),
                    "action": np.array(action),
                    "info_field": np.array(
                        info["field"]
                    ),  # Field snapshot
                    "field_origin": np.array(info["field_origin"]),
                }
            )

        # Termination Check (Near Goal)
        pos = state[:3]
        dist_goal = jnp.linalg.norm(pos - GOAL_POS)
        if dist_goal < 0.5:
            print(f"Goal reached at step {t}!")
            break

    return history_x, history_info, actions, trajectory_data, grid_map


if __name__ == "__main__":
    from viz_utils import (
        plot_control_inputs,
        plot_info_levels,
        plot_trajectory_2d,
    )

    history_x, history_info, actions, traj_data, grid_map = run_simulation()

    # 5. Visualization
    # Convert grid to numpy for plotting
    grid_array = np.array(grid_map.grid)
    map_resolution = grid_map.resolution

    # Plot 2D Trajectory
    plot_trajectory_2d(
        history_x, grid_array, map_resolution, title="I-MPPI Trajectory"
    )
    # Assign to variable to suppress lint warning, though unused
    _fig_info = plot_info_levels(history_info, DT)
    _fig_controls = plot_control_inputs(actions, DT)

    os.makedirs(MEDIA_DIR, exist_ok=True)

    # Create GIF
    gif_path = os.path.join(MEDIA_DIR, "i_mppi_parallel_demo.gif")
    print(f"Creating animation at {gif_path}...")
    create_trajectory_gif(
        traj_data,
        grid_array,
        map_resolution,
        filename=gif_path,
        dt=DT * 5,  # 5x subsampling
    )

    if IN_COLAB:
        from IPython.display import Image, display

        display(Image(filename=gif_path))
    else:
        plt.show()
