import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from env_setup import create_grid_map
from sim_utils import (
    DT,
    NOISE_SIGMA,
    NU,
    NX,
    U_INIT,
    U_MAX,
    U_MIN,
    build_parallel_sim_fn,
)
from tqdm import tqdm
from viz_utils import create_trajectory_gif, plot_trajectory_2d

from jax_mppi import mppi
from jax_mppi.i_mppi.environment import GOAL_POS

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
    """Run the Parallel I-MPPI simulation."""

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
    start_pos = jnp.array([1.0, 6.0, -2.0])
    start_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    start_vel = jnp.zeros(3)
    start_omega = jnp.zeros(3)
    # Info levels for 3 zones
    start_info = jnp.array([100.0, 100.0, 100.0])

    state = jnp.concatenate(
        [start_pos, start_vel, start_quat, start_omega, start_info]
    )

    # 4. Simulation Loop
    history_x = []
    actions = []
    trajectory_data = []

    print("Starting simulation...")
    start_time = time.time()

    for t in tqdm(range(max_steps)):
        state, action, mppi_state, info = sim_step_fn(t, state, mppi_state)

        # Store data
        history_x.append(np.array(state[:13]))
        actions.append(np.array(action))

        # Store trajectory for visualization (every 5 steps)
        if t % 5 == 0:
            trajectory_data.append(
                {
                    "time": t * DT,
                    "state": np.array(state),
                    "action": np.array(action),
                    "info_field": np.array(
                        info.get("field", np.zeros((10, 10)))
                    ),
                    "field_origin": np.array(
                        info.get("field_origin", np.zeros(2))
                    ),
                }
            )

        # Termination Check
        pos = state[:3]
        dist_goal = jnp.linalg.norm(pos - GOAL_POS)
        if dist_goal < 0.5:
            print(f"Goal reached at step {t}!")
            break

    total_time = time.time() - start_time
    print(f"Simulation finished in {total_time:.2f}s")
    print(f"Average FPS: {len(history_x) / total_time:.2f}")

    return history_x, actions, trajectory_data, grid_map


def make_gif(history_x, trajectory_data, grid_map):
    """Create animated GIF showing trajectory + reference trajectory + info field heatmap."""
    from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: F401
    from matplotlib.patches import Polygon  # noqa: F401
    from viz_utils import _INFO_GAIN_CMAP, _fov_polygon  # noqa: F401

    # (implementation detail - reusing create_trajectory_gif from viz_utils in practice)
    # For this script, we just assume viz_utils.create_trajectory_gif handles it
    pass


if __name__ == "__main__":
    history_x, actions, traj_data, grid_map = run_simulation()

    # 5. Visualization
    grid_array = np.array(grid_map.grid)
    map_resolution = grid_map.resolution

    plot_trajectory_2d(
        history_x,
        grid_array,
        map_resolution,
        title="Parallel I-MPPI Trajectory",
    )

    os.makedirs(MEDIA_DIR, exist_ok=True)

    # Create GIF
    gif_path = os.path.join(MEDIA_DIR, "i_mppi_parallel_demo.gif")
    print(f"Creating animation at {gif_path}...")
    create_trajectory_gif(
        traj_data,
        grid_array,
        map_resolution,
        filename=gif_path,
        dt=DT * 5,
    )

    if IN_COLAB:
        from IPython.display import Image, display

        display(Image(filename=gif_path))
    else:
        plt.show()
