"""
Grid-based FSMI demonstration script.

This script demonstrates the actual FSMI algorithm from Zhang et al. (2020)
using occupancy grid maps instead of geometric zones.

Features:
- Creates a synthetic occupancy grid with obstacles
- Computes FSMI at different robot poses
- Visualizes information gain heatmaps
- Compares grid FSMI vs legacy geometric method
"""

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax_mppi.i_mppi.environment import INFO_ZONES
from jax_mppi.i_mppi.fsmi import FSMIConfig, FSMIModule, FSMITrajectoryGenerator


def create_synthetic_grid(width=100, height=100, resolution=0.1):
    """
    Create a synthetic occupancy grid with high-entropy regions.

    Args:
        width: Grid width in cells
        height: Grid height in cells
        resolution: Meters per cell

    Returns:
        grid_map: (height, width) occupancy probability grid
        map_origin: (x, y) world origin
    """
    # Start with unknown (0.5 probability)
    grid = 0.5 * jnp.ones((height, width))

    # Add some known free space (low occupancy)
    grid = grid.at[20:80, 20:40].set(0.2)

    # Add some obstacles (high occupancy)
    grid = grid.at[40:50, 50:60].set(0.8)
    grid = grid.at[60:70, 30:40].set(0.7)

    # Add frontiers (high entropy regions at boundaries)
    # These are the most informative areas
    grid = grid.at[30:35, 55:70].set(0.5)  # Unknown region
    grid = grid.at[50:55, 20:35].set(0.5)  # Unknown region

    map_origin = jnp.array([0.0, 0.0])

    return grid, map_origin, resolution


def visualize_grid_with_fsmi(grid_map, map_origin, resolution, config):
    """
    Visualize occupancy grid and FSMI values at different poses.

    Args:
        grid_map: Occupancy probability grid
        map_origin: World origin
        resolution: Meters per cell
        config: FSMI configuration
    """
    fsmi = FSMIModule(config, map_origin, resolution)

    # Sample poses across the grid
    H, W = grid_map.shape
    world_width = W * resolution
    world_height = H * resolution

    # Create grid of poses (every meter)
    x_samples = jnp.arange(1.0, world_width - 1.0, 1.0)
    y_samples = jnp.arange(1.0, world_height - 1.0, 1.0)
    xx, yy = jnp.meshgrid(x_samples, y_samples)
    poses = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    # Compute FSMI for each pose (looking right: yaw=0)
    yaws = jnp.zeros(poses.shape[0])

    print(f"Computing FSMI for {poses.shape[0]} poses...")
    fsmi_values = jax.vmap(fsmi.compute_fsmi, in_axes=(None, 0, 0))(
        grid_map, poses, yaws
    )

    # Reshape for visualization
    fsmi_grid = fsmi_values.reshape(xx.shape)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Occupancy Grid
    ax = axes[0]
    extent = [0, world_width, 0, world_height]
    im1 = ax.imshow(
        np.array(grid_map),
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax.set_title("Occupancy Grid\n(0=free, 0.5=unknown, 1=occupied)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im1, ax=ax, label="Occupancy Probability")

    # Plot 2: FSMI Heatmap
    ax = axes[1]
    im2 = ax.contourf(
        np.array(xx),
        np.array(yy),
        np.array(fsmi_grid),
        levels=20,
        cmap="hot",
    )
    ax.set_title("FSMI Heatmap\n(Higher = More informative)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im2, ax=ax, label="Mutual Information")

    # Plot 3: Overlay
    ax = axes[2]
    ax.imshow(
        np.array(grid_map),
        origin="lower",
        extent=extent,
        cmap="gray",
        alpha=0.5,
    )
    im3 = ax.contourf(
        np.array(xx),
        np.array(yy),
        np.array(fsmi_grid),
        levels=20,
        cmap="hot",
        alpha=0.6,
    )
    ax.set_title("Overlay: Grid + FSMI")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im3, ax=ax, label="Mutual Information")

    plt.tight_layout()
    os.makedirs("docs/_media", exist_ok=True)
    plt.savefig("docs/_media/fsmi_grid_demo.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to docs/_media/fsmi_grid_demo.png")
    plt.close()


def compare_fsmi_modes():
    """
    Compare grid-based FSMI with legacy geometric method.
    """
    # Create grid
    grid_map, map_origin, resolution = create_synthetic_grid()

    # Setup both modes
    config_grid = FSMIConfig(
        use_grid_fsmi=True,
        fov_rad=1.57,
        num_beams=16,
        max_range=5.0,
        ray_step=0.1,
        trajectory_subsample_rate=5,
    )

    config_legacy = FSMIConfig(
        use_grid_fsmi=False,
        info_weight=10.0,
    )

    # Test trajectory (straight line)
    horizon = 40
    dt = 0.05
    pos0 = jnp.array([2.0, 5.0, -2.0])
    target = jnp.array([8.0, 5.0, -2.0])

    # Generate trajectory
    direction = target - pos0
    dist = jnp.linalg.norm(direction)
    unit = direction / dist
    step_d = 2.0 * dt * jnp.arange(horizon)  # ref_speed = 2.0
    step_d = jnp.minimum(step_d, dist)
    ref_traj = pos0[None, :] + step_d[:, None] * unit[None, :]

    view_dir_xy = target[:2] - pos0[:2]

    # Grid-based FSMI
    print("\n=== Grid-based FSMI ===")
    fsmi_gen_grid = FSMITrajectoryGenerator(
        config_grid, INFO_ZONES, map_origin, resolution
    )
    info_gain_grid = fsmi_gen_grid._info_gain_grid(
        ref_traj, view_dir_xy, grid_map, dt
    )
    print(f"Grid FSMI info gain: {info_gain_grid:.4f}")

    # Legacy geometric FSMI
    print("\n=== Legacy Geometric FSMI ===")
    fsmi_gen_legacy = FSMITrajectoryGenerator(config_legacy, INFO_ZONES)
    info_levels = jnp.array([100.0, 100.0])
    info_gain_legacy = fsmi_gen_legacy._info_gain_legacy(
        ref_traj, view_dir_xy, info_levels, dt
    )
    print(f"Legacy info gain: {info_gain_legacy:.4f}")

    print("\nKey differences:")
    print("- Grid FSMI: Uses ray casting and occupancy probabilities")
    print("- Legacy: Uses geometric gates (FOV, range, proximity)")
    print(
        f"- Ratio (Grid/Legacy): {info_gain_grid / (info_gain_legacy + 1e-6):.2f}"
    )


def demo_beam_computation():
    """
    Demonstrate the core FSMI beam computation.
    """
    print("\n=== FSMI Beam Computation Demo ===")

    # Create simple ray with varying occupancy
    N = 50  # 5 meters at 10cm resolution
    cell_dists = jnp.arange(N) * 0.1

    # Scenario: Free space, then frontier, then obstacle
    cell_probs = jnp.concatenate([
        0.2 * jnp.ones(20),  # Free
        0.5 * jnp.ones(10),  # Unknown (frontier)
        0.8 * jnp.ones(20),  # Occupied
    ])

    config = FSMIConfig()
    map_origin = jnp.array([0.0, 0.0])
    fsmi = FSMIModule(config, map_origin, 0.1)

    mi = fsmi._compute_beam_fsmi(cell_probs, cell_dists)

    print(f"Single beam MI: {mi:.6f}")
    print(f"Cell probabilities shape: {cell_probs.shape}")
    print(
        f"Cell distances range: [{cell_dists[0]:.2f}, {cell_dists[-1]:.2f}] m"
    )

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot cell probabilities
    ax = axes[0]
    ax.plot(np.array(cell_dists), np.array(cell_probs), "b-", linewidth=2)
    ax.axhline(0.5, color="r", linestyle="--", label="Unknown threshold")
    ax.fill_between(
        np.array(cell_dists),
        np.array(cell_probs),
        alpha=0.3,
    )
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Occupancy Probability")
    ax.set_title("Ray Occupancy Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Compute P(e_j) for visualization
    not_occ = 1.0 - cell_probs
    cum_not_occ = jnp.cumprod(not_occ)
    shifted_cum = jnp.concatenate([jnp.array([1.0]), cum_not_occ[:-1]])
    P_e = cell_probs * shifted_cum

    ax = axes[1]
    ax.plot(np.array(cell_dists), np.array(P_e), "g-", linewidth=2)
    ax.fill_between(
        np.array(cell_dists),
        np.array(P_e),
        alpha=0.3,
        color="g",
    )
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("P(beam stops at cell)")
    ax.set_title(f"Beam Termination Probability (Total MI = {mi:.6f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("docs/_media", exist_ok=True)
    plt.savefig("docs/_media/fsmi_beam_demo.png", dpi=150, bbox_inches="tight")
    print("Saved beam visualization to docs/_media/fsmi_beam_demo.png")
    plt.close()


def main():
    print("=" * 60)
    print("Grid-based FSMI Demonstration")
    print("Zhang et al. (2020): Fast Entropy-Based Informative Planning")
    print("=" * 60)

    # Demo 1: Single beam computation
    demo_beam_computation()

    # Demo 2: Full grid visualization
    print("\n" + "=" * 60)
    grid_map, map_origin, resolution = create_synthetic_grid()
    config = FSMIConfig(
        use_grid_fsmi=True,
        num_beams=16,
        max_range=5.0,
    )
    visualize_grid_with_fsmi(grid_map, map_origin, resolution, config)

    # Demo 3: Comparison
    print("\n" + "=" * 60)
    compare_fsmi_modes()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("Generated files:")
    print("  - docs/_media/fsmi_beam_demo.png: Single beam computation")
    print("  - docs/_media/fsmi_grid_demo.png: Grid FSMI heatmap")
    print("=" * 60)


if __name__ == "__main__":
    main()
