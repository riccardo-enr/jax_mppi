#!/usr/bin/env python3
"""Debug script to visualize FSMI field across the full map and depletion effects.

Computes FSMI information gain at a coarse grid over the entire map,
shows how depletion changes the field (depleted areas → brown),
and overlays a gradient ascent reference trajectory.

Run::
    pixi run python examples/i_mppi/debug_fsmi_field.py
"""

import os
import sys

_candidates = [
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(os.getcwd(), "examples", "i_mppi"),
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
    _FREE_PROB,
    _OCC_PROB,
    interp2d,
)

from jax_mppi.i_mppi.environment import GOAL_POS, INFO_ZONES  # noqa: E402
from jax_mppi.i_mppi.fsmi import FSMIConfig, FSMIModule  # noqa: E402

# --- Parameters ---
START_X, START_Y = 1.0, 5.0
ALTITUDE = -2.0
FSMI_BEAMS = 12
FSMI_RANGE = 5.0

FIELD_RES = 0.5  # meters per cell for the evaluation grid
N_YAW = 8  # yaw angles to evaluate per position

REF_SPEED = 2.0
DT = 0.05
HORIZON = 40


def _build_field_grid(grid_shape, map_origin, map_resolution, field_res):
    """Build the coarse evaluation grid (pure NumPy, no JIT)."""
    H, W = grid_shape
    map_w = W * map_resolution
    map_h = H * map_resolution
    margin = field_res
    field_xs = np.arange(margin, map_w - margin, field_res)
    field_ys = np.arange(margin, map_h - margin, field_res)
    return field_xs, field_ys


def compute_full_map_fsmi(
    fsmi_module,
    grid_array,
    flat_positions,
    yaws,
    Nx,
    Ny,
    map_origin,
    map_resolution,
):
    """Compute FSMI at precomputed positions (JIT-friendly).

    Returns:
        field: (Nx, Ny) FSMI values (max over yaw)
    """
    H, W = grid_array.shape

    # Mask out wall positions
    gx_idx = ((flat_positions[:, 0] - map_origin[0]) / map_resolution).astype(
        jnp.int32
    )
    gy_idx = ((flat_positions[:, 1] - map_origin[1]) / map_resolution).astype(
        jnp.int32
    )
    gx_idx = jnp.clip(gx_idx, 0, W - 1)
    gy_idx = jnp.clip(gy_idx, 0, H - 1)
    grid_vals = grid_array[gy_idx, gx_idx]
    is_wall = grid_vals >= 0.7

    # Compute FSMI for all (position, yaw) pairs via double vmap
    def fsmi_fn(pos, yaw):
        return fsmi_module.compute_fsmi(grid_array, pos, yaw)

    fsmi_vmap_pos = jax.vmap(fsmi_fn, in_axes=(0, None))
    fsmi_vmap_yaw = jax.vmap(fsmi_vmap_pos, in_axes=(None, 0))

    gains = fsmi_vmap_yaw(flat_positions, yaws)  # (n_yaw, N)
    field_flat = gains.max(axis=0)  # max over yaw

    # Zero out wall positions
    field_flat = jnp.where(is_wall, 0.0, field_flat)

    return field_flat.reshape(Nx, Ny)


def field_gradient_trajectory(
    field, field_origin, field_res, start_xy, horizon, ref_speed, dt, altitude
):
    """Generate reference trajectory via gradient ascent on the info field."""
    grad_x = jnp.gradient(field, field_res, axis=0)
    grad_y = jnp.gradient(field, field_res, axis=1)

    step_dist = ref_speed * dt

    def step_fn(carry, _):
        xy = carry
        gx = interp2d(grad_x, field_origin, field_res, xy)
        gy = interp2d(grad_y, field_origin, field_res, xy)
        grad = jnp.array([gx, gy])
        grad_norm = jnp.linalg.norm(grad)
        direction = grad / jnp.maximum(grad_norm, 1e-6)
        next_xy = xy + step_dist * direction
        return next_xy, next_xy

    _, trajectory_xy = jax.lax.scan(step_fn, start_xy, None, length=horizon)
    z_col = jnp.full((horizon, 1), altitude)
    return jnp.concatenate([trajectory_xy, z_col], axis=1)


def plot_field_panel(
    ax,
    fig,
    grid_array,
    map_resolution,
    field_np,
    field_origin,
    field_res,
    title,
    annotate_points=None,
    trajectory=None,
):
    """Plot grid + FSMI field heatmap on a single axis."""
    H, W = grid_array.shape
    grid_extent = [0, W * map_resolution, 0, H * map_resolution]
    ax.imshow(
        np.array(grid_array),
        origin="lower",
        extent=grid_extent,
        cmap="gray_r",
        vmin=0,
        vmax=1,
        alpha=0.5,
    )

    Nx, Ny = field_np.shape
    fx0, fy0 = float(field_origin[0]), float(field_origin[1])
    field_extent = [fx0, fx0 + Nx * field_res, fy0, fy0 + Ny * field_res]

    im = ax.imshow(
        field_np.T,
        origin="lower",
        extent=field_extent,
        cmap="YlOrBr_r",
        alpha=0.7,
        interpolation="bilinear",
        vmin=0,
    )
    fig.colorbar(im, ax=ax, shrink=0.7, label="FSMI gain")

    # Mark start and goal
    ax.plot(START_X, START_Y, "go", markersize=10, label="Start", zorder=5)
    ax.plot(
        float(GOAL_POS[0]),
        float(GOAL_POS[1]),
        "r*",
        markersize=14,
        label="Goal",
        zorder=5,
    )

    # Draw info zones
    for i in range(len(INFO_ZONES)):
        cx, cy = float(INFO_ZONES[i, 0]), float(INFO_ZONES[i, 1])
        w, h = float(INFO_ZONES[i, 2]), float(INFO_ZONES[i, 3])
        rect = plt.Rectangle(  # pyright: ignore[reportPrivateImportUsage]
            (cx - w / 2, cy - h / 2),
            w,
            h,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            cx,
            cy + h / 2 + 0.3,
            f"Zone {i + 1}",
            ha="center",
            fontsize=8,
            color="cyan",
            fontweight="bold",
        )

    # Annotate specific sample points with FSMI values
    if annotate_points is not None:
        for px, py, val in annotate_points:
            ax.plot(px, py, "k+", markersize=8, markeredgewidth=1.5, zorder=7)
            ax.annotate(
                f"{val:.2f}",
                (px, py),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    # Gradient trajectory
    if trajectory is not None:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "c-o",
            markersize=3,
            linewidth=2,
            label="Gradient traj",
            zorder=6,
        )

    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_aspect("equal")


def main():
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    # --- Environment ---
    grid_map_obj, grid_array, map_origin, map_resolution = create_grid_map()

    # --- FSMI module ---
    fsmi_config = FSMIConfig(
        num_beams=FSMI_BEAMS,
        max_range=FSMI_RANGE,
        ray_step=0.15,
        fov_rad=1.57,
    )
    fsmi_module = FSMIModule(fsmi_config, map_origin, map_resolution)

    # --- Build evaluation grid (outside JIT) ---
    field_xs, field_ys = _build_field_grid(
        grid_array.shape, map_origin, map_resolution, FIELD_RES
    )
    Nx, Ny = len(field_xs), len(field_ys)
    positions_grid = np.stack(
        np.meshgrid(field_xs, field_ys, indexing="ij"), axis=-1
    )
    flat_positions = jnp.array(positions_grid.reshape(-1, 2))
    yaws = jnp.linspace(0, 2 * jnp.pi, N_YAW, endpoint=False)
    field_origin = jnp.array([field_xs[0], field_ys[0]])

    # --- Step 1: Compute full-map FSMI (undepleted) ---
    print("Computing full-map FSMI field (undepleted)...")
    _compute_field = jax.jit(
        lambda g: compute_full_map_fsmi(
            fsmi_module,
            g,
            flat_positions,
            yaws,
            Nx,
            Ny,
            map_origin,
            map_resolution,
        )
    )
    field = _compute_field(grid_array)
    field.block_until_ready()
    print(f"  Field shape: {field.shape}")
    print(
        f"  Field range: [{float(field.min()):.4f}, {float(field.max()):.4f}]"
    )

    # Sample N annotated points (zone centers + corridors + start/goal area)
    sample_points_xy = [
        (2.5, 6.0),  # Zone 1 center
        (11.5, 6.0),  # Zone 2 center
        (11.5, 2.0),  # Zone 3 center
        (1.0, 5.0),  # Start
        (9.0, 5.0),  # Goal
        (5.0, 5.0),  # Mid-corridor
        (7.0, 3.0),  # Lower corridor
        (7.0, 8.0),  # Upper corridor
    ]
    annotate_pts = []
    for px, py in sample_points_xy:
        val = float(
            interp2d(field, field_origin, FIELD_RES, jnp.array([px, py]))
        )
        annotate_pts.append((px, py, val))
        print(f"  FSMI at ({px:.1f}, {py:.1f}): {val:.4f}")

    # --- Step 2: Simulate zone 1 fully observed ---
    print(
        f"\nSimulating zone 1 observed (non-obstacle cells → {_FREE_PROB})..."
    )
    H, W = grid_array.shape
    # Build mask: cells inside zone 1 that are not obstacles
    z = INFO_ZONES[0]  # (cx, cy, w, h)
    cx, cy, zw, zh = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    ys = jnp.arange(H) * map_resolution + map_origin[1]
    xs = jnp.arange(W) * map_resolution + map_origin[0]
    xx, yy = jnp.meshgrid(xs, ys, indexing="xy")
    in_zone = (jnp.abs(xx - cx) < zw / 2) & (jnp.abs(yy - cy) < zh / 2)
    not_obstacle = grid_array < 0.7
    free_mask = in_zone & not_obstacle
    occ_mask = in_zone & ~not_obstacle
    depleted_grid = jnp.where(free_mask, _FREE_PROB, grid_array)
    depleted_grid = jnp.where(occ_mask, _OCC_PROB, depleted_grid)

    print("Computing full-map FSMI field (zone 1 depleted)...")
    field_depleted = _compute_field(depleted_grid)
    field_depleted.block_until_ready()
    print(
        f"  Field range: [{float(field_depleted.min()):.4f}, {float(field_depleted.max()):.4f}]"
    )

    # Annotated points on depleted field
    annotate_pts_depleted = []
    for px, py in sample_points_xy:
        val = float(
            interp2d(
                field_depleted, field_origin, FIELD_RES, jnp.array([px, py])
            )
        )
        annotate_pts_depleted.append((px, py, val))
        print(f"  FSMI at ({px:.1f}, {py:.1f}): {val:.4f}")

    # --- Step 3: Gradient trajectory on undepleted field ---
    print(f"\nGenerating gradient ascent trajectory (horizon={HORIZON})...")
    uav_xy = jnp.array([START_X, START_Y])
    ref_traj = field_gradient_trajectory(
        field,
        field_origin,
        FIELD_RES,
        uav_xy,
        HORIZON,
        REF_SPEED,
        DT,
        ALTITUDE,
    )
    ref_traj_np = np.array(ref_traj)
    print(
        f"  Trajectory: ({ref_traj_np[0, 0]:.2f}, {ref_traj_np[0, 1]:.2f}) -> "
        f"({ref_traj_np[-1, 0]:.2f}, {ref_traj_np[-1, 1]:.2f})"
    )

    # --- Visualization ---
    field_np = np.array(field)
    field_depleted_np = np.array(field_depleted)
    grid_np = np.array(grid_array)
    depleted_grid_np = np.array(depleted_grid)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel 1: Full-map FSMI (undepleted) with sample points + gradient trajectory
    plot_field_panel(
        axes[0],
        fig,
        grid_np,
        map_resolution,
        field_np,
        field_origin,
        FIELD_RES,
        "FSMI Field (Undepleted) + Gradient Trajectory",
        annotate_points=annotate_pts,
        trajectory=ref_traj_np,
    )

    # Panel 2: Full-map FSMI (zone 1 depleted) with sample points
    plot_field_panel(
        axes[1],
        fig,
        depleted_grid_np,
        map_resolution,
        field_depleted_np,
        field_origin,
        FIELD_RES,
        "FSMI Field (Zone 1 Depleted)",
        annotate_points=annotate_pts_depleted,
    )

    # Panel 3: Difference field (how much FSMI dropped)
    diff_np = field_np - field_depleted_np
    ax = axes[2]
    H_g, W_g = grid_np.shape
    grid_extent = [0, W_g * map_resolution, 0, H_g * map_resolution]
    ax.imshow(
        grid_np,
        origin="lower",
        extent=grid_extent,
        cmap="gray_r",
        vmin=0,
        vmax=1,
        alpha=0.5,
    )
    Nx, Ny = diff_np.shape
    fx0, fy0 = float(field_origin[0]), float(field_origin[1])
    diff_extent = [fx0, fx0 + Nx * FIELD_RES, fy0, fy0 + Ny * FIELD_RES]
    im = ax.imshow(
        diff_np.T,
        origin="lower",
        extent=diff_extent,
        cmap="RdYlGn",
        alpha=0.7,
        interpolation="bilinear",
    )
    fig.colorbar(im, ax=ax, shrink=0.7, label="FSMI drop (before - after)")
    ax.plot(START_X, START_Y, "go", markersize=10, zorder=5)
    ax.plot(
        float(GOAL_POS[0]), float(GOAL_POS[1]), "r*", markersize=14, zorder=5
    )
    for i in range(len(INFO_ZONES)):
        cx, cy = float(INFO_ZONES[i, 0]), float(INFO_ZONES[i, 1])
        w, h = float(INFO_ZONES[i, 2]), float(INFO_ZONES[i, 3])
        rect = plt.Rectangle(  # pyright: ignore[reportPrivateImportUsage]
            (cx - w / 2, cy - h / 2),
            w,
            h,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(rect)
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_title("FSMI Drop (Zone 1 Depleted)")
    ax.set_xlabel("X (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "docs",
        "_media",
        "i_mppi",
        "debug_fsmi_field.png",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
