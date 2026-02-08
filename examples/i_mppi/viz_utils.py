"""Visualization utilities for the I-MPPI interactive simulation."""

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from jax_mppi.i_mppi.environment import (
    GOAL_POS,
    INFO_ZONES,
    SENSOR_FOV_RAD,
    SENSOR_MAX_RANGE,
)

# --- FOV visualisation helpers ---
_FOV_NUM_RAYS = 40  # angular resolution of the FOV wedge
_FOV_RAY_STEP = 0.1  # metres between samples along each ray
_OCC_THRESHOLD = 0.7  # occupancy value considered an obstacle


def _cast_ray(x, y, angle, grid, resolution, max_range, origin=(0.0, 0.0)):
    """Cast a single ray and return the endpoint (truncated by obstacles)."""
    steps = int(max_range / _FOV_RAY_STEP)
    for s in range(1, steps + 1):
        d = s * _FOV_RAY_STEP
        rx = x + d * np.cos(angle)
        ry = y + d * np.sin(angle)
        col = int((rx - origin[0]) / resolution)
        row = int((ry - origin[1]) / resolution)
        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            return rx, ry
        if grid[row, col] >= _OCC_THRESHOLD:
            return rx, ry
    d = max_range
    return x + d * np.cos(angle), y + d * np.sin(angle)


def _fov_polygon(
    x, y, yaw, grid, resolution, max_range=SENSOR_MAX_RANGE, origin=(0.0, 0.0)
):
    """Return (N, 2) polygon vertices for the visible FOV wedge."""
    half_fov = SENSOR_FOV_RAD / 2.0
    angles = np.linspace(yaw - half_fov, yaw + half_fov, _FOV_NUM_RAYS)
    pts = [(x, y)]  # start at UAV
    for a in angles:
        pts.append(_cast_ray(x, y, a, grid, resolution, max_range, origin))
    pts.append((x, y))  # close polygon
    return np.array(pts)


_SEEN_NUM_RAYS = 80  # denser than FOV polygon for better cell coverage


def _compute_seen_mask(x, y, yaw, grid, resolution, max_range, origin=(0.0, 0.0)):
    """Return (H, W) boolean mask of grid cells visible from the given pose.

    Casts ``_SEEN_NUM_RAYS`` rays across the sensor FOV.  Each ray is
    sampled at ``_FOV_RAY_STEP`` intervals and stopped by obstacles
    (occupancy >= ``_OCC_THRESHOLD``) or the grid boundary.  All cells
    visited along visible ray segments are marked True.

    Fully vectorised with numpy — no Python loops over rays or steps.
    """
    H, W = grid.shape
    half_fov = SENSOR_FOV_RAD / 2.0
    n_steps = int(max_range / _FOV_RAY_STEP) + 1
    dists = np.arange(n_steps) * _FOV_RAY_STEP
    angles = np.linspace(yaw - half_fov, yaw + half_fov, _SEEN_NUM_RAYS)

    # Ray sample world coordinates: (n_rays, n_steps)
    ray_x = x + dists[None, :] * np.cos(angles[:, None])
    ray_y = y + dists[None, :] * np.sin(angles[:, None])

    cols = np.int32(np.floor((ray_x - origin[0]) / resolution))
    rows = np.int32(np.floor((ray_y - origin[1]) / resolution))

    valid = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    safe_c = np.clip(cols, 0, W - 1)
    safe_r = np.clip(rows, 0, H - 1)

    occ = grid[safe_r, safe_c]

    # A step is blocking if out of bounds or hits an obstacle
    blocking = (~valid) | (occ >= _OCC_THRESHOLD)
    # Propagate along each ray: once blocked, all subsequent steps are blocked
    cum_block = np.maximum.accumulate(blocking.astype(np.uint8), axis=1)
    # A cell is visible if valid and no *prior* step was blocking
    # (the first blocked cell itself IS visible — you see the wall)
    prev_block = np.zeros_like(cum_block)
    prev_block[:, 1:] = cum_block[:, :-1]
    visible = valid & (prev_block == 0)

    mask = np.zeros((H, W), dtype=bool)
    mask[safe_r[visible], safe_c[visible]] = True
    return mask


def plot_environment(ax, grid, resolution, show_labels=True):
    """Plot the occupancy grid with walls, info zones, start, and goal."""
    extent = [0, grid.shape[1] * resolution, 0, grid.shape[0] * resolution]
    ax.imshow(
        np.array(grid),
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=1,
        alpha=0.8,
    )

    for i in range(len(INFO_ZONES)):
        cx, cy = float(INFO_ZONES[i, 0]), float(INFO_ZONES[i, 1])
        w, h = float(INFO_ZONES[i, 2]), float(INFO_ZONES[i, 3])
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.05",
            facecolor="yellow",
            alpha=0.3,
            edgecolor="orange",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        if show_labels:
            ax.text(
                cx, cy, f"Info {i + 1}", ha="center", va="center", fontsize=8
            )

    ax.plot(1.0, 5.0, "go", markersize=10, label="Start", zorder=5)
    ax.plot(
        float(GOAL_POS[0]),
        float(GOAL_POS[1]),
        "r*",
        markersize=15,
        label="Goal",
        zorder=5,
    )

    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")


def plot_trajectory_2d(
    ax, history_x, grid, resolution, title="I-MPPI Trajectory"
):
    """Plot 2D trajectory over the environment."""
    plot_environment(ax, grid, resolution, show_labels=False)

    positions = np.array(history_x[:, :2])
    n_steps = len(positions)
    colors = plt.colormaps["viridis"](np.linspace(0, 1, n_steps))
    for i in range(n_steps - 1):
        ax.plot(
            positions[i : i + 2, 0],
            positions[i : i + 2, 1],
            color=colors[i],
            linewidth=2,
        )

    ax.plot(positions[0, 0], positions[0, 1], "go", markersize=10, zorder=5)
    ax.plot(positions[-1, 0], positions[-1, 1], "bs", markersize=8, zorder=5)
    ax.plot(
        float(GOAL_POS[0]), float(GOAL_POS[1]), "r*", markersize=15, zorder=5
    )
    ax.set_title(title)


def plot_info_levels(ax, history_info, dt):
    """Plot info zone depletion over time."""
    info = np.array(history_info)
    t = np.arange(len(info)) * dt
    for i in range(info.shape[1]):
        ax.plot(t, info[:, i], linewidth=2, label=f"Info Zone {i + 1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Information Level")
    ax.set_title("Information Zone Depletion")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_control_inputs(axes, actions, dt):
    """Plot control inputs over time (4 subplots)."""
    acts = np.array(actions)
    t = np.arange(len(acts)) * dt
    labels = [
        "Thrust (N)",
        "Omega X (rad/s)",
        "Omega Y (rad/s)",
        "Omega Z (rad/s)",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(t, acts[:, i], color=color, linewidth=1, alpha=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Control Inputs")


def plot_position_3d(ax, history_x):
    """Plot 3D trajectory in ENU frame (z = altitude)."""
    pos = np.array(history_x[:, :3])
    pos[:, 2] = -pos[:, 2]  # NED → ENU: negate z so altitude is positive
    n = len(pos)
    colors = plt.colormaps["viridis"](np.linspace(0, 1, n))
    for i in range(n - 1):
        ax.plot(
            pos[i : i + 2, 0],
            pos[i : i + 2, 1],
            pos[i : i + 2, 2],
            color=colors[i],
            linewidth=1.5,
        )
    ax.scatter(*pos[0], color="green", s=60, zorder=5)
    ax.scatter(
        float(GOAL_POS[0]),
        float(GOAL_POS[1]),
        -float(GOAL_POS[2]),
        color="red",
        marker="*",
        s=100,
        zorder=5,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_zlim(0, 5)
    ax.set_title("3D Trajectory")


def create_trajectory_gif(
    history_x,
    history_info,
    grid,
    resolution,
    dt,
    save_path="i_mppi_trajectory.gif",
    fps=20,
    step_skip=5,
    origin=(0.0, 0.0),
):
    """Create an animated GIF of the UAV trajectory.

    Args:
        history_x: (N, 16) state history.
        history_info: (N, 3) info level history.
        grid: (H, W) occupancy grid.
        resolution: Grid resolution in m/cell.
        dt: Simulation timestep.
        save_path: Output GIF file path.
        fps: Frames per second in the GIF.
        step_skip: Show every N-th simulation step as a frame.
        origin: (x, y) world coordinates of the grid origin.
    """
    positions = np.array(history_x[:, :2])
    # Extract yaw from quaternion (indices 6-9: qw, qx, qy, qz)
    quats = np.array(history_x[:, 6:10])
    yaws = np.arctan2(
        2 * (quats[:, 0] * quats[:, 3] + quats[:, 1] * quats[:, 2]),
        1 - 2 * (quats[:, 2] ** 2 + quats[:, 3] ** 2),
    )
    info = np.array(history_info)
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

    ax_map.plot(1.0, 5.0, "go", markersize=10, zorder=5)
    ax_map.plot(
        float(GOAL_POS[0]), float(GOAL_POS[1]), "r*", markersize=15, zorder=5
    )
    ax_map.set_xlim(-0.5, 14.5)
    ax_map.set_ylim(-0.5, 12.5)
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_aspect("equal")

    # Animated elements
    (trail_line,) = ax_map.plot([], [], "c-", linewidth=1.5, alpha=0.6)
    (uav_dot,) = ax_map.plot([], [], "ro", markersize=8, zorder=10)
    heading_arrow = mpatches.FancyArrowPatch(
        (0, 0),
        (0, 0),
        arrowstyle="-|>",
        mutation_scale=12,
        color="red",
        linewidth=2,
        zorder=11,
    )
    ax_map.add_patch(heading_arrow)
    # FOV wedge (filled polygon, updated each frame)
    grid_np = np.array(grid)
    fov_patch = Polygon(
        [[0, 0]],
        closed=True,
        facecolor="cyan",
        alpha=0.2,
        edgecolor="cyan",
        linewidth=0.5,
        zorder=9,
    )
    ax_map.add_patch(fov_patch)

    # --- Explored overlay (cumulative seen cells) ---
    # Precompute the cumulative seen mask at each frame index
    cumulative_seen = np.zeros(grid_np.shape, dtype=bool)
    seen_snapshots = []
    fi = 0
    for k in range(n_steps):
        seen_k = _compute_seen_mask(
            positions[k, 0], positions[k, 1], yaws[k],
            grid_np, resolution, SENSOR_MAX_RANGE, origin,
        )
        cumulative_seen = cumulative_seen | seen_k
        if fi < len(frame_indices) and frame_indices[fi] == k:
            seen_snapshots.append(cumulative_seen.copy())
            fi += 1
    # Fill remaining frames (if last frame_index < n_steps - 1)
    while len(seen_snapshots) < len(frame_indices):
        seen_snapshots.append(cumulative_seen.copy())

    explored_rgba = np.zeros((*grid_np.shape, 4))
    explored_img = ax_map.imshow(
        explored_rgba, origin="lower", extent=extent, zorder=1,
    )

    title = ax_map.set_title("")

    # Info level plot setup
    t_all = np.arange(n_steps) * dt
    zone_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    info_lines = []
    for i in range(info.shape[1]):
        (line,) = ax_info.plot(
            [], [], linewidth=2, color=zone_colors[i], label=f"Zone {i + 1}"
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

    arrow_len = 0.5  # heading arrow length in metres

    def update(frame_idx):
        k = frame_indices[frame_idx]
        # Trail
        trail_line.set_data(positions[: k + 1, 0], positions[: k + 1, 1])
        # UAV position
        uav_dot.set_data([positions[k, 0]], [positions[k, 1]])
        # Heading arrow
        x, y = positions[k, 0], positions[k, 1]
        yaw = yaws[k]
        dx = arrow_len * np.cos(yaw)
        dy = arrow_len * np.sin(yaw)
        heading_arrow.set_positions((x, y), (x + dx, y + dy))
        # FOV wedge
        fov_verts = _fov_polygon(x, y, yaw, grid_np, resolution, origin=origin)
        fov_patch.set_xy(fov_verts)
        # Explored overlay (green tint on seen cells)
        seen = seen_snapshots[frame_idx]
        rgba = np.zeros((*grid_np.shape, 4))
        rgba[seen] = [0.2, 0.8, 0.2, 0.3]
        explored_img.set_data(rgba)
        # Title with time
        title.set_text(f"I-MPPI Trajectory  t = {k * dt:.1f}s")
        # Info zone opacity (fade as depleted)
        for i, patch in enumerate(zone_patches):
            alpha = max(0.05, info[min(k, len(info) - 1), i] / 100.0 * 0.4)
            patch.set_alpha(alpha)
        # Info level lines
        for i, line in enumerate(info_lines):
            line.set_data(t_all[: k + 1], info[: k + 1, i])
        return (
            [trail_line, uav_dot, heading_arrow, fov_patch, explored_img, title]
            + zone_patches
            + info_lines
        )

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 // fps, blit=True
    )
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    return save_path
