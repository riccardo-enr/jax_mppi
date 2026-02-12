"""Visualization utilities for the I-MPPI simulation using Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Rectangle

from jax_mppi.i_mppi.environment import (
    GOAL_POS,
    INFO_ZONES,
    SENSOR_FOV_RAD,
    SENSOR_MAX_RANGE,
)

# --- Style setup ---
_SCIENCE_STYLE = ["science", "no-latex"]

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


def _draw_environment(ax, grid, resolution, show_labels=True):
    """Draw occupancy grid, info zones, start and goal on an axes."""
    H, W = grid.shape
    extent = [0, W * resolution, 0, H * resolution]

    # Occupancy grid
    ax.imshow(
        np.array(grid),
        cmap="Greys",
        origin="lower",
        extent=extent,
        vmin=0,
        vmax=1,
        alpha=0.8,
        aspect="equal",
    )

    # Information zones
    for i in range(len(INFO_ZONES)):
        cx, cy = float(INFO_ZONES[i, 0]), float(INFO_ZONES[i, 1])
        w, h = float(INFO_ZONES[i, 2]), float(INFO_ZONES[i, 3])
        rect = Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            linewidth=2,
            edgecolor="orange",
            facecolor="yellow",
            alpha=0.3,
        )
        ax.add_patch(rect)
        if show_labels:
            ax.text(
                cx, cy, f"Info {i + 1}", ha="center", va="center", fontsize=8
            )

    # Start position
    ax.plot(1.0, 5.0, "o", color="green", markersize=8, label="Start")

    # Goal position
    ax.plot(
        float(GOAL_POS[0]),
        float(GOAL_POS[1]),
        "*",
        color="red",
        markersize=12,
        label="Goal",
    )

    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_aspect("equal")


def plot_environment(grid, resolution, show_labels=True):
    """Plot the occupancy grid with walls, info zones, start, and goal."""
    with plt.style.context(_SCIENCE_STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))
        _draw_environment(ax, grid, resolution, show_labels)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right")
        fig.tight_layout()
    return fig


def plot_trajectory_2d(history_x, grid, resolution, title="I-MPPI Trajectory"):
    """Plot 2D trajectory over the environment."""
    with plt.style.context(_SCIENCE_STYLE):
        fig, ax = plt.subplots(figsize=(9, 7))
        _draw_environment(ax, grid, resolution, show_labels=False)

        positions = np.array(history_x[:, :2])
        n_steps = len(positions)
        colors = np.linspace(0, 1, n_steps)

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=colors,
            cmap="viridis",
            s=4,
            zorder=5,
        )
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            color="gray",
            linewidth=1,
            alpha=0.3,
            zorder=4,
        )

        # Start and end markers
        ax.plot(
            positions[0, 0],
            positions[0, 1],
            "o",
            color="green",
            markersize=10,
            zorder=6,
            label="Start",
        )
        ax.plot(
            positions[-1, 0],
            positions[-1, 1],
            "s",
            color="blue",
            markersize=8,
            zorder=6,
            label="End",
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        fig.tight_layout()
    return fig


def plot_info_levels(history_info, dt):
    """Plot info zone depletion over time."""
    with plt.style.context(_SCIENCE_STYLE):
        info = np.array(history_info)
        t = np.arange(len(info)) * dt

        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(info.shape[1]):
            ax.plot(t, info[:, i], linewidth=2, label=f"Info Zone {i + 1}")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Information Level")
        ax.set_title("Information Zone Depletion")
        ax.legend()
        fig.tight_layout()
    return fig


def plot_control_inputs(actions, dt):
    """Plot control inputs over time (4 subplots)."""
    with plt.style.context(_SCIENCE_STYLE):
        acts = np.array(actions)
        t = np.arange(len(acts)) * dt
        labels = [
            "Thrust (N)",
            "Omega X (rad/s)",
            "Omega Y (rad/s)",
            "Omega Z (rad/s)",
        ]

        fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(t, acts[:, i], linewidth=1.5)
            ax.set_ylabel(label)
            ax.set_title(label)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Control Inputs", y=1.02)
        fig.tight_layout()
    return fig


def plot_position_3d(history_x):
    """Plot 3D trajectory in ENU frame (z = altitude)."""
    with plt.style.context(_SCIENCE_STYLE):
        pos = np.array(history_x[:, :3])
        pos[:, 2] = -pos[:, 2]  # NED -> ENU: negate z so altitude is positive
        n = len(pos)
        colors = np.linspace(0, 1, n)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Trajectory colored by progress
        for i in range(n - 1):
            ax.plot(
                pos[i : i + 2, 0],
                pos[i : i + 2, 1],
                pos[i : i + 2, 2],
                color=plt.cm.viridis(colors[i]),  # pyright: ignore[reportAttributeAccessIssue]
                linewidth=2,
            )

        # Start marker
        ax.scatter(
            *pos[0], color="green", s=60, label="Start", depthshade=False
        )

        # Goal marker
        ax.scatter(
            float(GOAL_POS[0]),
            float(GOAL_POS[1]),
            -float(GOAL_POS[2]),  # pyright: ignore[reportArgumentType]
            color="red",
            s=80,
            marker="D",
            label="Goal",
            depthshade=False,
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Altitude (m)")
        ax.set_zlim(0, 5)
        ax.set_title("3D Trajectory")
        ax.legend()
    return fig


def create_trajectory_gif(
    history_x,
    grid,
    resolution,
    dt,
    save_path="i_mppi_trajectory.gif",
    fps=20,
    step_skip=5,
    origin=(0.0, 0.0),
):
    """Create an animated GIF showing the UAV trajectory and FOV on the map.

    Args:
        history_x: (N, 13+) state history.
        grid: (H, W) occupancy grid.
        resolution: Grid resolution in m/cell.
        dt: Simulation timestep.
        save_path: Output file path (.gif).
        fps: Frames per second.
        step_skip: Show every N-th simulation step as a frame.
        origin: (x, y) world coordinates of the grid origin.
    """
    positions = np.array(history_x[:, :2])
    quats = np.array(history_x[:, 6:10])
    yaws = np.arctan2(
        2 * (quats[:, 0] * quats[:, 3] + quats[:, 1] * quats[:, 2]),
        1 - 2 * (quats[:, 2] ** 2 + quats[:, 3] ** 2),
    )
    n_steps = len(positions)
    frame_indices = list(range(0, n_steps, step_skip))
    if frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)

    grid_np = np.array(grid)
    H, W = grid_np.shape
    extent = (0, W * resolution, 0, H * resolution)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Occupancy grid
    ax.imshow(
        grid_np,
        cmap="Greys",
        origin="lower",
        extent=extent,
        vmin=0,
        vmax=1,
        alpha=0.8,
        aspect="equal",
    )

    # Info zones
    for i in range(len(INFO_ZONES)):
        cx, cy = float(INFO_ZONES[i, 0]), float(INFO_ZONES[i, 1])
        w, h = float(INFO_ZONES[i, 2]), float(INFO_ZONES[i, 3])
        rect = Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            linewidth=2,
            edgecolor="orange",
            facecolor="yellow",
            alpha=0.3,
        )
        ax.add_patch(rect)

    ax.plot(1.0, 5.0, "o", color="green", markersize=8)
    ax.plot(
        float(GOAL_POS[0]), float(GOAL_POS[1]), "*", color="red", markersize=12
    )
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    # Trail
    (trail_line,) = ax.plot(
        [], [], color="cyan", linewidth=2, label="Trajectory"
    )
    # UAV marker
    (uav_marker,) = ax.plot([], [], "o", color="cyan", markersize=8)
    # Heading arrow
    arrow_len = 0.5
    (heading_line,) = ax.plot([], [], color="cyan", linewidth=2.5)
    # FOV wedge
    fov_patch = Polygon(
        _fov_polygon(
            positions[0, 0],
            positions[0, 1],
            yaws[0],
            grid_np,
            resolution,
            origin=origin,
        ),
        closed=True,
        facecolor="cyan",
        alpha=0.2,
        edgecolor="cyan",
        linewidth=0.5,
    )
    ax.add_patch(fov_patch)
    ax.legend(loc="upper right")

    title_text = fig.suptitle("I-MPPI Trajectory  t = 0.0s")

    def update(frame_idx):
        k = frame_indices[frame_idx]
        x, y = positions[k, 0], positions[k, 1]
        yaw = yaws[k]

        trail_line.set_data(positions[: k + 1, 0], positions[: k + 1, 1])
        uav_marker.set_data([x], [y])

        dx = arrow_len * np.cos(yaw)
        dy = arrow_len * np.sin(yaw)
        heading_line.set_data([x, x + dx], [y, y + dy])

        fov_verts = _fov_polygon(x, y, yaw, grid_np, resolution, origin=origin)
        fov_patch.set_xy(fov_verts)

        title_text.set_text(f"I-MPPI Trajectory  t = {k * dt:.1f}s")

        return [trail_line, uav_marker, heading_line, fov_patch, title_text]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 // fps,
        blit=False,
    )

    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return save_path
