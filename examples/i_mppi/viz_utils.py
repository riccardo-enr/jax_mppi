"""Visualization utilities for the I-MPPI interactive simulation using Plotly."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jax_mppi.i_mppi.environment import (
    GOAL_POS,
    INFO_ZONES,
    SENSOR_FOV_RAD,
    SENSOR_MAX_RANGE,
)

# --- Information gain colormap (dark green → white → dark blue) ---
_INFO_GAIN_COLORSCALE = [
    [0.0, "#0d2905"],
    [0.1, "#2d5016"],
    [0.2, "#4a7c2e"],
    [0.3, "#c8e6c9"],
    [0.5, "#ffffff"],
    [0.7, "#b3d9f2"],
    [0.8, "#66b3cc"],
    [0.9, "#3366aa"],
    [0.95, "#1a4488"],
    [1.0, "#003366"],
]

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


def _compute_seen_mask(
    x, y, yaw, grid, resolution, max_range, origin=(0.0, 0.0)
):
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

    cols = np.floor((ray_x - origin[0]) / resolution).astype(np.intp)
    rows = np.floor((ray_y - origin[1]) / resolution).astype(np.intp)

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


def _create_environment_traces(grid, resolution, show_labels=True):
    """Create plotly traces for the occupancy grid with walls, info zones, start, and goal."""
    traces = []

    # Occupancy grid as heatmap
    extent = [0, grid.shape[1] * resolution, 0, grid.shape[0] * resolution]
    traces.append(
        go.Heatmap(
            z=np.array(grid),
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

    # Information zones
    for i in range(len(INFO_ZONES)):
        cx, cy = float(INFO_ZONES[i, 0]), float(INFO_ZONES[i, 1])
        w, h = float(INFO_ZONES[i, 2]), float(INFO_ZONES[i, 3])
        traces.append(
            go.Scatter(
                x=[cx - w/2, cx + w/2, cx + w/2, cx - w/2, cx - w/2],
                y=[cy - h/2, cy - h/2, cy + h/2, cy + h/2, cy - h/2],
                fill="toself",
                fillcolor="rgba(255, 255, 0, 0.3)",
                line=dict(color="orange", width=2),
                mode="lines",
                name=f"Info Zone {i + 1}",
                hoverinfo="name",
            )
        )
        if show_labels:
            traces.append(
                go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="text",
                    text=[f"Info {i + 1}"],
                    textposition="middle center",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Start position
    traces.append(
        go.Scatter(
            x=[1.0],
            y=[5.0],
            mode="markers",
            marker=dict(size=12, color="green", symbol="circle"),
            name="Start",
        )
    )

    # Goal position
    traces.append(
        go.Scatter(
            x=[float(GOAL_POS[0])],
            y=[float(GOAL_POS[1])],
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name="Goal",
        )
    )

    return traces


def plot_environment(grid, resolution, show_labels=True):
    """Plot the occupancy grid with walls, info zones, start, and goal."""
    traces = _create_environment_traces(grid, resolution, show_labels)

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis=dict(title="X (m)", range=[-0.5, 14.5]),
        yaxis=dict(title="Y (m)", range=[-0.5, 12.5], scaleanchor="x", scaleratio=1),
        width=800,
        height=700,
        showlegend=True,
    )

    return fig


def plot_trajectory_2d(history_x, grid, resolution, title="I-MPPI Trajectory"):
    """Plot 2D trajectory over the environment."""
    traces = _create_environment_traces(grid, resolution, show_labels=False)

    positions = np.array(history_x[:, :2])
    n_steps = len(positions)

    # Trajectory with color gradient
    colors = np.linspace(0, 1, n_steps)
    traces.append(
        go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode="lines",
            line=dict(color=colors, colorscale="Viridis", width=3),
            name="Trajectory",
            hovertemplate="Step: %{pointNumber}<br>X: %{x:.2f}m<br>Y: %{y:.2f}m",
        )
    )

    # Start and end markers
    traces.append(
        go.Scatter(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            mode="markers",
            marker=dict(size=12, color="green", symbol="circle"),
            name="Start",
        )
    )
    traces.append(
        go.Scatter(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            mode="markers",
            marker=dict(size=10, color="blue", symbol="square"),
            name="End",
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis=dict(title="X (m)", range=[-0.5, 14.5]),
        yaxis=dict(title="Y (m)", range=[-0.5, 12.5], scaleanchor="x", scaleratio=1),
        width=900,
        height=700,
        showlegend=True,
    )

    return fig


def plot_info_levels(history_info, dt):
    """Plot info zone depletion over time."""
    info = np.array(history_info)
    t = np.arange(len(info)) * dt

    fig = go.Figure()

    for i in range(info.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=info[:, i],
                mode="lines",
                name=f"Info Zone {i + 1}",
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="Information Zone Depletion",
        xaxis_title="Time (s)",
        yaxis_title="Information Level",
        showlegend=True,
        width=800,
        height=500,
        hovermode="x unified",
    )

    return fig


def plot_control_inputs(actions, dt):
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

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=labels,
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    for i, (label, color) in enumerate(zip(labels, colors)):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=acts[:, i],
                mode="lines",
                line=dict(color=color, width=1.5),
                name=label,
                showlegend=False,
            ),
            row=i + 1,
            col=1,
        )
        fig.update_yaxis(title_text=label, row=i + 1, col=1)

    fig.update_xaxis(title_text="Time (s)", row=4, col=1)
    fig.update_layout(
        title_text="Control Inputs",
        height=800,
        width=900,
        hovermode="x unified",
    )

    return fig


def plot_position_3d(history_x):
    """Plot 3D trajectory in ENU frame (z = altitude)."""
    pos = np.array(history_x[:, :3])
    pos[:, 2] = -pos[:, 2]  # NED → ENU: negate z so altitude is positive
    n = len(pos)

    # Color by progress
    colors = np.linspace(0, 1, n)

    fig = go.Figure()

    # Trajectory
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines",
            line=dict(color=colors, colorscale="Viridis", width=4),
            name="Trajectory",
            hovertemplate="Step: %{pointNumber}<br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m",
        )
    )

    # Start marker
    fig.add_trace(
        go.Scatter3d(
            x=[pos[0, 0]],
            y=[pos[0, 1]],
            z=[pos[0, 2]],
            mode="markers",
            marker=dict(size=8, color="green", symbol="circle"),
            name="Start",
        )
    )

    # Goal marker
    fig.add_trace(
        go.Scatter3d(
            x=[float(GOAL_POS[0])],
            y=[float(GOAL_POS[1])],
            z=[-float(GOAL_POS[2])],
            mode="markers",
            marker=dict(size=10, color="red", symbol="diamond"),
            name="Goal",
        )
    )

    fig.update_layout(
        title="3D Trajectory",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Altitude (m)",
            zaxis=dict(range=[0, 5]),
            aspectmode="data",
        ),
        width=900,
        height=700,
        showlegend=True,
    )

    return fig


def create_trajectory_gif(
    history_x,
    history_info,
    grid,
    resolution,
    dt,
    save_path="i_mppi_trajectory.html",
    fps=20,
    step_skip=5,
    origin=(0.0, 0.0),
    info_gain_field=None,
):
    """Create an animated HTML visualization of the UAV trajectory.

    Args:
        history_x: (N, 16) state history.
        history_info: (N, 3) info level history.
        grid: (H, W) occupancy grid.
        resolution: Grid resolution in m/cell.
        dt: Simulation timestep.
        save_path: Output file path (.html).
        fps: Frames per second.
        step_skip: Show every N-th simulation step as a frame.
        origin: (x, y) world coordinates of the grid origin.
        info_gain_field: Optional (N, H, W) or (H, W) information gain field.
            If 3D, uses time-varying field; if 2D, uses static field.
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

    grid_np = np.array(grid)
    extent = [0, grid.shape[1] * resolution, 0, grid.shape[0] * resolution]

    # Precompute cumulative seen masks
    cumulative_seen = np.zeros(grid_np.shape, dtype=bool)
    seen_snapshots = []
    fi = 0
    for k in range(n_steps):
        seen_k = _compute_seen_mask(
            positions[k, 0],
            positions[k, 1],
            yaws[k],
            grid_np,
            resolution,
            SENSOR_MAX_RANGE,
            origin,
        )
        cumulative_seen = cumulative_seen | seen_k
        if fi < len(frame_indices) and frame_indices[fi] == k:
            seen_snapshots.append(cumulative_seen.copy())
            fi += 1
    while len(seen_snapshots) < len(frame_indices):
        seen_snapshots.append(cumulative_seen.copy())

    # Create subplot with map and info plot
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=("I-MPPI Trajectory", "Info Zone Depletion"),
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    # --- Left plot: Environment and trajectory ---

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
        ),
        row=1,
        col=1,
    )

    # Information gain field overlay (if provided)
    if info_gain_field is not None:
        info_gain_arr = np.array(info_gain_field)
        is_time_varying = info_gain_arr.ndim == 3

        # Create frames for each timestep
        ig_frames = []
        for frame_idx in range(len(frame_indices)):
            k = frame_indices[frame_idx]
            if is_time_varying:
                ig_frame = info_gain_arr[min(k, len(info_gain_arr) - 1)]
            else:
                ig_frame = info_gain_arr

            # Mask zero values for transparency
            ig_masked = np.where(ig_frame > 0, ig_frame, np.nan)
            ig_frames.append(ig_masked)

        # Add initial info gain field
        fig.add_trace(
            go.Heatmap(
                z=ig_frames[0],
                x=np.linspace(0, extent[1], grid.shape[1]),
                y=np.linspace(0, extent[3], grid.shape[0]),
                colorscale=_INFO_GAIN_COLORSCALE,
                opacity=0.6,
                showscale=True,
                colorbar=dict(title="Info Gain", x=0.45),
                hoverinfo="z",
                zmin=0,
                zmax=np.nanmax([np.nanmax(f) for f in ig_frames]) if any(np.nanmax(f) > 0 for f in ig_frames if not np.all(np.isnan(f))) else 1,
            ),
            row=1,
            col=1,
        )

    # Information zones
    for i in range(len(INFO_ZONES)):
        cx, cy = float(INFO_ZONES[i, 0]), float(INFO_ZONES[i, 1])
        w, h = float(INFO_ZONES[i, 2]), float(INFO_ZONES[i, 3])
        fig.add_trace(
            go.Scatter(
                x=[cx - w/2, cx + w/2, cx + w/2, cx - w/2, cx - w/2],
                y=[cy - h/2, cy - h/2, cy + h/2, cy + h/2, cy - h/2],
                fill="toself",
                fillcolor=f"rgba(255, 255, 0, {max(0.05, info[0, i] / 100.0 * 0.4)})",
                line=dict(color="orange", width=2),
                mode="lines",
                name=f"Info Zone {i + 1}",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Start and goal
    fig.add_trace(
        go.Scatter(
            x=[1.0],
            y=[5.0],
            mode="markers",
            marker=dict(size=10, color="green", symbol="circle"),
            name="Start",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[float(GOAL_POS[0])],
            y=[float(GOAL_POS[1])],
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name="Goal",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Trajectory trail (animated)
    fig.add_trace(
        go.Scatter(
            x=positions[:1, 0],
            y=positions[:1, 1],
            mode="lines",
            line=dict(color="cyan", width=2),
            name="Trail",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # UAV position (animated)
    fig.add_trace(
        go.Scatter(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="circle"),
            name="UAV",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Heading arrow (animated)
    arrow_len = 0.5
    dx = arrow_len * np.cos(yaws[0])
    dy = arrow_len * np.sin(yaws[0])
    fig.add_trace(
        go.Scatter(
            x=[positions[0, 0], positions[0, 0] + dx],
            y=[positions[0, 1], positions[0, 1] + dy],
            mode="lines",
            line=dict(color="red", width=3),
            name="Heading",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # FOV wedge (animated)
    fov_verts = _fov_polygon(
        positions[0, 0], positions[0, 1], yaws[0], grid_np, resolution, origin=origin
    )
    fig.add_trace(
        go.Scatter(
            x=fov_verts[:, 0],
            y=fov_verts[:, 1],
            fill="toself",
            fillcolor="rgba(0, 255, 255, 0.2)",
            line=dict(color="cyan", width=1),
            mode="lines",
            name="FOV",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Explored overlay (animated)
    seen = seen_snapshots[0]
    explored_z = np.where(seen, 0.5, np.nan)
    fig.add_trace(
        go.Heatmap(
            z=explored_z,
            x=np.linspace(0, extent[1], grid.shape[1]),
            y=np.linspace(0, extent[3], grid.shape[0]),
            colorscale=[[0, "rgba(50, 200, 50, 0)"], [1, "rgba(50, 200, 50, 0.3)"]],
            showscale=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # --- Right plot: Info levels ---
    t_all = np.arange(n_steps) * dt
    zone_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i in range(info.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=t_all[:1],
                y=info[:1, i],
                mode="lines",
                line=dict(color=zone_colors[i], width=2),
                name=f"Zone {i + 1}",
            ),
            row=1,
            col=2,
        )

    # Create animation frames
    frames = []
    for frame_idx in range(len(frame_indices)):
        k = frame_indices[frame_idx]
        x, y = positions[k, 0], positions[k, 1]
        yaw = yaws[k]

        frame_data = []

        # Update occupancy grid (static, but needs to be in frame)
        frame_data.append(
            go.Heatmap(
                z=grid_np,
                x=np.linspace(0, extent[1], grid.shape[1]),
                y=np.linspace(0, extent[3], grid.shape[0]),
            )
        )

        # Update info gain field if present
        if info_gain_field is not None:
            frame_data.append(
                go.Heatmap(z=ig_frames[frame_idx])
            )

        # Update info zones with fading alpha
        for i in range(len(INFO_ZONES)):
            cx, cy = float(INFO_ZONES[i, 0]), float(INFO_ZONES[i, 1])
            w, h = float(INFO_ZONES[i, 2]), float(INFO_ZONES[i, 3])
            alpha = max(0.05, info[min(k, len(info) - 1), i] / 100.0 * 0.4)
            frame_data.append(
                go.Scatter(
                    x=[cx - w/2, cx + w/2, cx + w/2, cx - w/2, cx - w/2],
                    y=[cy - h/2, cy - h/2, cy + h/2, cy + h/2, cy - h/2],
                    fillcolor=f"rgba(255, 255, 0, {alpha})",
                )
            )

        # Static start/goal
        frame_data.extend([
            go.Scatter(x=[1.0], y=[5.0]),
            go.Scatter(x=[float(GOAL_POS[0])], y=[float(GOAL_POS[1])]),
        ])

        # Update trajectory trail
        frame_data.append(
            go.Scatter(x=positions[: k + 1, 0], y=positions[: k + 1, 1])
        )

        # Update UAV position
        frame_data.append(go.Scatter(x=[x], y=[y]))

        # Update heading arrow
        dx = arrow_len * np.cos(yaw)
        dy = arrow_len * np.sin(yaw)
        frame_data.append(go.Scatter(x=[x, x + dx], y=[y, y + dy]))

        # Update FOV wedge
        fov_verts = _fov_polygon(x, y, yaw, grid_np, resolution, origin=origin)
        frame_data.append(go.Scatter(x=fov_verts[:, 0], y=fov_verts[:, 1]))

        # Update explored overlay
        seen = seen_snapshots[frame_idx]
        explored_z = np.where(seen, 0.5, np.nan)
        frame_data.append(go.Heatmap(z=explored_z))

        # Update info level lines
        for i in range(info.shape[1]):
            frame_data.append(
                go.Scatter(x=t_all[: k + 1], y=info[: k + 1, i])
            )

        frames.append(
            go.Frame(
                data=frame_data,
                name=f"frame_{frame_idx}",
                layout=go.Layout(
                    title_text=f"I-MPPI Trajectory  t = {k * dt:.1f}s"
                ),
            )
        )

    fig.frames = frames

    # Add play/pause buttons
    fig.update_layout(
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

    # Update layout
    fig.update_xaxes(title_text="X (m)", range=[-0.5, 14.5], row=1, col=1)
    fig.update_yaxes(
        title_text="Y (m)", range=[-0.5, 12.5], scaleanchor="x", scaleratio=1, row=1, col=1
    )
    fig.update_xaxes(title_text="Time (s)", range=[0, t_all[-1]], row=1, col=2)
    fig.update_yaxes(title_text="Information Level", range=[-5, 105], row=1, col=2)

    fig.update_layout(
        width=1400,
        height=700,
        showlegend=True,
        hovermode="closest",
    )

    # Save as HTML
    fig.write_html(save_path)
    return save_path
