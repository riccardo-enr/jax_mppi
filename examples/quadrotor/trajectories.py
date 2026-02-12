
import jax.numpy as jnp
from jaxtyping import Array, Float


def generate_hover_trajectory(
    center: Float[Array, "3"],
    duration: float,
    dt: float,
) -> Float[Array, "T 6"]:
    """Generate hover setpoint trajectory (constant position, zero velocity).

    Args:
        center: [x, y, z] position.
        duration: trajectory duration in seconds.
        dt: time step.

    Returns:
        (T, 6) trajectory [pos, vel].
    """
    num_steps = int(duration / dt)
    pos = jnp.tile(center, (num_steps, 1))
    vel = jnp.zeros((num_steps, 3))
    return jnp.concatenate([pos, vel], axis=1)


def generate_circle_trajectory(
    radius: float = 2.0,
    height: float = -2.0,
    speed: float = 1.0,
    duration: float = 10.0,
    dt: float = 0.01,
    center: Float[Array, "2"] | None = None,
    phase: float = 0.0,
) -> Float[Array, "T 6"]:
    """Generate circular trajectory in horizontal plane (NED frame).

    Args:
        radius: circle radius (m).
        height: constant altitude (negative for up in NED).
        speed: tangential speed (m/s).
        duration: trajectory duration (s).
        dt: time step (s).
        center: [x, y] center (default [0, 0]).
        phase: starting phase (rad).

    Returns:
        (T, 6) trajectory [pos, vel].
    """
    if center is None:
        center = jnp.zeros(2)

    num_steps = int(duration / dt)
    t = jnp.arange(num_steps) * dt
    omega = speed / radius

    # Position
    x = center[0] + radius * jnp.cos(omega * t + phase)
    y = center[1] + radius * jnp.sin(omega * t + phase)
    z = jnp.full_like(x, height)

    # Velocity
    vx = -radius * omega * jnp.sin(omega * t + phase)
    vy = radius * omega * jnp.cos(omega * t + phase)
    vz = jnp.zeros_like(vx)

    pos = jnp.stack([x, y, z], axis=1)
    vel = jnp.stack([vx, vy, vz], axis=1)

    return jnp.concatenate([pos, vel], axis=1)


def generate_figure8_trajectory(
    size_x: float = 2.0,
    size_y: float = 1.0,
    height: float = -2.0,
    period: float = 10.0,
    duration: float = 20.0,
    dt: float = 0.01,
    center: Float[Array, "2"] | None = None,
    axis: str = "xy",
) -> Float[Array, "T 6"]:
    """Generate figure-8 (lemniscate of Gerono) trajectory.

    x = A * sin(w*t)
    y = B * sin(w*t) * cos(w*t)

    Args:
        size_x: Width (A).
        size_y: Height (B).
        height: Altitude.
        period: Time for full cycle.
        duration: Total duration.
        dt: Time step.
        center: [x, y] center.
        axis: Plane of figure-8 ('xy' for horizontal).

    Returns:
        (T, 6) trajectory.
    """
    if center is None:
        center = jnp.zeros(2)

    num_steps = int(duration / dt)
    t = jnp.arange(num_steps) * dt
    omega = 2 * jnp.pi / period

    sin_wt = jnp.sin(omega * t)
    cos_wt = jnp.cos(omega * t)

    # Lemniscate of Gerono
    x_local = size_x * sin_wt
    y_local = size_y * sin_wt * cos_wt

    # Velocity chain rule
    # dx/dt = A * w * cos(wt)
    # dy/dt = B * w * (cos^2(wt) - sin^2(wt)) = B * w * cos(2wt)
    vx_local = size_x * omega * cos_wt
    vy_local = size_y * omega * jnp.cos(2 * omega * t)

    x = x_local + center[0]
    y = y_local + center[1]
    z = jnp.full_like(x, height)
    vx = vx_local
    vy = vy_local
    vz = jnp.zeros_like(vx)

    pos = jnp.stack([x, y, z], axis=1)
    vel = jnp.stack([vx, vy, vz], axis=1)

    return jnp.concatenate([pos, vel], axis=1)


def generate_helical_trajectory(
    radius: float = 2.0,
    speed: float = 1.0,
    climb_rate: float = 0.5,
    duration: float = 10.0,
    dt: float = 0.01,
    start_height: float = 0.0,
    center: Float[Array, "2"] | None = None,
) -> Float[Array, "T 6"]:
    """Generate helical (spiral) trajectory.

    Args:
        radius: Radius.
        speed: Tangential speed.
        climb_rate: Vertical speed (m/s).
        duration: Duration.
        dt: Time step.
        start_height: Initial z (NED, so negative is up).
        center: [x, y] center.

    Returns:
        (T, 6) trajectory.
    """
    if center is None:
        center = jnp.zeros(2)

    num_steps = int(duration / dt)
    t = jnp.arange(num_steps) * dt
    omega = speed / radius

    x = center[0] + radius * jnp.cos(omega * t)
    y = center[1] + radius * jnp.sin(omega * t)
    # Climb up (negative z direction for NED)
    # If climb_rate is positive (upward speed), z decreases
    z = start_height - climb_rate * t

    vx = -radius * omega * jnp.sin(omega * t)
    vy = radius * omega * jnp.cos(omega * t)
    vz = jnp.full_like(vx, -climb_rate)

    pos = jnp.stack([x, y, z], axis=1)
    vel = jnp.stack([vx, vy, vz], axis=1)

    return jnp.concatenate([pos, vel], axis=1)


def generate_waypoint_trajectory(
    waypoints: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"] | None = None,
    segment_duration: float = 5.0,
    dt: float = 0.01,
    blend_time: float = 0.5,
) -> Float[Array, "T 6"]:
    """Generate trajectory through waypoints with smooth transitions.

    Uses simple linear interpolation with optional blending or polynomial segments.
    For simplicity, implements piecewise linear with velocity smoothing.

    Args:
        waypoints: List of [x, y, z] points.
        velocities: Optional velocities at waypoints.
        segment_duration: Time between waypoints.
        dt: Time step.
        blend_time: Not implemented in simple version.

    Returns:
        (T, 6) trajectory.
    """
    num_points = waypoints.shape[0]
    steps_per_seg = int(segment_duration / dt)
    # Removed unused total_steps variable

    pos_list = []
    vel_list = []

    for i in range(num_points - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        v_seg = (p1 - p0) / segment_duration

        # Generate segment
        t = jnp.linspace(0, segment_duration, steps_per_seg, endpoint=False)
        # Linear pos: p0 + v*t
        seg_pos = p0[None, :] + v_seg[None, :] * t[:, None]
        seg_vel = jnp.tile(v_seg, (steps_per_seg, 1))

        pos_list.append(seg_pos)
        vel_list.append(seg_vel)

    # Add final point
    pos_list.append(waypoints[-1][None, :])
    vel_list.append(jnp.zeros((1, 3)))

    pos = jnp.concatenate(pos_list, axis=0)
    vel = jnp.concatenate(vel_list, axis=0)

    # Handle slight length mismatch due to endpoint
    return jnp.concatenate([pos, vel], axis=1)


def get_reference_state(
    trajectory: Float[Array, "T 6"], t: int
) -> Float[Array, "6"]:
    """Extract reference [pos, vel] at time index t with clamping."""
    horizon = trajectory.shape[0]
    idx = jnp.clip(t, 0, horizon - 1)
    return trajectory[idx]


def slice_trajectory_window(
    trajectory: Float[Array, "T 6"], t_start: int, horizon: int
) -> Float[Array, "H 6"]:
    """Extract a window of the trajectory [t_start, t_start + horizon].

    Clamps indices to the end of the trajectory.

    Args:
        trajectory: Full reference trajectory.
        t_start: Current time index.
        horizon: Length of window.

    Returns:
        (horizon, 6) trajectory window.
    """
    T = trajectory.shape[0]
    indices = jnp.arange(t_start, t_start + horizon)
    indices = jnp.clip(indices, 0, T - 1)
    return trajectory[indices]


def compute_trajectory_metrics(
    trajectory: Float[Array, "T 6"],
    dt: float,
) -> dict[str, float]:
    """Compute metrics like path length, duration, average speed."""
    pos = trajectory[:, :3]
    vel = trajectory[:, 3:6]

    # Path length
    diffs = jnp.diff(pos, axis=0)
    dists = jnp.linalg.norm(diffs, axis=1)
    length = jnp.sum(dists)

    # Average speed
    speeds = jnp.linalg.norm(vel, axis=1)
    avg_speed = jnp.mean(speeds)
    max_speed = jnp.max(speeds)

    duration = trajectory.shape[0] * dt

    return {
        "length": float(length),
        "duration": float(duration),
        "avg_speed": float(avg_speed),
        "max_speed": float(max_speed),
    }
