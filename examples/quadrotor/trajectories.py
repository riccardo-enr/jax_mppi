"""Trajectory generation utilities for quadrotor control.

This module provides functions to generate reference trajectories for quadrotor
trajectory tracking tasks. All trajectories follow the NED (North-East-Down)
coordinate frame convention.

Frame Convention:
    - NED (North-East-Down): World/global frame
    - X: North, Y: East, Z: Down (positive downward)
    - Altitude: Negative Z values indicate height above ground
      (e.g., z = -5.0 means 5m altitude)
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


def generate_hover_setpoint(
    position: Float[Array, "3"],
    duration: float,
    dt: float,
) -> Float[Array, "T 6"]:  # noqa: F722
    """Generate hover setpoint trajectory (constant position, zero velocity).

    Args:
        position: Hover position in NED frame [px, py, pz]
                 (e.g., [0, 0, -5.0] for hovering at 5m altitude)
        duration: Total trajectory duration (s)
        dt: Time step (s)

    Returns:
        Trajectory array [T x 6] with [px, py, pz, vx, vy, vz]
        where T = ceil(duration / dt)

    Example:
        >>> # Hover at 5m altitude above origin
        >>> traj = generate_hover_setpoint([0,0,-5], duration=10.0, dt=0.01)
        >>> traj.shape
        (1000, 6)
    """
    num_steps = int(jnp.ceil(duration / dt))

    # Constant position, zero velocity
    trajectory = jnp.tile(
        jnp.concatenate([position, jnp.zeros(3)]),
        (num_steps, 1)
    )

    return trajectory


def generate_circle_trajectory(
    radius: float,
    height: float,
    period: float,
    duration: float,
    dt: float,
    center: Float[Array, "2"] | None = None,
    phase: float = 0.0,
) -> Float[Array, "T 6"]:  # noqa: F722
    """Generate circular trajectory in horizontal plane (NED frame).

    The trajectory follows a circle in the xy-plane at constant altitude.

    Args:
        radius: Circle radius in xy plane (m)
        height: Altitude in NED frame (m)
               Negative values = above ground (e.g., -5.0 for 5m altitude)
               Positive values = below ground (uncommon)
        period: Period of one complete revolution (s)
        duration: Total trajectory duration (s)
        dt: Time step (s)
        center: Circle center in xy plane [cx, cy] (default: origin [0, 0])
        phase: Initial phase angle (rad) (default: 0)

    Returns:
        Trajectory array [T x 6] with [px, py, pz, vx, vy, vz]
        where T = ceil(duration / dt)

    Example:
        >>> # Circle at 5m altitude, 3m radius, 10s period
        >>> traj = generate_circle_trajectory(
        ...     radius=3.0,
        ...     height=-5.0,
        ...     period=10.0,
        ...     duration=20.0,
        ...     dt=0.01
        ... )
    """
    if center is None:
        center = jnp.array([0.0, 0.0])

    num_steps = int(jnp.ceil(duration / dt))
    t = jnp.arange(num_steps) * dt
    omega = 2 * jnp.pi / period

    # Position
    x = center[0] + radius * jnp.cos(omega * t + phase)
    y = center[1] + radius * jnp.sin(omega * t + phase)
    z = jnp.ones_like(t) * height  # Constant altitude

    # Velocity (time derivatives)
    vx = -radius * omega * jnp.sin(omega * t + phase)
    vy = radius * omega * jnp.cos(omega * t + phase)
    vz = jnp.zeros_like(t)

    # Stack into trajectory array [T x 6]
    trajectory = jnp.stack([x, y, z, vx, vy, vz], axis=1)

    return trajectory


def generate_lemniscate_trajectory(
    scale: float,
    height: float,
    period: float,
    duration: float,
    dt: float,
    center: Float[Array, "2"] | None = None,
    axis: str = "xy",
) -> Float[Array, "T 6"]:  # noqa: F722
    """Generate figure-8 (lemniscate of Gerono) trajectory.

    The lemniscate is a figure-8 pattern often used for aggressive
    maneuvering tests in quadrotor control.

    Parametric equations:
        x(t) = scale * sin(ωt)
        y(t) = scale * sin(ωt) * cos(ωt)

    Args:
        scale: Size of the figure-8 (m)
        height: Altitude in NED frame (m)
               Negative = above ground (e.g., -5.0 for 5m altitude)
        period: Period of one complete figure-8 (s)
        duration: Total trajectory duration (s)
        dt: Time step (s)
        center: Pattern center in xy plane [cx, cy] (default: origin)
        axis: Plane of motion - "xy" (horizontal) or "xz" (vertical)
              (default: "xy")

    Returns:
        Trajectory array [T x 6] with [px, py, pz, vx, vy, vz]

    Example:
        >>> # Figure-8 at 5m altitude, 4m scale, 15s period
        >>> traj = generate_lemniscate_trajectory(
        ...     scale=4.0,
        ...     height=-5.0,
        ...     period=15.0,
        ...     duration=30.0,
        ...     dt=0.01
        ... )
    """
    if center is None:
        center = jnp.array([0.0, 0.0])

    num_steps = int(jnp.ceil(duration / dt))
    t = jnp.arange(num_steps) * dt
    omega = 2 * jnp.pi / period

    if axis == "xy":
        # Horizontal figure-8 in xy plane
        # Lemniscate of Gerono
        x = center[0] + scale * jnp.sin(omega * t)
        y = center[1] + scale * jnp.sin(omega * t) * jnp.cos(omega * t)
        z = jnp.ones_like(t) * height

        # Velocities (derivatives)
        vx = scale * omega * jnp.cos(omega * t)
        vy = scale * omega * (jnp.cos(omega * t)**2 - jnp.sin(omega * t)**2)
        vz = jnp.zeros_like(t)

    elif axis == "xz":
        # Vertical figure-8 in xz plane
        x = center[0] + scale * jnp.sin(omega * t)
        y = jnp.ones_like(t) * center[1]
        z = height + scale * jnp.sin(omega * t) * jnp.cos(omega * t)

        # Velocities
        vx = scale * omega * jnp.cos(omega * t)
        vy = jnp.zeros_like(t)
        vz = scale * omega * (jnp.cos(omega * t)**2 - jnp.sin(omega * t)**2)

    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'xy' or 'xz'")

    # Stack into trajectory array
    trajectory = jnp.stack([x, y, z, vx, vy, vz], axis=1)

    return trajectory


def generate_helix_trajectory(
    radius: float,
    height_rate: float,
    period: float,
    duration: float,
    dt: float,
    start_height: float = 0.0,
    center: Float[Array, "2"] | None = None,
) -> Float[Array, "T 6"]:  # noqa: F722
    """Generate helical (spiral) trajectory.

    Combines circular motion in xy-plane with constant vertical velocity.

    Args:
        radius: Helix radius in xy plane (m)
        height_rate: Vertical velocity (m/s)
                    Negative = climbing (moving up)
                    Positive = descending (moving down)
        period: Period of one horizontal revolution (s)
        duration: Total trajectory duration (s)
        dt: Time step (s)
        start_height: Starting altitude in NED frame (m)
        center: Helix center in xy plane [cx, cy] (default: origin)

    Returns:
        Trajectory array [T x 6] with [px, py, pz, vx, vy, vz]

    Example:
        >>> # Climbing helix starting at 2m altitude
        >>> traj = generate_helix_trajectory(
        ...     radius=2.0,
        ...     height_rate=-0.5,  # climbing at 0.5 m/s
        ...     period=10.0,
        ...     duration=20.0,
        ...     dt=0.01,
        ...     start_height=-2.0
        ... )
    """
    if center is None:
        center = jnp.array([0.0, 0.0])

    num_steps = int(jnp.ceil(duration / dt))
    t = jnp.arange(num_steps) * dt
    omega = 2 * jnp.pi / period

    # Position
    x = center[0] + radius * jnp.cos(omega * t)
    y = center[1] + radius * jnp.sin(omega * t)
    z = start_height + height_rate * t  # Linear altitude change

    # Velocity
    vx = -radius * omega * jnp.sin(omega * t)
    vy = radius * omega * jnp.cos(omega * t)
    vz = jnp.ones_like(t) * height_rate  # Constant vertical velocity

    trajectory = jnp.stack([x, y, z, vx, vy, vz], axis=1)

    return trajectory


def generate_waypoint_trajectory(
    waypoints: Float[Array, "N 3"],  # noqa: F722
    velocities: Float[Array, "N 3"] | None = None,  # noqa: F722
    segment_duration: float = 5.0,
    dt: float = 0.01,
    blend_time: float = 0.5,
) -> Float[Array, "T 6"]:  # noqa: F722
    """Generate trajectory through waypoints with smooth transitions.

    Uses cubic interpolation between waypoints with optional velocity
    specification at each waypoint.

    Args:
        waypoints: Array of waypoint positions [N x 3] in NED frame
                  Each row is [px, py, pz]
        velocities: Array of velocities at waypoints [N x 3] (optional)
                   If None, velocities are computed for smooth transitions
        segment_duration: Time to travel between waypoints (s)
        dt: Time step (s)
        blend_time: Time for smooth blending at waypoints (s)

    Returns:
        Trajectory array [T x 6] with [px, py, pz, vx, vy, vz]

    Example:
        >>> waypoints = jnp.array([
        ...     [0.0, 0.0, -2.0],  # Start at 2m altitude
        ...     [5.0, 0.0, -5.0],  # Move to 5m altitude
        ...     [5.0, 5.0, -5.0],  # Turn
        ...     [0.0, 5.0, -2.0],  # Return lower
        ... ])
        >>> traj = generate_waypoint_trajectory(waypoints, segment_duration=5.0)
    """
    N = waypoints.shape[0]

    if N < 2:
        raise ValueError("Need at least 2 waypoints")

    # If velocities not provided, compute them for smooth transitions
    if velocities is None:
        velocities = jnp.zeros_like(waypoints)
        # Start and end with zero velocity
        # Middle waypoints: average of incoming and outgoing directions
        for i in range(1, N - 1):
            direction = waypoints[i + 1] - waypoints[i - 1]
            velocities = velocities.at[i].set(
                direction / (2 * segment_duration)
            )

    # Generate trajectory for each segment
    segments = []
    num_steps_per_segment = int(jnp.ceil(segment_duration / dt))

    for i in range(N - 1):
        t = jnp.linspace(0, 1, num_steps_per_segment)

        # Cubic Hermite interpolation
        p0, p1 = waypoints[i], waypoints[i + 1]
        v0 = velocities[i] * segment_duration
        v1 = velocities[i + 1] * segment_duration

        # Hermite basis functions
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2

        # Position
        pos = (
            h00[:, None] * p0[None, :] +
            h10[:, None] * v0[None, :] +
            h01[:, None] * p1[None, :] +
            h11[:, None] * v1[None, :]
        )

        # Velocity (derivative of position)
        dh00 = 6 * t**2 - 6 * t
        dh10 = 3 * t**2 - 4 * t + 1
        dh01 = -6 * t**2 + 6 * t
        dh11 = 3 * t**2 - 2 * t

        vel = (
            dh00[:, None] * p0[None, :] +
            dh10[:, None] * v0[None, :] +
            dh01[:, None] * p1[None, :] +
            dh11[:, None] * v1[None, :]
        ) / segment_duration

        # Combine position and velocity
        segment = jnp.concatenate([pos, vel], axis=1)
        segments.append(segment)

    # Concatenate all segments
    trajectory = jnp.concatenate(segments, axis=0)

    return trajectory


def compute_trajectory_metrics(
    trajectory: Float[Array, "T 6"],  # noqa: F722
    dt: float,
) -> dict[str, float]:
    """Compute metrics for trajectory analysis.

    Args:
        trajectory: Trajectory array [T x 6] with [px, py, pz, vx, vy, vz]
        dt: Time step (s)

    Returns:
        Dictionary with trajectory metrics:
            - total_distance: Total path length (m)
            - max_velocity: Maximum velocity magnitude (m/s)
            - avg_velocity: Average velocity magnitude (m/s)
            - max_acceleration: Maximum acceleration magnitude (m/s²)
            - avg_acceleration: Average acceleration magnitude (m/s²)

    Example:
        >>> traj = generate_circle_trajectory(3.0, -5.0, 10.0, 20.0, 0.01)
        >>> metrics = compute_trajectory_metrics(traj, dt=0.01)
        >>> print(f"Max velocity: {metrics['max_velocity']:.2f} m/s")
    """
    pos = trajectory[:, 0:3]
    vel = trajectory[:, 3:6]

    # Distance traveled
    pos_diff = jnp.diff(pos, axis=0)
    distances = jnp.linalg.norm(pos_diff, axis=1)
    total_distance = float(jnp.sum(distances))

    # Velocity magnitudes
    vel_magnitudes = jnp.linalg.norm(vel, axis=1)
    max_velocity = float(jnp.max(vel_magnitudes))
    avg_velocity = float(jnp.mean(vel_magnitudes))

    # Acceleration (finite difference)
    acc = jnp.diff(vel, axis=0) / dt
    acc_magnitudes = jnp.linalg.norm(acc, axis=1)
    max_acceleration = float(jnp.max(acc_magnitudes))
    avg_acceleration = float(jnp.mean(acc_magnitudes))

    return {
        "total_distance": total_distance,
        "max_velocity": max_velocity,
        "avg_velocity": avg_velocity,
        "max_acceleration": max_acceleration,
        "avg_acceleration": avg_acceleration,
    }
