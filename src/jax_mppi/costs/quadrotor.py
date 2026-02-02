"""Cost functions for quadrotor trajectory tracking.

This module provides cost functions for quadrotor control tasks including:
- Trajectory tracking (position and velocity)
- Attitude tracking (quaternion-based)
- Hover control (stabilization)
- Control effort penalties
"""

from typing import Callable, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

CostFn = Callable[
    [Float[Array, "nx"], Optional[Float[Array, "nu"]]], Float[Array, ""]
]


def quaternion_distance(
    q1: Float[Array, "4"], q2: Float[Array, "4"]
) -> Float[Array, ""]:
    """Compute distance between two unit quaternions.

    Uses: d = 1 - |q1^T q2|

    This metric is 0 when quaternions represent the same orientation
    and approaches 1 as they diverge.

    Args:
        q1: First unit quaternion [qw, qx, qy, qz]
        q2: Second unit quaternion [qw, qx, qy, qz]

    Returns:
        Distance metric in [0, 1]
    """
    # Inner product
    dot = jnp.abs(jnp.dot(q1, q2))
    # Clamp to avoid numerical issues
    dot = jnp.clip(dot, 0.0, 1.0)
    return 1.0 - dot


def create_trajectory_tracking_cost(
    Q_pos: Float[Array, "3 3"],
    Q_vel: Float[Array, "3 3"],
    R: Float[Array, "4 4"],
    reference_trajectory: Float[Array, "T 6"] | None = None,
    dt: float = 0.01,
) -> CostFn:
    """Create trajectory tracking cost function.

    Cost: ||p - p_ref||²_Q_pos + ||v - v_ref||²_Q_vel + ||u||²_R

    Args:
        Q_pos: Position error weight matrix (3x3)
        Q_vel: Velocity error weight matrix (3x3)
        R: Control effort weight matrix (4x4)
        reference_trajectory: Reference trajectory array [T x 6] with [px, py, pz, vx, vy, vz]
                             If None, uses zero reference (hover at origin)
        dt: Time step for indexing into reference trajectory

    Returns:
        Cost function for trajectory tracking

    Example:
        >>> Q_pos = jnp.eye(3) * 10.0
        >>> Q_vel = jnp.eye(3) * 1.0
        >>> R = jnp.eye(4) * 0.01
        >>> reference = jnp.zeros((100, 6))  # hover at origin
        >>> cost_fn = create_trajectory_tracking_cost(Q_pos, Q_vel, R, reference)
    """

    def cost_fn(
        state: Float[Array, "13"], action: Optional[Float[Array, "4"]] = None
    ) -> Float[Array, ""]:
        # Extract position and velocity
        pos = state[0:3]
        vel = state[3:6]

        # Get reference (default to zeros if no trajectory provided)
        if reference_trajectory is not None:
            # Use current position as time index (assumes trajectory is time-indexed)
            # For now, use first reference point (can be extended with time parameter)
            ref = reference_trajectory[0]
        else:
            ref = jnp.zeros(6)

        pos_ref = ref[0:3]
        vel_ref = ref[3:6]

        # Position tracking error
        pos_error = pos - pos_ref
        cost_pos = pos_error @ Q_pos @ pos_error

        # Velocity tracking error
        vel_error = vel - vel_ref
        cost_vel = vel_error @ Q_vel @ vel_error

        # Control effort
        cost_control = 0.0
        if action is not None:
            cost_control = action @ R @ action

        return cost_pos + cost_vel + cost_control  # type: ignore

    return cost_fn


def create_time_indexed_trajectory_cost(
    Q_pos: Float[Array, "3 3"],
    Q_vel: Float[Array, "3 3"],
    R: Float[Array, "4 4"],
    reference_trajectory: Float[Array, "T 6"],
    dt: float = 0.01,
) -> Callable[
    [Float[Array, "13"], Optional[Float[Array, "4"]], int], Float[Array, ""]
]:
    """Create time-indexed trajectory tracking cost function.

    This version explicitly takes a time index to lookup the reference trajectory.

    Args:
        Q_pos: Position error weight matrix (3x3)
        Q_vel: Velocity error weight matrix (3x3)
        R: Control effort weight matrix (4x4)
        reference_trajectory: Reference trajectory [T x 6] with [px, py, pz, vx, vy, vz]
        dt: Time step

    Returns:
        Cost function that takes (state, action, time_index)
    """
    T = reference_trajectory.shape[0]

    def cost_fn(
        state: Float[Array, "13"],
        action: Optional[Float[Array, "4"]] = None,
        t: int = 0,
    ) -> Float[Array, ""]:
        # Extract position and velocity
        pos = state[0:3]
        vel = state[3:6]

        # Get reference at time t (with bounds checking)
        t_idx = jnp.clip(t, 0, T - 1)
        ref = reference_trajectory[t_idx]

        pos_ref = ref[0:3]
        vel_ref = ref[3:6]

        # Position tracking error
        pos_error = pos - pos_ref
        cost_pos = pos_error @ Q_pos @ pos_error

        # Velocity tracking error
        vel_error = vel - vel_ref
        cost_vel = vel_error @ Q_vel @ vel_error

        # Control effort
        cost_control = 0.0
        if action is not None:
            cost_control = action @ R @ action

        return cost_pos + cost_vel + cost_control  # type: ignore

    return cost_fn


def create_hover_cost(
    Q_pos: Float[Array, "3 3"],
    Q_vel: Float[Array, "3 3"],
    Q_att: Float[Array, "4 4"],
    R: Float[Array, "4 4"],
    hover_position: Float[Array, "3"],
    hover_quaternion: Float[Array, "4"] | None = None,
) -> CostFn:
    """Create cost function for hover control (stabilization).

    Cost: ||p - p_hover||²_Q + ||v||²_Q_vel + ||q - q_hover||²_Q_att + ||u||²_R

    Args:
        Q_pos: Position error weight matrix (3x3)
        Q_vel: Velocity penalty matrix (3x3)
        Q_att: Attitude error weight matrix (4x4)
        R: Control effort weight matrix (4x4)
        hover_position: Desired hover position in NED frame
        hover_quaternion: Desired hover orientation (default: level flight [1,0,0,0])

    Returns:
        Cost function for hover control
    """
    if hover_quaternion is None:
        hover_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])  # level flight

    def cost_fn(
        state: Float[Array, "13"], action: Optional[Float[Array, "4"]] = None
    ) -> Float[Array, ""]:
        # Extract state components
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]

        # Position error
        pos_error = pos - hover_position
        cost_pos = pos_error @ Q_pos @ pos_error

        # Velocity penalty (want zero velocity)
        cost_vel = vel @ Q_vel @ vel

        # Attitude error (quaternion distance)
        att_dist = quaternion_distance(quat, hover_quaternion)
        cost_att = att_dist * jnp.trace(Q_att)

        # Control effort
        cost_control = 0.0
        if action is not None:
            cost_control = action @ R @ action

        return cost_pos + cost_vel + cost_att + cost_control  # type: ignore

    return cost_fn


def create_terminal_cost(
    Q_pos: Float[Array, "3 3"],
    Q_vel: Float[Array, "3 3"],
    Q_att: Float[Array, "4 4"],
    goal_position: Float[Array, "3"],
    goal_quaternion: Float[Array, "4"] | None = None,
) -> CostFn:
    """Create terminal cost for goal reaching.

    Terminal cost penalizes deviation from goal state at end of horizon.

    Args:
        Q_pos: Terminal position error weight matrix (3x3)
        Q_vel: Terminal velocity weight matrix (3x3)
        Q_att: Terminal attitude error weight matrix (4x4)
        goal_position: Goal position in NED frame
        goal_quaternion: Goal orientation (default: level flight)

    Returns:
        Terminal cost function
    """
    if goal_quaternion is None:
        goal_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

    def terminal_cost(
        state: Float[Array, "13"],
        last_action: Optional[Float[Array, "4"]] = None,
    ) -> Float[Array, ""]:
        # Extract state components
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]

        # Position error
        pos_error = pos - goal_position
        cost_pos = pos_error @ Q_pos @ pos_error

        # Velocity penalty (want to arrive with zero velocity)
        cost_vel = vel @ Q_vel @ vel

        # Attitude error
        att_dist = quaternion_distance(quat, goal_quaternion)
        cost_att = att_dist * jnp.trace(Q_att)

        return cost_pos + cost_vel + cost_att  # type: ignore

    return terminal_cost
