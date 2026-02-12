from typing import Callable, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

# Define cost function signature
# Takes: state (nx,), action (nu,)
# Returns: scalar cost
CostFn = Callable[
    [Float[Array, "nx"], Optional[Float[Array, "nu"]]], Float[Array, ""]
]


def quaternion_distance(
    q1: Float[Array, "4"], q2: Float[Array, "4"]
) -> Float[Array, ""]:
    """Compute distance between two unit quaternions.

    d(q1, q2) = 1 - <q1, q2>^2
    Matches the geodesic distance on SO(3).
    """
    # Ensure unit norm
    q1 = q1 / (jnp.linalg.norm(q1) + 1e-6)
    q2 = q2 / (jnp.linalg.norm(q2) + 1e-6)
    dot = jnp.dot(q1, q2)
    return 1.0 - dot**2


def create_trajectory_tracking_cost(
    Q_pos: Float[Array, "3 3"],
    Q_vel: Float[Array, "3 3"],
    R: Float[Array, "4 4"],
    reference_trajectory: Float[Array, "T 6"] | None = None,
    dt: float = 0.01,
) -> CostFn:
    """Create a trajectory tracking cost function.

    Args:
        Q_pos: Position error weight matrix (3x3).
        Q_vel: Velocity error weight matrix (3x3).
        R: Control effort weight matrix (4x4).
        reference_trajectory: Optional reference trajectory (T, 6).
            If None, tracks origin (0, 0, 0).
        dt: Time step for trajectory indexing.

    Returns:
        Cost function: (state, action) -> scalar cost.
    """

    def cost_fn(
        state: Float[Array, "13"], action: Optional[Float[Array, "4"]] = None
    ) -> Float[Array, ""]:
        # Extract position and velocity
        pos = state[0:3]
        vel = state[3:6]

        # Determine reference
        # For simple tracking, we just track the first point or origin
        if reference_trajectory is not None:
            ref_pos = reference_trajectory[0, 0:3]
            ref_vel = reference_trajectory[0, 3:6]
        else:
            ref_pos = jnp.zeros(3)
            ref_vel = jnp.zeros(3)

        # State errors
        e_pos = pos - ref_pos
        e_vel = vel - ref_vel

        # Quadratic costs
        cost = e_pos @ Q_pos @ e_pos + e_vel @ Q_vel @ e_vel

        # Control cost
        if action is not None:
            cost += action @ R @ action

        return cost

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

    This function returns a cost that depends on the time step `t`.
    Requires `step_dependent_dynamics=True` in MPPIConfig.

    Args:
        Q_pos: Position error weight matrix.
        Q_vel: Velocity error weight matrix.
        R: Control weight matrix.
        reference_trajectory: Array of shape (T, 6) containing [pos, vel].
        dt: Time step (unused here, but consistent with API).

    Returns:
        Cost function: (state, action, t) -> scalar cost.
    """
    traj_len = reference_trajectory.shape[0]

    def cost_fn(
        state: Float[Array, "13"],
        action: Optional[Float[Array, "4"]] = None,
        t: int = 0,
    ) -> Float[Array, ""]:
        # Extract position and velocity
        pos = state[0:3]
        vel = state[3:6]

        # Get reference at time t (clamp to end)
        idx = jnp.minimum(t, traj_len - 1)
        ref_state = reference_trajectory[idx]
        ref_pos = ref_state[0:3]
        ref_vel = ref_state[3:6]

        # Errors
        e_pos = pos - ref_pos
        e_vel = vel - ref_vel

        # Cost
        cost = e_pos @ Q_pos @ e_pos + e_vel @ Q_vel @ e_vel

        if action is not None:
            cost += action @ R @ action

        return cost

    return cost_fn


def create_hover_cost(
    Q_pos: Float[Array, "3 3"],
    Q_vel: Float[Array, "3 3"],
    Q_att: Float[Array, "4 4"],
    R: Float[Array, "4 4"],
    hover_position: Float[Array, "3"],
    hover_quaternion: Float[Array, "4"] | None = None,
) -> CostFn:
    """Create a hover stabilization cost function.

    Penalizes deviation from a fixed position and optionally orientation.

    Args:
        Q_pos: Position weight.
        Q_vel: Velocity weight.
        Q_att: Attitude (quaternion) weight.
        R: Control weight.
        hover_position: Target [x, y, z].
        hover_quaternion: Target [qw, qx, qy, qz] (default: identity/upright).

    Returns:
        Cost function.
    """
    if hover_quaternion is None:
        hover_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

    def cost_fn(
        state: Float[Array, "13"], action: Optional[Float[Array, "4"]] = None
    ) -> Float[Array, ""]:
        # Extract state components
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]

        # Errors
        e_pos = pos - hover_position
        e_vel = vel  # Target velocity is zero
        # Quaternion distance (1 - <q, q_ref>^2)
        # We multiply by a scalar weight, often trace(Q_att) or similar scalar
        # For matrix Q_att, we might treat q as a vector 4
        # Here, let's treat Q_att as diagonal weights on q components difference
        # But standard quadrotor control often uses geodesic distance.
        # Let's use simple weighted vector difference for consistency with Q matrices
        e_quat = quat - hover_quaternion

        cost = e_pos @ Q_pos @ e_pos + e_vel @ Q_vel @ e_vel
        cost += e_quat @ Q_att @ e_quat

        if action is not None:
            # Hover thrust is approx mg.
            # We can penalize deviation from hover thrust or just total effort.
            # Usually penalize deviation from hover thrust (mg)
            # For simplicity, penalize total effort here as per standard MPPI
            cost += action @ R @ action

        return cost

    return cost_fn


def create_terminal_cost(
    Q_pos: Float[Array, "3 3"],
    Q_vel: Float[Array, "3 3"],
    Q_att: Float[Array, "4 4"],
    goal_position: Float[Array, "3"],
    goal_quaternion: Float[Array, "4"] | None = None,
) -> Callable[
    [Float[Array, "13"], Optional[Float[Array, "4"]]], Float[Array, ""]
]:
    """Create a terminal cost function (no action penalty).

    Used for the final state in the horizon.
    """
    if goal_quaternion is None:
        goal_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

    def cost_fn(
        state: Float[Array, "13"],
        last_action: Optional[Float[Array, "4"]] = None,
    ) -> Float[Array, ""]:
        # Extract state components
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]

        e_pos = pos - goal_position
        e_vel = vel
        e_quat = quat - goal_quaternion

        cost = e_pos @ Q_pos @ e_pos + e_vel @ Q_vel @ e_vel
        cost += e_quat @ Q_att @ e_quat

        return cost

    return cost_fn
