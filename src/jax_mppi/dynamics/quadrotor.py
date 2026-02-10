"""Quadrotor 6-DOF dynamics with quaternion representation.

This module implements quadrotor dynamics following the NED-FRD convention:
- NED (North-East-Down): World/global frame where Z-axis points down
- FRD (Forward-Right-Down): Body frame where X-axis points forward, Y right, Z down

State (13D): [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    - Position/velocity in NED world frame
    - Quaternion: body FRD to world NED (unit norm)
    - Angular velocity in FRD body frame

Control (4D): [T, wx_cmd, wy_cmd, wz_cmd]
    - T: thrust magnitude (positive, acts in -Z body direction/upward)
    - w_cmd: angular rate commands in FRD body frame
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


def quaternion_to_rotation_matrix(q: Float[Array, "4"]) -> Float[Array, "3 3"]:
    """Convert unit quaternion to rotation matrix (body FRD to world NED).

    Args:
        q: Unit quaternion [qw, qx, qy, qz]

    Returns:
        R: 3x3 rotation matrix from FRD body to NED world frame
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Rotation matrix from quaternion (body to world)
    R = jnp.array(
        [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx**2 + qy**2),
            ],
        ]
    )

    return R


def normalize_quaternion(q: Float[Array, "4"]) -> Float[Array, "4"]:
    """Normalize quaternion to unit norm.

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        Normalized unit quaternion
    """
    norm = jnp.linalg.norm(q)
    # Add small epsilon to avoid division by zero
    return q / (norm + 1e-8)


def quaternion_multiply(
    q1: Float[Array, "4"], q2: Float[Array, "4"]
) -> Float[Array, "4"]:
    """Multiply two quaternions: q1 * q2.

    Args:
        q1: First quaternion [qw, qx, qy, qz]
        q2: Second quaternion [qw, qx, qy, qz]

    Returns:
        Product quaternion q1 * q2
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_derivative(
    q: Float[Array, "4"], omega: Float[Array, "3"]
) -> Float[Array, "4"]:
    """Compute quaternion time derivative from angular velocity.

    Uses: q_dot = 0.5 * q ⊗ [0, omega]

    Args:
        q: Current quaternion [qw, qx, qy, qz]
        omega: Angular velocity in body frame [wx, wy, wz]

    Returns:
        Time derivative of quaternion
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    wx, wy, wz = omega[0], omega[1], omega[2]

    # q_dot = 0.5 * Omega(omega) @ q
    q_dot = 0.5 * jnp.array(
        [
            -wx * qx - wy * qy - wz * qz,  # qw_dot
            wx * qw + wz * qy - wy * qz,  # qx_dot
            wy * qw - wz * qx + wx * qz,  # qy_dot
            wz * qw + wy * qx - wx * qy,  # qz_dot
        ]
    )

    return q_dot


def quadrotor_dynamics_dt(
    state: Float[Array, "13"],
    action: Float[Array, "4"],
    dt: float,
    mass: float = 1.0,
    gravity: float = 9.81,
    tau_omega: float = 0.05,
) -> Float[Array, "13"]:
    """Compute state derivative for quadrotor dynamics.

    This computes dx/dt for the 13D quadrotor state.

    Args:
        state: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        action: [T, wx_cmd, wy_cmd, wz_cmd]
        dt: Time step (not used in derivative, but kept for API consistency)
        mass: Quadrotor mass (kg)
        gravity: Gravity constant (m/s², positive down in NED)
        tau_omega: Time constant for angular velocity tracking (s)

    Returns:
        State derivative dx/dt
    """
    # Extract state components
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]  # [qw, qx, qy, qz]
    omega = state[10:13]  # angular velocity in FRD body frame

    # Extract control
    thrust = action[0]  # positive magnitude
    omega_cmd = action[1:4]

    # Rotation matrix from FRD body to NED world frame
    R = quaternion_to_rotation_matrix(quat)

    # Translational dynamics (NED world frame)
    # Gravity: positive Z in NED (downward)
    f_gravity = jnp.array([0.0, 0.0, mass * gravity])
    # Thrust in body frame: [0, 0, -T] (upward in FRD)
    # Transform to world frame
    f_thrust = R @ jnp.array([0.0, 0.0, -thrust])
    accel = (f_gravity + f_thrust) / mass

    # Rotational dynamics (FRD body frame, first-order model)
    omega_dot = (omega_cmd - omega) / tau_omega

    # Quaternion kinematics
    q_dot = quaternion_derivative(quat, omega)

    # State derivative
    state_dot = jnp.concatenate([vel, accel, q_dot, omega_dot])

    return state_dot


def rk4_step(
    state: Float[Array, "13"],
    action: Float[Array, "4"],
    dt: float,
    mass: float,
    gravity: float,
    tau_omega: float,
) -> Float[Array, "13"]:
    """Single RK4 integration step for quadrotor dynamics.

    Args:
        state: Current state
        action: Control input
        dt: Time step
        mass: Quadrotor mass
        gravity: Gravity constant
        tau_omega: Angular velocity time constant

    Returns:
        Next state after dt using RK4 integration
    """
    # RK4 integration
    k1 = quadrotor_dynamics_dt(state, action, dt, mass, gravity, tau_omega)
    k2 = quadrotor_dynamics_dt(
        state + 0.5 * dt * k1, action, dt, mass, gravity, tau_omega
    )
    k3 = quadrotor_dynamics_dt(
        state + 0.5 * dt * k2, action, dt, mass, gravity, tau_omega
    )
    k4 = quadrotor_dynamics_dt(
        state + dt * k3, action, dt, mass, gravity, tau_omega
    )

    next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return next_state


def create_quadrotor_dynamics(
    dt: float = 0.01,
    mass: float = 1.0,
    gravity: float = 9.81,
    tau_omega: float = 0.05,
    u_min: Float[Array, "4"] | None = None,
    u_max: Float[Array, "4"] | None = None,
):
    """Create quadrotor dynamics function with specified parameters.

    This factory function creates a dynamics function following the library's pattern,
    with RK4 integration for better accuracy.

    Frame conventions: NED (world), FRD (body)

    Args:
        dt: Integration time step (s)
        mass: Quadrotor mass (kg)
        gravity: Gravity constant (m/s², positive down in NED)
        tau_omega: Time constant for angular velocity tracking (s)
        u_min: Minimum control bounds [T_min, wx_min, wy_min, wz_min]
        u_max: Maximum control bounds [T_max, wx_max, wy_max, wz_max]

    Returns:
        Dynamics function that takes (state, action) and returns next_state

    Example:
        >>> dynamics = create_quadrotor_dynamics(dt=0.01, mass=1.0)
        >>> state = jnp.zeros(13)
        >>> state = state.at[6].set(1.0)  # qw = 1 (identity quaternion)
        >>> action = jnp.array([mass * 9.81, 0.0, 0.0, 0.0])  # hover thrust
        >>> next_state = dynamics(state, action)
    """
    # Default control bounds if not provided
    if u_min is None:
        u_min = jnp.array([0.0, -10.0, -10.0, -10.0])
    if u_max is None:
        u_max = jnp.array([4.0 * mass * gravity, 10.0, 10.0, 10.0])

    def dynamics(
        state: Float[Array, "13"], action: Float[Array, "4"]
    ) -> Float[Array, "13"]:
        """Quadrotor dynamics with RK4 integration.

        State: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz] (13D)
            - Position/velocity in NED world frame
            - Quaternion: body FRD to world NED
            - Angular velocity in FRD body frame

        Action: [T, wx_cmd, wy_cmd, wz_cmd] (4D)
            - T: thrust magnitude (positive, acts in -Z body direction)
            - w_cmd: angular rate commands in FRD body frame

        Returns:
            next_state after dt using RK4 integration
        """
        # Clip control to bounds
        action_clipped = jnp.clip(action, u_min, u_max)

        # RK4 integration
        next_state = rk4_step(
            state, action_clipped, dt, mass, gravity, tau_omega
        )

        # Normalize quaternion to maintain unit norm
        next_quat = next_state[6:10]
        next_quat = normalize_quaternion(next_quat)
        next_state = next_state.at[6:10].set(next_quat)

        return next_state

    return dynamics
