
import jax.numpy as jnp
from jaxtyping import Array, Float

# -----------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------


def quaternion_to_rotation_matrix(
    q: Float[Array, "4"],
) -> Float[Array, "3 3"]:
    """Convert unit quaternion to rotation matrix (body FRD to world NED).

    Args:
        q: Unit quaternion [qw, qx, qy, qz].

    Returns:
        Rotation matrix R_wb (3x3).
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    return jnp.array(
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


def normalize_quaternion(q: Float[Array, "4"]) -> Float[Array, "4"]:
    """Normalize quaternion to unit length."""
    return q / (jnp.linalg.norm(q) + 1e-6)


def rk4_step(
    state: Float[Array, "13"],
    action: Float[Array, "4"],
    dt: float,
    mass: float = 1.0,
    g: float = 9.81,
    tau_omega: float = 0.05,
) -> Float[Array, "13"]:
    """Perform a single RK4 integration step.

    Args:
        state: Current state [pos(3), vel(3), quat(4), omega(3)].
        action: Control input [thrust, omega_cmd(3)].
        dt: Time step.
        mass: Quadrotor mass.
        g: Gravity acceleration.
        tau_omega: Time constant for angular rate tracking.

    Returns:
        Next state.
    """

    def dynamics_fn(s, u):
        return quadrotor_dynamics(s, u, mass, g, tau_omega)

    k1 = dynamics_fn(state, action)
    k2 = dynamics_fn(state + 0.5 * dt * k1, action)
    k3 = dynamics_fn(state + 0.5 * dt * k2, action)
    k4 = dynamics_fn(state + dt * k3, action)

    next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Re-normalize quaternion
    next_quat = normalize_quaternion(next_state[6:10])
    next_state = next_state.at[6:10].set(next_quat)

    return next_state


# -----------------------------------------------------------------------
# Dynamics Model
# -----------------------------------------------------------------------


def quadrotor_dynamics(
    state: Float[Array, "13"],
    action: Float[Array, "4"],
    mass: float = 1.0,
    g: float = 9.81,
    tau_omega: float = 0.05,
) -> Float[Array, "13"]:
    """Compute time derivative of the quadrotor state.

    State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    Action: [thrust, cmd_wx, cmd_wy, cmd_wz]

    Coordinate system: NED (North-East-Down)
    - x: North
    - y: East
    - z: Down (Gravity points in +z)

    Args:
        state: Current state vector.
        action: Control input.
        mass: Mass (kg).
        g: Gravity (m/s^2).
        tau_omega: Angular rate time constant (1/gain).

    Returns:
        State derivative (x_dot).
    """
    # Extract state components
    # pos = state[0:3] # Unused in dynamics (pos dot = vel)
    vel = state[3:6]
    quat = state[6:10]  # [qw, qx, qy, qz]
    omega = state[10:13]

    # Extract inputs
    thrust_cmd = action[0]
    omega_cmd = action[1:4]

    # 1. Position derivative: velocity
    pos_dot = vel

    # 2. Velocity derivative: F = ma
    # Forces: Gravity (down/positive z) + Thrust (up/negative z in body frame)
    # R_wb rotates vector from Body to World
    R_wb = quaternion_to_rotation_matrix(quat)

    # Thrust vector in body frame: [0, 0, -thrust] (points up in FRD)
    # Note: If thrust_cmd > 0, it pushes up (-z body).
    thrust_body = jnp.array([0.0, 0.0, -thrust_cmd])
    thrust_world = R_wb @ thrust_body

    gravity_world = jnp.array([0.0, 0.0, g])

    accel = gravity_world + thrust_world / mass
    vel_dot = accel

    # 3. Quaternion derivative: q_dot = 0.5 * q * omega
    # Quaternion multiplication of q and pure quaternion [0, omega]
    # q = [qw, qv], omega_q = [0, omega]
    # q_dot = 0.5 * (qw*omega - qv.dot(omega), qw*omega + qv x omega)
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    p, q, r = omega[0], omega[1], omega[2]

    quat_dot = 0.5 * jnp.array(
        [
            -qx * p - qy * q - qz * r,
            qw * p + qy * r - qz * q,
            qw * q - qx * r + qz * p,
            qw * r + qx * q - qy * p,
        ]
    )

    # 4. Angular velocity derivative: First order lag
    # w_dot = (w_cmd - w) / tau
    # Simplified model assuming low-level controller handles body rates
    omega_dot = (omega_cmd - omega) / tau_omega

    return jnp.concatenate([pos_dot, vel_dot, quat_dot, omega_dot])
