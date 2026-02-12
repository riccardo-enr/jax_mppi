"""Differential drive UGV dynamics.

State (5D): [x, y, θ, v, ω]
    - x, y: position in world frame
    - θ: heading angle (radians)
    - v: linear velocity (m/s)
    - ω: angular velocity (rad/s)

Control (2D): [a, α]
    - a: linear acceleration command (m/s²)
    - α: angular acceleration command (rad/s²)
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


def diffdrive_dynamics_dt(
    state: Float[Array, "5"],
    action: Float[Array, "2"],
    dt: float,
    max_v: float = 2.0,
    max_omega: float = 2.0,
    drag: float = 0.1,
) -> Float[Array, "5"]:
    """State derivative for differential drive dynamics.

    Args:
        state: [x, y, θ, v, ω]
        action: [a, α] linear and angular acceleration
        dt: time step (unused in derivative, kept for API consistency)
        max_v: maximum linear velocity
        max_omega: maximum angular velocity
        drag: viscous drag coefficient

    Returns:
        State derivative dx/dt
    """
    theta = state[2]
    v = state[3]
    omega = state[4]

    a = action[0]
    alpha = action[1]

    # Position kinematics
    x_dot = v * jnp.cos(theta)
    y_dot = v * jnp.sin(theta)
    theta_dot = omega

    # Velocity dynamics with drag
    v_dot = a - drag * v
    omega_dot = alpha - drag * omega

    return jnp.array([x_dot, y_dot, theta_dot, v_dot, omega_dot])


def rk4_step(
    state: Float[Array, "5"],
    action: Float[Array, "2"],
    dt: float,
    max_v: float,
    max_omega: float,
    drag: float,
) -> Float[Array, "5"]:
    """Single RK4 integration step for differential drive dynamics.

    Args:
        state: Current state [x, y, θ, v, ω]
        action: Control input [a, α]
        dt: Time step
        max_v: Maximum linear velocity
        max_omega: Maximum angular velocity
        drag: Viscous drag coefficient

    Returns:
        Next state after dt using RK4 integration
    """
    k1 = diffdrive_dynamics_dt(state, action, dt, max_v, max_omega, drag)
    k2 = diffdrive_dynamics_dt(
        state + 0.5 * dt * k1, action, dt, max_v, max_omega, drag
    )
    k3 = diffdrive_dynamics_dt(
        state + 0.5 * dt * k2, action, dt, max_v, max_omega, drag
    )
    k4 = diffdrive_dynamics_dt(
        state + dt * k3, action, dt, max_v, max_omega, drag
    )

    next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Clip velocities to limits
    next_v = jnp.clip(next_state[3], -max_v, max_v)
    next_omega = jnp.clip(next_state[4], -max_omega, max_omega)
    next_state = next_state.at[3].set(next_v)
    next_state = next_state.at[4].set(next_omega)

    # Wrap heading to [-π, π]
    next_theta = jnp.arctan2(jnp.sin(next_state[2]), jnp.cos(next_state[2]))
    next_state = next_state.at[2].set(next_theta)

    return next_state


def create_diffdrive_dynamics(
    dt: float = 0.05,
    max_v: float = 2.0,
    max_omega: float = 2.0,
    max_accel: float = 1.0,
    max_alpha: float = 2.0,
    drag: float = 0.1,
):
    """Create differential drive dynamics function.

    Args:
        dt: Integration time step (s)
        max_v: Maximum linear velocity (m/s)
        max_omega: Maximum angular velocity (rad/s)
        max_accel: Maximum linear acceleration (m/s²)
        max_alpha: Maximum angular acceleration (rad/s²)
        drag: Viscous drag coefficient

    Returns:
        Dynamics function: (state, action) -> next_state

    Example:
        >>> dynamics = create_diffdrive_dynamics(dt=0.05)
        >>> state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        >>> action = jnp.array([1.0, 0.0])  # accelerate forward
        >>> next_state = dynamics(state, action)
    """
    u_min = jnp.array([-max_accel, -max_alpha])
    u_max = jnp.array([max_accel, max_alpha])

    def dynamics(
        state: Float[Array, "5"], action: Float[Array, "2"]
    ) -> Float[Array, "5"]:
        """Differential drive dynamics with RK4 integration.

        State: [x, y, θ, v, ω] (5D)
        Action: [a, α] (2D) — linear and angular acceleration

        Returns:
            next_state after dt using RK4 integration
        """
        action_clipped = jnp.clip(action, u_min, u_max)
        return rk4_step(state, action_clipped, dt, max_v, max_omega, drag)

    return dynamics
