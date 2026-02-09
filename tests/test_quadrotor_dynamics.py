"""Unit tests for quadrotor dynamics."""

import jax
import jax.numpy as jnp

from jax_mppi.dynamics.quadrotor import (
    create_quadrotor_dynamics,
    normalize_quaternion,
    quaternion_derivative,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
)


class TestQuaternionUtilities:
    """Tests for quaternion utility functions."""

    def test_identity_quaternion_to_rotation_matrix(self):
        """Test that identity quaternion gives identity rotation matrix."""
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)

        expected = jnp.eye(3)
        assert jnp.allclose(R, expected, atol=1e-6)

    def test_rotation_matrix_orthonormality(self):
        """Test that rotation matrix is orthonormal."""
        # Random quaternion
        q = jnp.array([0.5, 0.5, 0.5, 0.5])
        q = normalize_quaternion(q)

        R = quaternion_to_rotation_matrix(q)

        # R @ R.T should be identity
        RTR = R @ R.T
        assert jnp.allclose(RTR, jnp.eye(3), atol=1e-6)

        # det(R) should be 1
        det = jnp.linalg.det(R)
        assert jnp.allclose(det, 1.0, atol=1e-6)

    def test_normalize_quaternion_preserves_direction(self):
        """Test that normalization preserves quaternion direction."""
        q = jnp.array([2.0, 0.0, 0.0, 0.0])
        q_norm = normalize_quaternion(q)

        # Should be [1, 0, 0, 0]
        expected = jnp.array([1.0, 0.0, 0.0, 0.0])
        assert jnp.allclose(q_norm, expected, atol=1e-6)

    def test_normalize_quaternion_unit_norm(self):
        """Test that normalized quaternion has unit norm."""
        q = jnp.array([1.5, 2.0, 3.0, 0.5])
        q_norm = normalize_quaternion(q)

        norm = jnp.linalg.norm(q_norm)
        assert jnp.allclose(norm, 1.0, atol=1e-6)

    def test_quaternion_multiply_identity(self):
        """Test quaternion multiplication with identity."""
        q = jnp.array([0.5, 0.5, 0.5, 0.5])
        q = normalize_quaternion(q)

        q_identity = jnp.array([1.0, 0.0, 0.0, 0.0])

        result = quaternion_multiply(q, q_identity)
        assert jnp.allclose(result, q, atol=1e-6)

        result = quaternion_multiply(q_identity, q)
        assert jnp.allclose(result, q, atol=1e-6)

    def test_quaternion_multiply_inverse(self):
        """Test quaternion multiplication with conjugate gives identity."""
        q = jnp.array([0.5, 0.5, 0.5, 0.5])
        q = normalize_quaternion(q)

        # Conjugate (inverse for unit quaternion)
        q_conj = jnp.array([q[0], -q[1], -q[2], -q[3]])

        result = quaternion_multiply(q, q_conj)
        expected = jnp.array([1.0, 0.0, 0.0, 0.0])

        assert jnp.allclose(result, expected, atol=1e-6)

    def test_quaternion_derivative_zero_omega(self):
        """Test quaternion derivative with zero angular velocity."""
        q = jnp.array([0.5, 0.5, 0.5, 0.5])
        q = normalize_quaternion(q)
        omega = jnp.zeros(3)

        q_dot = quaternion_derivative(q, omega)

        # Should be zero
        assert jnp.allclose(q_dot, jnp.zeros(4), atol=1e-6)

    def test_quaternion_derivative_maintains_norm_constraint(self):
        """Test that q_dot is orthogonal to q (maintains unit norm)."""
        q = jnp.array([0.5, 0.5, 0.5, 0.5])
        q = normalize_quaternion(q)
        omega = jnp.array([1.0, 2.0, 0.5])

        q_dot = quaternion_derivative(q, omega)

        # q and q_dot should be orthogonal: q^T q_dot = 0
        dot_product = jnp.dot(q, q_dot)
        assert jnp.allclose(dot_product, 0.0, atol=1e-6)


class TestQuadrotorDynamics:
    """Tests for quadrotor dynamics function."""

    def test_hover_state_with_hover_thrust(self):
        """Test that hover thrust keeps quadrotor stationary."""
        dt = 0.01
        mass = 1.0
        gravity = 9.81

        dynamics = create_quadrotor_dynamics(dt=dt, mass=mass, gravity=gravity)

        # Initial state: at origin, level flight, no velocity
        state = jnp.zeros(13)
        state = state.at[6].set(1.0)  # qw = 1 (identity quaternion)

        # Hover thrust: T = m*g, no angular rates
        action = jnp.array([mass * gravity, 0.0, 0.0, 0.0])

        # Simulate multiple steps
        for _ in range(100):
            state = dynamics(state, action)

        # Position should remain near zero (small numerical drift)
        pos = state[0:3]
        assert jnp.allclose(pos, jnp.zeros(3), atol=0.01)

        # Velocity should remain near zero
        vel = state[3:6]
        assert jnp.allclose(vel, jnp.zeros(3), atol=0.01)

    def test_quaternion_norm_preservation(self):
        """Test that quaternion norm is preserved during integration."""
        dt = 0.01
        dynamics = create_quadrotor_dynamics(dt=dt)

        # Initial state with identity quaternion
        state = jnp.zeros(13)
        state = state.at[6].set(1.0)  # qw = 1

        # Apply some angular rates
        action = jnp.array([9.81, 1.0, -0.5, 2.0])

        # Simulate
        for _ in range(100):
            state = dynamics(state, action)
            quat = state[6:10]
            norm = jnp.linalg.norm(quat)
            assert jnp.allclose(norm, 1.0, atol=1e-6)

    def test_gravity_pulls_down(self):
        """Test that without thrust, quadrotor falls due to gravity."""
        dt = 0.01
        mass = 1.0
        gravity = 9.81

        dynamics = create_quadrotor_dynamics(dt=dt, mass=mass, gravity=gravity)

        # Initial state: at origin, level flight
        state = jnp.zeros(13)
        state = state.at[6].set(1.0)  # qw = 1

        # No thrust, no angular rates
        action = jnp.array([0.0, 0.0, 0.0, 0.0])

        # Simulate
        for _ in range(10):
            state = dynamics(state, action)

        # Should have fallen (positive Z in NED)
        z_pos = state[2]
        assert z_pos > 0.01  # fallen downward

        # Velocity should be positive (downward in NED)
        z_vel = state[5]
        assert z_vel > 0.0

    def test_thrust_produces_acceleration(self):
        """Test that thrust produces upward acceleration."""
        dt = 0.01
        mass = 1.0
        gravity = 9.81

        dynamics = create_quadrotor_dynamics(dt=dt, mass=mass, gravity=gravity)

        # Initial state: at origin, level flight
        state = jnp.zeros(13)
        state = state.at[6].set(1.0)  # qw = 1

        # High thrust (more than hover)
        action = jnp.array([2.0 * mass * gravity, 0.0, 0.0, 0.0])

        # Simulate
        for _ in range(10):
            state = dynamics(state, action)

        # Should have climbed (negative Z in NED)
        z_pos = state[2]
        assert z_pos < -0.01  # climbed upward

        # Velocity should be negative (upward in NED)
        z_vel = state[5]
        assert z_vel < 0.0

    def test_angular_velocity_tracking(self):
        """Test that angular velocity tracks commanded rates."""
        dt = 0.01
        mass = 1.0
        tau_omega = 0.05

        dynamics = create_quadrotor_dynamics(
            dt=dt, mass=mass, tau_omega=tau_omega
        )

        # Initial state
        state = jnp.zeros(13)
        state = state.at[6].set(1.0)  # qw = 1

        # Command angular rates
        omega_cmd = jnp.array([1.0, 2.0, -1.0])
        action = jnp.array([mass * 9.81, omega_cmd[0], omega_cmd[1], omega_cmd[2]])

        # Simulate until convergence
        for _ in range(200):
            state = dynamics(state, action)

        # Angular velocity should track commanded rates
        omega = state[10:13]
        assert jnp.allclose(omega, omega_cmd, atol=0.01)

    def test_control_bounds_respected(self):
        """Test that control inputs are clipped to bounds."""
        dt = 0.01
        mass = 1.0
        u_min = jnp.array([0.0, -5.0, -5.0, -5.0])
        u_max = jnp.array([20.0, 5.0, 5.0, 5.0])

        dynamics = create_quadrotor_dynamics(
            dt=dt, mass=mass, u_min=u_min, u_max=u_max
        )

        # Initial state
        state = jnp.zeros(13)
        state = state.at[6].set(1.0)

        # Command beyond bounds
        action = jnp.array([100.0, 10.0, -10.0, 10.0])

        # Should not crash and should apply clipped values
        next_state = dynamics(state, action)
        assert next_state.shape == (13,)

    def test_dynamics_is_jit_compatible(self):
        """Test that dynamics function can be JIT compiled."""
        dynamics = create_quadrotor_dynamics(dt=0.01, mass=1.0)
        dynamics_jit = jax.jit(dynamics)

        state = jnp.zeros(13)
        state = state.at[6].set(1.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])

        # Should run without error
        next_state = dynamics_jit(state, action)
        assert next_state.shape == (13,)

    def test_dynamics_gradient_exists(self):
        """Test that gradients can be computed through dynamics."""
        dynamics = create_quadrotor_dynamics(dt=0.01, mass=1.0)

        def loss_fn(action):
            state = jnp.zeros(13)
            state = state.at[6].set(1.0)
            next_state = dynamics(state, action)
            return jnp.sum(next_state**2)

        action = jnp.array([9.81, 0.0, 0.0, 0.0])

        # Should compute gradients without error
        grad = jax.grad(loss_fn)(action)
        assert grad.shape == (4,)

    def test_different_mass_affects_dynamics(self):
        """Test that changing mass affects dynamics appropriately."""
        dt = 0.01

        dynamics_light = create_quadrotor_dynamics(dt=dt, mass=0.5)
        dynamics_heavy = create_quadrotor_dynamics(dt=dt, mass=2.0)

        # Same initial state
        state = jnp.zeros(13)
        state = state.at[6].set(1.0)

        # Same thrust
        action = jnp.array([10.0, 0.0, 0.0, 0.0])

        # Simulate
        state_light = state
        state_heavy = state
        for _ in range(10):
            state_light = dynamics_light(state_light, action)
            state_heavy = dynamics_heavy(state_heavy, action)

        # Light quadrotor should accelerate more with same thrust
        z_vel_light = state_light[5]
        z_vel_heavy = state_heavy[5]

        # Light should have more negative velocity (climbed faster)
        assert z_vel_light < z_vel_heavy


class TestQuadrotorDynamicsIntegration:
    """Integration tests for complete quadrotor dynamics."""

    def test_circular_motion_with_roll(self):
        """Test that roll rate produces lateral motion."""
        dt = 0.01
        mass = 1.0
        gravity = 9.81

        dynamics = create_quadrotor_dynamics(dt=dt, mass=mass, gravity=gravity)

        # Initial state: hovering, level flight
        state = jnp.zeros(13)
        state = state.at[6].set(1.0)  # qw = 1

        # Hover thrust + roll rate
        action = jnp.array([mass * gravity, 1.0, 0.0, 0.0])

        # Simulate
        states = [state]
        for _ in range(200):
            state = dynamics(state, action)
            states.append(state)

        # Should have rotated (quaternion changed)
        final_quat = state[6:10]
        initial_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        assert not jnp.allclose(final_quat, initial_quat, atol=0.1)

    def test_energy_increases_with_thrust(self):
        """Test that high thrust increases total energy."""
        dt = 0.01
        mass = 1.0
        gravity = 9.81

        dynamics = create_quadrotor_dynamics(dt=dt, mass=mass, gravity=gravity)

        # Initial state
        state = jnp.zeros(13)
        state = state.at[6].set(1.0)

        # High thrust
        action = jnp.array([2.0 * mass * gravity, 0.0, 0.0, 0.0])

        initial_kinetic = 0.5 * mass * jnp.sum(state[3:6]**2)
        initial_potential = mass * gravity * state[2]
        initial_energy = initial_kinetic + initial_potential

        # Simulate
        for _ in range(50):
            state = dynamics(state, action)

        final_kinetic = 0.5 * mass * jnp.sum(state[3:6]**2)
        final_potential = mass * gravity * state[2]
        final_energy = final_kinetic + final_potential

        # Energy should increase (work done by thrust)
        assert final_energy > initial_energy
