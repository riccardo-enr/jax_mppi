"""Integration tests for pendulum example with MPPI control."""

import jax
import jax.numpy as jnp
import pytest

from jax_mppi import mppi


def pendulum_dynamics(state: jax.Array, action: jax.Array) -> jax.Array:
    """Pendulum dynamics (copied from examples/pendulum.py for testing)."""
    g = 10.0
    m = 1.0
    length = 1.0
    dt = 0.05

    theta, theta_dot = state[0], state[1]
    torque = action[0]

    torque = jnp.clip(torque, -2.0, 2.0)
    theta_ddot = (torque - m * g * length * jnp.sin(theta)) / (m * length * length)

    theta_dot_next = theta_dot + theta_ddot * dt
    theta_next = theta + theta_dot_next * dt
    theta_next = ((theta_next + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    return jnp.array([theta_next, theta_dot_next])


def pendulum_cost(state: jax.Array, action: jax.Array) -> jax.Array:
    """Running cost for pendulum."""
    theta, theta_dot = state[0], state[1]
    torque = action[0]

    angle_cost = theta**2
    velocity_cost = 0.1 * theta_dot**2
    control_cost = 0.01 * torque**2

    return angle_cost + velocity_cost + control_cost


def pendulum_terminal_cost(state: jax.Array, last_action: jax.Array) -> jax.Array:
    """Terminal cost for pendulum.

    Args:
        state: (2,) terminal state [theta, theta_dot]
        last_action: (1,) last action [torque]

    Returns:
        cost: scalar terminal cost
    """
    theta, theta_dot = state[0], state[1]
    return 10.0 * theta**2 + theta_dot**2


class TestPendulumIntegration:
    """Integration tests for pendulum swing-up with MPPI."""

    def test_pendulum_dynamics_shape(self):
        """Test that pendulum dynamics returns correct shape."""
        state = jnp.array([0.0, 0.0])
        action = jnp.array([1.0])

        next_state = pendulum_dynamics(state, action)

        assert next_state.shape == (2,)

    def test_pendulum_dynamics_equilibrium(self):
        """Test that upright position with no torque stays near upright."""
        state = jnp.array([0.0, 0.0])  # upright, no velocity
        action = jnp.array([0.0])  # no torque

        next_state = pendulum_dynamics(state, action)

        # Should stay close to upright (small drift due to discretization)
        assert jnp.abs(next_state[0]) < 0.1
        assert jnp.abs(next_state[1]) < 0.1

    def test_pendulum_cost_minimum_at_upright(self):
        """Test that cost is minimized at upright position."""
        upright = jnp.array([0.0, 0.0])
        hanging = jnp.array([jnp.pi, 0.0])
        action = jnp.array([0.0])

        cost_upright = pendulum_cost(upright, action)
        cost_hanging = pendulum_cost(hanging, action)

        assert cost_upright < cost_hanging

    def test_mppi_pendulum_stabilization(self):
        """Test that MPPI can stabilize pendulum near upright position."""
        key = jax.random.PRNGKey(42)

        nx = 2
        nu = 1
        noise_sigma = jnp.array([[0.5]])

        config, mppi_state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=500,
            horizon=20,
            lambda_=1.0,
            u_min=jnp.array([-2.0]),
            u_max=jnp.array([2.0]),
            key=key,
        )

        # Start near upright with small perturbation
        state = jnp.array([0.2, 0.0])

        command_fn = jax.jit(
            lambda mppi_state, obs: mppi.command(
                config=config,
                mppi_state=mppi_state,
                current_obs=obs,
                dynamics=pendulum_dynamics,
                running_cost=pendulum_cost,
                terminal_cost=pendulum_terminal_cost,
                shift=True,
            )
        )

        # Run for 50 steps
        for _ in range(50):
            action, mppi_state = command_fn(mppi_state, state)
            state = pendulum_dynamics(state, action)

        # Should be close to upright
        assert jnp.abs(state[0]) < 0.3, f"Failed to stabilize: theta={state[0]}"
        assert jnp.abs(state[1]) < 1.0, f"High velocity: theta_dot={state[1]}"

    def test_mppi_pendulum_swing_up(self):
        """Test that MPPI can swing up pendulum from hanging position."""
        key = jax.random.PRNGKey(123)

        nx = 2
        nu = 1
        noise_sigma = jnp.array([[1.0]])

        config, mppi_state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=1000,
            horizon=30,
            lambda_=1.0,
            u_min=jnp.array([-2.0]),
            u_max=jnp.array([2.0]),
            key=key,
        )

        # Start hanging down with small perturbation
        state = jnp.array([jnp.pi + 0.1, 0.0])

        command_fn = jax.jit(
            lambda mppi_state, obs: mppi.command(
                config=config,
                mppi_state=mppi_state,
                current_obs=obs,
                dynamics=pendulum_dynamics,
                running_cost=pendulum_cost,
                terminal_cost=pendulum_terminal_cost,
                shift=True,
            )
        )

        # Track cost reduction
        initial_cost = pendulum_cost(state, jnp.array([0.0]))
        costs = []

        # Run for 100 steps (swing-up is harder, needs more steps)
        for _ in range(100):
            action, mppi_state = command_fn(mppi_state, state)
            state = pendulum_dynamics(state, action)
            costs.append(pendulum_cost(state, action))

        final_cost = jnp.mean(jnp.array(costs[-10:]))

        # Should reduce cost significantly (at least 50% reduction in final average)
        assert final_cost < initial_cost * 0.5, (
            f"Failed to swing up: initial_cost={initial_cost:.2f}, final_avg_cost={final_cost:.2f}"
        )

        # Should be closer to upright than hanging
        assert jnp.abs(state[0]) < jnp.abs(jnp.pi), f"Still far from upright: theta={state[0]:.2f}"

    def test_mppi_respects_torque_bounds(self):
        """Test that MPPI respects control bounds."""
        key = jax.random.PRNGKey(0)

        nx = 2
        nu = 1
        noise_sigma = jnp.array([[1.0]])

        config, mppi_state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=100,
            horizon=10,
            lambda_=1.0,
            u_min=jnp.array([-2.0]),
            u_max=jnp.array([2.0]),
            key=key,
        )

        state = jnp.array([jnp.pi, 0.0])

        command_fn = jax.jit(
            lambda mppi_state, obs: mppi.command(
                config=config,
                mppi_state=mppi_state,
                current_obs=obs,
                dynamics=pendulum_dynamics,
                running_cost=pendulum_cost,
                shift=True,
            )
        )

        # Run a few steps and check bounds
        for _ in range(10):
            action, mppi_state = command_fn(mppi_state, state)
            assert jnp.all(action >= -2.0), f"Action below min: {action}"
            assert jnp.all(action <= 2.0), f"Action above max: {action}"
            state = pendulum_dynamics(state, action)

    def test_pendulum_cost_function_continuity(self):
        """Test that cost function is continuous (no jumps)."""
        states = jnp.linspace(-jnp.pi, jnp.pi, 100)
        action = jnp.array([0.0])

        costs = jnp.array([pendulum_cost(jnp.array([theta, 0.0]), action) for theta in states])

        # Check no large jumps
        cost_diffs = jnp.abs(jnp.diff(costs))
        assert jnp.max(cost_diffs) < 1.0, "Cost function has discontinuities"

    def test_get_rollouts_pendulum(self):
        """Test that get_rollouts works with pendulum dynamics."""
        key = jax.random.PRNGKey(0)

        nx = 2
        nu = 1
        noise_sigma = jnp.array([[1.0]])

        config, mppi_state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=10,
            horizon=15,
            lambda_=1.0,
            key=key,
        )

        state = jnp.array([0.5, 0.0])

        # Get rollouts
        rollouts = mppi.get_rollouts(
            config=config,
            mppi_state=mppi_state,
            current_obs=state,
            dynamics=pendulum_dynamics,
            num_rollouts=3,
        )

        # Check shape: (num_rollouts, horizon+1, nx)
        assert rollouts.shape == (3, 16, 2)

        # First state should be close to initial state
        assert jnp.allclose(rollouts[:, 0, :], state, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
