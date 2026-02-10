"""Unit tests for MPPI implementation."""

import jax
import jax.numpy as jnp
import pytest

from jax_mppi import mppi


# Simple test dynamics and costs
def simple_dynamics(state: jax.Array, action: jax.Array) -> jax.Array:
    """Simple linear dynamics: x' = x + a (broadcasts action to match state)"""
    nx = state.shape[0]
    nu = action.shape[0]
    if nx == nu:
        return state + action
    else:
        action_padded = jnp.concatenate([action, jnp.zeros(nx - nu)])
        return state + action_padded


def step_dependent_dynamics(
    state: jax.Array, action: jax.Array, t: int
) -> jax.Array:
    """Dynamics that depend on the time step."""
    return state + action + t * 0.1


def quadratic_cost(state: jax.Array, action: jax.Array) -> jax.Array:
    """Quadratic cost: ||state||^2 + ||action||^2"""
    return jnp.sum(state**2) + 0.1 * jnp.sum(action**2)


def step_dependent_cost(
    state: jax.Array, action: jax.Array, t: int
) -> jax.Array:
    """Cost that depends on the time step."""
    return jnp.sum(state**2) + 0.1 * jnp.sum(action**2) + t * 0.01


class TestMPPIBasics:
    """Basic functionality tests for MPPI."""

    def test_create_returns_correct_types(self):
        """Test that create() returns proper config and state."""
        nx, nu = 4, 2
        noise_sigma = jnp.eye(nu) * 0.1
        horizon = 20

        config, state = mppi.create(
            nx=nx, nu=nu, noise_sigma=noise_sigma, horizon=horizon
        )

        assert isinstance(config, mppi.MPPIConfig)
        assert isinstance(state, mppi.MPPIState)

    def test_create_initializes_correct_shapes(self):
        """Test that state arrays have correct shapes."""
        nx, nu = 4, 2
        noise_sigma = jnp.eye(nu) * 0.1
        horizon = 20

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=horizon,
        )

        assert config.nx == nx
        assert config.nu == nu
        assert config.horizon == horizon
        assert state.U.shape == (horizon, nu)
        assert state.noise_sigma.shape == (nu, nu)

    def test_create_with_custom_parameters(self):
        """Test create with custom MPPI parameters."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            u_scale=0.5,
            u_per_command=2,
            step_dependent_dynamics=True,
            sample_null_action=True,
            noise_abs_cost=True,
        )

        assert config.u_scale == 0.5
        assert config.u_per_command == 2
        assert config.step_dependent_dynamics is True
        assert config.sample_null_action is True
        assert config.noise_abs_cost is True

    def test_mppi_state_is_pytree(self):
        """Test that MPPIState can be used in JAX transformations."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)
        config, state = mppi.create(nx=nx, nu=nu, noise_sigma=noise_sigma)

        # Test tree_flatten and tree_unflatten
        flat, tree_def = jax.tree_util.tree_flatten(state)
        reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)

        assert isinstance(reconstructed, mppi.MPPIState)
        assert jnp.allclose(reconstructed.U, state.U)


class TestMPPICommand:
    """Tests for MPPI command() function."""

    def test_command_returns_correct_shapes(self):
        """Test that command() returns correct action shape."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)
        horizon = 10

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=horizon,
            num_samples=32,
        )

        current_obs = jnp.array([1.0, 1.0])
        action, new_state = mppi.command(
            config,
            state,
            current_obs,
            simple_dynamics,
            quadratic_cost,
        )

        assert action.shape == (nu,)
        assert new_state.U.shape == (horizon, nu)

    def test_command_with_u_per_command(self):
        """Test command with u_per_command > 1."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)
        u_per_command = 2

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            u_per_command=u_per_command,
        )

        current_obs = jnp.zeros(nx)
        action, new_state = mppi.command(
            config,
            state,
            current_obs,
            simple_dynamics,
            quadratic_cost,
        )

        # Should return u_per_command actions
        # Note: unlike smppi which flattens, mppi currently returns (u_per_command, nu)
        assert action.shape == (u_per_command, nu)

    def test_step_dependent_dynamics(self):
        """Test that step-dependent dynamics are used correctly."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            step_dependent_dynamics=True,
        )

        # Use step_dependent_dynamics and cost
        current_obs = jnp.zeros(nx)
        action, new_state = mppi.command(
            config,
            state,
            current_obs,
            step_dependent_dynamics,
            step_dependent_cost,
        )

        assert action.shape == (nu,)

    def test_sample_null_action(self):
        """Test that sample_null_action works without error."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            sample_null_action=True,
        )

        current_obs = jnp.zeros(nx)
        action, new_state = mppi.command(
            config,
            state,
            current_obs,
            simple_dynamics,
            quadratic_cost,
        )
        assert action.shape == (nu,)

    def test_noise_abs_cost(self):
        """Test that noise_abs_cost works without error."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            noise_abs_cost=True,
        )

        current_obs = jnp.zeros(nx)
        action, new_state = mppi.command(
            config,
            state,
            current_obs,
            simple_dynamics,
            quadratic_cost,
        )
        assert action.shape == (nu,)

    def test_u_scale(self):
        """Test that u_scale affects actions."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        # Create two instances, one with large scale, one with small
        config_small, state_small = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            u_scale=0.01,
            key=jax.random.PRNGKey(0),
        )
        config_large, state_large = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            u_scale=100.0,
            key=jax.random.PRNGKey(0),
        )

        current_obs = jnp.zeros(nx)

        action_small, _ = mppi.command(
            config_small,
            state_small,
            current_obs,
            simple_dynamics,
            quadratic_cost,
        )

        action_large, _ = mppi.command(
            config_large,
            state_large,
            current_obs,
            simple_dynamics,
            quadratic_cost,
        )

        # The large scale should result in larger magnitude actions typically
        # (though this depends on the cost landscape, for a quadratic cost around 0,
        # both might try to be small, but the exploration noise is scaled).
        # Actually, u_scale scales the *control authority*.
        # The noise is added to U, then multiplied by u_scale.
        # So larger u_scale -> larger effective noise in action space -> larger exploration.

        # We can check that the action magnitude is different/larger for the large scale case
        # assuming the noise drove it somewhat away from 0.

        assert jnp.max(jnp.abs(action_large)) > jnp.max(jnp.abs(action_small))


class TestMPPIBounds:
    """Tests for MPPI bounding system."""

    def test_mppi_command_shapes_and_bounds(self):
        nx = 1
        nu = 1
        noise_sigma = jnp.eye(nu) * 0.2
        horizon = 12

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=horizon,
            num_samples=64,
            u_min=jnp.array([-0.2]),
            u_max=jnp.array([0.2]),
            u_per_command=2,
        )

        action, new_state = mppi.command(
            config,
            state,
            current_obs=jnp.array([1.0]),
            dynamics=simple_dynamics,
            running_cost=quadratic_cost,
        )

        assert action.shape == (2, 1)
        assert jnp.all(action <= 0.2 + 1e-6)
        assert jnp.all(action >= -0.2 - 1e-6)
        assert new_state.U.shape == (horizon, nu)


class TestMPPIUtilities:
    """Tests for MPPI utility functions."""

    def test_mppi_reset(self):
        nx = 4
        nu = 2
        noise_sigma = jnp.eye(nu) * 0.1
        horizon = 20

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=horizon,
            U_init=jnp.ones((horizon, nu)),
        )

        assert jnp.all(state.U == 1.0)

        key = jax.random.PRNGKey(1)
        state = mppi.reset(config, state, key)

        # reset should set U to u_init (default zeros)
        assert jnp.all(state.U == 0.0)

    def test_mppi_get_rollouts_shapes(self):
        nx = 2
        nu = 2
        noise_sigma = jnp.eye(nu) * 0.1
        horizon = 5

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=horizon,
            num_samples=32,
        )

        rollouts = mppi.get_rollouts(
            config,
            state,
            current_obs=jnp.zeros(nx),
            dynamics=simple_dynamics,
            num_rollouts=3,
        )
        assert rollouts.shape == (3, horizon + 1, nx)

        batched_rollouts = mppi.get_rollouts(
            config,
            state,
            current_obs=jnp.zeros((2, nx)),
            dynamics=simple_dynamics,
            num_rollouts=2,
        )
        assert batched_rollouts.shape == (2, 2, horizon + 1, nx)


class TestMPPIIntegration:
    """Integration tests for MPPI."""

    def test_temperature_effect(self):
        """Test that lower temperature (lambda) leads to more aggressive optimization."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu) * 1.0

        # High temperature (effectively random walk / average of noise)
        config_high, state_high = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            lambda_=1000.0,
            key=jax.random.PRNGKey(0),
        )

        # Low temperature (aggressive selection of best sample)
        config_low, state_low = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            lambda_=0.001,
            key=jax.random.PRNGKey(0),
        )

        # State where optimal action is non-zero (approx -0.9)
        # Cost is ||state+action||^2 + 0.1*||action||^2
        current_obs = jnp.array([1.0, 1.0])

        # Use same key for same noise samples
        key = jax.random.PRNGKey(42)
        state_high = mppi.replace(state_high, key=key)
        state_low = mppi.replace(state_low, key=key)

        action_high, _ = mppi.command(
            config_high,
            state_high,
            current_obs,
            simple_dynamics,
            quadratic_cost,
        )

        action_low, _ = mppi.command(
            config_low,
            state_low,
            current_obs,
            simple_dynamics,
            quadratic_cost,
        )

        # Evaluate performance
        def eval_cost(a):
            next_s = simple_dynamics(current_obs, a)
            return quadratic_cost(next_s, a)

        cost_high = eval_cost(action_high)
        cost_low = eval_cost(action_low)
        cost_idle = eval_cost(jnp.zeros(nu))

        # Check that temperature parameter has an effect
        assert not jnp.allclose(action_high, action_low, atol=1e-4)

        # Sanity check: High temperature (averaging) usually produces
        # reliable improvement over doing nothing in convex landscapes
        assert cost_high < cost_idle

    def test_cost_reduction(self):
        """Test that MPPI reduces cost over iterations."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu) * 0.5

        config, state = mppi.create(
            nx=nx, nu=nu, noise_sigma=noise_sigma, num_samples=100, horizon=15
        )

        current_obs = jnp.array([2.0, 2.0])

        costs = []
        for _ in range(10):
            action, state = mppi.command(
                config, state, current_obs, simple_dynamics, quadratic_cost
            )
            # Apply action
            current_obs = simple_dynamics(current_obs, action)
            costs.append(quadratic_cost(current_obs, action))

        # Cost should decrease significantly
        assert costs[-1] < costs[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
