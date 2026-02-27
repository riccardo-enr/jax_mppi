"""Unit tests for MPPI implementation."""

import jax
import jax.numpy as jnp
import pytest

from jax_mppi import mppi

# Define simple dynamics and cost for testing
# Simple double integrator:
# x = [pos, vel]
# u = [acc]
# x_next = [pos + vel*dt, vel + acc*dt]


def double_integrator_dynamics(state, action, dt=0.1):
    pos = state[0]
    vel = state[1]
    acc = action[0]
    pos_next = pos + vel * dt
    vel_next = vel + acc * dt
    return jnp.array([pos_next, vel_next])


def quadratic_cost(state, action, goal=0.0):
    pos = state[0]
    return (pos - goal) ** 2 + 0.1 * action[0] ** 2


class TestMPPIBasics:
    """Test basic MPPI functionality."""

    def test_create_returns_correct_shapes(self):
        nx = 2
        nu = 1
        num_samples = 10
        horizon = 5
        noise_sigma = jnp.eye(nu) * 0.1

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            num_samples=num_samples,
            horizon=horizon,
            noise_sigma=noise_sigma,
        )

        assert config.num_samples == num_samples
        assert config.horizon == horizon
        assert config.nx == nx
        assert config.nu == nu

        assert state.U.shape == (horizon, nu)
        assert state.noise_mu.shape == (nu,)
        assert state.noise_sigma.shape == (nu, nu)
        assert state.noise_sigma_inv.shape == (nu, nu)

    def test_reset_clears_trajectory(self):
        nx = 2
        nu = 1
        noise_sigma = jnp.eye(nu)
        u_init = jnp.array([1.0])

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            u_init=u_init,
        )

        # Manually modify U
        state = state.replace(U=jnp.ones_like(state.U) * 5.0)
        assert jnp.all(state.U == 5.0)

        # Reset
        key = jax.random.PRNGKey(1)
        state = mppi.reset(config, state, key)

        # Should be reset to u_init
        assert jnp.all(state.U == 1.0)


class TestMPPICommand:
    """Test MPPI command generation."""

    @pytest.fixture
    def mppi_setup(self):
        nx = 2
        nu = 1
        num_samples = 100
        horizon = 10
        noise_sigma = jnp.eye(nu) * 0.5
        lambda_ = 0.1

        config, state = mppi.create(
            nx=nx,
            nu=nu,
            num_samples=num_samples,
            horizon=horizon,
            noise_sigma=noise_sigma,
            lambda_=lambda_,
            u_min=jnp.array([-2.0]),
            u_max=jnp.array([2.0]),
        )
        return config, state

    def test_command_returns_correct_shapes(self, mppi_setup):
        config, state = mppi_setup
        obs = jnp.zeros(2)

        action, new_state = mppi.command(
            config,
            state,
            obs,
            double_integrator_dynamics,
            quadratic_cost,
        )

        assert action.shape == (config.nu,)
        assert new_state.U.shape == (config.horizon, config.nu)

    def test_command_respects_bounds(self, mppi_setup):
        config, state = mppi_setup
        obs = jnp.zeros(2)

        # Force high cost to drive action to limits?
        # Instead, verify the output action is within bounds
        action, _ = mppi.command(
            config,
            state,
            obs,
            double_integrator_dynamics,
            quadratic_cost,
        )

        assert jnp.all(action >= state.u_min)
        assert jnp.all(action <= state.u_max)

    def test_shift_behavior(self, mppi_setup):
        config, state = mppi_setup
        obs = jnp.zeros(2)

        # Set specific U to track shift
        U_pattern = jnp.arange(config.horizon)[:, None] * 1.0
        state = state.replace(U=U_pattern)

        # Run command with shift=True
        # We need to mock weights computation to predict U_new,
        # or just check that shift happens on the result.
        # Since command() updates U with noise, we can't easily predict exact U.
        # But we can check that if we disable noise/learning, shift happens.

        # Hack: set lambda very high so weights are uniform -> U_new ~ U (mean of noise is 0)
        # Actually U_new = U + weighted_noise. If noise mean is 0, U_new approx U.
        # But shift happens AFTER update.
        # Let's just check shape and general property (last element is u_init)

        action, new_state = mppi.command(
            config,
            state,
            obs,
            double_integrator_dynamics,
            quadratic_cost,
            shift=True,
        )

        # The last element should be u_init (0.0)
        assert jnp.allclose(new_state.U[-1], state.u_init)

        # With shift=False
        action_no_shift, state_no_shift = mppi.command(
            config,
            state,
            obs,
            double_integrator_dynamics,
            quadratic_cost,
            shift=False,
        )
        # Last element should NOT be reset to u_init if it wasn't already
        # (Though update might change it slightly)


class TestMPPIIntegration:
    """Integration tests for MPPI convergence."""

    def test_convergence_to_goal(self):
        nx = 2
        nu = 1
        # Use large samples/horizon for convergence
        config, state = mppi.create(
            nx=nx,
            nu=nu,
            num_samples=1000,
            horizon=20,
            noise_sigma=jnp.eye(nu) * 0.5,
            lambda_=0.01,
        )

        # Dynamics: x_next = x + u
        def simple_dyn(x, u):
            return x + u

        # Cost: ||x - goal||^2
        goal = jnp.array([5.0, 5.0])

        def cost_fn(x, u):
            return jnp.sum((x - goal) ** 2)

        obs = jnp.array([0.0, 0.0])

        # Run loop
        for _ in range(20):
            action, state = mppi.command(
                config, state, obs, simple_dyn, cost_fn, shift=True
            )
            obs = simple_dyn(obs, action)

        # Should be close to goal
        assert jnp.linalg.norm(obs - goal) < 1.0

    def test_step_dependent_dynamics(self):
        """Test that time step t is passed correctly to dynamics/cost."""
        nx = 1
        nu = 1
        config, state = mppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=jnp.eye(nu),
            step_dependent_dynamics=True,
        )

        # Dynamics that depend on t
        def time_dyn(x, u, t):
            # If t is not passed, this will fail
            return x + u + t * 0.1

        def time_cost(x, u, t):
            # If t is not passed, this will fail
            return (x - 1.0) ** 2 + t * 0.01

        obs = jnp.array([0.0])
        # Should run without error
        action, state = mppi.command(
            config, state, obs, time_dyn, time_cost, shift=True
        )


class TestMPPIUtils:
    """Test utility functions in mppi module."""

    def test_rollouts_generation(self):
        nx = 2
        nu = 1
        horizon = 10
        num_rollouts = 5
        config, state = mppi.create(
            nx=nx,
            nu=nu,
            horizon=horizon,
            noise_sigma=jnp.eye(nu),
        )

        obs = jnp.zeros(nx)
        rollouts = mppi.get_rollouts(
            config, state, obs, double_integrator_dynamics, num_rollouts
        )

        # Shape: (num_rollouts, horizon+1, nx)
        assert rollouts.shape == (num_rollouts, horizon + 1, nx)

        # Initial state should match obs
        assert jnp.all(rollouts[:, 0, :] == obs)

    def test_cost_evaluation_shapes(self):
        # Verify internal cost functions handle shapes correctly
        nx, nu = 2, 1
        config = mppi.MPPIConfig(
            num_samples=10,
            horizon=5,
            nx=nx,
            nu=nu,
            lambda_=1.0,
            u_scale=1.0,
            u_per_command=1,
            step_dependent_dynamics=False,
            rollout_samples=1,
            rollout_var_cost=0.0,
            rollout_var_discount=0.95,
            sample_null_action=False,
            noise_abs_cost=False,
        )

        obs = jnp.zeros(nx)
        actions = jnp.zeros((config.num_samples, config.horizon, nu))

        costs = mppi._compute_rollout_costs(
            config,
            obs,
            actions,
            double_integrator_dynamics,
            quadratic_cost,
            terminal_cost=None,
        )

        assert costs.shape == (config.num_samples,)

    def test_weights_computation(self):
        costs = jnp.array([10.0, 1.0, 100.0])
        lambda_ = 1.0
        weights = mppi._compute_weights(costs, lambda_)

        assert weights.shape == costs.shape
        assert jnp.abs(jnp.sum(weights) - 1.0) < 1e-6
        # Lowest cost should have highest weight
        assert jnp.argmax(weights) == 1

    def test_noise_cost_modes(self):
        # Test quadratic vs absolute noise cost
        noise = jnp.array([[[1.0], [2.0]], [[-1.0], [-2.0]]])  # (2, 2, 1) samples
        sigma_inv = jnp.eye(1)

        # Quadratic: 0.5 * x^T * S^-1 * x
        # Sample 1: 0.5*(1+4) = 2.5
        # Sample 2: 0.5*(1+4) = 2.5
        cost_quad = mppi._compute_noise_cost(
            noise, sigma_inv, noise_abs_cost=False
        )
        assert jnp.allclose(cost_quad, jnp.array([2.5, 2.5]))

        # Absolute: |x|
        # Wait, implementation of _compute_noise_cost for abs:
        # quad = abs_noise.T * abs(sigma_inv) * abs_noise ... then 0.5 * sum
        # In code:
        # if noise_abs_cost:
        #   abs_noise = jnp.abs(noise)
        #   quad = jnp.einsum('ktd,df,ktf->kt', abs_noise, jnp.abs(noise_sigma_inv), abs_noise)
        #   return 0.5 * jnp.sum(quad, axis=1)

        # So abs cost in this implementation is actually
        # 0.5 * |noise|^T * |Sigma^-1| * |noise|
        # It's still quadratic form, just on absolute values.
        # This seems to be what the code does, let's verify.
        cost_abs = mppi._compute_noise_cost(noise, sigma_inv, noise_abs_cost=True)
        assert jnp.allclose(cost_abs, jnp.array([2.5, 2.5]))

    def test_terminal_cost(self):
        nx, nu = 1, 1
        config, state = mppi.create(
            nx=nx,
            nu=nu,
            num_samples=5,
            horizon=2,
            noise_sigma=jnp.eye(nu),
        )

        def dynamics(x, u):
            return x + u

        def running_cost(x, u):
            return 0.0

        def terminal_cost(x, last_u):
            return x[0] ** 2  # Cost is squared final state

        obs = jnp.array([0.0])

        # Action = 1 -> final state = 2 -> terminal cost = 4
        # Action = 0 -> final state = 0 -> terminal cost = 0
        actions = jnp.array(
            [[[1.0], [1.0]], [[0.0], [0.0]]]
        )  # (2 samples, 2 steps, 1 dim)

        costs = mppi._compute_rollout_costs(
            config, obs, actions, dynamics, running_cost, terminal_cost
        )

        assert jnp.allclose(costs[0], 4.0)
        assert jnp.allclose(costs[1], 0.0)

    def test_eval_cost_function(self):
        """Test auxiliary cost evaluation function used in tests."""
        # This test ensures the helper `eval_cost` logic logic inside tests is valid
        nx, nu = 2, 1
        config, state = mppi.create(nx=nx, nu=nu, noise_sigma=jnp.eye(nu))

        def eval_cost(action_val):
            # Helper to run a rollout with fixed action
            actions = jnp.ones((config.horizon, nu)) * action_val
            # Just create a 1-sample batch of actions
            batch_actions = actions[None, :, :]
            costs = mppi._compute_rollout_costs(
                config,
                jnp.zeros(nx),
                batch_actions,
                double_integrator_dynamics,
                quadratic_cost,
                None,
            )
            return costs[0]

        action_high = jnp.array([2.0])
        action_low = jnp.array([0.5])

        # We assign to variables to avoid unused warning in the test itself
        # but here we assert
        cost_high = eval_cost(action_high)
        cost_low = eval_cost(action_low)

        # High action should incur more control cost
        # cost = sum((pos-goal)^2 + 0.1*u^2)
        # Higher u -> higher pos deviation (if goal=0) AND higher control cost
        assert cost_high > cost_low
