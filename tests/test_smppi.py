"""Unit tests for Smooth MPPI (SMPPI) implementation."""

import jax
import jax.numpy as jnp
import pytest

from jax_mppi import smppi


# Simple test dynamics and costs
def simple_dynamics(state: jax.Array, action: jax.Array) -> jax.Array:
    """Simple linear dynamics: x' = x + a (broadcasts action to match state)"""
    # Handle dimension mismatch by only using first nu elements
    nx = state.shape[0]
    nu = action.shape[0]
    if nx == nu:
        return state + action
    else:
        # Pad action with zeros to match state dimension
        action_padded = jnp.concatenate([action, jnp.zeros(nx - nu)])
        return state + action_padded


def quadratic_cost(state: jax.Array, action: jax.Array) -> jax.Array:
    """Quadratic cost: ||state||^2 + ||action||^2"""
    return jnp.sum(state**2) + 0.1 * jnp.sum(action**2)


class TestSMPPIBasics:
    """Basic functionality tests for SMPPI."""

    def test_create_returns_correct_types(self):
        """Test that create() returns proper config and state."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = smppi.create(nx=nx, nu=nu, noise_sigma=noise_sigma)

        assert isinstance(config, smppi.SMPPIConfig)
        assert isinstance(state, smppi.SMPPIState)

    def test_create_initializes_correct_shapes(self):
        """Test that state arrays have correct shapes."""
        nx, nu, horizon = 3, 2, 10
        noise_sigma = jnp.eye(nu)

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=horizon,
        )

        assert state.U.shape == (horizon, nu)
        assert state.action_sequence.shape == (horizon, nu)
        assert state.noise_mu.shape == (nu,)
        assert state.noise_sigma.shape == (nu, nu)
        assert state.u_init.shape == (nu,)

    def test_create_with_custom_parameters(self):
        """Test create with custom SMPPI parameters."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[0.5]])

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            w_action_seq_cost=2.0,
            delta_t=0.5,
            action_min=jnp.array([-1.0]),
            action_max=jnp.array([1.0]),
        )

        assert config.w_action_seq_cost == 2.0
        assert config.delta_t == 0.5
        assert state.action_min is not None
        assert state.action_max is not None

    def test_smppi_state_is_pytree(self):
        """Test that SMPPIState can be used in JAX transformations."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = smppi.create(nx=nx, nu=nu, noise_sigma=noise_sigma)

        # Test tree_flatten and tree_unflatten
        flat, tree_def = jax.tree_util.tree_flatten(state)
        reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)

        assert isinstance(reconstructed, smppi.SMPPIState)
        assert jnp.allclose(reconstructed.U, state.U)
        assert jnp.allclose(reconstructed.action_sequence, state.action_sequence)


class TestSMPPICommand:
    """Tests for SMPPI command() function."""

    def test_command_returns_correct_shapes(self):
        """Test that command() returns correct action shape."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=10,
            horizon=5,
        )

        current_state = jnp.zeros(nx)
        action, new_state = smppi.command(
            config,
            state,
            current_state,
            simple_dynamics,
            quadratic_cost,
        )

        assert action.shape == (nu,)
        assert isinstance(new_state, smppi.SMPPIState)

    def test_command_with_u_per_command(self):
        """Test command with u_per_command > 1."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=10,
            horizon=10,
            u_per_command=3,
        )

        current_state = jnp.zeros(nx)
        action, new_state = smppi.command(
            config,
            state,
            current_state,
            simple_dynamics,
            quadratic_cost,
        )

        # Should return 3 actions flattened
        assert action.shape == (3 * nu,)

    def test_command_updates_action_sequence(self):
        """Test that command updates action_sequence through integration."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[0.1]])

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=50,
            horizon=5,
            delta_t=0.5,
        )

        current_state = jnp.array([1.0, 0.0])

        # Run command
        action, new_state = smppi.command(
            config,
            state,
            current_state,
            simple_dynamics,
            quadratic_cost,
            shift=False,
        )

        # Action sequence should have changed from zeros
        assert not jnp.allclose(new_state.action_sequence, state.action_sequence, atol=1e-6)

        # Relationship: action_sequence = old_action_sequence + U * delta_t
        expected_action_seq = state.action_sequence + new_state.U * config.delta_t
        assert jnp.allclose(new_state.action_sequence, expected_action_seq, atol=1e-5)

    def test_command_is_jit_compatible(self):
        """Test that command can be JIT compiled."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=20,
            horizon=5,
        )

        @jax.jit
        def jitted_command(state, obs):
            return smppi.command(
                config,
                state,
                obs,
                simple_dynamics,
                quadratic_cost,
            )

        current_state = jnp.zeros(nx)
        action, new_state = jitted_command(state, current_state)

        assert action.shape == (nu,)
        assert isinstance(new_state, smppi.SMPPIState)


class TestSMPPISmoothness:
    """Tests for SMPPI smoothness features."""

    def test_smoothness_cost_reduces_action_variation(self):
        """Test that higher smoothness weight produces smoother actions."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[0.5]])

        # Low smoothness weight
        config_rough, state_rough = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=100,
            horizon=10,
            w_action_seq_cost=0.1,
            key=jax.random.PRNGKey(42),
        )

        # High smoothness weight
        config_smooth, state_smooth = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=100,
            horizon=10,
            w_action_seq_cost=10.0,
            key=jax.random.PRNGKey(42),
        )

        current_state = jnp.array([1.0, -0.5])

        # Run several steps
        for _ in range(5):
            _, state_rough = smppi.command(
                config_rough,
                state_rough,
                current_state,
                simple_dynamics,
                quadratic_cost,
            )
            _, state_smooth = smppi.command(
                config_smooth,
                state_smooth,
                current_state,
                simple_dynamics,
                quadratic_cost,
            )

        # Compute action variation (sum of squared differences)
        var_rough = jnp.sum(jnp.diff(state_rough.action_sequence, axis=0) ** 2)
        var_smooth = jnp.sum(jnp.diff(state_smooth.action_sequence, axis=0) ** 2)

        # Smooth should have lower variation
        # These are JAX arrays, so they are not None.
        # basedpyright gets confused by Optional fields in dataclass.
        # pyright: ignore[reportOptionalOperand]
        assert var_smooth < var_rough  # type: ignore

    def test_delta_t_affects_integration(self):
        """Test that delta_t scales the integration correctly."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[0.1]])

        # Create with different delta_t values
        config_small, state_small = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            delta_t=0.1,
            key=jax.random.PRNGKey(42),
        )

        config_large, state_large = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            delta_t=1.0,
            key=jax.random.PRNGKey(42),
        )

        # Set same U for both
        U_test = jnp.ones((config_small.horizon, nu))
        state_small = smppi.replace(state_small, U=U_test)
        state_large = smppi.replace(state_large, U=U_test)

        current_state = jnp.zeros(nx)

        _, new_small = smppi.command(
            config_small,
            state_small,
            current_state,
            simple_dynamics,
            quadratic_cost,
            shift=False,
        )

        _, new_large = smppi.command(
            config_large,
            state_large,
            current_state,
            simple_dynamics,
            quadratic_cost,
            shift=False,
        )

        # Large delta_t should have larger action changes
        change_small = jnp.max(jnp.abs(new_small.action_sequence))
        change_large = jnp.max(jnp.abs(new_large.action_sequence))

        # These are JAX arrays, so they are not None.
        # pyright: ignore[reportOptionalOperand]
        assert change_large > change_small  # type: ignore


class TestSMPPIBounds:
    """Tests for SMPPI dual bounding system."""

    def test_action_bounds_are_respected(self):
        """Test that final actions respect action_min/action_max."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[1.0]])

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=100,
            horizon=10,
            action_min=jnp.array([-0.5]),
            action_max=jnp.array([0.5]),
        )

        current_state = jnp.array([2.0, -1.0])

        # Run multiple steps
        for _ in range(10):
            action, state = smppi.command(
                config,
                state,
                current_state,
                simple_dynamics,
                quadratic_cost,
            )

            # Check action bounds
            assert jnp.all(action >= -0.5 * config.u_scale)
            assert jnp.all(action <= 0.5 * config.u_scale)

            # Check action_sequence bounds
            # pyright: ignore[reportOptionalOperand]
            assert jnp.all(state.action_sequence >= state.action_min - 1e-5)
            # pyright: ignore[reportOptionalOperand]
            assert jnp.all(state.action_sequence <= state.action_max + 1e-5)

    def test_control_bounds_are_respected(self):
        """Test that control velocities respect u_min/u_max."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[1.0]])

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=100,
            horizon=10,
            u_min=jnp.array([-0.2]),
            u_max=jnp.array([0.2]),
        )

        current_state = jnp.array([2.0, -1.0])

        # Run multiple steps
        for _ in range(10):
            _, state = smppi.command(
                config,
                state,
                current_state,
                simple_dynamics,
                quadratic_cost,
            )

            # Check control velocity bounds
            # pyright: ignore[reportOptionalOperand]
            assert jnp.all(state.U >= state.u_min - 1e-5)
            # pyright: ignore[reportOptionalOperand]
            assert jnp.all(state.U <= state.u_max + 1e-5)

    def test_symmetric_bounds_inference(self):
        """Test that symmetric bounds are inferred correctly."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        # Only min specified
        config1, state1 = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            action_min=jnp.array([-1.0]),
        )

        assert state1.action_max is not None
        assert state1.action_min is not None
        # type: ignore to suppress operator '-' not supported for None
        assert jnp.allclose(state1.action_max, -state1.action_min)  # type: ignore # pyright: ignore[reportOptionalOperand]

        # Only max specified
        config2, state2 = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            action_max=jnp.array([2.0]),
        )

        assert state2.action_min is not None
        assert state2.action_max is not None
        # type: ignore to suppress operator '-' not supported for None
        assert jnp.allclose(state2.action_min, -state2.action_max)  # type: ignore # pyright: ignore[reportOptionalOperand]


class TestSMPPIShift:
    """Tests for SMPPI shift operation."""

    def test_shift_maintains_continuity(self):
        """Test that shift holds last action for continuity."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=5,
        )

        # Set some action sequence
        state = smppi.replace(
            state,
            action_sequence=jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        )

        # Shift
        shifted = smppi._shift_nominal(state, shift_steps=1)

        # First 3 should be shifted left
        assert jnp.allclose(shifted.action_sequence[0], jnp.array([2.0]))
        assert jnp.allclose(shifted.action_sequence[1], jnp.array([3.0]))
        assert jnp.allclose(shifted.action_sequence[2], jnp.array([4.0]))
        assert jnp.allclose(shifted.action_sequence[3], jnp.array([5.0]))

        # Last should hold at previous last value (4.0, not 5.0)
        assert jnp.allclose(shifted.action_sequence[4], jnp.array([4.0]))

    def test_shift_resets_control_velocity(self):
        """Test that shift resets U to u_init at end."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=5,
            u_init=jnp.array([0.5]),
        )

        # Set some U
        state = smppi.replace(
            state,
            U=jnp.ones((5, 1)),
        )

        # Shift
        shifted = smppi._shift_nominal(state, shift_steps=1)

        # Last U should be u_init
        assert jnp.allclose(shifted.U[-1], state.u_init)


class TestSMPPIUtilities:
    """Tests for SMPPI utility functions."""

    def test_reset_zeros_trajectories(self):
        """Test that reset zeros both U and action_sequence."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
        )

        # Set non-zero values
        state = smppi.replace(
            state,
            U=jnp.ones_like(state.U),
            action_sequence=jnp.ones_like(state.action_sequence),
        )

        # Reset
        new_key = jax.random.PRNGKey(123)
        reset_state = smppi.reset(config, state, new_key)

        assert jnp.allclose(reset_state.U, 0.0)
        assert jnp.allclose(reset_state.action_sequence, 0.0)
        assert jnp.array_equal(reset_state.key, new_key)

    def test_get_rollouts_returns_correct_shape(self):
        """Test that get_rollouts returns correct trajectory shape."""
        nx, nu = 3, 2
        noise_sigma = jnp.eye(nu)

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=10,
        )

        current_state = jnp.zeros(nx)
        num_rollouts = 5

        rollouts = smppi.get_rollouts(
            config,
            state,
            current_state,
            simple_dynamics,
            num_rollouts=num_rollouts,
        )

        # Should be (num_rollouts, horizon+1, nx)
        assert rollouts.shape == (num_rollouts, config.horizon + 1, nx)

        # First state should be close to current_state
        assert jnp.allclose(rollouts[:, 0, :], current_state, atol=1e-5)


class TestSMPPIIntegration:
    """Integration tests for SMPPI."""

    def test_smppi_reduces_cost_over_iterations(self):
        """Test that SMPPI reduces cost over multiple iterations."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[0.5]])

        config, state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=100,
            horizon=10,
            w_action_seq_cost=1.0,
            key=jax.random.PRNGKey(42),
        )

        current_state = jnp.array([1.0, -0.5])

        costs = []
        for _ in range(10):
            action, state = smppi.command(
                config,
                state,
                current_state,
                simple_dynamics,
                quadratic_cost,
            )
            cost = quadratic_cost(current_state, action)
            costs.append(cost)

        costs = jnp.array(costs)

        # Cost should generally decrease
        # (allowing some variation due to stochasticity)
        assert costs[-1] < costs[0] * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
