"""Unit tests for Kernel MPPI (KMPPI) implementation."""

import jax
import jax.numpy as jnp
import pytest

from jax_mppi import kmppi


# Simple test dynamics and costs
def simple_dynamics(state: jax.Array, action: jax.Array) -> jax.Array:
    """Simple linear dynamics: x' = x + a (with dimension handling)"""
    nx = state.shape[0]
    nu = action.shape[0]
    if nx == nu:
        return state + action
    else:
        action_padded = jnp.concatenate([action, jnp.zeros(nx - nu)])
        return state + action_padded


def quadratic_cost(state: jax.Array, action: jax.Array) -> jax.Array:
    """Quadratic cost: ||state||^2 + ||action||^2"""
    return jnp.sum(state**2) + 0.1 * jnp.sum(action**2)


class TestRBFKernel:
    """Tests for RBF kernel implementation."""

    def test_rbf_kernel_shape(self):
        """Test that RBF kernel returns correct shape."""
        kernel = kmppi.RBFKernel(sigma=1.0)

        t = jnp.array([0.0, 1.0, 2.0])  # 3 query points
        tk = jnp.array([0.0, 2.0])  # 2 control points

        K = kernel(t, tk)

        assert K.shape == (3, 2)

    def test_rbf_kernel_identity(self):
        """Test that RBF kernel evaluates to 1 at identical points."""
        kernel = kmppi.RBFKernel(sigma=1.0)

        t = jnp.array([1.0, 2.0, 3.0])

        K = kernel(t, t)

        # Diagonal should be all 1s
        assert jnp.allclose(jnp.diag(K), 1.0)

    def test_rbf_kernel_decreases_with_distance(self):
        """Test that RBF kernel decreases with distance."""
        kernel = kmppi.RBFKernel(sigma=1.0)

        t = jnp.array([0.0])
        tk = jnp.array([0.0, 1.0, 2.0, 3.0])

        K = kernel(t, tk)[0]  # Get first row

        # Should decrease monotonically as distance increases
        assert K[0] > K[1] > K[2] > K[3]

    def test_rbf_kernel_sigma_effect(self):
        """Test that larger sigma produces wider kernels."""
        kernel_narrow = kmppi.RBFKernel(sigma=0.5)
        kernel_wide = kmppi.RBFKernel(sigma=2.0)

        t = jnp.array([0.0])
        tk = jnp.array([0.0, 2.0])

        K_narrow = kernel_narrow(t, tk)[0]
        K_wide = kernel_wide(t, tk)[0]

        # At distance 2, wider kernel should have higher value
        assert K_wide[1] > K_narrow[1]


class TestKMPPIBasics:
    """Basic functionality tests for KMPPI."""

    def test_create_returns_correct_types(self):
        """Test that create() returns proper config, state, and kernel."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state, kernel_fn = kmppi.create(nx=nx, nu=nu, noise_sigma=noise_sigma)

        assert isinstance(config, kmppi.KMPPIConfig)
        assert isinstance(state, kmppi.KMPPIState)
        assert isinstance(kernel_fn, kmppi.RBFKernel)

    def test_create_initializes_correct_shapes(self):
        """Test that state arrays have correct shapes."""
        nx, nu, horizon = 3, 2, 10
        noise_sigma = jnp.eye(nu)
        num_support_pts = 5

        config, state, _ = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=horizon,
            num_support_pts=num_support_pts,
        )

        assert state.U.shape == (horizon, nu)
        assert state.theta.shape == (num_support_pts, nu)
        assert state.Tk.shape == (num_support_pts,)
        assert state.Hs.shape == (horizon,)
        assert config.num_support_pts == num_support_pts

    def test_create_with_custom_kernel(self):
        """Test create with custom kernel."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[0.5]])
        custom_kernel = kmppi.RBFKernel(sigma=2.0)

        config, state, kernel_fn = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            kernel=custom_kernel,
        )

        # kernel_fn is typed as TimeKernel (Protocol), so basedpyright doesn't know about sigma
        assert isinstance(kernel_fn, kmppi.RBFKernel)
        assert kernel_fn.sigma == 2.0

    def test_default_num_support_pts(self):
        """Test that default num_support_pts is horizon // 2."""
        nx, nu, horizon = 2, 1, 20
        noise_sigma = jnp.eye(nu)

        config, state, _ = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=horizon,
        )

        assert config.num_support_pts == horizon // 2

    def test_kmppi_state_is_pytree(self):
        """Test that KMPPIState can be used in JAX transformations."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state, _ = kmppi.create(nx=nx, nu=nu, noise_sigma=noise_sigma)

        # Test tree_flatten and tree_unflatten
        flat, tree_def = jax.tree_util.tree_flatten(state)
        reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)

        assert isinstance(reconstructed, kmppi.KMPPIState)
        assert jnp.allclose(reconstructed.U, state.U)
        assert jnp.allclose(reconstructed.theta, state.theta)


class TestKernelInterpolation:
    """Tests for kernel interpolation functionality."""

    def test_kernel_interpolate_shape(self):
        """Test that kernel interpolation returns correct shape."""
        kernel_fn = kmppi.RBFKernel(sigma=1.0)

        t = jnp.linspace(0, 10, 20)  # 20 query points
        tk = jnp.linspace(0, 10, 5)  # 5 control points
        control_values = jnp.ones((5, 2))  # 5 control points, 2D actions

        interpolated, K = kmppi._kernel_interpolate(t, tk, control_values, kernel_fn)

        assert interpolated.shape == (20, 2)
        assert K.shape == (20, 5)

    def test_kernel_interpolate_preserves_control_points(self):
        """Test that interpolation passes through control points."""
        kernel_fn = kmppi.RBFKernel(sigma=1.0)

        tk = jnp.array([0.0, 5.0, 10.0])
        control_values = jnp.array([[1.0], [2.0], [3.0]])

        interpolated, _ = kmppi._kernel_interpolate(tk, tk, control_values, kernel_fn)

        # Should pass through control points (within numerical precision)
        assert jnp.allclose(interpolated, control_values, atol=1e-5)

    def test_interpolation_is_smooth(self):
        """Test that interpolation produces smooth trajectories."""
        kernel_fn = kmppi.RBFKernel(sigma=1.0)

        # Create control points with a step
        tk = jnp.array([0.0, 10.0])
        control_values = jnp.array([[0.0], [1.0]])

        # Interpolate to fine grid
        t = jnp.linspace(0, 10, 100)
        interpolated, _ = kmppi._kernel_interpolate(t, tk, control_values, kernel_fn)

        # Check that trajectory is smooth (bounded second derivative)
        diffs = jnp.diff(interpolated[:, 0])
        second_diffs = jnp.diff(diffs)

        # Second derivative should be bounded (smooth transition)
        assert jnp.max(jnp.abs(second_diffs)) < 0.1


class TestKMPPICommand:
    """Tests for KMPPI command() function."""

    def test_command_returns_correct_shapes(self):
        """Test that command() returns correct action shape."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state, kernel_fn = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=10,
            horizon=10,
            num_support_pts=5,
        )

        current_state = jnp.zeros(nx)
        action, new_state = kmppi.command(
            config,
            state,
            current_state,
            simple_dynamics,
            quadratic_cost,
            kernel_fn,
        )

        assert action.shape == (nu,)
        assert isinstance(new_state, kmppi.KMPPIState)

    def test_command_updates_theta(self):
        """Test that command updates control points."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[0.5]])

        config, state, kernel_fn = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=50,
            horizon=10,
            num_support_pts=5,
        )

        current_state = jnp.array([1.0, -0.5])

        action, new_state = kmppi.command(
            config,
            state,
            current_state,
            simple_dynamics,
            quadratic_cost,
            kernel_fn,
            shift=False,
        )

        # Control points should have changed
        assert not jnp.allclose(new_state.theta, state.theta, atol=1e-6)

    def test_command_updates_U_via_interpolation(self):
        """Test that U is updated via interpolation of theta."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[0.5]])

        config, state, kernel_fn = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=50,
            horizon=10,
            num_support_pts=5,
        )

        current_state = jnp.array([1.0, 0.0])

        action, new_state = kmppi.command(
            config,
            state,
            current_state,
            simple_dynamics,
            quadratic_cost,
            kernel_fn,
            shift=False,
        )

        # Manually interpolate theta to check U
        expected_U, _ = kmppi._kernel_interpolate(
            new_state.Hs, new_state.Tk, new_state.theta, kernel_fn
        )

        # These are JAX arrays, so they are not None.
        assert jnp.allclose(new_state.U, expected_U, atol=1e-5)  # type: ignore

    def test_command_is_jit_compatible(self):
        """Test that command can be JIT compiled."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state, kernel_fn = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=20,
            horizon=10,
            num_support_pts=5,
        )

        @jax.jit
        def jitted_command(state, obs):
            return kmppi.command(
                config,
                state,
                obs,
                simple_dynamics,
                quadratic_cost,
                kernel_fn,
            )

        current_state = jnp.zeros(nx)
        action, new_state = jitted_command(state, current_state)

        assert action.shape == (nu,)
        assert isinstance(new_state, kmppi.KMPPIState)


class TestKMPPISmoothness:
    """Tests for KMPPI smoothness properties."""

    def test_fewer_control_points_smoother_trajectories(self):
        """Test that fewer control points produce smoother trajectories."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[1.0]])
        horizon = 20

        # Many control points (closer to base MPPI)
        config_many, state_many, kernel_many = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=100,
            horizon=horizon,
            num_support_pts=15,  # 15 out of 20
            key=jax.random.PRNGKey(42),
        )

        # Few control points (smoother)
        config_few, state_few, kernel_few = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=100,
            horizon=horizon,
            num_support_pts=5,  # 5 out of 20
            key=jax.random.PRNGKey(42),
        )

        current_state = jnp.array([1.0, -0.5])

        # Run several iterations
        for _ in range(5):
            _, state_many = kmppi.command(
                config_many,
                state_many,
                current_state,
                simple_dynamics,
                quadratic_cost,
                kernel_many,
            )
            _, state_few = kmppi.command(
                config_few,
                state_few,
                current_state,
                simple_dynamics,
                quadratic_cost,
                kernel_few,
            )

        # Compute trajectory variation (sum of squared differences)
        var_many = jnp.sum(jnp.diff(state_many.U, axis=0) ** 2)
        var_few = jnp.sum(jnp.diff(state_few.U, axis=0) ** 2)

        # Fewer control points should produce smoother trajectories
        assert var_few < var_many * 0.8  # At least 20% smoother


class TestKMPPIShift:
    """Tests for KMPPI shift operation."""

    def test_shift_maintains_interpolation_consistency(self):
        """Test that shift maintains consistency between theta and U."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state, kernel_fn = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=10,
            num_support_pts=5,
        )

        # Set some control points
        state = kmppi.replace(
            state,
            theta=jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        )

        # Manually shift
        shifted_theta = kmppi._shift_control_points(
            state.theta, state.Tk, state.u_init, shift_steps=1, kernel_fn=kernel_fn
        )

        # Interpolate shifted theta
        shifted_U, _ = kmppi._kernel_interpolate(
            state.Hs, state.Tk, shifted_theta, kernel_fn
        )

        # Shapes should be preserved
        assert shifted_theta.shape == state.theta.shape
        assert shifted_U.shape == state.U.shape


class TestKMPPIUtilities:
    """Tests for KMPPI utility functions."""

    def test_reset_zeros_trajectories(self):
        """Test that reset zeros both theta and U."""
        nx, nu = 2, 1
        noise_sigma = jnp.eye(nu)

        config, state, kernel_fn = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
        )

        # Set non-zero values
        state = kmppi.replace(
            state,
            theta=jnp.ones_like(state.theta),
        )

        # Reset
        new_key = jax.random.PRNGKey(123)
        reset_state = kmppi.reset(config, state, kernel_fn, new_key)

        assert jnp.allclose(reset_state.theta, 0.0)
        assert jnp.allclose(reset_state.U, 0.0, atol=1e-5)
        assert jnp.array_equal(reset_state.key, new_key)

    def test_get_rollouts_returns_correct_shape(self):
        """Test that get_rollouts returns correct trajectory shape."""
        nx, nu = 3, 2
        noise_sigma = jnp.eye(nu)

        config, state, kernel_fn = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            horizon=10,
        )

        current_state = jnp.zeros(nx)
        num_rollouts = 5

        rollouts = kmppi.get_rollouts(
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


class TestKMPPIBounds:
    """Tests for KMPPI action bounding."""

    def test_action_bounds_are_respected(self):
        """Test that actions respect u_min/u_max bounds."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[1.0]])

        config, state, kernel_fn = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=100,
            horizon=10,
            num_support_pts=5,
            u_min=jnp.array([-0.5]),
            u_max=jnp.array([0.5]),
        )

        current_state = jnp.array([2.0, -1.0])

        # Run multiple steps
        for _ in range(10):
            action, state = kmppi.command(
                config,
                state,
                current_state,
                simple_dynamics,
                quadratic_cost,
                kernel_fn,
            )

            # Check action bounds
            assert jnp.all(action >= -0.5 * config.u_scale)
            assert jnp.all(action <= 0.5 * config.u_scale)

            # Check U bounds
            # type: ignore to suppress operator '-' and '+' not supported for None
            assert jnp.all(state.U >= state.u_min - 1e-5)  # type: ignore
            assert jnp.all(state.U <= state.u_max + 1e-5)  # type: ignore


class TestKMPPIIntegration:
    """Integration tests for KMPPI."""

    def test_kmppi_reduces_cost_over_iterations(self):
        """Test that KMPPI reduces cost over multiple iterations."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[0.5]])

        config, state, kernel_fn = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=100,
            horizon=10,
            num_support_pts=5,
            key=jax.random.PRNGKey(42),
        )

        current_state = jnp.array([1.0, -0.5])

        costs = []
        for _ in range(10):
            action, state = kmppi.command(
                config,
                state,
                current_state,
                simple_dynamics,
                quadratic_cost,
                kernel_fn,
            )
            cost = quadratic_cost(current_state, action)
            costs.append(cost)

        costs = jnp.array(costs)

        # Cost should generally decrease
        assert costs[-1] < costs[0]

    def test_kmppi_with_small_num_support_pts(self):
        """Test KMPPI with very few control points."""
        nx, nu = 2, 1
        noise_sigma = jnp.array([[0.5]])

        config, state, kernel_fn = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=50,
            horizon=20,
            num_support_pts=3,  # Only 3 control points
        )

        current_state = jnp.zeros(nx)

        # Should still work
        action, new_state = kmppi.command(
            config,
            state,
            current_state,
            simple_dynamics,
            quadratic_cost,
            kernel_fn,
        )

        assert action.shape == (nu,)
        assert new_state.theta.shape == (3, nu)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
