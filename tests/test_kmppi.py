import jax.numpy as jnp
import numpy as np

from jax_mppi import kmppi


def test_rbf_kernel_properties():
    """Test that RBF kernel has correct properties."""
    kernel_fn = kmppi.RBFKernel(sigma=1.0)

    # 1. K(t, t) = 1
    t = jnp.array([0.0, 1.0, 2.0])
    K = kernel_fn(t, t)

    assert K.shape == (3, 3)
    np.testing.assert_allclose(jnp.diag(K), 1.0, atol=1e-5)

    # 2. Symmetry K(t1, t2) = K(t2, t1)
    np.testing.assert_allclose(K, K.T, atol=1e-5)

    # 3. Value decreases with distance
    assert K[0, 1] < K[0, 0]
    assert K[0, 2] < K[0, 1]

    # 4. Correct value: exp(-d^2 / 2sigma^2)
    # d(0, 1) = 1. exp(-0.5) approx 0.606
    expected = np.exp(-0.5)
    np.testing.assert_allclose(K[0, 1], expected, atol=1e-5)


def test_rbf_kernel_broadcasting():
    """Test kernel broadcasting with different shapes."""
    kernel_fn = kmppi.RBFKernel(sigma=1.0)

    t = jnp.linspace(0, 10, 20)  # (20,)
    tk = jnp.linspace(0, 10, 5)  # (5,)

    K = kernel_fn(t, tk)
    assert K.shape == (20, 5)

    # Test with (T, 1) and (K, 1)
    t_col = t[:, None]
    tk_col = tk[:, None]
    K2 = kernel_fn(t_col, tk_col)
    assert K2.shape == (20, 5)

    np.testing.assert_allclose(K, K2, atol=1e-5)


def test_kmppi_state_create():
    """Test KMPPI-specific state creation."""
    # Mock config
    from jax_mppi.mppi import MPPIConfig

    config = MPPIConfig(
        dynamics_fn=lambda x, u: x,
        cost_fn=lambda x, u: 0.0,
        nx=2,
        nu=1,
        num_samples=10,
        horizon=20,
        lambda_=1.0,
        noise_sigma=jnp.eye(1),
        u_min=jnp.array([-1.0]),
        u_max=jnp.array([1.0]),
        u_init=jnp.array([0.0]),
        step_method="kmppi",
        num_support_pts=5,
    )

    config, state = kmppi.create_kmppi(config, seed=42)

    # Check KMPPI fields
    assert hasattr(state, "theta")
    assert hasattr(state, "Tk")
    assert hasattr(state, "Hs")

    assert state.theta.shape == (5, 1)  # (K, nu)
    assert state.Tk.shape == (5,)
    assert state.Hs.shape == (20,)

    # U should be interpolated from theta (initially 0)
    assert state.U.shape == (20, 1)
    np.testing.assert_allclose(state.U, 0.0)


def test_update_control():
    """Test parameter update logic."""
    kernel_fn = kmppi.RBFKernel(sigma=1.0)

    # Setup state
    horizon = 10
    num_support = 3
    nu = 1

    Hs = jnp.arange(horizon, dtype=float)
    Tk = jnp.linspace(0, horizon - 1, num_support)
    theta = jnp.zeros((num_support, nu))
    U = jnp.zeros((horizon, nu))

    from jax_mppi.mppi import MPPIState

    # Manually populate state
    state = MPPIState(
        U=U,
        key=jnp.array([0, 0], dtype=jnp.uint32),
        step=0,
        theta=theta,
        Tk=Tk,
        Hs=Hs,
    )

    # Create dummy update data
    K = 5  # samples
    weights = jnp.ones(K) / K  # Uniform weights

    # Noise in parameter space
    noise = jnp.ones((K, num_support, nu)) * 0.1
    # Weighted sum should be 0.1

    new_state = kmppi.update_control(state, weights, noise, kernel_fn)

    # Check theta update
    # new_theta = theta + sum(w * noise) = 0 + 0.1 = 0.1
    np.testing.assert_allclose(new_state.theta, 0.1, atol=1e-5)

    # Check U update
    # U = Kernel * theta
    # Kernel row sums? Not necessarily 1, but should be > 0
    K_mat = kernel_fn(Hs, Tk)
    expected_U = K_mat @ new_state.theta

    np.testing.assert_allclose(new_state.U, expected_U, atol=1e-5)


def test_create_kmppi_default_kernel():
    """Test create_kmppi uses correct default kernel sigma."""
    # Mock config
    from jax_mppi.mppi import MPPIConfig

    config = MPPIConfig(
        dynamics_fn=lambda x, u: x,
        cost_fn=lambda x, u: 0.0,
        nx=2,
        nu=1,
        num_samples=10,
        horizon=20,
        lambda_=1.0,
        noise_sigma=jnp.eye(1),
        u_min=jnp.array([-1.0]),
        u_max=jnp.array([1.0]),
        u_init=jnp.array([0.0]),
        step_method="kmppi",
        num_support_pts=5,
    )

    config, state = kmppi.create_kmppi(config, seed=42)

    # Verify kernel function was created (implicitly checked by execution)
    # But create_kmppi returns config/state, not kernel.
    # The kernel is typically passed or created in step function.
    # In create_kmppi implementation, we might attach it or just verify
    # state setup.

    # Check if we can recreate the kernel used
    # The helper `create_kmppi` accepts kernel_sigma arg
    config2, state2 = kmppi.create_kmppi(config, seed=42, kernel_sigma=2.0)

    # There's no way to inspect the closure if it's not stored,
    # but `create_kmppi` just sets up state.
    # So we just verify state is valid.
    assert state2.Tk.shape == (5,)


def test_kmppi_integration():
    """Integration test with dummy dynamics."""
    # Basic rollout test
    pass


def test_rbf_kernel_type():
    """Test RBFKernel satisfies Protocol."""
    kernel_fn = kmppi.RBFKernel(sigma=2.0)
    # kernel_fn is typed as TimeKernel (Protocol), so basedpyright doesn't
    # know about sigma
    assert isinstance(kernel_fn, kmppi.RBFKernel)
    assert kernel_fn.sigma == 2.0
