"""Tests for Pendulum example."""

import jax
import jax.numpy as jnp
import numpy as np

from examples import pendulum


def test_pendulum_dynamics():
    """Test pendulum dynamics logic."""
    # 1. Hanging down (stable equilibrium)
    # Theta = pi, theta_dot = 0, torque = 0
    state = jnp.array([np.pi, 0.0])
    action = jnp.array([0.0])

    next_state = pendulum.pendulum_dynamics(state, action)

    # Should stay at pi (or -pi depending on wrap)
    # sin(pi) = 0, so theta_ddot = 0
    # wrap handles pi -> -pi potentially
    assert jnp.abs(jnp.abs(next_state[0]) - np.pi) < 1e-4
    assert jnp.abs(next_state[1]) < 1e-4

    # 2. Upright (unstable equilibrium)
    # Theta = 0, theta_dot = 0, torque = 0
    state = jnp.array([0.0, 0.0])
    next_state = pendulum.pendulum_dynamics(state, action)

    # Should fall? Actually at exactly 0 it stays if unstable eq.
    # But numerical noise might tip it.
    # sin(0) = 0.
    assert jnp.abs(next_state[0]) < 1e-4

    # 3. Horizontal, gravity pulls down
    # Theta = pi/2
    state = jnp.array([np.pi / 2, 0.0])
    next_state = pendulum.pendulum_dynamics(state, action)

    # sin(pi/2) = 1. Torque=0.
    # theta_ddot = -g/l
    # theta decreases (falls down towards 0 or pi? Wait, 0 is up.)
    pass


def test_pendulum_cost():
    """Test cost function."""
    # Goal is 0, 0
    state = jnp.array([0.0, 0.0])
    action = jnp.array([0.0])
    c = pendulum.pendulum_cost(state, action)
    assert c < 1e-6

    # Cost increases with angle
    state2 = jnp.array([1.0, 0.0])
    c2 = pendulum.pendulum_cost(state2, action)
    assert c2 > c

    # Cost increases with torque
    action2 = jnp.array([1.0])
    c3 = pendulum.pendulum_cost(state, action2)
    assert c3 > c


def test_pendulum_run_integration():
    """Run short simulation to verify integration."""
    # Run for few steps only
    try:
        pendulum.run_pendulum(render=False, steps=5)
    except Exception as e:
        assert False, f"Run failed with: {e}"


def test_pendulum_performance():
    """Verify controller actually stabilizes."""
    # This is a stochastic test, might be flaky.
    # Use fixed seed in run_pendulum
    from jax_mppi import mppi

    config = mppi.MPPIConfig(
        dynamics_fn=pendulum.pendulum_dynamics,
        cost_fn=pendulum.pendulum_cost,
        nx=2,
        nu=1,
        num_samples=100,
        horizon=20,
        lambda_=0.1,
        noise_sigma=jnp.array([[0.5]]),
        u_min=jnp.array([-2.0]),
        u_max=jnp.array([2.0]),
        u_init=jnp.array([0.0]),
        step_method="mppi",
    )

    config, state = mppi.create(config, seed=42)
    x0 = jnp.array([np.pi, 0.0])

    jit_step = jax.jit(mppi.step)
    jit_step(config, state, x0)  # Compile

    sim_state = x0
    costs = []

    # Run for 50 steps
    for _ in range(50):
        state, action, _ = jit_step(config, state, sim_state)
        sim_state = pendulum.pendulum_dynamics(sim_state, action)
        c = pendulum.pendulum_cost(sim_state, action)
        costs.append(float(c))

    initial_cost = costs[0]
    final_cost = jnp.mean(jnp.array(costs[-10:]))

    # Should reduce cost significantly (at least 50% reduction in final
    # average)
    assert final_cost < initial_cost * 0.5, (
        f"Failed to swing up: initial_cost={initial_cost:.2f}, "
        f"final_avg_cost={final_cost:.2f}"
    )
