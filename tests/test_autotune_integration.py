"""Integration tests for autotune with MPPI variants."""

import jax.numpy as jnp
import numpy as np

from jax_mppi import autotune, mppi


def test_autotune_integration_basic():
    """Test basic autotune workflow with mock evaluation."""
    # 1. Setup MPPI
    config = mppi.MPPIConfig(
        dynamics_fn=lambda x, u: x + u,
        cost_fn=lambda x, u: jnp.sum(x**2),
        nx=1,
        nu=1,
        num_samples=10,
        horizon=10,
        lambda_=1.0,
        noise_sigma=jnp.eye(1),
        u_min=jnp.array([-1.0]),
        u_max=jnp.array([1.0]),
        u_init=jnp.array([0.0]),
        step_method="mppi",
    )
    config, state = mppi.create(config, seed=42)
    holder = autotune.ConfigStateHolder(config, state)

    # 2. Evaluation function
    def evaluate():
        # Mock evaluation: cost depends on parameters
        # Target: lambda=0.5, sigma=1.0, horizon=15
        lambda_err = (holder.config.lambda_ - 0.5) ** 2
        sigma_err = (holder.state.noise_sigma[0, 0] - 1.0) ** 2
        horizon_err = (holder.config.horizon - 15) ** 2

        cost = lambda_err + sigma_err + horizon_err
        return autotune.EvaluationResult(
            mean_cost=float(cost),
            rollouts=jnp.zeros((1, 1)),
            params={},
            iteration=0,
        )

    # 3. Create Tuner
    # Use dummy optimizer that just samples near 0.5, 1.0, 15
    class MockOptimizer(autotune.Optimizer):
        def setup_optimization(self, initial_params, evaluate_fn):
            self.eval_fn = evaluate_fn
            self.iter = 0

        def optimize_step(self):
            # Deterministic sequence towards solution
            if self.iter == 0:
                # Bad guess
                p = [2.0, 2.0, 10.0]
            else:
                # Good guess
                p = [0.5, 1.0, 15.0]

            self.iter += 1
            return self.eval_fn(np.array(p))

    tuner = autotune.Autotune(
        params_to_tune=[
            autotune.LambdaParameter(holder),
            autotune.NoiseSigmaParameter(holder),
            autotune.HorizonParameter(holder, min_value=5, max_value=20),
        ],
        evaluate_fn=evaluate,
        optimizer=MockOptimizer(),
    )

    # 4. Optimize
    best = tuner.optimize_all(iterations=2)

    # Check best result
    assert best.mean_cost < 1e-6
    # Check if parameters were applied
    assert abs(holder.config.lambda_ - 0.5) < 1e-3
    assert abs(holder.state.noise_sigma[0, 0] - 1.0) < 1e-3
    assert holder.config.horizon == 15
    assert holder.state.U.shape[0] == 15


def test_horizon_resizing():
    """Test horizon parameter correctly resizes U."""
    config = mppi.MPPIConfig(
        dynamics_fn=lambda x, u: x + u,
        cost_fn=lambda x, u: 0.0,
        nx=1,
        nu=1,
        num_samples=10,
        horizon=10,
        lambda_=1.0,
        noise_sigma=jnp.eye(1),
        u_min=jnp.array([-1.0]),
        u_max=jnp.array([1.0]),
        u_init=jnp.array([0.5]),
        step_method="mppi",
    )
    config, state = mppi.create(config, seed=42)
    holder = autotune.ConfigStateHolder(config, state)

    param = autotune.HorizonParameter(holder, min_value=5, max_value=20)

    # Increase horizon
    param.apply_parameter_value(np.array([15.0]))
    assert holder.config.horizon == 15
    assert holder.state.U.shape == (15, 1)
    # Check extension with u_init
    np.testing.assert_allclose(holder.state.U[10:], 0.5)

    # Decrease horizon
    param.apply_parameter_value(np.array([8.0]))
    assert holder.config.horizon == 8
    assert holder.state.U.shape == (8, 1)
