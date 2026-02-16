"""Unit tests for autotune core functionality."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_mppi import autotune, mppi

# Check if optional autotuning dependencies are available
try:
    import cma  # noqa: F401

    HAS_CMA = True
except ImportError:
    HAS_CMA = False

requires_cma = pytest.mark.skipif(
    not HAS_CMA, reason="requires cma package (pip install cma)"
)


class TestParameterBasics:
    """Test basic parameter interface and properties."""

    def test_tunable_parameter_is_abstract(self):
        """TunableParameter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # Try to instantiate without implementing abstract methods
            class IncompleteParameter(autotune.TunableParameter):
                pass
            IncompleteParameter()

    def test_optimizer_is_abstract(self):
        """Optimizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # Try to instantiate without implementing abstract methods
            class IncompleteOptimizer(autotune.Optimizer):
                pass
            IncompleteOptimizer()

    def test_evaluation_result_creation(self):
        """EvaluationResult can be created with all fields."""
        result = autotune.EvaluationResult(
            mean_cost=1.5,
            rollouts=jnp.zeros((10, 20, 4)),
            params={"lambda": np.array([1.0])},
            iteration=5,
        )

        assert result.mean_cost == 1.5
        assert result.rollouts.shape == (10, 20, 4)
        assert "lambda" in result.params
        assert result.iteration == 5

    def test_config_state_holder(self):
        """ConfigStateHolder stores config and state."""
        config, state = mppi.create(
            nx=2, nu=1, horizon=10, noise_sigma=jnp.eye(1) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)

        assert holder.config is config
        assert holder.state is state


class TestParameterFlattening:
    """Test parameter extraction and flattening."""

    def test_lambda_parameter_extraction(self):
        """LambdaParameter extracts lambda from config."""
        config, state = mppi.create(
            nx=2, nu=1, horizon=10, lambda_=2.5, noise_sigma=jnp.eye(1) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)
        param = autotune.LambdaParameter(holder)

        value = param.get_current_parameter_value()
        assert value.shape == (1,)
        assert np.isclose(value[0], 2.5)

    def test_lambda_parameter_validation(self):
        """LambdaParameter enforces minimum value."""
        config, state = mppi.create(
            nx=2, nu=1, horizon=10, noise_sigma=jnp.eye(1) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)
        param = autotune.LambdaParameter(holder, min_value=0.5)

        # Value below minimum should be clamped
        validated = param.ensure_valid_value(np.array([0.1]))
        assert validated[0] >= 0.5

    def test_lambda_parameter_application(self):
        """LambdaParameter updates config.lambda_."""
        config, state = mppi.create(
            nx=2, nu=1, horizon=10, lambda_=1.0, noise_sigma=jnp.eye(1) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)
        param = autotune.LambdaParameter(holder)

        param.apply_parameter_value(np.array([3.5]))
        assert np.isclose(holder.config.lambda_, 3.5)

    def test_noise_sigma_parameter_extraction(self):
        """NoiseSigmaParameter extracts diagonal from noise_sigma."""
        config, state = mppi.create(
            nx=2, nu=2, horizon=10, noise_sigma=jnp.eye(2) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)
        param = autotune.NoiseSigmaParameter(holder)

        value = param.get_current_parameter_value()
        assert value.shape == (2,)
        assert np.allclose(value, [0.5, 0.5])

    def test_noise_sigma_parameter_application(self):
        """NoiseSigmaParameter updates state.noise_sigma."""
        config, state = mppi.create(
            nx=2, nu=2, horizon=10, noise_sigma=jnp.eye(2) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)
        param = autotune.NoiseSigmaParameter(holder)

        new_sigma = np.array([1.2, 0.8])
        param.apply_parameter_value(new_sigma)

        # Check diagonal values
        sigma_diag = np.diag(np.array(holder.state.noise_sigma))
        assert np.allclose(sigma_diag, new_sigma)

        # Check inverse
        sigma_inv_diag = np.diag(np.array(holder.state.noise_sigma_inv))
        assert np.allclose(sigma_inv_diag, 1.0 / new_sigma)

    def test_mu_parameter_extraction(self):
        """MuParameter extracts noise_mu from state."""
        config, state = mppi.create(
            nx=2,
            nu=2,
            horizon=10,
            noise_sigma=jnp.eye(2) * 0.5,
            noise_mu=jnp.array([0.1, -0.2]),
        )
        holder = autotune.ConfigStateHolder(config, state)
        param = autotune.MuParameter(holder)

        value = param.get_current_parameter_value()
        assert value.shape == (2,)
        assert np.allclose(value, [0.1, -0.2])

    def test_mu_parameter_application(self):
        """MuParameter updates state.noise_mu."""
        config, state = mppi.create(
            nx=2, nu=2, horizon=10, noise_sigma=jnp.eye(2) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)
        param = autotune.MuParameter(holder)

        new_mu = np.array([0.5, -0.3])
        param.apply_parameter_value(new_mu)

        assert np.allclose(np.array(holder.state.noise_mu), new_mu)

    def test_horizon_parameter_extraction(self):
        """HorizonParameter extracts horizon from config."""
        config, state = mppi.create(
            nx=2, nu=1, horizon=15, noise_sigma=jnp.eye(1) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)
        param = autotune.HorizonParameter(holder)

        value = param.get_current_parameter_value()
        assert value.shape == (1,)
        assert value[0] == 15.0

    def test_horizon_parameter_validation(self):
        """HorizonParameter clips and rounds horizon."""
        config, state = mppi.create(
            nx=2, nu=1, horizon=10, noise_sigma=jnp.eye(1) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)
        param = autotune.HorizonParameter(holder, min_value=5, max_value=20)

        # Test clipping
        assert param.ensure_valid_value(np.array([3.0]))[0] == 5.0
        assert param.ensure_valid_value(np.array([25.0]))[0] == 20.0

        # Test rounding
        assert param.ensure_valid_value(np.array([12.7]))[0] == 13.0

    def test_horizon_parameter_application(self):
        """HorizonParameter updates config.horizon and resizes U."""
        config, state = mppi.create(
            nx=2, nu=2, horizon=10, noise_sigma=jnp.eye(2) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)
        param = autotune.HorizonParameter(holder)

        # Increase horizon
        param.apply_parameter_value(np.array([15.0]))
        assert holder.config.horizon == 15
        assert holder.state.U.shape == (15, 2)

        # Decrease horizon
        param.apply_parameter_value(np.array([8.0]))
        assert holder.config.horizon == 8
        assert holder.state.U.shape == (8, 2)

    def test_flatten_multiple_parameters(self):
        """Flattening multiple parameters concatenates values."""
        config, state = mppi.create(
            nx=2, nu=2, horizon=10, lambda_=1.5, noise_sigma=jnp.eye(2) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)

        params = [
            autotune.LambdaParameter(holder),  # dim=1
            autotune.NoiseSigmaParameter(holder),  # dim=2
        ]

        flattened = autotune.flatten_params(params)
        assert flattened.shape == (3,)  # 1 + 2
        assert flattened[0] == 1.5  # lambda


@requires_cma
class TestCMAESOpt:
    """Test CMA-ES optimizer."""

    def test_cmaes_creation(self):
        """CMAESOpt can be created with default parameters."""
        optimizer = autotune.CMAESOpt(population=8, sigma=0.2)
        assert optimizer.population == 8
        assert optimizer.sigma == 0.2

    def test_cmaes_requires_setup(self):
        """CMAESOpt requires setup_optimization before optimize_step."""
        optimizer = autotune.CMAESOpt()

        with pytest.raises(RuntimeError, match="setup_optimization"):
            optimizer.optimize_step()

    def test_cmaes_simple_quadratic(self):
        """CMA-ES converges on simple quadratic function."""

        # Minimize f(x) = (x - 3)^2
        def evaluate(x):
            cost = float((x[0] - 3.0) ** 2)
            return autotune.EvaluationResult(
                mean_cost=cost,
                rollouts=jnp.zeros((1, 1, 1)),
                params={"x": x},
                iteration=0,
            )

        optimizer = autotune.CMAESOpt(population=10, sigma=1.0)
        optimizer.setup_optimization(np.array([0.0]), evaluate)

        # Run optimization
        best = optimizer.optimize_all(iterations=20)

        # Should converge near x=3
        assert abs(best.params["x"][0] - 3.0) < 0.5
        assert best.mean_cost < 0.5


@requires_cma
class TestAutotuneCore:
    """Test core Autotune functionality."""

    def test_autotune_with_single_param(self):
        """Autotune works with a single parameter."""
        config, state = mppi.create(
            nx=2, nu=1, horizon=10, noise_sigma=jnp.eye(1) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)

        def dummy_evaluate():
            return autotune.EvaluationResult(
                mean_cost=1.0,
                rollouts=jnp.zeros((1, 1, 1)),
                params={},
                iteration=0,
            )

        tuner = autotune.Autotune(
            params_to_tune=[autotune.LambdaParameter(holder)],
            evaluate_fn=dummy_evaluate,
        )
        assert len(tuner.params_to_tune) == 1

    def test_autotune_tracks_best_result(self):
        """Autotune tracks best result across iterations."""
        config, state = mppi.create(
            nx=2, nu=1, horizon=10, lambda_=5.0, noise_sigma=jnp.eye(1) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)

        # Evaluation returns cost = (lambda - 2)^2
        def evaluate():
            lam = holder.config.lambda_
            return autotune.EvaluationResult(
                mean_cost=float((lam - 2.0) ** 2),
                rollouts=jnp.zeros((1, 1, 1)),
                params={"lambda": np.array([lam])},
                iteration=0,
            )

        tuner = autotune.Autotune(
            params_to_tune=[autotune.LambdaParameter(holder, min_value=0.1)],
            evaluate_fn=evaluate,
            optimizer=autotune.CMAESOpt(population=6, sigma=0.5),
        )

        # Run optimization
        best = tuner.optimize_all(iterations=15)

        # Should find minimum near lambda=2
        assert abs(best.params["lambda"][0] - 2.0) < 0.5
        assert best.mean_cost < 0.5

    def test_get_best_result_before_optimization(self):
        """get_best_result raises error before any optimization."""

        # Create a simple mock optimizer that doesn't initialize CMA-ES
        class MockOptimizer(autotune.Optimizer):
            def setup_optimization(self, initial_params, evaluate_fn):
                pass  # Don't do anything

            def optimize_step(self):
                raise NotImplementedError()

        config, state = mppi.create(
            nx=2, nu=1, horizon=10, noise_sigma=jnp.eye(1) * 0.5
        )
        holder = autotune.ConfigStateHolder(config, state)

        def dummy_evaluate():
            return autotune.EvaluationResult(
                mean_cost=1.0,
                rollouts=jnp.zeros((1, 1, 1)),
                params={},
                iteration=0,
            )

        tuner = autotune.Autotune(
            params_to_tune=[autotune.LambdaParameter(holder)],
            evaluate_fn=dummy_evaluate,
            optimizer=MockOptimizer(),
        )

        # Should raise before optimization
        with pytest.raises(RuntimeError, match="No results"):
            tuner.get_best_result()
