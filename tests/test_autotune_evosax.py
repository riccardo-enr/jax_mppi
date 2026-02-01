"""Unit tests for evosax-based autotuning functionality."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_mppi import autotune, autotune_evosax, mppi


class TestEvoSaxOptimizer:
    """Test EvoSaxOptimizer base class."""

    def test_evosax_creation(self):
        """EvoSaxOptimizer can be created with default parameters."""
        pytest.importorskip("evosax")
        optimizer = autotune_evosax.EvoSaxOptimizer(
            strategy="CMA_ES", population=10, sigma_init=0.1
        )
        assert optimizer.strategy == "CMA_ES"
        assert optimizer.population == 10
        assert optimizer.sigma_init == 0.1

    def test_evosax_import_error(self):
        """EvoSaxOptimizer raises ImportError if evosax not available."""
        try:
            import evosax  # noqa: F401

            pytest.skip("evosax is installed")
        except ImportError:
            with pytest.raises(ImportError, match="evosax"):
                autotune_evosax.EvoSaxOptimizer()

    def test_evosax_requires_setup(self):
        """EvoSaxOptimizer requires setup before optimize_step."""
        pytest.importorskip("evosax")
        optimizer = autotune_evosax.EvoSaxOptimizer()

        with pytest.raises(RuntimeError, match="setup_optimization"):
            optimizer.optimize_step()

    def test_evosax_invalid_strategy(self):
        """EvoSaxOptimizer raises error for invalid strategy."""
        pytest.importorskip("evosax")
        optimizer = autotune_evosax.EvoSaxOptimizer(strategy="InvalidStrategy")

        def dummy_eval(x):
            return autotune.EvaluationResult(
                mean_cost=1.0,
                rollouts=jnp.zeros((1, 1, 1)),
                params={},
                iteration=0,
            )

        with pytest.raises(ValueError, match="Unknown strategy"):
            optimizer.setup_optimization(np.array([0.0]), dummy_eval)

    def test_evosax_simple_quadratic(self):
        """EvoSaxOptimizer with CMA-ES converges on simple quadratic."""
        pytest.importorskip("evosax")

        # Minimize f(x) = (x - 3)^2
        def evaluate(x):
            cost = float((x[0] - 3.0) ** 2)
            return autotune.EvaluationResult(
                mean_cost=cost,
                rollouts=jnp.zeros((1, 1, 1)),
                params={"x": x},
                iteration=0,
            )

        optimizer = autotune_evosax.EvoSaxOptimizer(
            strategy="CMA_ES", population=10, sigma_init=1.0
        )
        optimizer.setup_optimization(np.array([0.0]), evaluate)

        # Run optimization
        best = optimizer.optimize_all(iterations=20)

        # Should converge near x=3
        assert abs(best.params["x"][0] - 3.0) < 1.0
        assert best.mean_cost < 1.0

    def test_evosax_maximize_option(self):
        """EvoSaxOptimizer respects maximize flag."""
        pytest.importorskip("evosax")

        # Maximize f(x) = -||x - 3||^2
        def evaluate(x):
            cost = float(-np.sum((x - 3.0) ** 2))
            return autotune.EvaluationResult(
                mean_cost=cost,
                rollouts=jnp.zeros((1, 1, 3)),
                params={"x": x},
                iteration=0,
            )

        optimizer = autotune_evosax.EvoSaxOptimizer(
            strategy="CMA_ES",
            population=10,
            sigma_init=1.0,
            maximize=True,
        )
        optimizer.setup_optimization(np.array([0.0, 0.0, 0.0]), evaluate)

        best = optimizer.optimize_all(iterations=20)

        # Should maximize near x=[3, 3, 3]
        assert best.mean_cost < 0.0  # Cost is negative
        assert abs(best.params["x"][0] - 3.0) < 1.5

    def test_evosax_high_dimensional(self):
        """EvoSaxOptimizer works with high-dimensional problems."""
        pytest.importorskip("evosax")

        dim = 10

        def evaluate(x):
            # Minimize sum of squared differences from 2
            cost = float(np.sum((x - 2.0) ** 2))
            return autotune.EvaluationResult(
                mean_cost=cost,
                rollouts=jnp.zeros((1, 1, dim)),
                params={"x": x},
                iteration=0,
            )

        optimizer = autotune_evosax.EvoSaxOptimizer(
            strategy="CMA_ES", population=20, sigma_init=0.5
        )
        optimizer.setup_optimization(np.zeros(dim), evaluate)

        best = optimizer.optimize_all(iterations=30)

        # Should converge toward all 2.0
        assert best.mean_cost < 5.0
        assert np.allclose(best.params["x"], 2.0, atol=1.0)


class TestCMAESOpt:
    """Test CMA-ES optimizer via evosax."""

    def test_cmaes_creation(self):
        """CMAESOpt can be created with default parameters."""
        pytest.importorskip("evosax")
        optimizer = autotune_evosax.CMAESOpt(population=10, sigma=0.1)
        assert optimizer.strategy == "CMA_ES"
        assert optimizer.population == 10
        assert optimizer.sigma_init == 0.1

    def test_cmaes_custom_params(self):
        """CMAESOpt passes kwargs to base class."""
        pytest.importorskip("evosax")
        optimizer = autotune_evosax.CMAESOpt(
            population=15, sigma=0.2, maximize=True
        )
        assert optimizer.population == 15
        assert optimizer.sigma_init == 0.2
        assert optimizer.maximize is True

    def test_cmaes_convergence(self):
        """CMA-ES converges on simple function."""
        pytest.importorskip("evosax")

        def evaluate(x):
            cost = float(np.sum(x**2))
            return autotune.EvaluationResult(
                mean_cost=cost,
                rollouts=jnp.zeros((1, 1, len(x))),
                params={"x": x},
                iteration=0,
            )

        optimizer = autotune_evosax.CMAESOpt(population=15, sigma=1.0)
        optimizer.setup_optimization(np.ones(3), evaluate)

        best = optimizer.optimize_all(iterations=25)

        # Should converge toward origin
        assert best.mean_cost < 0.5
        assert np.allclose(best.params["x"], 0.0, atol=0.5)


class TestSepCMAESOpt:
    """Test Separable CMA-ES optimizer via evosax."""

    def test_sepcmaes_creation(self):
        """SepCMAESOpt can be created."""
        pytest.importorskip("evosax")
        optimizer = autotune_evosax.SepCMAESOpt(population=10, sigma=0.1)
        assert optimizer.strategy == "Sep_CMA_ES"
        assert optimizer.population == 10

    def test_sepcmaes_convergence(self):
        """SepCMAES converges on separable function."""
        pytest.importorskip("evosax")

        def evaluate(x):
            # Separable: f(x) = sum(x_i^2)
            cost = float(np.sum(x**2))
            return autotune.EvaluationResult(
                mean_cost=cost,
                rollouts=jnp.zeros((1, 1, len(x))),
                params={"x": x},
                iteration=0,
            )

        optimizer = autotune_evosax.SepCMAESOpt(population=10, sigma=1.0)
        optimizer.setup_optimization(np.ones(5), evaluate)

        best = optimizer.optimize_all(iterations=20)

        # Should converge toward origin
        assert best.mean_cost < 0.5


class TestOpenESOpt:
    """Test OpenES optimizer via evosax."""

    def test_openes_creation(self):
        """OpenESOpt can be created."""
        pytest.importorskip("evosax")
        optimizer = autotune_evosax.OpenESOpt(population=50, sigma=0.1)
        assert optimizer.strategy == "Open_ES"
        assert optimizer.population == 50

    def test_openes_convergence(self):
        """OpenES converges on simple function."""
        pytest.importorskip("evosax")

        def evaluate(x):
            cost = float((x[0] - 1.5) ** 2)
            return autotune.EvaluationResult(
                mean_cost=cost,
                rollouts=jnp.zeros((1, 1, 1)),
                params={"x": x},
                iteration=0,
            )

        optimizer = autotune_evosax.OpenESOpt(population=30, sigma=1.0)
        optimizer.setup_optimization(np.array([0.0]), evaluate)

        best = optimizer.optimize_all(iterations=25)

        # Should converge toward x=1.5
        assert abs(best.params["x"][0] - 1.5) < 1.0


class TestSNESOpt:
    """Test SNES optimizer via evosax."""

    def test_snes_creation(self):
        """SNESOpt can be created."""
        pytest.importorskip("evosax")
        optimizer = autotune_evosax.SNESOpt(population=20, sigma=0.1)
        assert optimizer.strategy == "SNES"
        assert optimizer.population == 20

    def test_snes_convergence(self):
        """SNES converges on simple function."""
        pytest.importorskip("evosax")

        def evaluate(x):
            cost = float(np.sum((x - 1.0) ** 2))
            return autotune.EvaluationResult(
                mean_cost=cost,
                rollouts=jnp.zeros((1, 1, len(x))),
                params={"x": x},
                iteration=0,
            )

        optimizer = autotune_evosax.SNESOpt(population=15, sigma=1.0)
        optimizer.setup_optimization(np.zeros(2), evaluate)

        best = optimizer.optimize_all(iterations=20)

        # Should converge toward [1, 1]
        assert best.mean_cost < 1.0


class TestxNESOpt:
    """Test xNES optimizer via evosax."""

    def test_xnes_creation(self):
        """xNESOpt can be created."""
        pytest.importorskip("evosax")
        optimizer = autotune_evosax.xNESOpt(population=20, sigma=0.1)
        assert optimizer.strategy == "xNES"
        assert optimizer.population == 20

    def test_xnes_convergence(self):
        """xNES converges on simple function."""
        pytest.importorskip("evosax")

        def evaluate(x):
            cost = float(np.sum((x - 0.5) ** 2))
            return autotune.EvaluationResult(
                mean_cost=cost,
                rollouts=jnp.zeros((1, 1, len(x))),
                params={"x": x},
                iteration=0,
            )

        optimizer = autotune_evosax.xNESOpt(population=15, sigma=1.0)
        optimizer.setup_optimization(np.zeros(2), evaluate)

        best = optimizer.optimize_all(iterations=20)

        # Should converge toward [0.5, 0.5]
        assert best.mean_cost < 1.0


class TestEvoSaxIntegration:
    """Integration tests with Autotune framework."""

    def test_evosax_with_autotune(self):
        """EvoSaxOptimizer works with Autotune framework."""
        pytest.importorskip("evosax")
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
            optimizer=autotune_evosax.CMAESOpt(population=10, sigma=0.5),
        )

        # Run optimization
        best = tuner.optimize_all(iterations=15)

        # Should find minimum near lambda=2
        assert abs(best.params["lambda"][0] - 2.0) < 1.0
        assert best.mean_cost < 1.0

    def test_evosax_with_multiple_parameters(self):
        """EvoSaxOptimizer works with multiple parameters."""
        pytest.importorskip("evosax")
        config, state = mppi.create(
            nx=2,
            nu=2,
            horizon=10,
            lambda_=1.0,
            noise_sigma=jnp.eye(2) * 0.5,
        )
        holder = autotune.ConfigStateHolder(config, state)

        # Evaluation function
        def evaluate():
            lam = holder.config.lambda_
            sigma = np.diag(np.array(holder.state.noise_sigma))
            # Cost is distance from target (lambda=2, sigma=[0.3, 0.3])
            cost = float((lam - 2.0) ** 2 + np.sum((sigma - 0.3) ** 2))
            return autotune.EvaluationResult(
                mean_cost=cost,
                rollouts=jnp.zeros((1, 1, 1)),
                params={
                    "lambda": np.array([lam]),
                    "sigma": sigma,
                },
                iteration=0,
            )

        tuner = autotune.Autotune(
            params_to_tune=[
                autotune.LambdaParameter(holder, min_value=0.1),
                autotune.NoiseSigmaParameter(holder, min_value=0.1),
            ],
            evaluate_fn=evaluate,
            optimizer=autotune_evosax.CMAESOpt(population=15, sigma=0.3),
        )

        # Run optimization
        best = tuner.optimize_all(iterations=20)

        # Should improve from initial configuration
        assert best.mean_cost < 2.0

    def test_evosax_optimizer_inheritance(self):
        """EvoSaxOptimizer properly implements Optimizer ABC."""
        pytest.importorskip("evosax")
        optimizer = autotune_evosax.CMAESOpt()

        # Should have required methods
        assert hasattr(optimizer, "setup_optimization")
        assert hasattr(optimizer, "optimize_step")
        assert hasattr(optimizer, "optimize_all")

        # Should be callable
        assert callable(optimizer.setup_optimization)
        assert callable(optimizer.optimize_step)
        assert callable(optimizer.optimize_all)
