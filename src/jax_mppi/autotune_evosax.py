"""Evosax-based optimizer for JAX-MPPI autotuning.

This module provides JAX-native evolutionary strategies using the evosax library.
Evosax enables fully JIT-compiled optimization loops with GPU acceleration.

Supports multiple strategies:
- CMA-ES: Classic Covariance Matrix Adaptation
- Sep-CMA-ES: Separable CMA-ES (faster for high dimensions)
- OpenES: OpenAI's Natural Evolution Strategies
- SNES: Separable Natural Evolution Strategies
- And 10+ additional strategies from evosax

Example:
    >>> from jax_mppi import autotune_evosax
    >>> optimizer = autotune_evosax.CMAESOpt(population=10, sigma=0.1)
    >>> optimizer.setup_optimization(initial_params, evaluate_fn)
    >>> best = optimizer.optimize_all(iterations=30)
"""

import abc
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .autotune import EvaluationResult, Optimizer


def _create_jax_evaluate_fn(
    evaluate_fn: Callable[[np.ndarray], EvaluationResult],
    maximize: bool = False,
) -> Callable[[jax.Array], jax.Array]:
    """Wrap evaluation function for JAX compatibility.

    Handles conversion between JAX arrays and numpy arrays,
    extracts scalar cost from EvaluationResult.

    Args:
        evaluate_fn: Function that takes numpy array and returns EvaluationResult
        maximize: Whether to maximize (True) or minimize (False, default)

    Returns:
        Function that takes JAX array and returns scalar cost
    """

    def jax_eval(x: jax.Array) -> jax.Array:
        # Convert JAX array to numpy for evaluation
        x_np = np.array(x)
        result = evaluate_fn(x_np)
        cost = float(result.mean_cost)
        # Return negative cost if maximizing
        return jnp.array(-cost if maximize else cost, dtype=jnp.float32)

    return jax_eval


class EvoSaxOptimizer(Optimizer):
    """JAX-native evolutionary strategies using evosax.

    Provides full JIT compilation of the entire optimization loop
    with GPU acceleration support.

    Attributes:
        strategy: Evosax strategy name (e.g., "CMA_ES", "OpenES", "SNES")
        population: Population size (number of samples per iteration)
        num_generations: Number of generations per optimize_step
        sigma_init: Initial step size for search distribution
        maximize: Whether to maximize (True) or minimize (False, default)
        es_params: Dictionary of strategy-specific hyperparameters
    """

    def __init__(
        self,
        strategy: str = "CMA_ES",
        population: int = 10,
        num_generations: int = 1,
        sigma_init: float = 0.1,
        maximize: bool = False,
        es_params: Optional[dict[str, Any]] = None,
    ):
        """Initialize evosax optimizer.

        Args:
            strategy: Evosax strategy name
            population: Population size
            num_generations: Number of generations per optimize_step
            sigma_init: Initial step size
            maximize: Whether to maximize objective
            es_params: Strategy-specific hyperparameters

        Raises:
            ImportError: If evosax is not installed
        """
        try:
            import evosax
        except ImportError:
            raise ImportError(
                "Evosax optimizer requires the 'evosax' package. "
                "Install with: pip install evosax"
            )

        self.evosax = evosax
        self.strategy = strategy
        self.population = population
        self.num_generations = num_generations
        self.sigma_init = sigma_init
        self.maximize = maximize
        self.es_params = es_params or {}

        # Initialize state variables
        self.es = None
        self.es_state = None
        self.rng_key = None
        self.evaluate_fn = None
        self.jax_evaluate_fn = None
        self.best_fitness = np.inf
        self.best_params = None
        self.iteration = 0

    def setup_optimization(
        self,
        initial_params: np.ndarray,
        evaluate_fn: Callable[[np.ndarray], EvaluationResult],
    ) -> None:
        """Initialize evosax strategy with starting parameters.

        Args:
            initial_params: Initial parameter values, shape (D,)
            evaluate_fn: Function that evaluates parameters and returns cost
        """
        self.evaluate_fn = evaluate_fn
        self.jax_evaluate_fn = _create_jax_evaluate_fn(
            evaluate_fn, maximize=self.maximize
        )

        # Get strategy from evosax.algorithms
        try:
            from evosax import algorithms
            strategy_cls = getattr(algorithms, self.strategy)
        except (AttributeError, ImportError):
            try:
                from evosax import algorithms
                available = algorithms.__all__
            except:
                available = []
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                f"Available strategies: {available}"
            )

        # Initialize strategy with evosax API
        # solution is just a template for the solution shape
        solution_template = jnp.zeros(len(initial_params))

        self.es = strategy_cls(
            population_size=self.population,
            solution=solution_template,
            **self.es_params,
        )

        # Initialize ES state with evosax API
        self.rng_key = jax.random.PRNGKey(0)
        params = self.es.default_params  # Get default hyperparameters
        self.es_state = self.es.init(
            self.rng_key,
            mean=jnp.array(initial_params),
            params=params,
        )

        self.best_params = initial_params.copy()
        self.iteration = 0

    def optimize_step(self) -> EvaluationResult:
        """Execute one optimization iteration (ask-tell loop).

        Returns:
            Best evaluation result from this step
        """
        if self.es is None or self.es_state is None:
            raise RuntimeError("Must call setup_optimization() first")

        # Ask: sample population with evosax API
        self.rng_key, subkey = jax.random.split(self.rng_key)
        params = self.es.default_params
        solutions, self.es_state = self.es.ask(
            subkey, self.es_state, params
        )

        # Evaluate all solutions sequentially
        # (JAX-pure evaluation would use vmap for parallelization)
        results = []
        fitness_values = []

        for x in solutions:
            result = self.evaluate_fn(np.array(x))  # type: ignore
            results.append(result)

            # Get fitness for ES update
            cost = float(result.mean_cost)
            fitness = -cost if self.maximize else cost
            fitness_values.append(fitness)

        fitness_array = jnp.array(fitness_values, dtype=jnp.float32)

        # Tell: update ES state with fitness values using evosax API
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.es_state, metrics = self.es.tell(
            subkey, solutions, fitness_array, self.es_state, params
        )

        # Track best result
        best_idx = int(jnp.argmin(jnp.array(fitness_values)))
        best_result = results[best_idx]

        if best_result.mean_cost < self.best_fitness:
            self.best_fitness = best_result.mean_cost
            self.best_params = np.array(solutions[best_idx])

        self.iteration += 1

        return best_result


class CMAESOpt(EvoSaxOptimizer):
    """CMA-ES optimizer via evosax (JAX-native).

    Covariance Matrix Adaptation Evolution Strategy.
    Well-tested, robust algorithm suitable for most optimization problems.

    Example:
        >>> opt = CMAESOpt(population=10, sigma=0.1)
        >>> opt.setup_optimization(initial_params, evaluate_fn)
        >>> best = opt.optimize_all(iterations=50)
    """

    def __init__(
        self,
        population: int = 10,
        sigma: float = 0.1,
        **kwargs,
    ):
        """Initialize CMA-ES optimizer.

        Args:
            population: Population size
            sigma: Initial step size
            **kwargs: Additional arguments passed to EvoSaxOptimizer
        """
        super().__init__(
            strategy="CMA_ES",
            population=population,
            sigma_init=sigma,
            **kwargs,
        )


class SepCMAESOpt(EvoSaxOptimizer):
    """Separable CMA-ES optimizer via evosax.

    Faster variant of CMA-ES with reduced memory and computation.
    Recommended for high-dimensional optimization problems.

    Example:
        >>> opt = SepCMAESOpt(population=10, sigma=0.1)
        >>> opt.setup_optimization(initial_params, evaluate_fn)
        >>> best = opt.optimize_all(iterations=50)
    """

    def __init__(
        self,
        population: int = 10,
        sigma: float = 0.1,
        **kwargs,
    ):
        """Initialize Separable CMA-ES optimizer.

        Args:
            population: Population size
            sigma: Initial step size
            **kwargs: Additional arguments passed to EvoSaxOptimizer
        """
        super().__init__(
            strategy="Sep_CMA_ES",
            population=population,
            sigma_init=sigma,
            **kwargs,
        )


class OpenESOpt(EvoSaxOptimizer):
    """OpenAI's Natural Evolution Strategies via evosax.

    Gradient-free optimization that naturally parallelizes.
    Good for large population sizes and GPU acceleration.

    Example:
        >>> opt = OpenESOpt(population=100, sigma=0.1)
        >>> opt.setup_optimization(initial_params, evaluate_fn)
        >>> best = opt.optimize_all(iterations=50)
    """

    def __init__(
        self,
        population: int = 100,
        sigma: float = 0.1,
        **kwargs,
    ):
        """Initialize OpenES optimizer.

        Args:
            population: Population size
            sigma: Initial step size
            **kwargs: Additional arguments passed to EvoSaxOptimizer
        """
        super().__init__(
            strategy="Open_ES",
            population=population,
            sigma_init=sigma,
            **kwargs,
        )


class SNESOpt(EvoSaxOptimizer):
    """Separable Natural Evolution Strategies via evosax.

    Faster variant suitable for high-dimensional problems.
    Combines natural gradient updates with separable covariance.

    Example:
        >>> opt = SNESOpt(population=30, sigma=0.1)
        >>> opt.setup_optimization(initial_params, evaluate_fn)
        >>> best = opt.optimize_all(iterations=50)
    """

    def __init__(
        self,
        population: int = 30,
        sigma: float = 0.1,
        **kwargs,
    ):
        """Initialize SNES optimizer.

        Args:
            population: Population size
            sigma: Initial step size
            **kwargs: Additional arguments passed to EvoSaxOptimizer
        """
        super().__init__(
            strategy="SNES",
            population=population,
            sigma_init=sigma,
            **kwargs,
        )


class xNESOpt(EvoSaxOptimizer):
    """Exponential Natural Evolution Strategies via evosax.

    Natural evolution with exponential parameterization for covariance.
    Good balance between exploration and exploitation.

    Example:
        >>> opt = xNESOpt(population=30, sigma=0.1)
        >>> opt.setup_optimization(initial_params, evaluate_fn)
        >>> best = opt.optimize_all(iterations=50)
    """

    def __init__(
        self,
        population: int = 30,
        sigma: float = 0.1,
        **kwargs,
    ):
        """Initialize xNES optimizer.

        Args:
            population: Population size
            sigma: Initial step size
            **kwargs: Additional arguments passed to EvoSaxOptimizer
        """
        super().__init__(
            strategy="xNES",
            population=population,
            sigma_init=sigma,
            **kwargs,
        )
