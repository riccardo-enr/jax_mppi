"""Global search extensions for JAX-MPPI autotuning.

This module provides Ray Tune integration for distributed global hyperparameter search.
Requires optional dependencies: ray[tune], hyperopt, bayesian-optimization

Example:
    >>> from ray import tune
    >>> from jax_mppi import autotune_global as autog
    >>>
    >>> # Define parameters with search spaces
    >>> params = [
    ...     autog.GlobalLambdaParameter(
    ...         holder,
    ...         search_space=tune.loguniform(0.1, 10.0)
    ...     ),
    ...     autog.GlobalNoiseSigmaParameter(
    ...         holder,
    ...         search_space=tune.uniform(0.1, 2.0)
    ...     ),
    ... ]
    >>>
    >>> tuner = autog.AutotuneGlobal(
    ...     params_to_tune=params,
    ...     evaluate_fn=evaluate,
    ...     optimizer=autog.RayOptimizer(num_samples=100),
    ... )
    >>> best = tuner.optimize_all(iterations=100)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .autotune import (
    Autotune,
    ConfigStateHolder,
    EvaluationResult,
    HorizonParameter,
    LambdaParameter,
    MuParameter,
    NoiseSigmaParameter,
    Optimizer,
    TunableParameter,
)


class GlobalTunableParameter(TunableParameter):
    """Parameter with Ray Tune search space definition.

    Extends TunableParameter with search space information for global optimization.
    """

    def __init__(self, search_space: Any):
        """Initialize with Ray Tune search space.

        Args:
            search_space: Ray Tune search space (e.g., tune.loguniform(0.1, 10))
        """
        self.search_space = search_space

    def get_search_space_dict(self) -> Dict[str, Any]:
        """Return dictionary mapping parameter name to search space.

        Returns:
            Dictionary for Ray Tune configuration
        """
        return {self.name(): self.search_space}


@dataclass
class GlobalLambdaParameter(LambdaParameter, GlobalTunableParameter):
    """Lambda parameter with global search space."""

    search_space: Any = None

    def __init__(
        self,
        holder: ConfigStateHolder,
        search_space: Any,
        min_value: float = 0.0001,
    ):
        """Initialize global lambda parameter.

        Args:
            holder: Config/state holder
            search_space: Ray Tune search space
            min_value: Minimum value constraint
        """
        LambdaParameter.__init__(self, holder, min_value)
        self.search_space = search_space

    def get_search_space_dict(self) -> Dict[str, Any]:
        return {self.name(): self.search_space}


@dataclass
class GlobalNoiseSigmaParameter(NoiseSigmaParameter, GlobalTunableParameter):
    """Noise sigma parameter with global search space."""

    search_space: Any = None

    def __init__(
        self,
        holder: ConfigStateHolder,
        search_space: Any,
        min_value: float = 0.0001,
    ):
        """Initialize global noise sigma parameter.

        Args:
            holder: Config/state holder
            search_space: Ray Tune search space (applied per-dimension)
            min_value: Minimum value constraint
        """
        NoiseSigmaParameter.__init__(self, holder, min_value)
        self.search_space = search_space

    def get_search_space_dict(self) -> Dict[str, Any]:
        """Return search space dict with dimensionality.

        For multi-dimensional parameters, creates separate search spaces
        for each dimension.
        """
        nu = self.holder.config.nu
        if nu == 1:
            return {self.name(): self.search_space}
        else:
            # Create search space for each dimension
            return {f"{self.name()}_{i}": self.search_space for i in range(nu)}


@dataclass
class GlobalMuParameter(MuParameter, GlobalTunableParameter):
    """Noise mu parameter with global search space."""

    search_space: Any = None

    def __init__(self, holder: ConfigStateHolder, search_space: Any):
        """Initialize global noise mu parameter.

        Args:
            holder: Config/state holder
            search_space: Ray Tune search space (applied per-dimension)
        """
        MuParameter.__init__(self, holder)
        self.search_space = search_space

    def get_search_space_dict(self) -> Dict[str, Any]:
        nu = self.holder.config.nu
        if nu == 1:
            return {self.name(): self.search_space}
        else:
            return {f"{self.name()}_{i}": self.search_space for i in range(nu)}


@dataclass
class GlobalHorizonParameter(HorizonParameter, GlobalTunableParameter):
    """Horizon parameter with global search space."""

    search_space: Any = None

    def __init__(
        self,
        holder: ConfigStateHolder,
        search_space: Any,
        min_value: int = 5,
        max_value: int = 100,
    ):
        """Initialize global horizon parameter.

        Args:
            holder: Config/state holder
            search_space: Ray Tune search space (e.g., tune.randint(5, 20))
            min_value: Minimum value constraint
            max_value: Maximum value constraint
        """
        HorizonParameter.__init__(self, holder, min_value, max_value)
        self.search_space = search_space

    def get_search_space_dict(self) -> Dict[str, Any]:
        return {self.name(): self.search_space}


class RayOptimizer(Optimizer):
    """Ray Tune optimizer for distributed global search.

    Uses Ray Tune's hyperparameter optimization algorithms (HyperOpt, BayesOpt, etc.)
    for distributed search over large parameter spaces.

    Note: Only supports optimize_all(), not step-wise optimization.
    """

    def __init__(
        self,
        search_alg: str = "hyperopt",
        num_samples: int = 100,
        metric: str = "mean_cost",
        mode: str = "min",
    ):
        """Initialize Ray Tune optimizer.

        Args:
            search_alg: Search algorithm ("hyperopt", "bayesopt", or "random")
            num_samples: Number of parameter configurations to try
            metric: Metric to optimize (always "mean_cost")
            mode: Optimization mode (always "min")
        """
        try:
            import ray
            from ray import tune
        except ImportError:
            raise ImportError(
                "Ray Tune optimizer requires 'ray[tune]'. "
                "Install with: pip install 'ray[tune]' hyperopt bayesian-optimization"
            )

        self.search_alg = search_alg
        self.num_samples = num_samples
        self.metric = metric
        self.mode = mode
        self.ray = ray
        self.tune = tune
        self.evaluate_fn = None
        self.best_result = None

    def setup_optimization(
        self,
        initial_params: np.ndarray,
        evaluate_fn: Callable[[np.ndarray], EvaluationResult],
    ) -> None:
        """Setup Ray Tune optimization.

        Args:
            initial_params: Initial parameters (not used by Ray Tune)
            evaluate_fn: Evaluation function
        """
        self.evaluate_fn = evaluate_fn

    def optimize_step(self) -> EvaluationResult:
        """Not supported for Ray Tune (requires full run)."""
        raise NotImplementedError(
            "RayOptimizer only supports optimize_all(), not step-wise optimization"
        )

    def optimize_all(self, iterations: int) -> EvaluationResult:
        """Run full Ray Tune search.

        Args:
            iterations: Number of trials (overrides num_samples if provided)

        Returns:
            Best evaluation result found
        """
        if self.evaluate_fn is None:
            raise RuntimeError("Must call setup_optimization() first")

        # Use iterations if provided, otherwise use num_samples
        num_trials = iterations if iterations > 0 else self.num_samples

        # This will be set by AutotuneGlobal
        raise NotImplementedError(
            "RayOptimizer.optimize_all() must be called through AutotuneGlobal"
        )


class AutotuneGlobal(Autotune):
    """Extended Autotune with global search space support.

    Integrates Ray Tune for distributed hyperparameter search over
    large search spaces using advanced algorithms like HyperOpt and BayesOpt.

    Example:
        >>> from ray import tune
        >>> params = [
        ...     GlobalLambdaParameter(holder, search_space=tune.loguniform(0.1, 10)),
        ...     GlobalNoiseSigmaParameter(holder, search_space=tune.uniform(0.1, 2.0)),
        ... ]
        >>> tuner = AutotuneGlobal(
        ...     params_to_tune=params,
        ...     evaluate_fn=evaluate,
        ...     optimizer=RayOptimizer(num_samples=100),
        ... )
        >>> best = tuner.optimize_all(iterations=100)
    """

    def __init__(
        self,
        params_to_tune: list[GlobalTunableParameter],
        evaluate_fn: Callable[[], EvaluationResult],
        optimizer: Optional[RayOptimizer] = None,
        reload_state_fn: Optional[Callable] = None,
    ):
        """Initialize global autotuner.

        Args:
            params_to_tune: List of global tunable parameters
            evaluate_fn: Evaluation function
            optimizer: RayOptimizer instance
            reload_state_fn: Optional state reload function
        """
        if optimizer is None:
            optimizer = RayOptimizer()

        if not isinstance(optimizer, RayOptimizer):
            raise ValueError("AutotuneGlobal requires RayOptimizer")

        # Don't call parent __init__ since we handle setup differently
        self.params_to_tune = params_to_tune
        self.evaluate_fn = evaluate_fn
        self.reload_state_fn = reload_state_fn
        self.optimizer = optimizer
        self.best_result = None
        self.iteration_count = 0

    def define_search_space(self) -> dict:
        """Define Ray Tune search space from parameters.

        Returns:
            Dictionary mapping parameter names to search spaces
        """
        search_space = {}
        for param in self.params_to_tune:
            if isinstance(param, GlobalTunableParameter):
                search_space.update(param.get_search_space_dict())
        return search_space

    def optimize_all(self, iterations: int) -> EvaluationResult:
        """Run Ray Tune global search.

        Args:
            iterations: Number of trials to run

        Returns:
            Best evaluation result found
        """
        from ray import tune
        from ray.tune.search import ConcurrencyLimiter

        # Define search space
        search_space = self.define_search_space()

        # Create trainable function for Ray Tune
        def trainable(config: dict):
            """Ray Tune trainable function."""
            # Unflatten config to parameter values
            param_values = {}

            for param in self.params_to_tune:
                if param.dim() == 1:
                    # Single-valued parameter
                    param_values[param.name()] = np.array([config[param.name()]])
                else:
                    # Multi-valued parameter
                    values = []
                    for i in range(param.dim()):
                        key = f"{param.name()}_{i}"
                        values.append(config[key])
                    param_values[param.name()] = np.array(values)

            # Apply parameters
            for param in self.params_to_tune:
                validated = param.ensure_valid_value(param_values[param.name()])
                param.apply_parameter_value(validated)

            # Reload state if needed
            if self.reload_state_fn is not None:
                self.reload_state_fn()

            # Evaluate
            result = self.evaluate_fn()

            # Report to Ray Tune
            tune.report(mean_cost=result.mean_cost)

        # Setup search algorithm
        if self.optimizer.search_alg == "hyperopt":
            from ray.tune.search.hyperopt import HyperOptSearch

            search_alg = HyperOptSearch(metric="mean_cost", mode="min")
        elif self.optimizer.search_alg == "bayesopt":
            from ray.tune.search.bayesopt import BayesOptSearch

            search_alg = BayesOptSearch(metric="mean_cost", mode="min")
        else:
            search_alg = None  # Random search

        # Limit concurrency to avoid overwhelming the system
        if search_alg is not None:
            search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)

        # Run tuning
        analysis = tune.run(
            trainable,
            config=search_space,
            num_samples=iterations,
            search_alg=search_alg,
            verbose=1,
        )

        # Get best result
        best_config = analysis.get_best_config(metric="mean_cost", mode="min")

        # Apply best config and evaluate one more time to get full result
        for param in self.params_to_tune:
            if param.dim() == 1:
                value = np.array([best_config[param.name()]])
            else:
                value = np.array([best_config[f"{param.name()}_{i}"] for i in range(param.dim())])
            validated = param.ensure_valid_value(value)
            param.apply_parameter_value(validated)

        final_result = self.evaluate_fn()
        self.best_result = final_result._replace(params=best_config, iteration=iterations)

        return self.best_result

    def get_best_result(self) -> EvaluationResult:
        """Get best result found.

        Returns:
            Best evaluation result

        Raises:
            RuntimeError: If no results available yet
        """
        if self.best_result is None:
            raise RuntimeError("No results available yet")
        return self.best_result


def save_search_progress_plot(
    iteration_costs: list[float],
    output_path: str | Path = "docs/media/autotune_global_progress.png",
    title: str = "Global Hyperparameter Search Progress",
    **kwargs,
) -> None:
    """Save global search progress plot to docs/media.

    Args:
        iteration_costs: List of best costs at each iteration
        output_path: Path to save the plot
        title: Plot title
        **kwargs: Additional arguments passed to plt.savefig
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=kwargs.get("figsize", (10, 6)))
    plt.plot(iteration_costs, marker="o", linewidth=2, markersize=6, label="Best Cost")
    plt.fill_between(range(len(iteration_costs)), iteration_costs, alpha=0.3, label="Running Best")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cost", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    dpi = kwargs.get("dpi", 150)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved search progress plot to: {output_path}")
