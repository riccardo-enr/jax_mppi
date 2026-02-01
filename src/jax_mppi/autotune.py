"""Autotuning framework for JAX-MPPI.

This module provides automatic hyperparameter optimization for MPPI controllers
using multiple optimization backends. Supports tuning of:
- Lambda (temperature parameter)
- Noise sigma (exploration covariance)
- Noise mu (exploration mean)
- Horizon (planning horizon)

Compatible with MPPI, SMPPI, and KMPPI variants.

Optimizers:
- CMAESOpt (from `cma` library) - Classic CMA-ES
- CMAESOpt, SepCMAESOpt, OpenESOpt (from `evosax`) - JAX-native, GPU-accelerated

Example with cma:
    >>> import jax_mppi as jmppi
    >>> config, state = jmppi.mppi.create(...)
    >>>
    >>> holder = jmppi.autotune.ConfigStateHolder(config, state)
    >>> def evaluate():
    ...     # Run MPPI rollout, return cost
    ...     return jmppi.autotune.EvaluationResult(...)
    >>>
    >>> tuner = jmppi.autotune.Autotune(
    ...     params_to_tune=[
    ...         jmppi.autotune.LambdaParameter(holder),
    ...         jmppi.autotune.NoiseSigmaParameter(holder),
    ...     ],
    ...     evaluate_fn=evaluate,
    ...     optimizer=jmppi.autotune.CMAESOpt(population=10),
    ... )
    >>> best = tuner.optimize_all(iterations=30)

Example with evosax (JAX-native, GPU-accelerated):
    >>> from jax_mppi import autotune_evosax
    >>> tuner = jmppi.autotune.Autotune(
    ...     params_to_tune=[...],
    ...     evaluate_fn=evaluate,
    ...     optimizer=autotune_evosax.CMAESOpt(population=10),  # JAX-native
    ... )
    >>> best = tuner.optimize_all(iterations=30)
"""

import abc
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


class EvaluationResult(NamedTuple):
    """Result from evaluating a parameter configuration.

    Attributes:
        mean_cost: Average cost across rollouts (lower is better)
        rollouts: Trajectory rollouts, shape (N, H+1, nx)
        params: Parameter values used for this evaluation
        iteration: Iteration number in optimization
    """

    mean_cost: float
    rollouts: jax.Array
    params: dict
    iteration: int


class ConfigStateHolder:
    """Mutable holder for MPPI config and state.

    This allows parameters to access and update config/state through a shared reference.
    """

    def __init__(self, config: Any, state: Any):
        """Initialize holder.

        Args:
            config: MPPI configuration object
            state: MPPI state object
        """
        self.config = config
        self.state = state


class TunableParameter(abc.ABC):
    """Abstract base class for tunable MPPI parameters.

    Concrete implementations must define how to extract, validate,
    and apply parameter values to MPPI config/state.
    """

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """Unique identifier for this parameter type."""
        ...

    @abc.abstractmethod
    def dim(self) -> int:
        """Dimensionality when flattened to 1D array."""
        ...

    @abc.abstractmethod
    def get_current_parameter_value(self) -> np.ndarray:
        """Extract current parameter value from MPPI config/state.

        Returns:
            Parameter value as numpy array, shape (dim(),)
        """
        ...

    @abc.abstractmethod
    def ensure_valid_value(self, value: np.ndarray) -> np.ndarray:
        """Validate and constrain parameter value.

        Args:
            value: Raw parameter value, shape (dim(),)

        Returns:
            Validated and constrained value, shape (dim(),)
        """
        ...

    @abc.abstractmethod
    def apply_parameter_value(self, value: np.ndarray) -> None:
        """Update MPPI config/state with new parameter value.

        Args:
            value: Validated parameter value, shape (dim(),)

        Side effects:
            Updates holder.config and/or holder.state with new values
        """
        ...


class Optimizer(abc.ABC):
    """Abstract base class for parameter optimizers."""

    @abc.abstractmethod
    def setup_optimization(
        self,
        initial_params: np.ndarray,
        evaluate_fn: Callable[[np.ndarray], EvaluationResult],
    ) -> None:
        """Initialize optimizer with starting parameters and evaluation function.

        Args:
            initial_params: Initial parameter values, shape (D,)
            evaluate_fn: Function that evaluates parameter vector and returns cost
        """
        ...

    @abc.abstractmethod
    def optimize_step(self) -> EvaluationResult:
        """Execute one optimization iteration.

        Returns:
            Best evaluation result from this step
        """
        ...

    def optimize_all(self, iterations: int) -> EvaluationResult:
        """Run full optimization loop.

        Args:
            iterations: Number of optimization iterations

        Returns:
            Best evaluation result found
        """
        best_result = None
        for i in range(iterations):
            result = self.optimize_step()
            if best_result is None or result.mean_cost < best_result.mean_cost:
                best_result = result
        return best_result  # type: ignore


class LambdaParameter(TunableParameter):
    """Tunes temperature parameter (config.lambda_).

    Lower lambda_ values lead to more aggressive exploitation of low-cost trajectories.
    Higher values provide more exploration.
    """

    def __init__(self, holder: ConfigStateHolder, min_value: float = 0.0001):
        """Initialize lambda parameter.

        Args:
            holder: Config/state holder
            min_value: Minimum allowed lambda value
        """
        self.holder = holder
        self.min_value = min_value

    @staticmethod
    def name() -> str:
        return "lambda"

    def dim(self) -> int:
        return 1

    def get_current_parameter_value(self) -> np.ndarray:
        """Extract lambda from config."""
        return np.array([self.holder.config.lambda_])

    def ensure_valid_value(self, value: np.ndarray) -> np.ndarray:
        """Ensure lambda >= min_value."""
        return np.maximum(value, self.min_value)

    def apply_parameter_value(self, value: np.ndarray) -> None:
        """Update config.lambda_ with new value."""
        new_lambda = float(value[0])
        self.holder.config = replace(self.holder.config, lambda_=new_lambda)


class NoiseSigmaParameter(TunableParameter):
    """Tunes noise covariance diagonal (state.noise_sigma).

    Controls exploration in action space. Higher sigma values increase exploration.
    """

    def __init__(self, holder: ConfigStateHolder, min_value: float = 0.0001):
        """Initialize noise sigma parameter.

        Args:
            holder: Config/state holder
            min_value: Minimum allowed sigma value
        """
        self.holder = holder
        self.min_value = min_value

    @staticmethod
    def name() -> str:
        return "noise_sigma"

    def dim(self) -> int:
        """Returns nu (number of control dimensions)."""
        return self.holder.config.nu

    def get_current_parameter_value(self) -> np.ndarray:
        """Extract diagonal of noise covariance matrix."""
        # noise_sigma is (nu, nu), extract diagonal
        sigma_diag = np.diag(np.array(self.holder.state.noise_sigma))
        return sigma_diag

    def ensure_valid_value(self, value: np.ndarray) -> np.ndarray:
        """Ensure all sigma values >= min_value."""
        return np.maximum(value, self.min_value)

    def apply_parameter_value(self, value: np.ndarray) -> None:
        """Update state.noise_sigma with diagonal matrix.

        Also updates noise_sigma_inv (inverse covariance).
        """
        # Create diagonal covariance matrix
        noise_sigma = jnp.diag(jnp.array(value))
        noise_sigma_inv = jnp.diag(1.0 / jnp.array(value))

        self.holder.state = replace(
            self.holder.state,
            noise_sigma=noise_sigma,
            noise_sigma_inv=noise_sigma_inv,
        )


class MuParameter(TunableParameter):
    """Tunes noise mean (state.noise_mu).

    Shifts the exploration distribution. Can be used to bias exploration
    toward certain actions.
    """

    def __init__(self, holder: ConfigStateHolder):
        """Initialize noise mu parameter.

        Args:
            holder: Config/state holder
        """
        self.holder = holder

    @staticmethod
    def name() -> str:
        return "noise_mu"

    def dim(self) -> int:
        """Returns nu (number of control dimensions)."""
        return self.holder.config.nu

    def get_current_parameter_value(self) -> np.ndarray:
        """Extract noise mean from state."""
        return np.array(self.holder.state.noise_mu)

    def ensure_valid_value(self, value: np.ndarray) -> np.ndarray:
        """No constraints on mu (can be any value)."""
        return value

    def apply_parameter_value(self, value: np.ndarray) -> None:
        """Update state.noise_mu with new mean."""
        noise_mu = jnp.array(value)
        self.holder.state = replace(self.holder.state, noise_mu=noise_mu)


class HorizonParameter(TunableParameter):
    """Tunes planning horizon (config.horizon).

    Longer horizons allow for more foresight but increase computation.
    Changing horizon requires resizing the control trajectory U.
    """

    def __init__(
        self,
        holder: ConfigStateHolder,
        min_value: int = 5,
        max_value: int = 100,
    ):
        """Initialize horizon parameter.

        Args:
            holder: Config/state holder
            min_value: Minimum allowed horizon
            max_value: Maximum allowed horizon
        """
        self.holder = holder
        self.min_value = min_value
        self.max_value = max_value

    @staticmethod
    def name() -> str:
        return "horizon"

    def dim(self) -> int:
        return 1

    def get_current_parameter_value(self) -> np.ndarray:
        """Extract horizon from config."""
        return np.array([float(self.holder.config.horizon)])

    def ensure_valid_value(self, value: np.ndarray) -> np.ndarray:
        """Clip horizon to [min_value, max_value] and round to int."""
        value = np.clip(value, self.min_value, self.max_value)
        value = np.round(value)
        return value

    def apply_parameter_value(self, value: np.ndarray) -> None:
        """Update config.horizon and resize U trajectory.

        Handles horizon changes for all MPPI variants:
        - MPPI: Resize U
        - SMPPI: Resize U and action_sequence
        - KMPPI: Rebuild Tk/Hs grids, reinterpolate U from theta
        """
        new_horizon = int(value[0])
        old_horizon = self.holder.config.horizon

        if new_horizon == old_horizon:
            return  # No change needed

        # Update config
        self.holder.config = replace(self.holder.config, horizon=new_horizon)

        # Resize U trajectory
        old_U = self.holder.state.U

        if new_horizon > old_horizon:
            # Extend with u_init
            extension = jnp.tile(
                self.holder.state.u_init, (new_horizon - old_horizon, 1)
            )
            new_U = jnp.concatenate([old_U, extension], axis=0)
        else:
            # Truncate
            new_U = old_U[:new_horizon]

        # Update state with new U
        self.holder.state = replace(self.holder.state, U=new_U)

        # Handle variant-specific state updates
        if hasattr(self.holder.state, "action_sequence"):
            # SMPPI: also resize action_sequence
            old_seq = self.holder.state.action_sequence
            if new_horizon > old_horizon:
                extension = jnp.tile(
                    self.holder.state.u_init, (new_horizon - old_horizon, 1)
                )
                new_seq = jnp.concatenate([old_seq, extension], axis=0)
            else:
                new_seq = old_seq[:new_horizon]
            self.holder.state = replace(
                self.holder.state, action_sequence=new_seq
            )

        if hasattr(self.holder.state, "theta"):
            # KMPPI: rebuild time grids and reinterpolate
            # This is more complex - for now, keep theta unchanged and rebuild Hs
            # A full implementation would reinterpolate U from theta
            num_support_pts = self.holder.config.num_support_pts
            new_Tk = jnp.linspace(0, new_horizon - 1, num_support_pts)
            new_Hs = jnp.arange(new_horizon, dtype=jnp.float32)
            self.holder.state = replace(
                self.holder.state,
                Tk=new_Tk,
                Hs=new_Hs,
            )


def flatten_params(params: list[TunableParameter]) -> np.ndarray:
    """Flatten list of parameters to 1D array.

    Args:
        params: List of tunable parameters

    Returns:
        Concatenated parameter values, shape (total_dim,)
    """
    if not params:
        return np.array([])

    values = []
    for param in params:
        value = param.get_current_parameter_value()
        values.append(value.flatten())
    return np.concatenate(values)


def unflatten_params(
    x: np.ndarray,
    params: list[TunableParameter],
    apply: bool = True,
) -> dict[str, np.ndarray]:
    """Unflatten 1D array to parameter dict and optionally apply.

    Args:
        x: Flattened parameter vector, shape (total_dim,)
        params: List of tunable parameters (defines structure)
        apply: If True, apply values to parameters

    Returns:
        Dictionary mapping parameter names to values
    """
    param_dict = {}
    offset = 0

    for param in params:
        dim = param.dim()
        value = x[offset : offset + dim]
        value = param.ensure_valid_value(value)

        if apply:
            param.apply_parameter_value(value)

        param_dict[param.name()] = value
        offset += dim

    return param_dict


class CMAESOpt(Optimizer):
    """CMA-ES optimizer using the cma library.

    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a
    gradient-free optimization algorithm well-suited for hyperparameter tuning.

    Attributes:
        population: Population size (number of samples per iteration)
        sigma: Initial step size (exploration width)
    """

    def __init__(self, population: int = 10, sigma: float = 0.1):
        """Initialize CMA-ES optimizer.

        Args:
            population: Population size (popsize in cma)
            sigma: Initial standard deviation for sampling
        """
        try:
            import cma
        except ImportError:
            raise ImportError(
                "CMA-ES optimizer requires the 'cma' package. Install with: pip install cma"
            )

        self.population = population
        self.sigma = sigma
        self.cma = cma
        self.es = None
        self.evaluate_fn = None

    def setup_optimization(
        self,
        initial_params: np.ndarray,
        evaluate_fn: Callable[[np.ndarray], EvaluationResult],
    ) -> None:
        """Initialize CMA-ES with starting parameters.

        Args:
            initial_params: Initial parameter values
            evaluate_fn: Evaluation function
        """
        self.evaluate_fn = evaluate_fn

        # Initialize CMA-ES
        opts = {
            "popsize": self.population,
            "verbose": -9,  # Suppress output
        }

        self.es = self.cma.CMAEvolutionStrategy(
            initial_params.tolist(),
            self.sigma,
            opts,
        )

    def optimize_step(self) -> EvaluationResult:
        """Execute one CMA-ES iteration (ask-tell loop).

        Returns:
            Best result from this iteration
        """
        if self.es is None:
            raise RuntimeError("Must call setup_optimization() first")

        # Ask: sample population
        solutions = self.es.ask()

        # Evaluate all solutions
        results = []
        for x in solutions:
            result = self.evaluate_fn(np.array(x))  # type: ignore
            results.append(result)

        # Tell: update CMA-ES with costs
        costs = [r.mean_cost for r in results]
        self.es.tell(solutions, costs)

        # Return best result
        best_idx = np.argmin(costs)
        return results[best_idx]


class Autotune:
    """Main autotuning orchestrator.

    Manages parameter optimization using a specified optimizer.
    Handles flattening/unflattening of parameters and result tracking.

    Example:
        >>> holder = ConfigStateHolder(config, state)
        >>> tuner = Autotune(
        ...     params_to_tune=[LambdaParameter(holder), NoiseSigmaParameter(holder)],
        ...     evaluate_fn=my_evaluate_fn,
        ...     optimizer=CMAESOpt(population=10),
        ... )
        >>> best = tuner.optimize_all(iterations=30)
    """

    def __init__(
        self,
        params_to_tune: list[TunableParameter],
        evaluate_fn: Callable[[], EvaluationResult],
        optimizer: Optional[Optimizer] = None,
        reload_state_fn: Optional[Callable] = None,
    ):
        """Initialize autotuner.

        Args:
            params_to_tune: List of parameters to optimize
            evaluate_fn: Function that runs MPPI and returns EvaluationResult
            optimizer: Optimizer instance (defaults to CMAESOpt)
            reload_state_fn: Optional function to reload state (for multiprocessing)
        """
        self.params_to_tune = params_to_tune
        self.evaluate_fn = evaluate_fn
        self.reload_state_fn = reload_state_fn
        self.optimizer = optimizer if optimizer is not None else CMAESOpt()

        self.best_result: Optional[EvaluationResult] = None
        self.iteration_count = 0

        # Setup optimizer with initial parameters
        initial_params = self.flatten_params()

        def _wrapped_evaluate(x: np.ndarray) -> EvaluationResult:
            # Unflatten and apply parameters
            param_dict = self.unflatten_params(x, apply=True)

            # Reload state if needed (for multiprocessing)
            if self.reload_state_fn is not None:
                self.reload_state_fn()

            # Evaluate with current parameters
            result = self.evaluate_fn()

            # Track iteration
            result = result._replace(
                params=param_dict,
                iteration=self.iteration_count,
            )
            self.iteration_count += 1

            # Update best result
            if (
                self.best_result is None
                or result.mean_cost < self.best_result.mean_cost
            ):
                self.best_result = result

            return result

        self.optimizer.setup_optimization(initial_params, _wrapped_evaluate)

    def flatten_params(self) -> np.ndarray:
        """Flatten all parameters to 1D array."""
        return flatten_params(self.params_to_tune)

    def unflatten_params(
        self, x: np.ndarray, apply: bool = True
    ) -> dict[str, np.ndarray]:
        """Unflatten parameter vector and optionally apply."""
        return unflatten_params(x, self.params_to_tune, apply=apply)

    def apply_parameters(self, param_values: dict[str, np.ndarray]) -> None:
        """Apply parameter dictionary to MPPI config/state.

        Args:
            param_values: Dict mapping parameter names to values
        """
        for param in self.params_to_tune:
            if param.name() in param_values:
                value = param_values[param.name()]
                param.apply_parameter_value(value)

    def optimize_step(self) -> EvaluationResult:
        """Execute one optimization iteration.

        Returns:
            Best result from this step
        """
        return self.optimizer.optimize_step()

    def optimize_all(self, iterations: int) -> EvaluationResult:
        """Run full optimization loop.

        Args:
            iterations: Number of optimization iterations

        Returns:
            Best result found across all iterations
        """
        for _ in range(iterations):
            self.optimize_step()

        return self.get_best_result()

    def get_best_result(self) -> EvaluationResult:
        """Get best result found so far.

        Returns:
            Best evaluation result

        Raises:
            RuntimeError: If no results have been evaluated yet
        """
        if not isinstance(self.best_result, EvaluationResult):
            raise RuntimeError("No results available yet")
        return self.best_result


def save_convergence_plot(
    costs: list[float],
    initial_cost: float,
    output_path: str | Path = "docs/media/autotune_convergence.png",
    title: str = "Autotuning Convergence",
    **kwargs,
) -> None:
    """Save convergence plot to docs/media.

    Args:
        costs: List of costs at each iteration
        initial_cost: Initial cost before optimization
        output_path: Path to save the plot (default: docs/media/autotune_convergence.png)
        title: Plot title
        **kwargs: Additional arguments passed to plt.savefig (dpi, figsize, etc.)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=kwargs.get("figsize", (10, 6)))
    plt.plot(costs, marker="o", linewidth=2, markersize=6, label="Current")
    plt.axhline(
        y=initial_cost, color="r", linestyle="--", linewidth=2, label="Initial"
    )
    if costs:
        plt.axhline(
            y=min(costs), color="g", linestyle="--", linewidth=2, label="Best"
        )
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cost", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    dpi = kwargs.get("dpi", 150)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved convergence plot to: {output_path}")
