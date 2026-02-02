"""Quality diversity optimization for JAX-MPPI autotuning.

This module provides CMA-ME (Covariance Matrix Adaptation MAP-Elites) integration
for finding diverse, high-performing parameter configurations.

Requires optional dependency: ribs[all]

Example:
    >>> from jax_mppi import autotune, autotune_qd
    >>>
    >>> tuner = autotune.Autotune(
    ...     params_to_tune=[...],
    ...     evaluate_fn=evaluate,
    ...     optimizer=autotune_qd.CMAMEOpt(population=20, bins=10),
    ... )
    >>> best = tuner.optimize_all(iterations=50)
    >>>
    >>> # Get diverse set of solutions
    >>> diverse_params = tuner.optimizer.get_diverse_top_parameters(n=10)
"""

from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np

from .autotune import EvaluationResult, Optimizer


class CMAMEOpt(Optimizer):
    """CMA-ME optimizer for quality diversity optimization.

    CMA-ME (Covariance Matrix Adaptation MAP-Elites) maintains an archive
    of diverse, high-performing solutions. This is useful when you want
    to find multiple good parameter configurations that work well in
    different scenarios.

    The archive is organized by "behavior characteristics" - properties
    of the solution that define diversity. For MPPI, these could be:
    - Mean cost (performance)
    - Control variance (how aggressive the controller is)
    - Horizon used
    etc.

    Attributes:
        population: Population size per iteration
        sigma: Initial step size
        bins: Number of bins per behavior dimension
        archive: Solution archive (ribs Archive)
    """

    def __init__(
        self,
        population: int = 20,
        sigma: float = 0.1,
        bins: int = 10,
        behavior_dim: int = 1,
    ):
        """Initialize CMA-ME optimizer.

        Args:
            population: Population size per iteration
            sigma: Initial step size
            bins: Number of bins per behavior dimension
            behavior_dim: Dimensionality of behavior space
        """
        try:
            from ribs.archives import GridArchive  # type: ignore
            from ribs.emitters import EvolutionStrategyEmitter  # type: ignore
            from ribs.schedulers import Scheduler  # type: ignore
        except ImportError:
            raise ImportError(
                "CMA-ME optimizer requires 'ribs'. Install with: pip install 'ribs[all]'"
            )

        self.population = population
        self.sigma = sigma
        self.bins = bins
        self.behavior_dim = behavior_dim

        self.ribs = None  # Will be initialized in setup
        self.archive = None
        self.emitters = None
        self.scheduler = None
        self.evaluate_fn = None
        self.solution_dim = None

    def setup_optimization(
        self,
        initial_params: np.ndarray,
        evaluate_fn: Callable[[np.ndarray], EvaluationResult],
    ) -> None:
        """Initialize CMA-ME archive and emitters.

        Args:
            initial_params: Initial parameter values
            evaluate_fn: Evaluation function
        """
        from ribs.archives import GridArchive  # type: ignore
        from ribs.emitters import EvolutionStrategyEmitter  # type: ignore
        from ribs.schedulers import Scheduler  # type: ignore

        self.evaluate_fn = evaluate_fn
        self.solution_dim = len(initial_params)

        # Create archive with behavior dimensions
        # For MPPI, we use a simple 1D behavior: normalized cost
        # Range [0, 1] with bins
        bounds = [(0.0, 1.0) for _ in range(self.behavior_dim)]

        self.archive = GridArchive(
            solution_dim=self.solution_dim,
            dims=[self.bins] * self.behavior_dim,
            ranges=bounds,
        )

        # Create emitters (ES-based)
        self.emitters = [
            EvolutionStrategyEmitter(
                archive=self.archive,
                x0=initial_params,
                sigma0=self.sigma,
                batch_size=self.population,
            )
        ]

        # Create scheduler
        self.scheduler = Scheduler(self.archive, self.emitters)

    def _compute_behavior(self, result: EvaluationResult) -> np.ndarray:
        """Compute behavior characteristics from evaluation result.

        For simplicity, we use normalized cost as the behavior.
        In practice, you might want to use multiple characteristics.

        Args:
            result: Evaluation result

        Returns:
            Behavior vector (shape: (behavior_dim,))
        """
        # Simple behavior: map cost to [0, 1]
        # Use a heuristic normalization
        normalized_cost = 1.0 / (1.0 + result.mean_cost)
        return np.array([normalized_cost])

    def optimize_step(self) -> EvaluationResult:
        """Execute one CMA-ME iteration.

        Returns:
            Best result from this iteration
        """
        if self.scheduler is None:
            raise RuntimeError("Must call setup_optimization() first")

        # Ask for solutions
        solutions = self.scheduler.ask()

        # Evaluate all solutions
        results = []
        objectives = []
        behaviors = []

        for solution in solutions:
            result = self.evaluate_fn(solution)
            results.append(result)

            # Objective is negative cost (we want to maximize quality)
            objectives.append(-result.mean_cost)

            # Behavior characteristics
            behavior = self._compute_behavior(result)
            behaviors.append(behavior)

        # Tell scheduler about results
        self.scheduler.tell(
            objectives=np.array(objectives),
            behaviors=np.array(behaviors),
        )

        # Return best result from this iteration
        best_idx = np.argmax(objectives)
        return results[best_idx]

    def get_diverse_top_parameters(
        self, n: int = 10
    ) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        """Get diverse set of top-performing parameters from archive.

        Args:
            n: Number of diverse solutions to retrieve

        Returns:
            List of (parameters, cost, behavior) tuples
        """
        if self.archive is None:
            raise RuntimeError("Must run optimization first")

        # Get all elite solutions from archive
        df = self.archive.as_pandas(include_solutions=True)

        if len(df) == 0:
            return []

        # Sort by objective (negative cost, so higher is better)
        df = df.sort_values("objective", ascending=False)

        # Take top n diverse solutions
        results = []
        for idx in range(min(n, len(df))):
            row = df.iloc[idx]
            params = row["solution"]
            # Objective is negative cost, so negate to get cost
            cost = -row["objective"]
            # Behavior is stored in columns like "behavior_0", "behavior_1", etc.
            behavior = np.array([
                row[f"index_{i}"] for i in range(self.behavior_dim)
            ])
            results.append((params, cost, behavior))

        return results

    def get_archive_stats(self) -> dict:
        """Get statistics about the archive.

        Returns:
            Dictionary with archive statistics
        """
        if self.archive is None:
            raise RuntimeError("Must run optimization first")

        stats = self.archive.stats
        return {
            "num_elites": stats.num_elites,
            "coverage": stats.coverage,
            "qd_score": stats.qd_score,
            "best_objective": stats.obj_max,
        }


def save_qd_heatmap(
    costs: list[float],
    output_path: str | Path = "docs/media/autotune_qd_heatmap.png",
    title: str = "Quality Diversity Archive Heatmap",
    **kwargs: Any,
) -> None:
    """Save quality diversity optimization progress to docs/media.

    Args:
        costs: List of best costs at each iteration
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

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))

    # Plot convergence
    ax.plot(
        costs,
        marker="o",
        linewidth=2,
        markersize=6,
        label="Best Cost",
        color="blue",
    )
    ax.fill_between(range(len(costs)), costs, alpha=0.2, color="blue")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cost", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    dpi = kwargs.get("dpi", 150)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved QD progress plot to: {output_path}")
