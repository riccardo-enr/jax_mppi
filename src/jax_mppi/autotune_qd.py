"""Quality diversity optimization for JAX-MPPI autotuning.

This module provides CMA-ME (Covariance Matrix Adaptation MAP-Elites)
integration for finding diverse, high-performing parameter configurations.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from jax_mppi.autotune import (
    ConfigStateHolder,
    EvaluationResult,
    Optimizer,
    TunableParameter,
)


@dataclass
class QDParameter:
    """Wraps a TunableParameter with behavior descriptors."""

    param: TunableParameter
    behavior_fn: Callable[[float], float] = lambda x: x
    behavior_name: str = "val"


class CMAMEOpt(Optimizer):
    """CMA-ME (Covariance Matrix Adaptation MAP-Elites) optimizer.

    Combines CMA-ES for local search with MAP-Elites for maintaining
    a diversity of high-performing solutions.
    """

    def __init__(
        self,
        behavior_dim: int,
        grid_shape: Tuple[int, ...],
        min_bounds: List[float],
        max_bounds: List[float],
        population: int = 15,
        num_emitters: int = 5,
        sigma0: float = 0.1,
    ) -> None:
        """
        Args:
            behavior_dim: Number of behavior descriptors
            grid_shape: Shape of the behavior grid (e.g., (10, 10))
            min_bounds: Minimum bounds for behavior space
            max_bounds: Maximum bounds for behavior space
            population: Population size per emitter
            num_emitters: Number of parallel emitters
            sigma0: Initial step size
        """
        self.behavior_dim = behavior_dim
        self.grid_shape = grid_shape
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.population = population
        self.num_emitters = num_emitters
        self.sigma0 = sigma0

        self.archive = None
        self.emitters = []
        self.scheduler = None
        self.evaluate_fn: Optional[Callable[[np.ndarray], EvaluationResult]] = (
            None
        )

    def initialize(
        self,
        initial_params: np.ndarray,
        evaluate_fn: Callable[[np.ndarray], EvaluationResult],
    ) -> None:
        """Initialize the QD optimizer.

        Args:
            initial_params: Initial guess for parameters
            evaluate_fn: Evaluation function
        """
        try:
            from ribs.archives import GridArchive  # type: ignore  # noqa: F401
            from ribs.emitters import (
                EvolutionStrategyEmitter,  # type: ignore  # noqa: F401
            )
            from ribs.schedulers import Scheduler  # type: ignore  # noqa: F401
        except ImportError:
            raise ImportError(
                "CMA-ME optimizer requires 'ribs'. "
                "Install: pip install 'ribs[all]'"
            )

        self.evaluate_fn = evaluate_fn

        # Initialize archive
        self.archive = GridArchive(
            solution_dim=len(initial_params),
            dims=self.grid_shape,
            ranges=list(zip(self.min_bounds, self.max_bounds)),
        )

        # Initialize emitters
        self.emitters = [
            EvolutionStrategyEmitter(
                self.archive,
                x0=initial_params,
                sigma0=self.sigma0,
                ranker="2imp",
                batch_size=self.population,
            )
            for _ in range(self.num_emitters)
        ]

        # Initialize scheduler
        self.scheduler = Scheduler(self.archive, self.emitters)

    def step(self) -> Tuple[np.ndarray, float]:
        """Perform one optimization step.

        Returns:
            best_params: Best parameters found so far
            best_cost: Best cost found so far
        """
        if self.scheduler is None or self.evaluate_fn is None:
            raise RuntimeError("Optimizer not initialized. Call initialize()")

        # Ask for solutions
        solutions = self.scheduler.ask()

        # Evaluate solutions
        costs = []
        behaviors = []

        for sol in solutions:
            result = self.evaluate_fn(sol)
            costs.append(result.cost)
            # Assuming result.metadata contains behavior metrics
            if result.metadata and "behavior" in result.metadata:
                behaviors.append(result.metadata["behavior"])
            else:
                # Fallback behavior (should not happen in proper QD setup)
                behaviors.append(np.zeros(self.behavior_dim))

        # Convert to objective (fitness = -cost)
        objectives = -np.array(costs)
        behaviors_np = np.array(behaviors)

        # Tell results to scheduler
        self.scheduler.tell(objectives, behaviors_np)

        # Return best solution in archive
        if self.archive.best_elite is not None:
            return (
                self.archive.best_elite.solution,
                -self.archive.best_elite.objective,
            )

        # If no elite yet, return current best
        best_idx = np.argmax(objectives)
        return solutions[best_idx], costs[best_idx]

    def optimize_all(
        self, iterations: int = 100
    ) -> Tuple[np.ndarray, float, List[float]]:
        """Run full optimization loop."""
        history = []
        best_params = None
        best_cost = float("inf")

        for _ in range(iterations):
            params, cost = self.step()
            history.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_params = params

        return best_params, best_cost, history  # type: ignore


@dataclass
class AutotuneQD:
    """Quality Diversity Autotuning Manager.

    Wraps the standard Autotune class to support QD optimization,
    extracting behavior descriptors from evaluations.
    """

    holder: ConfigStateHolder
    params_to_tune: List[TunableParameter]
    optimizer: CMAMEOpt
    behavior_fn: Callable[[Any], List[float]]

    def evaluate_wrapper(self, params: np.ndarray) -> EvaluationResult:
        """Wrapper to extract behavior from evaluation."""
        # This needs to be hooked into the Autotune evaluation logic
        # For now, this is a placeholder structure
        raise NotImplementedError("QD Autotuning integration in progress")

    def get_archive_dataframe(self):
        """Return the archive as a pandas DataFrame."""
        if self.optimizer.archive is None:
            return None
        return self.optimizer.archive.as_pandas()

    def plot_archive(self, output_path: str = "qd_archive.png"):
        """Visualize the QD archive."""
        df = self.get_archive_dataframe()
        if df is None:
            print("No archive to plot.")
            return

        import matplotlib.pyplot as plt

        if self.optimizer.behavior_dim == 2:
            plt.figure(figsize=(10, 8))
            # Plot the grid
            plt.scatter(
                df["behavior_0"],
                df["behavior_1"],
                c=-df["objective"],  # Color by cost
                cmap="viridis",
                s=50,
            )
            plt.colorbar(label="Cost")
            plt.xlabel("Behavior 1")
            plt.ylabel("Behavior 2")
            plt.title("QD Archive (MAP-Elites)")
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path)
            plt.close()
        else:
            print("Plotting only supported for 2D behavior spaces.")

    def load_best_from_archive(self) -> None:
        """Load the single best parameter set from the archive."""
        if self.optimizer.archive is None:
            return

        elite = self.optimizer.archive.best_elite
        if elite is not None:
            print(
                f"Loading best elite with cost {-elite.objective:.4f}"
            )
            # Update holder with best params
            current_idx = 0
            for param in self.params_to_tune:
                count = param.get_count()
                values = elite.solution[current_idx : current_idx + count]
                param.set_values(values)
                current_idx += count

    def load_diverse_policies(self, num_policies: int = 5) -> List[np.ndarray]:
        """Select a diverse set of high-performing policies."""
        df = self.get_archive_dataframe()
        if df is None:
            return []

        # Sort by cost (objective is negative cost)
        df = df.sort_values("objective", ascending=False)

        # Pick top policies, potentially with distance filtering
        # For simplicity, just return top N
        top_n = df.head(num_policies)

        policies = []
        for _, row in top_n.iterrows():
            # Extract solution columns
            # Ribs stores solution as "solution_0", "solution_1", etc.
            sol_cols = [
                c for c in df.columns if c.startswith("solution_")
            ]
            # Ensure correct order
            sol_cols.sort(key=lambda x: int(x.split("_")[1]))
            policy = row[sol_cols].values.astype(np.float64)
            policies.append(policy)

            # Objective is negative cost, so negate to get cost
            cost = -row["objective"]
            # Behavior is stored in "behavior_0", etc.
            behavior = np.array([
                row[f"index_{i}"] for i in range(self.optimizer.behavior_dim)
            ])
            print(f"Policy: cost={cost:.2f}, behavior={behavior}")

        return policies
