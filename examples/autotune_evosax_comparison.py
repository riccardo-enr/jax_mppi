"""Compare evosax vs cma library for MPPI autotuning.

This example demonstrates the performance differences between:
1. CMA-ES from the `cma` library (Python-based)
2. CMA-ES from evosax (JAX-native, GPU-accelerated)
3. Other evosax strategies (Sep-CMA-ES, OpenES)

The comparison evaluates convergence speed, wall-clock time, and final performance
on a simple pendulum MPPI tuning task.
"""

import time
from typing import Callable

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jax_mppi as jmppi
from jax_mppi import autotune


def create_pendulum_system():
    """Create a simple pendulum system for testing."""

    # Pendulum dynamics
    def dynamics(x, u):
        theta, theta_dot = x[0], x[1]
        g = 9.81
        L = 1.0
        m = 1.0
        b = 0.1

        theta_ddot = (
            -g / L * jnp.sin(theta)
            - b * theta_dot / (m * L**2)
            + u[0] / (m * L**2)
        )
        return jnp.array([theta_dot, theta_ddot])

    # Cost function: reach upright position (theta=0)
    def cost(x, u):
        theta, theta_dot = x[0], x[1]
        # Quadratic cost on angle and angular velocity
        return theta**2 + 0.1 * theta_dot**2 + 0.01 * u[0] ** 2

    return dynamics, cost


def setup_mppi(lambda_val: float = 1.0, sigma: float = 0.5):
    """Setup MPPI controller with given hyperparameters."""
    config, state = jmppi.mppi.create(
        nx=2,
        nu=1,
        horizon=20,
        num_samples=100,
        lambda_=lambda_val,
        noise_sigma=jnp.eye(1) * sigma,
    )

    return config, state


def create_evaluation_function(
    initial_lambda: float = 1.0,
    initial_sigma: float = 0.5,
):
    """Create evaluation function for autotuning."""
    config, state = setup_mppi(initial_lambda, initial_sigma)
    holder = autotune.ConfigStateHolder(config, state)

    # Setup parameters to tune
    params = [
        autotune.LambdaParameter(holder, min_value=0.01),
        autotune.NoiseSigmaParameter(holder, min_value=0.01),
    ]

    # Initial state: pendulum hanging down
    x0 = jnp.array([jnp.pi, 0.0])

    # Get dynamics and cost functions
    dynamics, cost = create_pendulum_system()

    def evaluate_fn():
        """Evaluate MPPI controller with current parameters."""
        # Run MPPI for a few steps
        x = x0
        total_cost = 0.0

        for _ in range(10):  # 10 steps of control
            u, state_new = jmppi.mppi.command(
                holder.config, holder.state, x, dynamics, cost
            )
            holder.state = state_new

            # Simulate one step
            x_new = x + 0.1 * dynamics(x, u)
            step_cost = cost(x, u)

            total_cost += step_cost
            x = x_new

        # Return average cost
        mean_cost = float(total_cost / 10.0)

        return autotune.EvaluationResult(
            mean_cost=mean_cost,
            rollouts=jnp.zeros((1, 1, 2)),
            params={
                "lambda": holder.config.lambda_,
                "noise_sigma": np.diag(np.array(holder.state.noise_sigma)),
            },
            iteration=0,
        )

    return evaluate_fn, params, holder


def run_optimization(
    optimizer_name: str,
    optimizer,
    evaluate_fn: Callable,
    params,
    iterations: int = 20,
):
    """Run optimization and track convergence."""
    print(f"\n{'=' * 60}")
    print(f"Running {optimizer_name}")
    print(f"{'=' * 60}")

    # Setup autotune
    tuner = autotune.Autotune(
        params_to_tune=params,
        evaluate_fn=evaluate_fn,
        optimizer=optimizer,
    )

    # Track convergence
    costs = []
    times = []

    start_time = time.time()

    for i in range(iterations):
        iter_start = time.time()
        result = tuner.optimize_step()
        iter_time = time.time() - iter_start

        costs.append(result.mean_cost)
        times.append(time.time() - start_time)

        if i % 5 == 0 or i == iterations - 1:
            print(
                f"  Iteration {i + 1:2d}: cost = {result.mean_cost:8.4f} "
                f"(time: {iter_time:6.3f}s)"
            )

    total_time = time.time() - start_time
    best_result = tuner.get_best_result()
    best_lambda = float(np.asarray(best_result.params["lambda"]).item())
    best_sigma = best_result.params.get("noise_sigma", None)

    print("\nFinal Results:")
    print(f"  Best cost: {best_result.mean_cost:.6f}")
    print(f"  Lambda: {best_lambda:.4f}")
    print(f"  Sigma: {best_sigma}")
    print(f"  Total time: {total_time:.2f}s")

    return {
        "costs": costs,
        "times": times,
        "total_time": total_time,
        "best_cost": best_result.mean_cost,
        "best_params": best_result.params,
    }


def plot_results(results_dict):
    """Plot comparison of different optimizers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot convergence curves
    for name, results in results_dict.items():
        ax1.plot(results["costs"], marker="o", label=name, linewidth=2)

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Cost", fontsize=12)
    ax1.set_title("Convergence Comparison", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot time comparison
    names = list(results_dict.keys())
    times = [results_dict[name]["total_time"] for name in names]
    best_costs = [results_dict[name]["best_cost"] for name in names]

    x_pos = np.arange(len(names))
    bars = ax2.bar(
        x_pos, times, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    )

    # Add cost labels on bars
    for i, (bar, cost) in enumerate(zip(bars, best_costs)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"cost: {cost:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.set_xlabel("Optimizer", fontsize=12)
    ax2.set_ylabel("Total Time (seconds)", fontsize=12)
    ax2.set_title("Execution Time Comparison", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        "docs/media/autotune_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("\n✓ Plot saved to docs/media/autotune_comparison.png")
    plt.show()


def main():
    """Run comparison of different optimizers."""
    print("=" * 60)
    print("MPPI Autotuning: evosax vs cma Library Comparison")
    print("=" * 60)

    # Create evaluation function (shared across all optimizers)
    evaluate_fn, params, holder = create_evaluation_function()

    iterations = 20
    population = 50
    sigma = 0.3

    results = {}

    # 1. CMA-ES from cma library
    try:
        from jax_mppi.autotune import CMAESOpt as CMALibOpt

        # Reset holder for each optimizer
        config, state = setup_mppi()
        holder.config = config
        holder.state = state

        results["cma library"] = run_optimization(
            "CMA-ES (cma library)",
            CMALibOpt(population=population, sigma=sigma),
            evaluate_fn,
            params,
            iterations=iterations,
        )
    except ImportError:
        print("\n⚠ Warning: cma library not installed, skipping...")

    # 2. CMA-ES from evosax
    try:
        from jax_mppi import autotune_evosax

        # Reset holder
        config, state = setup_mppi()
        holder.config = config
        holder.state = state

        results["evosax CMA-ES"] = run_optimization(
            "CMA-ES (evosax)",
            autotune_evosax.CMAESOpt(population=population, sigma=sigma),
            evaluate_fn,
            params,
            iterations=iterations,
        )
    except ImportError:
        print("\n⚠ Warning: evosax not installed, skipping...")

    # 3. Sep-CMA-ES from evosax
    try:
        from jax_mppi import autotune_evosax

        config, state = setup_mppi()
        holder.config = config
        holder.state = state

        results["evosax Sep-CMA-ES"] = run_optimization(
            "Sep-CMA-ES (evosax)",
            autotune_evosax.SepCMAESOpt(population=population, sigma=sigma),
            evaluate_fn,
            params,
            iterations=iterations,
        )
    except ImportError:
        pass

    # 4. OpenES from evosax
    try:
        from jax_mppi import autotune_evosax

        config, state = setup_mppi()
        holder.config = config
        holder.state = state

        results["evosax OpenES"] = run_optimization(
            "OpenES (evosax)",
            autotune_evosax.OpenESOpt(population=30, sigma=sigma),
            evaluate_fn,
            params,
            iterations=iterations,
        )
    except ImportError:
        pass

    # Plot results
    if len(results) > 1:
        plot_results(results)
    else:
        print(
            "\n⚠ Not enough optimizers to compare. Install both cma and evosax."
        )


if __name__ == "__main__":
    main()
