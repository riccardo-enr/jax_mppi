"""Basic autotuning example for MPPI.

This example shows the minimal code needed to autotune MPPI parameters.

Usage:
    python examples/autotune_basic.py
"""

import jax.numpy as jnp

from jax_mppi import autotune, mppi


def main():
    """Minimal autotuning example."""

    # 1. Define your dynamics and cost functions
    def dynamics(state, action):
        """Simple linear dynamics: x' = x + u * dt"""
        return state + action * 0.1

    def running_cost(state, action):
        """Quadratic cost to origin."""
        return jnp.sum(state**2) + 0.01 * jnp.sum(action**2)

    def terminal_cost(state, action):
        """Terminal cost."""
        return jnp.sum(state**2) * 10.0

    # 2. Create MPPI controller
    config, state = mppi.create(
        nx=2,  # state dimension
        nu=1,  # action dimension
        horizon=15,
        num_samples=50,
        lambda_=5.0,  # We'll optimize this
        noise_sigma=jnp.eye(1) * 1.0,  # And this
    )

    # 3. Create config/state holder
    holder = autotune.ConfigStateHolder(config, state)

    # 4. Define evaluation function
    def evaluate():
        """Run MPPI and return performance."""
        total_cost = 0.0
        obs = jnp.array([1.0, -0.5])  # Initial state
        rollout_state = holder.state

        # Run for some timesteps
        for _ in range(20):
            action, rollout_state = mppi.command(
                holder.config,
                rollout_state,
                obs,
                dynamics,
                running_cost,
                terminal_cost,
            )
            obs = dynamics(obs, action)
            total_cost += running_cost(obs, action)

        return autotune.EvaluationResult(
            mean_cost=float(total_cost),
            rollouts=jnp.zeros((1, 1, 2)),
            params={},
            iteration=0,
        )

    # 5. Create autotuner
    tuner = autotune.Autotune(
        params_to_tune=[
            autotune.LambdaParameter(holder, min_value=0.1),
            autotune.NoiseSigmaParameter(holder, min_value=0.1),
        ],
        evaluate_fn=evaluate,
        optimizer=autotune.CMAESOpt(population=6, sigma=0.3),
    )

    # 6. Run optimization
    print("Starting autotuning...")
    best = tuner.optimize_all(iterations=10)

    # 7. Print results
    print(f"\nBest cost: {best.mean_cost:.3f}")
    print(f"Best lambda: {best.params['lambda'][0]:.3f}")
    print(f"Best noise_sigma: {best.params['noise_sigma'][0]:.3f}")


if __name__ == "__main__":
    main()
