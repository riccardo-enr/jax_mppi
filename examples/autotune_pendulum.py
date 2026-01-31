"""Autotune MPPI parameters for inverted pendulum control.

This example demonstrates automatic hyperparameter tuning using CMA-ES
to optimize MPPI performance on the inverted pendulum task.

The autotuner optimizes:
- lambda: Temperature parameter (controls exploration vs exploitation)
- noise_sigma: Exploration noise covariance

Usage:
    python examples/autotune_pendulum.py
"""

import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax_mppi import autotune, mppi


def pendulum_dynamics(state, action, dt=0.05):
    """Pendulum dynamics: [theta, theta_dot] -> [theta', theta_dot']

    Args:
        state: [theta, theta_dot] where theta=0 is upright
        action: [torque]
        dt: timestep

    Returns:
        next_state: [theta', theta_dot']
    """
    g = 10.0  # gravity
    m = 1.0  # mass
    l = 1.0  # length
    b = 0.1  # damping

    theta, theta_dot = state[0], state[1]
    u = action[0]

    # Dynamics: theta_ddot = (u - mgl*sin(theta) - b*theta_dot) / (ml^2)
    theta_ddot = (u - m * g * l * jnp.sin(theta) - b * theta_dot) / (m * l**2)

    # Euler integration
    theta_new = theta + theta_dot * dt
    theta_dot_new = theta_dot + theta_ddot * dt

    # Normalize theta to [-pi, pi]
    theta_new = ((theta_new + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    return jnp.array([theta_new, theta_dot_new])


def pendulum_cost(state, action):
    """Cost function for inverted pendulum.

    Args:
        state: [theta, theta_dot]
        action: [torque]

    Returns:
        cost: scalar cost
    """
    theta, theta_dot = state[0], state[1]
    u = action[0]

    # Cost for being away from upright (theta=0)
    angle_cost = theta**2

    # Cost for angular velocity
    velocity_cost = 0.1 * theta_dot**2

    # Control effort cost
    control_cost = 0.01 * u**2

    return angle_cost + velocity_cost + control_cost


def terminal_cost(state, action):
    """Terminal cost for pendulum.

    Args:
        state: [theta, theta_dot]
        action: [torque]

    Returns:
        cost: scalar terminal cost
    """
    return pendulum_cost(state, action) * 10.0


def evaluate_mppi_performance(config, state, num_episodes=5, episode_length=100):
    """Evaluate MPPI performance over multiple episodes.

    Args:
        config: MPPI configuration
        state: MPPI state
        num_episodes: Number of evaluation episodes
        episode_length: Steps per episode

    Returns:
        mean_cost: Average cost over all episodes
    """
    total_cost = 0.0

    for ep in range(num_episodes):
        # Random initial state (pendulum starts at random angle)
        key = jax.random.PRNGKey(ep)
        theta_init = jax.random.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
        obs = jnp.array([theta_init, 0.0])

        episode_cost = 0.0
        rollout_state = state

        for step in range(episode_length):
            # Get action from MPPI
            action, rollout_state = mppi.command(
                config,
                rollout_state,
                obs,
                pendulum_dynamics,
                pendulum_cost,
                terminal_cost,
            )

            # Apply action to environment
            obs = pendulum_dynamics(obs, action)
            episode_cost += pendulum_cost(obs, action)

        total_cost += episode_cost

    return total_cost / num_episodes


def main():
    """Run autotuning example."""
    parser = argparse.ArgumentParser(description="Autotune MPPI for pendulum.")
    parser.add_argument("--population", type=int, default=8, help="CMA-ES population size")
    args = parser.parse_args()

    print("=" * 60)
    print("MPPI Autotuning Example: Inverted Pendulum")
    print("=" * 60)

    # Create initial MPPI configuration with suboptimal parameters
    print("\n1. Creating MPPI controller with initial (suboptimal) parameters...")
    config, state = mppi.create(
        nx=2,  # state dim: [theta, theta_dot]
        nu=1,  # action dim: [torque]
        horizon=20,
        num_samples=100,
        lambda_=5.0,  # Intentionally suboptimal (too high)
        noise_sigma=jnp.eye(1) * 1.5,  # Intentionally suboptimal (too high)
        u_min=jnp.array([-2.0]),  # Torque limits
        u_max=jnp.array([2.0]),
    )

    print(f"   Initial lambda: {config.lambda_}")
    print(f"   Initial noise_sigma: {state.noise_sigma[0, 0]}")

    # Evaluate initial performance
    print("\n2. Evaluating initial performance...")
    initial_cost = evaluate_mppi_performance(config, state, num_episodes=3)
    print(f"   Initial average cost: {initial_cost:.2f}")

    # Setup autotuning
    print("\n3. Setting up autotuner...")
    holder = autotune.ConfigStateHolder(config, state)

    def evaluate():
        """Evaluation function for autotuner."""
        # Evaluate current parameters
        cost = evaluate_mppi_performance(
            holder.config, holder.state, num_episodes=2, episode_length=80
        )

        return autotune.EvaluationResult(
            mean_cost=float(cost),
            rollouts=jnp.zeros((1, 1, 2)),  # Placeholder
            params={},
            iteration=0,
        )

    # Create autotuner with CMA-ES
    tuner = autotune.Autotune(
        params_to_tune=[
            autotune.LambdaParameter(holder, min_value=0.1),
            autotune.NoiseSigmaParameter(holder, min_value=0.1),
        ],
        evaluate_fn=evaluate,
        optimizer=autotune.CMAESOpt(population=args.population, sigma=0.3),
    )

    print("   Tuning 2 parameters: lambda, noise_sigma")
    print(f"   Optimizer: CMA-ES (population={args.population})")

    # Run optimization
    print("\n4. Running optimization (this may take a few minutes)...")
    num_iterations = 15
    print(f"   Iterations: {num_iterations}")

    # Track progress
    costs = []

    for i in range(num_iterations):
        result = tuner.optimize_step()
        costs.append(result.mean_cost)
        if (i + 1) % 3 == 0:
            print(
                f"   Iteration {i + 1}/{num_iterations}: "
                f"cost = {result.mean_cost:.2f}, "
                f"lambda = {result.params['lambda'][0]:.3f}"
            )

    best = tuner.get_best_result()

    print("\n5. Optimization complete!")
    print(f"   Best cost: {best.mean_cost:.2f}")
    print(f"   Optimized lambda: {best.params['lambda'][0]:.3f}")
    print(f"   Optimized noise_sigma: {best.params['noise_sigma'][0]:.3f}")

    # Compare initial vs optimized
    print("\n6. Performance comparison:")
    print(f"   Initial cost:   {initial_cost:.2f}")
    print(f"   Optimized cost: {best.mean_cost:.2f}")
    improvement = (initial_cost - best.mean_cost) / initial_cost * 100
    print(f"   Improvement:    {improvement:.1f}%")

    # Plot convergence
    print("\n7. Generating convergence plot...")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(costs, marker="o", linewidth=2, markersize=4)
    plt.axhline(y=initial_cost, color="r", linestyle="--", label="Initial")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Autotuning Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Simulate rollout with optimized parameters
    obs = jnp.array([jnp.pi * 0.9, 0.0])  # Start near upright
    trajectory = [obs]
    rollout_state = holder.state

    for _ in range(100):
        action, rollout_state = mppi.command(
            holder.config,
            rollout_state,
            obs,
            pendulum_dynamics,
            pendulum_cost,
            terminal_cost,
        )
        obs = pendulum_dynamics(obs, action)
        trajectory.append(obs)

    trajectory = jnp.array(trajectory)
    plt.plot(trajectory[:, 0], label="theta (angle)")
    plt.plot(trajectory[:, 1], label="theta_dot (velocity)")
    plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Step")
    plt.ylabel("State")
    plt.title("Example Rollout (Optimized)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("autotune_pendulum_results.png", dpi=150)
    print("   Saved plot to: autotune_pendulum_results.png")

    print("\n" + "=" * 60)
    print("Autotuning complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
