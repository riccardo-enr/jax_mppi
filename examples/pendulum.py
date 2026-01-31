"""Pendulum swing-up example using JAX MPPI.

This example demonstrates using MPPI to control an inverted pendulum.
The goal is to swing up and stabilize the pendulum in the upright position.
"""

import jax
import jax.numpy as jnp

from typing import Any, cast
from jax_mppi import mppi


def pendulum_dynamics(state: jax.Array, action: jax.Array) -> jax.Array:
    """Pendulum dynamics.

    State: [theta, theta_dot]
        theta: angle from upright (0 = upright, pi = hanging down)
        theta_dot: angular velocity
    Action: [torque]
        torque: applied torque (control input)

    Args:
        state: (2,) array [theta, theta_dot]
        action: (1,) array [torque]

    Returns:
        next_state: (2,) array [theta_next, theta_dot_next]
    """
    g = 10.0  # gravity
    m = 1.0  # mass
    l = 1.0  # length
    dt = 0.05  # timestep

    theta, theta_dot = state[0], state[1]
    torque = action[0]

    # Clip torque to reasonable bounds
    torque = jnp.clip(torque, -2.0, 2.0)

    # Pendulum dynamics: theta_ddot = (torque - m*g*l*sin(theta)) / (m*l^2)
    theta_ddot = (torque - m * g * l * jnp.sin(theta)) / (m * l * l)

    # Euler integration
    theta_dot_next = theta_dot + theta_ddot * dt
    theta_next = theta + theta_dot_next * dt

    # Normalize angle to [-pi, pi]
    theta_next = ((theta_next + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    return jnp.array([theta_next, theta_dot_next])


def pendulum_cost(state: jax.Array, action: jax.Array) -> jax.Array:
    """Running cost for pendulum.

    Penalizes deviation from upright position and high velocities/torques.

    Args:
        state: (2,) array [theta, theta_dot]
        action: (1,) array [torque]

    Returns:
        cost: scalar cost
    """
    theta, theta_dot = state[0], state[1]
    torque = action[0]

    # Cost for being away from upright (theta=0)
    angle_cost = theta**2

    # Cost for high angular velocity
    velocity_cost = 0.1 * theta_dot**2

    # Cost for using torque
    control_cost = 0.01 * torque**2

    return angle_cost + velocity_cost + control_cost


def pendulum_terminal_cost(state: jax.Array, last_action: jax.Array) -> jax.Array:
    """Terminal cost for pendulum.

    Args:
        state: (2,) terminal state [theta, theta_dot]
        last_action: (1,) last action [torque]

    Returns:
        cost: scalar terminal cost
    """
    # Terminal state cost (want to end upright with low velocity)
    theta, theta_dot = state[0], state[1]

    return 10.0 * theta**2 + theta_dot**2


def run_pendulum_mppi(
    num_steps: int = 100,
    num_samples: int = 1000,
    horizon: int = 30,
    lambda_: float = 1.0,
    visualize: bool = False,
    render: bool = False,
    seed: int = 0,
):
    """Run MPPI on pendulum swing-up task.

    Args:
        num_steps: Number of control steps to simulate
        num_samples: Number of MPPI samples (K)
        horizon: MPPI planning horizon (T)
        lambda_: Temperature parameter for MPPI
        visualize: Whether to plot results (requires matplotlib)
        render: Whether to render with gymnasium (requires gymnasium)
        seed: Random seed

    Returns:
        states: (num_steps+1, 2) trajectory of states
        actions: (num_steps, 1) trajectory of actions
        costs: (num_steps,) trajectory of costs
    """
    # Initialize MPPI
    key = jax.random.PRNGKey(seed)

    nx = 2  # state dimension
    nu = 1  # action dimension

    # Noise covariance (exploration in torque space)
    noise_sigma = jnp.array([[1.0]])

    # Control bounds
    u_min = jnp.array([-2.0])
    u_max = jnp.array([2.0])

    # Create MPPI controller
    config, mppi_state = mppi.create(
        nx=nx,
        nu=nu,
        noise_sigma=noise_sigma,
        num_samples=num_samples,
        horizon=horizon,
        lambda_=lambda_,
        u_min=u_min,
        u_max=u_max,
        key=key,
    )

    # Initialize gymnasium environment for rendering if requested
    env = None
    if render:
        try:
            import gymnasium as gym

            env = gym.make("Pendulum-v1", render_mode="human")
            # Reset to get initial observation
            env.reset(seed=seed)
        except ImportError:
            print("Gymnasium not available for rendering")
            render = False

    # Initial state: hanging down with small perturbation
    state = jnp.array([jnp.pi + 0.1, 0.0])

    # Storage for trajectory
    states = [state]
    actions_taken = []
    costs_history = []

    # JIT compile the command function for speed
    command_fn = jax.jit(
        lambda mppi_state, obs: mppi.command(
            config=config,
            mppi_state=mppi_state,
            current_obs=obs,
            dynamics=pendulum_dynamics,
            running_cost=pendulum_cost,
            terminal_cost=pendulum_terminal_cost,
            shift=True,
        )
    )

    print("Running MPPI on pendulum swing-up task...")
    print(f"  Samples: {num_samples}, Horizon: {horizon}, Lambda: {lambda_}")
    print(f"  Initial state: theta={state[0]:.2f}, theta_dot={state[1]:.2f}")
    if render:
        print("  Press Ctrl+C to stop...")

    # Control loop
    step = 0
    try:
        while step < num_steps:
            # Compute optimal action
            action, mppi_state = command_fn(mppi_state, state)

            # Apply action to environment
            state = pendulum_dynamics(state, action)

            # Compute cost
            cost = pendulum_cost(state, action)

            # Render if gymnasium environment is available
            if env is not None:
                # Update gymnasium environment state to match our JAX state
                # Gymnasium Pendulum-v1 state: [cos(theta), sin(theta), theta_dot]
                theta, theta_dot = float(state[0]), float(state[1])
                # Accessing .state on unwrapped env is dynamic, cast to Any to satisfy static analysis
                unwrapped_env = cast(Any, env.unwrapped)
                unwrapped_env.state = jnp.array([theta, theta_dot])
                env.render()

            # Store
            states.append(state)
            actions_taken.append(action)
            costs_history.append(cost)

            # Print progress
            if step % 20 == 0:
                print(
                    f"Step {step:3d}: theta={state[0]:6.3f}, theta_dot={state[1]:6.3f}, cost={cost:.3f}"
                )

            step += 1

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at step {step}")

    states = jnp.stack(states)
    actions_taken = jnp.stack(actions_taken)
    costs_history = jnp.array(costs_history)

    # Close gymnasium environment if used
    if env is not None:
        env.close()

    print(f"\nFinal state: theta={states[-1, 0]:.3f}, theta_dot={states[-1, 1]:.3f}")
    print(f"Total cost: {jnp.sum(costs_history):.2f}")
    print(f"Final 10-step avg cost: {jnp.mean(costs_history[-10:]):.3f}")

    if visualize:
        try:
            import matplotlib.pyplot as plt
            from pathlib import Path

            fig, axes = plt.subplots(3, 1, figsize=(10, 8))

            # Plot angle
            time = jnp.arange(len(states)) * 0.05
            axes[0].plot(time, states[:, 0], label="theta")
            axes[0].axhline(0, color="k", linestyle="--", alpha=0.3)
            axes[0].set_ylabel("Angle (rad)")
            axes[0].set_title("Pendulum Swing-Up with MPPI")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot angular velocity
            axes[1].plot(time, states[:, 1], label="theta_dot", color="orange")
            axes[1].axhline(0, color="k", linestyle="--", alpha=0.3)
            axes[1].set_ylabel("Angular velocity (rad/s)")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Plot control
            time_actions = jnp.arange(len(actions_taken)) * 0.05
            axes[2].plot(
                time_actions, actions_taken[:, 0], label="torque", color="green"
            )
            axes[2].axhline(0, color="k", linestyle="--", alpha=0.3)
            axes[2].set_ylabel("Torque (N·m)")
            axes[2].set_xlabel("Time (s)")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save to docs/media directory
            output_dir = Path(__file__).parent.parent / "docs" / "media"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "pendulum_mppi.png"

            plt.savefig(output_path, dpi=150)
            print(f"\nPlot saved to {output_path}")
            plt.show()

        except ImportError:
            print("\nMatplotlib not available for visualization")

    return states, actions_taken, costs_history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pendulum swing-up with MPPI")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of control steps (default: 100, or infinite with --render)"
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of MPPI samples"
    )
    parser.add_argument("--horizon", type=int, default=30, help="MPPI planning horizon")
    parser.add_argument(
        "--lambda", type=float, default=1.0, dest="lambda_", help="MPPI temperature"
    )
    parser.add_argument("--visualize", action="store_true", help="Plot results with matplotlib")
    parser.add_argument("--render", action="store_true", help="Render with gymnasium (animated)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    # Set default steps: infinite (large number) when rendering, 100 otherwise
    num_steps = args.steps
    if num_steps is None:
        num_steps = float('inf') if args.render else 100

    states, actions, costs = run_pendulum_mppi(
        num_steps=int(num_steps) if num_steps != float('inf') else 10**9,
        num_samples=args.samples,
        horizon=args.horizon,
        lambda_=args.lambda_,
        visualize=args.visualize,
        render=args.render,
        seed=args.seed,
    )
