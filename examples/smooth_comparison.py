"""Smooth MPPI comparison example.

This example compares MPPI, SMPPI, and KMPPI on a 2D navigation task with obstacles.
The task is to navigate from [-3, -2] to [2, 2] while avoiding a Gaussian hill obstacle.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax_mppi import kmppi, mppi, smppi
from jax_mppi.costs import create_hill_cost, create_lqr_cost
from jax_mppi.dynamics.linear import create_linear_delta_dynamics


def create_combined_cost(lqr_cost_fn, hill_cost_fn):
    """Combine LQR and hill costs."""

    def combined_cost(state, action=None):
        return lqr_cost_fn(state, action) + hill_cost_fn(state, action)

    return combined_cost


def run_mppi_controller(
    dynamics_fn,
    running_cost_fn,
    terminal_cost_fn,
    start_state: jax.Array,
    num_steps: int = 20,
    num_samples: int = 500,
    horizon: int = 20,
    lambda_: float = 1.0,
    seed: int = 0,
):
    """Run standard MPPI controller."""
    key = jax.random.PRNGKey(seed)

    nx = 2
    nu = 2
    noise_sigma = jnp.eye(2)
    u_min = jnp.array([-1.0, -1.0])
    u_max = jnp.array([1.0, 1.0])

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

    command_fn = jax.jit(
        lambda state, obs: mppi.command(
            config=config,
            mppi_state=state,
            current_obs=obs,
            dynamics=dynamics_fn,
            running_cost=running_cost_fn,
            terminal_cost=terminal_cost_fn,
            shift=True,
        )
    )

    state = start_state
    states = [state]
    actions_taken = []
    costs_history = []

    for step in range(num_steps):
        action, mppi_state = command_fn(mppi_state, state)
        state = dynamics_fn(state, action)
        cost = running_cost_fn(state, action)

        states.append(state)
        actions_taken.append(action)
        costs_history.append(cost)

        print(
            f"  Step {step:2d}: state=[{state[0]:6.3f}, {state[1]:6.3f}], cost={cost:.3f}"
        )

    return jnp.stack(states), jnp.stack(actions_taken), jnp.array(costs_history)


def run_smppi_controller(
    dynamics_fn,
    running_cost_fn,
    terminal_cost_fn,
    start_state: jax.Array,
    num_steps: int = 20,
    num_samples: int = 500,
    horizon: int = 20,
    lambda_: float = 1.0,
    seed: int = 0,
    w_action_seq_cost: float = 10.0,
    delta_t: float = 1.0,
):
    """Run Smooth MPPI controller."""
    key = jax.random.PRNGKey(seed)

    nx = 2
    nu = 2
    noise_sigma = jnp.eye(2)
    u_min = jnp.array([-1.0, -1.0])
    u_max = jnp.array([1.0, 1.0])

    config, smppi_state = smppi.create(
        nx=nx,
        nu=nu,
        noise_sigma=noise_sigma,
        num_samples=num_samples,
        horizon=horizon,
        lambda_=lambda_,
        u_min=u_min,
        u_max=u_max,
        key=key,
        w_action_seq_cost=w_action_seq_cost,
        delta_t=delta_t,
    )

    command_fn = jax.jit(
        lambda state, obs: smppi.command(
            config=config,
            smppi_state=state,
            current_obs=obs,
            dynamics=dynamics_fn,
            running_cost=running_cost_fn,
            terminal_cost=terminal_cost_fn,
            shift=True,
        )
    )

    state = start_state
    states = [state]
    actions_taken = []
    costs_history = []

    for step in range(num_steps):
        action, smppi_state = command_fn(smppi_state, state)
        state = dynamics_fn(state, action)
        cost = running_cost_fn(state, action)

        states.append(state)
        actions_taken.append(action)
        costs_history.append(cost)

        print(
            f"  Step {step:2d}: state=[{state[0]:6.3f}, {state[1]:6.3f}], cost={cost:.3f}"
        )

    return jnp.stack(states), jnp.stack(actions_taken), jnp.array(costs_history)


def run_kmppi_controller(
    dynamics_fn,
    running_cost_fn,
    terminal_cost_fn,
    start_state: jax.Array,
    num_steps: int = 20,
    num_samples: int = 500,
    horizon: int = 20,
    lambda_: float = 1.0,
    seed: int = 0,
    num_support_pts: int = 5,
    kernel_sigma: float = 2.0,
):
    """Run Kernel MPPI controller."""
    key = jax.random.PRNGKey(seed)

    nx = 2
    nu = 2
    noise_sigma = jnp.eye(2)
    u_min = jnp.array([-1.0, -1.0])
    u_max = jnp.array([1.0, 1.0])

    # Create RBF kernel
    rbf_kernel = kmppi.RBFKernel(sigma=kernel_sigma)

    config, kmppi_state, kernel_fn = kmppi.create(
        nx=nx,
        nu=nu,
        noise_sigma=noise_sigma,
        num_samples=num_samples,
        horizon=horizon,
        lambda_=lambda_,
        u_min=u_min,
        u_max=u_max,
        key=key,
        num_support_pts=num_support_pts,
        kernel=rbf_kernel,
    )

    command_fn = jax.jit(
        lambda state, obs: kmppi.command(
            config=config,
            kmppi_state=state,
            current_obs=obs,
            dynamics=dynamics_fn,
            running_cost=running_cost_fn,
            kernel_fn=kernel_fn,
            terminal_cost=terminal_cost_fn,
            shift=True,
        )
    )

    state = start_state
    states = [state]
    actions_taken = []
    costs_history = []

    for step in range(num_steps):
        action, kmppi_state = command_fn(kmppi_state, state)
        state = dynamics_fn(state, action)
        cost = running_cost_fn(state, action)

        states.append(state)
        actions_taken.append(action)
        costs_history.append(cost)

        print(
            f"  Step {step:2d}: state=[{state[0]:6.3f}, {state[1]:6.3f}], cost={cost:.3f}"
        )

    return jnp.stack(states), jnp.stack(actions_taken), jnp.array(costs_history)


def visualize_results(
    results_dict,
    start_state,
    goal_state,
    running_cost_fn,
    state_ranges=((-5, 5), (-5, 5)),
    save_path: Optional[str] = None,
):
    """Visualize comparison of different controllers."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Trajectories on cost landscape
    ax = axes[0, 0]

    # Draw cost landscape
    resolution = 0.05
    x_coords = jnp.arange(
        state_ranges[0][0], state_ranges[0][1] + resolution, resolution
    )
    y_coords = jnp.arange(
        state_ranges[1][0], state_ranges[1][1] + resolution, resolution
    )
    X, Y = jnp.meshgrid(x_coords, y_coords)
    pts = jnp.stack([X.flatten(), Y.flatten()], axis=1)

    costs = jax.vmap(lambda s: running_cost_fn(s, None))(pts)
    Z = costs.reshape(X.shape)

    levels = [2, 4, 8, 16, 24, 32, 40, 50, 60, 80, 100, 150, 200, 250]
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap="Greys", alpha=0.6)
    ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        colors="k",
        linestyles="dashed",
        alpha=0.3,
        linewidths=0.5,
    )

    # Plot trajectories
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for (name, (states, _, _)), color in zip(results_dict.items(), colors):
        ax.plot(
            states[:, 0],
            states[:, 1],
            "-o",
            label=name,
            color=color,
            linewidth=2,
            markersize=4,
            alpha=0.8,
        )

    ax.scatter(
        start_state[0],
        start_state[1],
        color="red",
        s=200,
        marker="*",
        edgecolors="black",
        linewidths=2,
        label="Start",
        zorder=10,
    )
    ax.scatter(
        goal_state[0],
        goal_state[1],
        color="green",
        s=200,
        marker="*",
        edgecolors="black",
        linewidths=2,
        label="Goal",
        zorder=10,
    )

    ax.set_xlim(state_ranges[0])
    ax.set_ylim(state_ranges[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectories on Cost Landscape")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Plot 2: Running costs over time
    ax = axes[0, 1]
    for (name, (_, _, costs)), color in zip(results_dict.items(), colors):
        ax.plot(costs, "-o", label=name, color=color, linewidth=2, markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Running Cost")
    ax.set_title("Running Cost Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Control inputs
    ax = axes[1, 0]
    for (name, (_, actions, _)), color in zip(results_dict.items(), colors):
        steps = np.arange(len(actions))
        ax.plot(
            steps,
            actions[:, 0],
            "-",
            label=f"{name} u0",
            color=color,
            linewidth=2,
            alpha=0.7,
        )
        ax.plot(
            steps,
            actions[:, 1],
            "--",
            label=f"{name} u1",
            color=color,
            linewidth=2,
            alpha=0.7,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Control Input")
    ax.set_title("Control Inputs Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

    # Plot 4: Control smoothness
    ax = axes[1, 1]
    for (name, (_, actions, _)), color in zip(results_dict.items(), colors):
        action_diff = jnp.diff(actions, axis=0)
        diff_norm = jnp.linalg.norm(action_diff, axis=1)
        ax.plot(diff_norm, "-o", label=name, color=color, linewidth=2, markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("||Δu||")
    ax.set_title("Control Smoothness (L2 norm of action differences)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")

    plt.show()


def main():
    """Run smooth MPPI comparison."""

    # Environment setup
    start_state = jnp.array([-3.0, -2.0])
    goal_state = jnp.array([2.0, 2.0])

    # Dynamics: Linear delta dynamics with B = [[0.5, 0], [0, -0.5]]
    B = jnp.array([[0.5, 0.0], [0.0, -0.5]])
    dynamics_fn = create_linear_delta_dynamics(B)

    # Costs
    Q = jnp.eye(2)
    R = jnp.eye(2) * 0.01
    lqr_cost = create_lqr_cost(Q, R, goal_state)

    # Hill cost (obstacle at [-0.5, -1.0])
    hill_Q = jnp.array([[0.1, 0.05], [0.05, 0.1]]) * 2.5
    hill_center = jnp.array([-0.5, -1.0])
    hill_cost = create_hill_cost(hill_Q, hill_center, cost_at_center=200.0)

    # Combined cost
    running_cost = create_combined_cost(lqr_cost, hill_cost)

    # Terminal cost
    terminal_scale = 10.0

    def terminal_cost_fn(state, last_action):
        return terminal_scale * running_cost(state, None)

    # Controller parameters
    num_steps = 20
    num_samples = 500
    horizon = 20
    lambda_ = 1.0
    seed = 0

    print("=" * 60)
    print("Smooth MPPI Comparison")
    print("=" * 60)
    print(f"Start: {start_state}")
    print(f"Goal:  {goal_state}")
    print(f"Obstacle center: {hill_center}")
    print("\nController parameters:")
    print(f"  Samples: {num_samples}")
    print(f"  Horizon: {horizon}")
    print(f"  Lambda:  {lambda_}")
    print(f"  Steps:   {num_steps}")
    print("=" * 60)

    # Run controllers
    results = {}

    print("\n[1/3] Running standard MPPI...")
    states_mppi, actions_mppi, costs_mppi = run_mppi_controller(
        dynamics_fn,
        running_cost,
        terminal_cost_fn,
        start_state,
        num_steps=num_steps,
        num_samples=num_samples,
        horizon=horizon,
        lambda_=lambda_,
        seed=seed,
    )
    results["MPPI"] = (states_mppi, actions_mppi, costs_mppi)
    print(f"  Total cost: {jnp.sum(costs_mppi):.2f}")
    print(
        f"  Control smoothness: {jnp.sum(jnp.linalg.norm(jnp.diff(actions_mppi, axis=0), axis=1)):.3f}"
    )

    print("\n[2/3] Running Smooth MPPI (SMPPI)...")
    states_smppi, actions_smppi, costs_smppi = run_smppi_controller(
        dynamics_fn,
        running_cost,
        terminal_cost_fn,
        start_state,
        num_steps=num_steps,
        num_samples=num_samples,
        horizon=horizon,
        lambda_=lambda_,
        seed=seed,
        w_action_seq_cost=10.0,
        delta_t=1.0,
    )
    results["SMPPI"] = (states_smppi, actions_smppi, costs_smppi)
    print(f"  Total cost: {jnp.sum(costs_smppi):.2f}")
    print(
        f"  Control smoothness: {jnp.sum(jnp.linalg.norm(jnp.diff(actions_smppi, axis=0), axis=1)):.3f}"
    )

    print("\n[3/3] Running Kernel MPPI (KMPPI)...")
    states_kmppi, actions_kmppi, costs_kmppi = run_kmppi_controller(
        dynamics_fn,
        running_cost,
        terminal_cost_fn,
        start_state,
        num_steps=num_steps,
        num_samples=num_samples,
        horizon=horizon,
        lambda_=lambda_,
        seed=seed,
        num_support_pts=5,
        kernel_sigma=2.0,
    )
    results["KMPPI"] = (states_kmppi, actions_kmppi, costs_kmppi)
    print(f"  Total cost: {jnp.sum(costs_kmppi):.2f}")
    print(
        f"  Control smoothness: {jnp.sum(jnp.linalg.norm(jnp.diff(actions_kmppi, axis=0), axis=1)):.3f}"
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, (_, actions, costs) in results.items():
        total_cost = jnp.sum(costs)
        smoothness = jnp.sum(jnp.linalg.norm(jnp.diff(actions, axis=0), axis=1))
        print(
            f"{name:10s}: Total Cost = {total_cost:8.2f}, Smoothness = {smoothness:6.3f}"
        )
    print("=" * 60)

    # Visualize
    print("\nGenerating visualization...")
    visualize_results(
        results,
        start_state,
        goal_state,
        running_cost,
        save_path="docs/media/smooth_comparison.png",
    )


if __name__ == "__main__":
    main()
