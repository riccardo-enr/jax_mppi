"""Quadrotor figure-8 trajectory comparison: MPPI vs SMPPI vs KMPPI.

This example demonstrates the differences between three MPPI variants on an
aggressive figure-8 trajectory. It compares:
- MPPI: Standard model predictive path integral control
- SMPPI: Smooth MPPI with control rate penalties
- KMPPI: Kernel-based MPPI with information-theoretic updates

The comparison evaluates tracking accuracy, control smoothness, and energy efficiency.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

from examples.quadrotor.trajectories import (
    compute_trajectory_metrics,
    generate_lemniscate_trajectory,
)
from jax_mppi import kmppi, mppi, smppi
from jax_mppi.costs.quadrotor import create_terminal_cost
from jax_mppi.dynamics.quadrotor import create_quadrotor_dynamics


def create_tracking_cost(
    Q_pos: jax.Array,
    Q_vel: jax.Array,
    R: jax.Array,
    ref_pos: jax.Array,
    ref_vel: jax.Array,
):
    """Create trajectory tracking cost for a specific reference point."""

    def cost_fn(state: jax.Array, action: jax.Array) -> jax.Array:
        pos_error = state[0:3] - ref_pos
        cost_pos = pos_error @ Q_pos @ pos_error

        vel_error = state[3:6] - ref_vel
        cost_vel = vel_error @ Q_vel @ vel_vel

        cost_control = action @ R @ action

        return cost_pos + cost_vel + cost_control

    return cost_fn


def run_controller(
    controller_type: str,
    reference: jax.Array,
    num_steps: int,
    num_samples: int,
    horizon: int,
    lambda_: float,
    dt: float,
    seed: int,
):
    """Run a single controller on the figure-8 trajectory.

    Args:
        controller_type: "mppi", "smppi", or "kmppi"
        reference: Reference trajectory [T x 6]
        num_steps: Number of control steps
        num_samples: Number of MPPI samples
        horizon: Planning horizon
        lambda_: Temperature parameter
        dt: Time step
        seed: Random seed

    Returns:
        states: (num_steps+1, 13) trajectory
        actions: (num_steps, 4) control inputs
        costs: (num_steps,) running costs
    """
    key = jax.random.PRNGKey(seed)

    nx, nu = 13, 4
    mass, gravity = 1.0, 9.81

    # Create dynamics
    u_min = jnp.array([0.0, -8.0, -8.0, -8.0])
    u_max = jnp.array([4.0 * mass * gravity, 8.0, 8.0, 8.0])

    dynamics = create_quadrotor_dynamics(
        dt=dt, mass=mass, gravity=gravity, tau_omega=0.05,
        u_min=u_min, u_max=u_max
    )

    # Cost weights
    Q_pos = jnp.eye(3) * 100.0
    Q_vel = jnp.eye(3) * 20.0
    R = jnp.diag(jnp.array([0.01, 0.1, 0.1, 0.1]))

    # Terminal cost
    goal_position = reference[-1, 0:3]
    goal_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])
    terminal_cost_fn = create_terminal_cost(
        Q_pos * 10.0, Q_vel * 10.0, jnp.eye(4) * 5.0,
        goal_position, goal_quaternion
    )

    # Noise covariance
    noise_sigma = jnp.diag(jnp.array([5.0, 1.5, 1.5, 1.5]))

    # Create controller
    if controller_type == "mppi":
        config, controller_state = mppi.create(
            nx=nx, nu=nu, noise_sigma=noise_sigma,
            num_samples=num_samples, horizon=horizon,
            lambda_=lambda_, u_min=u_min, u_max=u_max, key=key
        )
        command_fn = mppi.command
    elif controller_type == "smppi":
        config, controller_state = smppi.create(
            nx=nx, nu=nu, noise_sigma=noise_sigma,
            num_samples=num_samples, horizon=horizon,
            lambda_=lambda_, u_min=u_min, u_max=u_max, key=key,
            smoothing_factor=0.5,  # Penalize control rate changes
        )
        command_fn = smppi.command
    elif controller_type == "kmppi":
        config, controller_state = kmppi.create(
            nx=nx, nu=nu, noise_sigma=noise_sigma,
            num_samples=num_samples, horizon=horizon,
            lambda_=lambda_, u_min=u_min, u_max=u_max, key=key,
            kernel_bandwidth=1.0,
        )
        command_fn = kmppi.command
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    # Initial state
    state = jnp.array([
        reference[0, 0], reference[0, 1], reference[0, 2],
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])

    # Storage
    states, actions_taken, costs_history = [state], [], []

    print(f"\nRunning {controller_type.upper()}...")

    # Control loop
    for step in range(num_steps):
        # Create cost for current reference
        ref_pos = reference[step, 0:3]
        ref_vel = reference[step, 3:6]
        running_cost_fn = create_tracking_cost(Q_pos, Q_vel, R, ref_pos, ref_vel)

        # Compute action
        action, controller_state = command_fn(
            config=config,
            mppi_state=controller_state,
            current_obs=state,
            dynamics=dynamics,
            running_cost=running_cost_fn,
            terminal_cost=terminal_cost_fn,
            shift=True,
        )

        # Apply dynamics
        state = dynamics(state, action)
        cost = running_cost_fn(state, action)

        states.append(state)
        actions_taken.append(action)
        costs_history.append(cost)

        if step % 100 == 0:
            pos_error = jnp.linalg.norm(state[0:3] - ref_pos)
            print(f"  Step {step:4d}: pos_err={pos_error:.3f}m, cost={cost:.2f}")

    return jnp.stack(states), jnp.stack(actions_taken), jnp.array(costs_history)


def compute_smoothness_metrics(actions: jax.Array, dt: float) -> dict:
    """Compute control smoothness metrics."""
    # Control rate (acceleration in control space)
    action_rate = jnp.diff(actions, axis=0) / dt
    action_rate_mag = jnp.linalg.norm(action_rate, axis=1)

    # Jerk (rate of control rate)
    action_jerk = jnp.diff(action_rate, axis=0) / dt
    action_jerk_mag = jnp.linalg.norm(action_jerk, axis=1)

    return {
        "mean_control_rate": float(jnp.mean(action_rate_mag)),
        "max_control_rate": float(jnp.max(action_rate_mag)),
        "mean_control_jerk": float(jnp.mean(action_jerk_mag)),
        "max_control_jerk": float(jnp.max(action_jerk_mag)),
    }


def compute_energy_consumption(actions: jax.Array, dt: float, mass: float) -> float:
    """Compute total energy consumption (simplified)."""
    # Energy = sum of squared thrust over time
    thrust = actions[:, 0]
    energy = float(jnp.sum(thrust**2) * dt)
    return energy


def run_quadrotor_figure8_comparison(
    num_steps: int = 1000,
    num_samples: int = 800,
    horizon: int = 25,
    lambda_: float = 1.0,
    scale: float = 4.0,
    period: float = 20.0,
    visualize: bool = False,
    seed: int = 0,
):
    """Compare MPPI variants on figure-8 trajectory.

    Args:
        num_steps: Number of control steps
        num_samples: Number of MPPI samples
        horizon: Planning horizon
        lambda_: Temperature parameter
        scale: Figure-8 scale (m)
        period: Figure-8 period (s)
        visualize: Whether to plot results
        seed: Random seed

    Returns:
        Dictionary with results for each controller
    """
    dt = 0.02  # 50 Hz
    height = -5.0  # 5m altitude

    # Generate figure-8 reference
    duration = num_steps * dt
    reference = generate_lemniscate_trajectory(
        scale=scale, height=height, period=period,
        duration=duration, dt=dt, axis="xy"
    )

    print("=" * 60)
    print("QUADROTOR FIGURE-8 TRAJECTORY COMPARISON")
    print("=" * 60)
    print(f"\nTrajectory: figure-8, scale={scale}m, period={period}s")
    print(f"Control: {num_samples} samples, horizon={horizon}, λ={lambda_}")
    print(f"Duration: {duration:.1f}s ({num_steps} steps @ {1/dt:.0f} Hz)")

    metrics = compute_trajectory_metrics(reference, dt)
    print("\nReference trajectory metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.3f}")

    # Run each controller
    results = {}
    for controller in ["mppi", "smppi", "kmppi"]:
        states, actions, costs = run_controller(
            controller, reference, num_steps, num_samples,
            horizon, lambda_, dt, seed
        )

        # Compute metrics
        pos_errors = jnp.linalg.norm(
            states[50:-1, 0:3] - reference[50:, 0:3], axis=1
        )
        vel_errors = jnp.linalg.norm(
            states[50:-1, 3:6] - reference[50:, 3:6], axis=1
        )

        smoothness = compute_smoothness_metrics(actions, dt)
        energy = compute_energy_consumption(actions, dt, mass=1.0)

        results[controller] = {
            "states": states,
            "actions": actions,
            "costs": costs,
            "mean_pos_error": float(jnp.mean(pos_errors)),
            "max_pos_error": float(jnp.max(pos_errors)),
            "rms_pos_error": float(jnp.sqrt(jnp.mean(pos_errors**2))),
            "mean_vel_error": float(jnp.mean(vel_errors)),
            "total_cost": float(jnp.sum(costs)),
            "energy": energy,
            **smoothness,
        }

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    metrics_to_compare = [
        ("Mean Position Error (m)", "mean_pos_error"),
        ("RMS Position Error (m)", "rms_pos_error"),
        ("Max Position Error (m)", "max_pos_error"),
        ("Mean Control Rate", "mean_control_rate"),
        ("Max Control Rate", "max_control_rate"),
        ("Energy Consumption", "energy"),
        ("Total Cost", "total_cost"),
    ]

    for metric_name, metric_key in metrics_to_compare:
        print(f"\n{metric_name}:")
        for controller in ["mppi", "smppi", "kmppi"]:
            value = results[controller][metric_key]
            print(f"  {controller.upper():6s}: {value:8.4f}")

    if visualize:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(18, 12))

            # 3D trajectories
            ax1 = fig.add_subplot(2, 3, 1, projection="3d")
            ax1.plot(
                reference[:, 0], reference[:, 1], reference[:, 2],
                "k--", linewidth=2, alpha=0.7, label="Reference"
            )
            colors = {"mppi": "C0", "smppi": "C1", "kmppi": "C2"}
            for controller, color in colors.items():
                states = results[controller]["states"]
                ax1.plot(
                    states[:, 0], states[:, 1], states[:, 2],
                    color=color, linewidth=1, alpha=0.8,
                    label=controller.upper()
                )
            ax1.set_xlabel("X (m)")
            ax1.set_ylabel("Y (m)")
            ax1.set_zlabel("Z (m)")
            ax1.legend()
            ax1.set_title("3D Trajectory Comparison")
            ax1.view_init(elev=20, azim=45)

            # XY trajectory (top view)
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(
                reference[:, 0], reference[:, 1],
                "k--", linewidth=2, alpha=0.7, label="Reference"
            )
            for controller, color in colors.items():
                states = results[controller]["states"]
                ax2.plot(
                    states[:, 0], states[:, 1],
                    color=color, linewidth=1, alpha=0.8,
                    label=controller.upper()
                )
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            ax2.axis("equal")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title("XY Trajectory (Top View)")

            # Position errors over time
            time = jnp.arange(num_steps) * dt
            ax3 = plt.subplot(2, 3, 3)
            for controller, color in colors.items():
                states = results[controller]["states"]
                errors = jnp.linalg.norm(
                    states[:-1, 0:3] - reference[:, 0:3], axis=1
                )
                ax3.plot(time, errors, color=color, label=controller.upper())
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Position Error (m)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_title("Tracking Error Over Time")

            # Control smoothness (thrust)
            ax4 = plt.subplot(2, 3, 4)
            for controller, color in colors.items():
                actions = results[controller]["actions"]
                ax4.plot(time, actions[:, 0], color=color, label=controller.upper())
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Thrust (N)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_title("Thrust Command")

            # Control rates
            ax5 = plt.subplot(2, 3, 5)
            for controller, color in colors.items():
                actions = results[controller]["actions"]
                thrust_rate = jnp.diff(actions[:, 0]) / dt
                ax5.plot(time[:-1], thrust_rate, color=color, label=controller.upper())
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("Thrust Rate (N/s)")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_title("Control Rate (Smoothness)")

            # Bar chart comparison
            ax6 = plt.subplot(2, 3, 6)
            metrics_keys = ["mean_pos_error", "mean_control_rate", "energy"]
            metrics_labels = ["Pos Error\n(m)", "Control Rate", "Energy"]

            x = jnp.arange(len(metrics_keys))
            width = 0.25

            for i, (controller, color) in enumerate(colors.items()):
                values = [results[controller][k] for k in metrics_keys]
                # Normalize for comparison
                values_norm = [v / max(results[c][k] for c in colors) for v, k in zip(values, metrics_keys)]
                ax6.bar(x + i * width, values_norm, width, label=controller.upper(), color=color)

            ax6.set_ylabel("Normalized Value")
            ax6.set_title("Performance Comparison (Normalized)")
            ax6.set_xticks(x + width)
            ax6.set_xticklabels(metrics_labels)
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()

            # Save
            output_dir = Path(__file__).parent.parent / "docs" / "media"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "quadrotor_figure8_comparison.png"

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved to {output_path}")
            plt.show()

        except ImportError:
            print("\nMatplotlib not available for visualization")

    return results, reference


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare MPPI variants on quadrotor figure-8 trajectory"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of control steps")
    parser.add_argument("--samples", type=int, default=800, help="Number of MPPI samples")
    parser.add_argument("--horizon", type=int, default=25, help="Planning horizon")
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_", help="Temperature")
    parser.add_argument("--scale", type=float, default=4.0, help="Figure-8 scale (m)")
    parser.add_argument("--period", type=float, default=20.0, help="Figure-8 period (s)")
    parser.add_argument("--visualize", action="store_true", help="Plot results")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    results, reference = run_quadrotor_figure8_comparison(
        num_steps=args.steps,
        num_samples=args.samples,
        horizon=args.horizon,
        lambda_=args.lambda_,
        scale=args.scale,
        period=args.period,
        visualize=args.visualize,
        seed=args.seed,
    )
