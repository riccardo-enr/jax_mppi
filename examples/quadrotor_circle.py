"""Quadrotor circular trajectory following example using JAX MPPI.

This example demonstrates using MPPI to track a circular trajectory.
The quadrotor follows a circle at constant altitude, demonstrating
trajectory tracking capabilities.
"""

from pathlib import Path

import jax
import jax.numpy as jnp

from examples.quadrotor.trajectories import (
    compute_trajectory_metrics,
    generate_circle_trajectory,
)
from jax_mppi import mppi
from jax_mppi.costs.quadrotor import create_terminal_cost
from jax_mppi.dynamics.quadrotor import create_quadrotor_dynamics


def create_tracking_cost_at_time(
    Q_pos: jax.Array,
    Q_vel: jax.Array,
    R: jax.Array,
    ref_pos: jax.Array,
    ref_vel: jax.Array,
):
    """Create trajectory tracking cost for a specific reference point.

    Args:
        Q_pos: Position error weight matrix
        Q_vel: Velocity error weight matrix
        R: Control effort weight matrix
        ref_pos: Reference position [3]
        ref_vel: Reference velocity [3]

    Returns:
        Cost function that takes (state, action)
    """

    def cost_fn(state: jax.Array, action: jax.Array) -> jax.Array:
        # Position tracking error
        pos_error = state[0:3] - ref_pos
        cost_pos = pos_error @ Q_pos @ pos_error

        # Velocity tracking error
        vel_error = state[3:6] - ref_vel
        cost_vel = vel_error @ Q_vel @ vel_error

        # Control effort
        cost_control = action @ R @ action

        return cost_pos + cost_vel + cost_control

    return cost_fn


def run_quadrotor_circle(
    num_steps: int = 1000,
    num_samples: int = 1000,
    horizon: int = 30,
    lambda_: float = 1.0,
    radius: float = 3.0,
    period: float = 15.0,
    visualize: bool = False,
    seed: int = 0,
):
    """Run MPPI on quadrotor circular trajectory tracking task.

    Args:
        num_steps: Number of control steps to simulate
        num_samples: Number of MPPI samples (K)
        horizon: MPPI planning horizon (T)
        lambda_: Temperature parameter for MPPI
        radius: Circle radius (m)
        period: Time for one complete circle (s)
        visualize: Whether to plot results (requires matplotlib)
        seed: Random seed

    Returns:
        states: (num_steps+1, 13) trajectory of states
        actions: (num_steps, 4) trajectory of actions
        costs: (num_steps,) trajectory of costs
        reference: (num_steps, 6) reference trajectory
    """
    # Initialize MPPI
    key = jax.random.PRNGKey(seed)

    # State and action dimensions
    nx = 13  # [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    nu = 4   # [thrust, wx_cmd, wy_cmd, wz_cmd]

    # Physical parameters
    mass = 1.0  # kg
    gravity = 9.81  # m/s^2
    dt = 0.02  # 50 Hz control rate

    # Generate circular reference trajectory
    height = -5.0  # 5m altitude in NED
    duration = num_steps * dt
    reference = generate_circle_trajectory(
        radius=radius,
        height=height,
        period=period,
        duration=duration,
        dt=dt,
    )

    print("\nTrajectory metrics:")
    metrics = compute_trajectory_metrics(reference, dt)
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.3f}")

    # Create dynamics
    u_min = jnp.array([0.0, -5.0, -5.0, -5.0])
    u_max = jnp.array([4.0 * mass * gravity, 5.0, 5.0, 5.0])

    dynamics = create_quadrotor_dynamics(
        dt=dt,
        mass=mass,
        gravity=gravity,
        tau_omega=0.05,
        u_min=u_min,
        u_max=u_max,
    )

    # Cost function weights
    Q_pos = jnp.eye(3) * 50.0   # Position error weight
    Q_vel = jnp.eye(3) * 10.0   # Velocity error weight
    R = jnp.diag(jnp.array([0.01, 0.1, 0.1, 0.1]))  # Control effort

    # Terminal cost (track last reference point)
    goal_position = reference[-1, 0:3]
    goal_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

    terminal_cost_fn = create_terminal_cost(
        Q_pos * 10.0, Q_vel * 10.0, jnp.eye(4) * 5.0,
        goal_position, goal_quaternion
    )

    # Noise covariance (exploration in control space)
    noise_sigma = jnp.diag(jnp.array([5.0, 1.0, 1.0, 1.0]))

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

    # Initial state: start at first reference point
    state = jnp.array([
        reference[0, 0], reference[0, 1], reference[0, 2],  # start position
        0.0, 0.0, 0.0,                                       # zero initial velocity
        1.0, 0.0, 0.0, 0.0,                                  # level quaternion
        0.0, 0.0, 0.0                                        # zero angular velocity
    ])

    # Storage for trajectory
    states = [state]
    actions_taken = []
    costs_history = []

    print("\nRunning MPPI on quadrotor circular trajectory tracking...")
    print(f"  Samples: {num_samples}, Horizon: {horizon}, Lambda: {lambda_}")
    print(f"  Circle: radius={radius}m, period={period}s, altitude={-height}m")
    print(f"  Control rate: {1/dt:.0f} Hz")

    # Control loop
    for step in range(num_steps):
        # Create cost function for current reference point
        ref_pos = reference[step, 0:3]
        ref_vel = reference[step, 3:6]
        running_cost_fn = create_tracking_cost_at_time(
            Q_pos, Q_vel, R, ref_pos, ref_vel
        )

        # Compute optimal action
        action, mppi_state = mppi.command(
            config=config,
            mppi_state=mppi_state,
            current_obs=state,
            dynamics=dynamics,
            running_cost=running_cost_fn,
            terminal_cost=terminal_cost_fn,
            shift=True,
        )

        # Apply action to environment
        state = dynamics(state, action)

        # Compute cost
        cost = running_cost_fn(state, action)

        # Store
        states.append(state)
        actions_taken.append(action)
        costs_history.append(cost)

        # Print progress
        if step % 100 == 0:
            ref_pos = reference[step, 0:3]
            pos_error = jnp.linalg.norm(state[0:3] - ref_pos)
            vel_error = jnp.linalg.norm(state[3:6] - reference[step, 3:6])
            print(
                f"Step {step:4d}: pos_err={pos_error:.3f}m, "
                f"vel_err={vel_error:.3f}m/s, cost={cost:.2f}"
            )

    states = jnp.stack(states)
    actions_taken = jnp.stack(actions_taken)
    costs_history = jnp.array(costs_history)

    # Compute tracking performance
    pos_errors = jnp.linalg.norm(states[:-1, 0:3] - reference[:, 0:3], axis=1)
    vel_errors = jnp.linalg.norm(states[:-1, 3:6] - reference[:, 3:6], axis=1)

    print(f"\nTracking performance:")
    print(f"  Mean position error: {jnp.mean(pos_errors):.4f}m")
    print(f"  Max position error: {jnp.max(pos_errors):.4f}m")
    print(f"  RMS position error: {jnp.sqrt(jnp.mean(pos_errors**2)):.4f}m")
    print(f"  Mean velocity error: {jnp.mean(vel_errors):.4f}m/s")
    print(f"  Total cost: {jnp.sum(costs_history):.2f}")

    if visualize:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            time = jnp.arange(len(states)) * dt

            fig = plt.figure(figsize=(16, 10))

            # 3D trajectory
            ax1 = fig.add_subplot(2, 3, 1, projection="3d")
            ax1.plot(
                reference[:, 0], reference[:, 1], reference[:, 2],
                "k--", linewidth=2, alpha=0.5, label="Reference"
            )
            ax1.plot(
                states[:, 0], states[:, 1], states[:, 2],
                "b-", linewidth=1, label="Actual"
            )
            ax1.scatter(
                states[0, 0], states[0, 1], states[0, 2],
                c="g", s=100, marker="o", label="Start"
            )
            ax1.set_xlabel("X (m)")
            ax1.set_ylabel("Y (m)")
            ax1.set_zlabel("Z (m)")
            ax1.legend()
            ax1.set_title("3D Trajectory")
            ax1.view_init(elev=20, azim=45)

            # XY trajectory (top view)
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(
                reference[:, 0], reference[:, 1],
                "k--", linewidth=2, alpha=0.5, label="Reference"
            )
            ax2.plot(states[:, 0], states[:, 1], "b-", linewidth=1, label="Actual")
            ax2.scatter(states[0, 0], states[0, 1], c="g", s=100, marker="o", label="Start")
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            ax2.axis("equal")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title("XY Trajectory (Top View)")

            # Tracking errors
            time_err = time[:-1]
            ax3 = plt.subplot(2, 3, 3)
            ax3.plot(time_err, pos_errors, label="Position error", color="C0")
            ax3.set_ylabel("Position Error (m)")
            ax3.set_xlabel("Time (s)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_title("Tracking Error")

            # Position components
            ax4 = plt.subplot(2, 3, 4)
            ax4.plot(time, states[:, 0], "b-", label="px (actual)")
            ax4.plot(time[:-1], reference[:, 0], "b--", alpha=0.5, label="px (ref)")
            ax4.plot(time, states[:, 1], "r-", label="py (actual)")
            ax4.plot(time[:-1], reference[:, 1], "r--", alpha=0.5, label="py (ref)")
            ax4.set_ylabel("Position (m)")
            ax4.set_xlabel("Time (s)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_title("XY Position Tracking")

            # Altitude
            ax5 = plt.subplot(2, 3, 5)
            ax5.plot(time, states[:, 2], "b-", label="pz (actual)")
            ax5.plot(time[:-1], reference[:, 2], "k--", alpha=0.5, label="pz (ref)")
            ax5.set_ylabel("Z Position (m)")
            ax5.set_xlabel("Time (s)")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_title("Altitude Tracking")

            # Control inputs
            time_actions = jnp.arange(len(actions_taken)) * dt
            ax6 = plt.subplot(2, 3, 6)
            ax6.plot(time_actions, actions_taken[:, 0], label="Thrust", color="C3")
            ax6.axhline(mass * gravity, color="k", linestyle="--", alpha=0.3, label="Hover")
            ax6.set_ylabel("Thrust (N)")
            ax6.set_xlabel("Time (s)")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_title("Control Input")

            plt.tight_layout()

            # Save to docs/media directory
            output_dir = Path(__file__).parent.parent / "docs" / "media"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "quadrotor_circle_mppi.png"

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved to {output_path}")
            plt.show()

        except ImportError:
            print("\nMatplotlib not available for visualization")

    return states, actions_taken, costs_history, reference


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quadrotor circle tracking with MPPI")
    parser.add_argument("--steps", type=int, default=1000, help="Number of control steps")
    parser.add_argument("--samples", type=int, default=1000, help="Number of MPPI samples")
    parser.add_argument("--horizon", type=int, default=30, help="MPPI planning horizon")
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_", help="MPPI temperature")
    parser.add_argument("--radius", type=float, default=3.0, help="Circle radius (m)")
    parser.add_argument("--period", type=float, default=15.0, help="Circle period (s)")
    parser.add_argument("--visualize", action="store_true", help="Plot results with matplotlib")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    states, actions, costs, reference = run_quadrotor_circle(
        num_steps=args.steps,
        num_samples=args.samples,
        horizon=args.horizon,
        lambda_=args.lambda_,
        radius=args.radius,
        period=args.period,
        visualize=args.visualize,
        seed=args.seed,
    )
