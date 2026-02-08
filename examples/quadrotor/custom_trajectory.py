"""Quadrotor custom trajectory following using waypoints.

This example demonstrates following user-defined trajectories through waypoints
with smooth interpolation. Users can specify a sequence of waypoints and the
quadrotor will generate and track a smooth trajectory through them using
cubic Hermite splines.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp

from examples.quadrotor.trajectories import (
    compute_trajectory_metrics,
    generate_waypoint_trajectory,
)
from jax_mppi import mppi
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
        cost_vel = vel_error @ Q_vel @ vel_error

        cost_control = action @ R @ action

        return cost_pos + cost_vel + cost_control

    return cost_fn


def run_quadrotor_custom_trajectory(
    waypoints: jax.Array,
    segment_duration: float = 5.0,
    num_samples: int = 1000,
    horizon: int = 30,
    lambda_: float = 1.0,
    visualize: bool = False,
    seed: int = 0,
):
    """Follow custom waypoint-based trajectory.

    Args:
        waypoints: Waypoint positions [N x 3] in NED frame
        segment_duration: Time between waypoints (s)
        num_samples: Number of MPPI samples
        horizon: Planning horizon
        lambda_: Temperature parameter
        visualize: Whether to plot results
        seed: Random seed

    Returns:
        states: (num_steps+1, 13) trajectory
        actions: (num_steps, 4) control inputs
        costs: (num_steps,) running costs
        reference: (num_steps, 6) reference trajectory
    """
    key = jax.random.PRNGKey(seed)

    nx, nu = 13, 4
    mass, gravity = 1.0, 9.81
    dt = 0.02  # 50 Hz

    # Generate smooth trajectory through waypoints
    reference = generate_waypoint_trajectory(
        waypoints=waypoints,
        segment_duration=segment_duration,
        dt=dt,
    )

    num_steps = reference.shape[0]
    duration = num_steps * dt

    print("=" * 60)
    print("QUADROTOR CUSTOM WAYPOINT TRAJECTORY")
    print("=" * 60)
    print(f"\nWaypoints: {waypoints.shape[0]} points")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i}: [{wp[0]:6.2f}, {wp[1]:6.2f}, {wp[2]:6.2f}] m")

    print(f"\nSegment duration: {segment_duration:.1f}s")
    print(f"Total duration: {duration:.1f}s ({num_steps} steps @ {1/dt:.0f} Hz)")
    print(f"Control: {num_samples} samples, horizon={horizon}, λ={lambda_}")

    metrics = compute_trajectory_metrics(reference, dt)
    print("\nTrajectory metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.3f}")

    # Create dynamics
    u_min = jnp.array([0.0, -6.0, -6.0, -6.0])
    u_max = jnp.array([4.0 * mass * gravity, 6.0, 6.0, 6.0])

    dynamics = create_quadrotor_dynamics(
        dt=dt, mass=mass, gravity=gravity, tau_omega=0.05,
        u_min=u_min, u_max=u_max
    )

    # Cost weights
    Q_pos = jnp.eye(3) * 80.0
    Q_vel = jnp.eye(3) * 15.0
    R = jnp.diag(jnp.array([0.01, 0.1, 0.1, 0.1]))

    # Terminal cost
    goal_position = waypoints[-1]
    goal_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])
    terminal_cost_fn = create_terminal_cost(
        Q_pos * 10.0, Q_vel * 10.0, jnp.eye(4) * 5.0,
        goal_position, goal_quaternion
    )

    # Noise covariance
    noise_sigma = jnp.diag(jnp.array([5.0, 1.0, 1.0, 1.0]))

    # Create MPPI controller
    config, mppi_state = mppi.create(
        nx=nx, nu=nu, noise_sigma=noise_sigma,
        num_samples=num_samples, horizon=horizon,
        lambda_=lambda_, u_min=u_min, u_max=u_max, key=key
    )

    # Initial state (start at first waypoint)
    state = jnp.array([
        waypoints[0, 0], waypoints[0, 1], waypoints[0, 2],
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])

    # Storage
    states, actions_taken, costs_history = [state], [], []

    print("\nRunning MPPI controller...")

    # Control loop
    for step in range(num_steps):
        # Create cost for current reference
        ref_pos = reference[step, 0:3]
        ref_vel = reference[step, 3:6]
        running_cost_fn = create_tracking_cost(Q_pos, Q_vel, R, ref_pos, ref_vel)

        # Compute action
        action, mppi_state = mppi.command(
            config=config,
            mppi_state=mppi_state,
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

    states = jnp.stack(states)
    actions_taken = jnp.stack(actions_taken)
    costs_history = jnp.array(costs_history)

    # Compute tracking performance
    pos_errors = jnp.linalg.norm(states[:-1, 0:3] - reference[:, 0:3], axis=1)
    vel_errors = jnp.linalg.norm(states[:-1, 3:6] - reference[:, 3:6], axis=1)

    print("\nTracking performance:")
    print(f"  Mean position error: {jnp.mean(pos_errors):.4f}m")
    print(f"  Max position error: {jnp.max(pos_errors):.4f}m")
    print(f"  RMS position error: {jnp.sqrt(jnp.mean(pos_errors**2)):.4f}m")
    print(f"  Mean velocity error: {jnp.mean(vel_errors):.4f}m/s")
    print(f"  Total cost: {jnp.sum(costs_history):.2f}")

    # Check waypoint passage
    print("\nWaypoint passage errors:")
    steps_per_segment = int(segment_duration / dt)
    for i in range(waypoints.shape[0]):
        step_idx = min(i * steps_per_segment, num_steps - 1)
        wp_error = jnp.linalg.norm(states[step_idx, 0:3] - waypoints[i])
        print(f"  WP{i}: {wp_error:.4f}m")

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
            # Plot waypoints
            ax1.scatter(
                waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                c="r", s=100, marker="o", label="Waypoints", zorder=10
            )
            # Label waypoints
            for i, wp in enumerate(waypoints):
                ax1.text(wp[0], wp[1], wp[2], f"  WP{i}", fontsize=9)

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
            ax2.scatter(
                waypoints[:, 0], waypoints[:, 1],
                c="r", s=100, marker="o", label="Waypoints", zorder=10
            )
            for i, wp in enumerate(waypoints):
                ax2.annotate(
                    f"WP{i}", (wp[0], wp[1]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9
                )
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            ax2.axis("equal")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title("XY Trajectory (Top View)")

            # Altitude profile
            ax3 = plt.subplot(2, 3, 3)
            ax3.plot(time, states[:, 2], "b-", label="Actual")
            ax3.plot(time[:-1], reference[:, 2], "k--", alpha=0.5, label="Reference")
            # Mark waypoint times
            for i in range(waypoints.shape[0]):
                t_wp = i * segment_duration
                if t_wp <= time[-1]:
                    ax3.axvline(t_wp, color="r", linestyle=":", alpha=0.5)
                    ax3.plot(t_wp, waypoints[i, 2], "ro", markersize=8)
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Z Position (m)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_title("Altitude Profile")

            # Position errors
            ax4 = plt.subplot(2, 3, 4)
            ax4.plot(time[:-1], pos_errors, "b-", label="Position error")
            for i in range(1, waypoints.shape[0]):
                t_wp = i * segment_duration
                if t_wp <= time[-1]:
                    ax4.axvline(t_wp, color="r", linestyle=":", alpha=0.3)
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Position Error (m)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_title("Tracking Error")

            # Velocity components
            ax5 = plt.subplot(2, 3, 5)
            vel_mag = jnp.linalg.norm(states[:, 3:6], axis=1)
            ref_vel_mag = jnp.linalg.norm(reference[:, 3:6], axis=1)
            ax5.plot(time, vel_mag, "b-", label="Actual")
            ax5.plot(time[:-1], ref_vel_mag, "k--", alpha=0.5, label="Reference")
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("Velocity Magnitude (m/s)")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_title("Velocity Profile")

            # Control inputs
            time_actions = jnp.arange(len(actions_taken)) * dt
            ax6 = plt.subplot(2, 3, 6)
            ax6.plot(time_actions, actions_taken[:, 0], label="Thrust", color="C3")
            ax6.axhline(mass * gravity, color="k", linestyle="--", alpha=0.3, label="Hover")
            ax6.set_xlabel("Time (s)")
            ax6.set_ylabel("Thrust (N)")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_title("Control Input")

            plt.tight_layout()

            # Save
            output_dir = Path(__file__).parent.parent / "docs" / "media"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "quadrotor_custom_trajectory.png"

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved to {output_path}")
            plt.show()

        except ImportError:
            print("\nMatplotlib not available for visualization")

    return states, actions_taken, costs_history, reference


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quadrotor custom waypoint trajectory following"
    )
    parser.add_argument(
        "--waypoints",
        type=str,
        default=None,
        help='Waypoints as "x1,y1,z1;x2,y2,z2;..." (in NED frame)',
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=5.0,
        help="Time between waypoints (s)",
    )
    parser.add_argument("--samples", type=int, default=1000, help="Number of MPPI samples")
    parser.add_argument("--horizon", type=int, default=30, help="Planning horizon")
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_", help="Temperature")
    parser.add_argument("--visualize", action="store_true", help="Plot results")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    # Parse waypoints or use default
    if args.waypoints:
        # Parse from string: "x1,y1,z1;x2,y2,z2;..."
        waypoint_list = []
        for wp_str in args.waypoints.split(";"):
            coords = [float(x) for x in wp_str.split(",")]
            if len(coords) != 3:
                raise ValueError(f"Invalid waypoint: {wp_str}")
            waypoint_list.append(coords)
        waypoints = jnp.array(waypoint_list)
    else:
        # Default: square pattern at 5m altitude
        waypoints = jnp.array([
            [0.0, 0.0, -2.0],   # Start low
            [5.0, 0.0, -5.0],   # Climb and move forward
            [5.0, 5.0, -5.0],   # Turn right
            [0.0, 5.0, -5.0],   # Turn back
            [0.0, 0.0, -2.0],   # Return home and descend
        ])

    states, actions, costs, reference = run_quadrotor_custom_trajectory(
        waypoints=waypoints,
        segment_duration=args.segment_duration,
        num_samples=args.samples,
        horizon=args.horizon,
        lambda_=args.lambda_,
        visualize=args.visualize,
        seed=args.seed,
    )
