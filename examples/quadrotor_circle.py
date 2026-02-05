"""Quadrotor circular trajectory following example using JAX MPPI.

This example demonstrates using MPPI to track a circular trajectory.
The quadrotor follows a circle at constant altitude, demonstrating
trajectory tracking capabilities.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

from examples.quadrotor.trajectories import (
    compute_trajectory_metrics,
    generate_circle_trajectory,
)
from jax_mppi import mppi
from jax_mppi.costs.quadrotor import create_terminal_cost
from jax_mppi.dynamics.quadrotor import create_quadrotor_dynamics


def create_tracking_cost(
    Q_pos: jax.Array,
    Q_vel: jax.Array,
    R: jax.Array,
):
    """Create trajectory tracking cost using time index to look up reference.

    Args:
        Q_pos: Position error weight matrix
        Q_vel: Velocity error weight matrix
        R: Control effort weight matrix

    Returns:
        Cost function builder that takes reference_horizon
    """

    def cost_builder(reference_horizon: jax.Array):
        def cost_fn(state: jax.Array, action: jax.Array, t: int) -> jax.Array:
            # Look up reference for current relative time t
            ref = reference_horizon[t]
            ref_pos = ref[0:3]
            ref_vel = ref[3:6]

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

    return cost_builder


def run_quadrotor_circle(
    num_steps: int = 1000,
    num_samples: int = 2000,  # Tuned: Increased samples
    horizon: int = 30,  # Tuned: Increased horizon
    lambda_: float = 0.1,  # Tuned: Lower temperature (exploitation)
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
    nu = 4  # [thrust, wx_cmd, wy_cmd, wz_cmd]

    # Physical parameters
    mass = 1.0  # kg
    gravity = 9.81  # m/s^2
    dt = 0.02  # 50 Hz control rate

    # Generate circular reference trajectory
    height = -5.0  # 5m altitude in NED
    duration = num_steps * dt
    # Generate reference with extra horizon padding for lookahead
    # Note: We need enough reference for the scan loop + lookahead at the end
    reference = generate_circle_trajectory(
        radius=radius,
        height=height,
        period=period,
        duration=duration + horizon * dt * 2.0,  # Extra safety margin
        dt=dt,
    )

    print("\nTrajectory metrics:")
    metrics = compute_trajectory_metrics(reference[:num_steps], dt)
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.3f}")

    # Create dynamics
    u_min = jnp.array([0.0, -3.0, -3.0, -3.0])
    u_max = jnp.array([2.0 * mass * gravity, 3.0, 3.0, 3.0])

    dynamics_fn = create_quadrotor_dynamics(
        dt=dt,
        mass=mass,
        gravity=gravity,
        tau_omega=0.05,
        u_min=u_min,
        u_max=u_max,
    )

    def dynamics(state, action, t):
        return dynamics_fn(state, action)

    # Cost function weights (Tuned)
    Q_pos = jnp.eye(3) * 500.0  # Increased position weight
    Q_vel = jnp.eye(3) * 50.0  # Increased velocity weight
    R = jnp.diag(jnp.array([0.01, 0.1, 0.1, 0.1]))  # Control effort

    # Terminal cost (track last reference point)
    goal_position = reference[-1, 0:3]
    goal_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

    terminal_cost_fn = create_terminal_cost(
        Q_pos * 10.0,
        Q_vel * 10.0,
        jnp.eye(4) * 5.0,
        goal_position,
        goal_quaternion,
    )

    # Noise covariance (exploration in control space)
    noise_sigma = jnp.diag(
        jnp.array([2.0, 0.5, 0.5, 0.5])
    )  # Tuned: Reduced noise

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
        step_dependent_dynamics=True,  # Enable passing t to cost function
    )

    # Create cost builder
    cost_builder = create_tracking_cost(Q_pos, Q_vel, R)

    # Initial state: start at first reference point
    state = jnp.array([
        reference[0, 0],
        reference[0, 1],
        reference[0, 2],  # start position
        0.0,
        0.0,
        0.0,  # zero initial velocity
        1.0,
        0.0,
        0.0,
        0.0,  # level quaternion
        0.0,
        0.0,
        0.0,  # zero angular velocity
    ])

    print("\nRunning MPPI on quadrotor circular trajectory tracking...")
    print(f"  Samples: {num_samples}, Horizon: {horizon}, Lambda: {lambda_}")
    print(f"  Circle: radius={radius}m, period={period}s, altitude={-height}m")
    print(f"  Control rate: {1 / dt:.0f} Hz")

    # ---------------------------------------------------------
    # JIT-compiled Simulation Loop using jax.lax.scan
    # ---------------------------------------------------------

    def simulation_step(carry, step_idx):
        mppi_state, state = carry

        # Slice reference for current horizon
        # Use dynamic_slice for JIT compatibility
        ref_horizon = jax.lax.dynamic_slice(
            reference, (step_idx, 0), (horizon, 6)
        )

        # Build cost function closing over ref_horizon
        running_cost_fn = cost_builder(ref_horizon)

        # Compute action
        action, new_mppi_state = mppi.command(
            config=config,
            mppi_state=mppi_state,
            current_obs=state,
            dynamics=dynamics,
            running_cost=running_cost_fn,
            terminal_cost=terminal_cost_fn,
            shift=True,
        )

        # Apply action
        next_state = dynamics_fn(state, action)

        # Calculate cost for logging (using first point of horizon)
        current_ref_pos = ref_horizon[0, 0:3]
        current_ref_vel = ref_horizon[0, 3:6]
        pos_error = state[0:3] - current_ref_pos
        cost_pos = pos_error @ Q_pos @ pos_error
        vel_error = state[3:6] - current_ref_vel
        cost_vel = vel_error @ Q_vel @ vel_error
        cost_control = action @ R @ action
        cost = cost_pos + cost_vel + cost_control

        return (new_mppi_state, next_state), (next_state, action, cost)

    # Run the scan
    step_indices = jnp.arange(num_steps)
    init_carry = (mppi_state, state)

    # JIT the entire loop
    scan_fn = jax.jit(lambda c, x: jax.lax.scan(simulation_step, c, x))

    print("Compiling and running simulation loop...")
    import time

    t0 = time.time()
    _, (states_traj, actions_traj, costs_traj) = scan_fn(
        init_carry, step_indices
    )
    # Block to ensure completion
    states_traj.block_until_ready()
    t1 = time.time()

    print(f"Simulation complete in {t1 - t0:.4f}s")

    # Prepend initial state
    states = jnp.concatenate([state[None, :], states_traj], axis=0)
    actions_taken = actions_traj
    costs_history = costs_traj

    # Compute tracking performance
    # Use matching reference slice
    ref_match = reference[:num_steps]
    pos_errors = jnp.linalg.norm(states[:-1, 0:3] - ref_match[:, 0:3], axis=1)
    vel_errors = jnp.linalg.norm(states[:-1, 3:6] - ref_match[:, 3:6], axis=1)

    print("\nTracking performance:")
    print(f"  Mean position error: {jnp.mean(pos_errors):.4f}m")
    print(f"  Max position error: {jnp.max(pos_errors):.4f}m")
    print(f"  RMS position error: {jnp.sqrt(jnp.mean(pos_errors**2)):.4f}m")
    print(f"  Mean velocity error: {jnp.mean(vel_errors):.4f}m/s")
    print(f"  Total cost: {jnp.sum(costs_history):.2f}")

    if visualize:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            time_axis = jnp.arange(len(states)) * dt

            fig = plt.figure(figsize=(16, 10))

            # 3D trajectory
            ax1 = fig.add_subplot(2, 3, 1, projection="3d")
            ax1.plot(
                ref_match[:, 0],
                ref_match[:, 1],
                ref_match[:, 2],
                "k--",
                linewidth=2,
                alpha=0.5,
                label="Reference",
            )
            ax1.plot(
                states[:, 0],
                states[:, 1],
                states[:, 2],
                "b-",
                linewidth=1,
                label="Actual",
            )
            ax1.scatter(
                states[0, 0],
                states[0, 1],
                states[0, 2],
                c="g",
                s=100,
                marker="o",
                label="Start",
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
                ref_match[:, 0],
                ref_match[:, 1],
                "k--",
                linewidth=2,
                alpha=0.5,
                label="Reference",
            )
            ax2.plot(
                states[:, 0], states[:, 1], "b-", linewidth=1, label="Actual"
            )
            ax2.scatter(
                states[0, 0],
                states[0, 1],
                c="g",
                s=100,
                marker="o",
                label="Start",
            )
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            ax2.axis("equal")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title("XY Trajectory (Top View)")

            # Tracking errors
            time_err = time_axis[:-1]
            ax3 = plt.subplot(2, 3, 3)
            ax3.plot(time_err, pos_errors, label="Position error", color="C0")
            ax3.set_ylabel("Position Error (m)")
            ax3.set_xlabel("Time (s)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_title("Tracking Error")

            # Position components
            ax4 = plt.subplot(2, 3, 4)
            ax4.plot(time_axis, states[:, 0], "b-", label="px (actual)")
            ax4.plot(
                time_axis[:-1],
                ref_match[:, 0],
                "b--",
                alpha=0.5,
                label="px (ref)",
            )
            ax4.plot(time_axis, states[:, 1], "r-", label="py (actual)")
            ax4.plot(
                time_axis[:-1],
                ref_match[:, 1],
                "r--",
                alpha=0.5,
                label="py (ref)",
            )
            ax4.set_ylabel("Position (m)")
            ax4.set_xlabel("Time (s)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_title("XY Position Tracking")

            # Altitude
            ax5 = plt.subplot(2, 3, 5)
            ax5.plot(time_axis, states[:, 2], "b-", label="pz (actual)")
            ax5.plot(
                time_axis[:-1],
                ref_match[:, 2],
                "k--",
                alpha=0.5,
                label="pz (ref)",
            )
            ax5.set_ylabel("Z Position (m)")
            ax5.set_xlabel("Time (s)")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_title("Altitude Tracking")

            # Control inputs
            time_actions = jnp.arange(len(actions_taken)) * dt
            ax6 = plt.subplot(2, 3, 6)
            ax6.plot(
                time_actions, actions_taken[:, 0], label="Thrust", color="C3"
            )
            ax6.axhline(
                mass * gravity,
                color="k",
                linestyle="--",
                alpha=0.3,
                label="Hover",
            )
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

    return states, actions_taken, costs_history, ref_match


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quadrotor circle tracking with MPPI"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of control steps"
    )
    parser.add_argument(
        "--samples", type=int, default=2000, help="Number of MPPI samples"
    )
    parser.add_argument(
        "--horizon", type=int, default=50, help="MPPI planning horizon"
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.001,
        dest="lambda_",
        help="MPPI temperature",
    )
    parser.add_argument(
        "--radius", type=float, default=3.0, help="Circle radius (m)"
    )
    parser.add_argument(
        "--period", type=float, default=15.0, help="Circle period (s)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Plot results with matplotlib"
    )
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
