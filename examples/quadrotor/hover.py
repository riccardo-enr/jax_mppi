"""Quadrotor hover control example using JAX MPPI.

This example demonstrates using MPPI to stabilize a quadrotor at a fixed
hover position. The quadrotor starts from a displaced position and must
stabilize to the setpoint with zero velocity.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp

from jax_mppi import mppi
from jax_mppi.costs.quadrotor import create_hover_cost, create_terminal_cost
from jax_mppi.dynamics.quadrotor import create_quadrotor_dynamics


def run_quadrotor_hover(
    num_steps: int = 500,
    num_samples: int = 1000,
    horizon: int = 30,
    lambda_: float = 1.0,
    visualize: bool = False,
    seed: int = 0,
):
    """Run MPPI on quadrotor hover control task.

    Args:
        num_steps: Number of control steps to simulate
        num_samples: Number of MPPI samples (K)
        horizon: MPPI planning horizon (T)
        lambda_: Temperature parameter for MPPI
        visualize: Whether to plot results (requires matplotlib)
        seed: Random seed

    Returns:
        states: (num_steps+1, 13) trajectory of states
        actions: (num_steps, 4) trajectory of actions
        costs: (num_steps,) trajectory of costs
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

    # Hover setpoint (5m altitude in NED frame: z = -5.0)
    hover_position = jnp.array([0.0, 0.0, -5.0])
    hover_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])  # level flight

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
    Q_pos = jnp.eye(3) * 100.0  # Position error weight
    Q_vel = jnp.eye(3) * 10.0   # Velocity weight
    Q_att = jnp.eye(4) * 5.0    # Attitude weight
    R = jnp.diag(jnp.array([0.01, 0.1, 0.1, 0.1]))  # Control effort

    # Running cost
    running_cost_fn = create_hover_cost(
        Q_pos, Q_vel, Q_att, R, hover_position, hover_quaternion
    )

    # Terminal cost
    Q_pos_terminal = Q_pos * 10.0
    Q_vel_terminal = Q_vel * 10.0
    Q_att_terminal = Q_att * 5.0

    terminal_cost_fn = create_terminal_cost(
        Q_pos_terminal, Q_vel_terminal, Q_att_terminal,
        hover_position, hover_quaternion
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

    # Initial state: displaced from hover position with some velocity
    # Start at [2, 1, -3] (2m altitude, displaced in xy)
    state = jnp.array([
        2.0, 1.0, -3.0,       # position (displaced)
        0.5, 0.3, 0.0,        # velocity (small initial velocity)
        1.0, 0.0, 0.0, 0.0,   # quaternion (level)
        0.0, 0.0, 0.0         # angular velocity (zero)
    ])

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
            dynamics=dynamics,
            running_cost=running_cost_fn,
            terminal_cost=terminal_cost_fn,
            shift=True,
        )
    )

    print("Running MPPI on quadrotor hover control...")
    print(f"  Samples: {num_samples}, Horizon: {horizon}, Lambda: {lambda_}")
    print(f"  Target: {hover_position}, Initial: {state[0:3]}")
    print(f"  Control rate: {1/dt:.0f} Hz")

    # Control loop
    for step in range(num_steps):
        # Compute optimal action
        action, mppi_state = command_fn(mppi_state, state)

        # Apply action to environment
        state = dynamics(state, action)

        # Compute cost
        cost = running_cost_fn(state, action)

        # Store
        states.append(state)
        actions_taken.append(action)
        costs_history.append(cost)

        # Print progress
        if step % 50 == 0:
            pos_error = jnp.linalg.norm(state[0:3] - hover_position)
            vel_mag = jnp.linalg.norm(state[3:6])
            print(
                f"Step {step:3d}: pos_error={pos_error:.3f}m, "
                f"vel={vel_mag:.3f}m/s, cost={cost:.2f}"
            )

    states = jnp.stack(states)
    actions_taken = jnp.stack(actions_taken)
    costs_history = jnp.array(costs_history)

    # Compute performance metrics
    pos_errors = jnp.linalg.norm(states[:, 0:3] - hover_position, axis=1)
    vel_magnitudes = jnp.linalg.norm(states[:, 3:6], axis=1)

    # Settling criteria: within 0.1m of target and velocity < 0.05 m/s
    settled_mask = (pos_errors < 0.1) & (vel_magnitudes < 0.05)
    if jnp.any(settled_mask):
        settling_time = jnp.argmax(settled_mask) * dt
        print(f"\nSettling time: {settling_time:.2f}s")
    else:
        print("\nDid not settle within simulation time")

    final_pos_error = pos_errors[-1]
    final_vel = vel_magnitudes[-1]

    print(f"Final position error: {final_pos_error:.4f}m")
    print(f"Final velocity: {final_vel:.4f}m/s")
    print(f"Total cost: {jnp.sum(costs_history):.2f}")
    print(f"Average cost (last 50 steps): {jnp.mean(costs_history[-50:]):.2f}")

    if visualize:
        try:
            import matplotlib.pyplot as plt

            time = jnp.arange(len(states)) * dt


            # Position tracking
            ax1 = plt.subplot(3, 3, 1)
            ax1.plot(time, states[:, 0], label="px", color="C0")
            ax1.axhline(hover_position[0], color="C0", ls="--", alpha=0.5)
            ax1.set_ylabel("X Position (m)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title("Position Tracking")

            ax2 = plt.subplot(3, 3, 4)
            ax2.plot(time, states[:, 1], label="py", color="C1")
            ax2.axhline(hover_position[1], color="C1", ls="--", alpha=0.5)
            ax2.set_ylabel("Y Position (m)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            ax3 = plt.subplot(3, 3, 7)
            ax3.plot(time, states[:, 2], label="pz", color="C2")
            ax3.axhline(hover_position[2], color="C2", ls="--", alpha=0.5)
            ax3.set_ylabel("Z Position (m)")
            ax3.set_xlabel("Time (s)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Velocity
            ax4 = plt.subplot(3, 3, 2)
            ax4.plot(time, states[:, 3], label="vx", color="C0")
            ax4.plot(time, states[:, 4], label="vy", color="C1")
            ax4.plot(time, states[:, 5], label="vz", color="C2")
            ax4.axhline(0, color="k", linestyle="--", alpha=0.3)
            ax4.set_ylabel("Velocity (m/s)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_title("Velocity")

            # Angular velocity
            ax5 = plt.subplot(3, 3, 5)
            ax5.plot(time, states[:, 10], label="ωx", color="C0")
            ax5.plot(time, states[:, 11], label="ωy", color="C1")
            ax5.plot(time, states[:, 12], label="ωz", color="C2")
            ax5.axhline(0, color="k", linestyle="--", alpha=0.3)
            ax5.set_ylabel("Angular Velocity (rad/s)")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_title("Angular Velocity")

            # Control inputs
            time_actions = jnp.arange(len(actions_taken)) * dt

            ax6 = plt.subplot(3, 3, 3)
            ax6.plot(time_actions, actions_taken[:, 0], label="T", color="C3")
            ax6.axhline(mass * gravity, color="k", alpha=0.3, label="H")
            ax6.set_ylabel("Thrust (N)")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_title("Control Inputs")

            ax7 = plt.subplot(3, 3, 6)
            ax7.plot(time_actions, actions_taken[:, 1], label="wx", color="C0")
            ax7.plot(time_actions, actions_taken[:, 2], label="wy", color="C1")
            ax7.plot(time_actions, actions_taken[:, 3], label="wz", color="C2")
            ax7.axhline(0, color="k", ls="--", alpha=0.3)
            ax7.set_ylabel("Angular Rate Cmd (rad/s)")
            ax7.legend()
            ax7.grid(True, alpha=0.3)

            # Tracking errors
            ax8 = plt.subplot(3, 3, 8)
            ax8.plot(time, pos_errors, label="Position error", color="C4")
            ax8.axhline(0.1, color="r", alpha=0.5, label="Thresh")
            ax8.set_ylabel("Position Error (m)")
            ax8.set_xlabel("Time (s)")
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            ax8.set_title("Tracking Error")

            # Cost
            ax9 = plt.subplot(3, 3, 9)
            ax9.plot(time_actions, costs_history, label="Cost", color="C5")
            ax9.set_ylabel("Cost")
            ax9.set_xlabel("Time (s)")
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            ax9.set_title("Cost over Time")
            ax9.set_yscale("log")

            plt.tight_layout()

            # Save to docs/media directory
            output_dir = Path(__file__).parent.parent / "docs" / "media"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "quadrotor_hover_mppi.png"

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved to {output_path}")
            plt.show()

        except ImportError:
            print("\nMatplotlib not available for visualization")

    return states, actions_taken, costs_history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(desc="Quadrotor hover")
    parser.add_argument("--steps", type=int, default=500, help="Steps")
    parser.add_argument("--samples", type=int, default=1000, help="K")
    parser.add_argument("--horizon", type=int, default=30, help="T")
    parser.add_argument("--lambda", type=float, default=1.0, help="Lambda")
    parser.add_argument("--visualize", action="store_true", help="Plot")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    states, actions, costs = run_quadrotor_hover(
        num_steps=args.steps,
        num_samples=args.samples,
        horizon=args.horizon,
        lambda_=args.lambda_,
        visualize=args.visualize,
        seed=args.seed,
    )
