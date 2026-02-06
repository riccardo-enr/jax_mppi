"""Quadrotor hover control comparison: MPPI vs SMPPI vs KMPPI.

This example compares three MPPI variants on the quadrotor hover task:
- MPPI: Standard Model Predictive Path Integral control
- SMPPI: Smooth MPPI with smoothness constraints on control trajectories
- KMPPI: Kernel MPPI with kernel-based trajectory interpolation
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp

from jax_mppi import kmppi, mppi, smppi
from jax_mppi.costs.quadrotor import create_hover_cost, create_terminal_cost
from jax_mppi.dynamics.quadrotor import create_quadrotor_dynamics


def run_controller(
    controller_type: str,
    num_steps: int,
    num_samples: int,
    horizon: int,
    lambda_: float,
    seed: int,
    mass: float,
    gravity: float,
    dt: float,
    hover_position: jax.Array,
    hover_quaternion: jax.Array,
    initial_state: jax.Array,
    dynamics,
    running_cost_fn,
    terminal_cost_fn,
    u_min: jax.Array,
    u_max: jax.Array,
    noise_sigma: jax.Array,
):
    """Run a single controller variant.

    Returns:
        states, actions, costs, controller_state
    """
    key = jax.random.PRNGKey(seed)
    nx = 13
    nu = 4

    if controller_type == "mppi":
        # Standard MPPI
        config, controller_state = mppi.create(
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
                dynamics=dynamics,
                running_cost=running_cost_fn,
                terminal_cost=terminal_cost_fn,
                shift=True,
            )
        )

    elif controller_type == "smppi":
        # Smooth MPPI - operates in lifted velocity space
        # U represents rates: thrust_rate (N/s), angular_acceleration (rad/s²)
        # action_sequence = integral of U over time
        # Relationship: action(t+1) = action(t) + U * dt

        # Noise scaling for velocity space:
        # Since action_change = velocity * dt, to get same exploration magnitude:
        # var(action_change) = var(velocity) * dt²
        # So: var(velocity) = var(action) / dt²
        # noise_sigma is covariance, so divide by dt²
        velocity_noise_sigma = noise_sigma / (dt * dt)

        # Velocity bounds: limit rate of change of actions
        # Need generous bounds to allow fast response
        # Max action values: thrust=[0,40]N, angular=[-5,5]rad/s
        # Over horizon dt*30=0.6s, we need to traverse full range
        # So min velocity: 40N/0.6s ≈ 67 N/s, 10rad/s / 0.6s ≈ 17 rad/s²
        # Use 2-3x safety margin for faster convergence
        u_vel_min = jnp.array([-200.0, -50.0, -50.0, -50.0])
        u_vel_max = jnp.array([200.0, 50.0, 50.0, 50.0])

        config, controller_state = smppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=velocity_noise_sigma,
            num_samples=num_samples,
            horizon=horizon,
            lambda_=lambda_,
            u_min=u_vel_min,  # Velocity bounds (rates)
            u_max=u_vel_max,
            action_min=u_min,  # Final action bounds (same as MPPI)
            action_max=u_max,
            w_action_seq_cost=0.1,  # Lower smoothness weight for more aggressive control
            delta_t=dt,  # Integration timestep
            key=key,
        )

        command_fn = jax.jit(
            lambda state, obs: smppi.command(
                config=config,
                smppi_state=state,
                current_obs=obs,
                dynamics=dynamics,
                running_cost=running_cost_fn,
                terminal_cost=terminal_cost_fn,
                shift=True,
            )
        )

    elif controller_type == "kmppi":
        # Kernel MPPI
        config, controller_state, kernel = kmppi.create(
            nx=nx,
            nu=nu,
            noise_sigma=noise_sigma,
            num_samples=num_samples,
            horizon=horizon,
            lambda_=lambda_,
            u_min=u_min,
            u_max=u_max,
            num_support_pts=horizon // 2,  # Half horizon support points
            kernel=kmppi.RBFKernel(sigma=1.0),
            key=key,
        )

        command_fn = jax.jit(
            lambda state, obs: kmppi.command(
                config=config,
                kmppi_state=state,
                current_obs=obs,
                dynamics=dynamics,
                running_cost=running_cost_fn,
                kernel_fn=kernel,
                terminal_cost=terminal_cost_fn,
                shift=True,
            )
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    # Run control loop
    state = initial_state
    states = [state]
    actions_taken = []
    costs_history = []

    for step in range(num_steps):
        # Compute optimal action
        action, controller_state = command_fn(controller_state, state)

        # Apply action to environment
        state = dynamics(state, action)

        # Compute cost
        cost = running_cost_fn(state, action)

        # Store
        states.append(state)
        actions_taken.append(action)
        costs_history.append(cost)

    return (
        jnp.stack(states),
        jnp.stack(actions_taken),
        jnp.array(costs_history),
    )


def run_quadrotor_hover_comparison(
    num_steps: int = 500,
    num_samples: int = 1000,
    horizon: int = 30,
    lambda_: float = 1.0,
    visualize: bool = False,
    seed: int = 0,
):
    """Compare MPPI, SMPPI, and KMPPI on quadrotor hover control.

    Args:
        num_steps: Number of control steps to simulate
        num_samples: Number of MPPI samples (K)
        horizon: MPPI planning horizon (T)
        lambda_: Temperature parameter for MPPI
        visualize: Whether to plot results (requires matplotlib)
        seed: Random seed

    Returns:
        Dictionary with results for each controller
    """
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
    Q_vel = jnp.eye(3) * 10.0  # Velocity weight
    Q_att = jnp.eye(4) * 5.0  # Attitude weight
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
        Q_pos_terminal,
        Q_vel_terminal,
        Q_att_terminal,
        hover_position,
        hover_quaternion,
    )

    # Noise covariance (exploration in control space)
    noise_sigma = jnp.diag(jnp.array([5.0, 1.0, 1.0, 1.0]))

    # Initial state: displaced from hover position with some velocity
    initial_state = jnp.array([
        2.0,
        1.0,
        -3.0,  # position (displaced)
        0.5,
        0.3,
        0.0,  # velocity (small initial velocity)
        1.0,
        0.0,
        0.0,
        0.0,  # quaternion (level)
        0.0,
        0.0,
        0.0,  # angular velocity (zero)
    ])

    print("\n" + "=" * 70)
    print("Quadrotor Hover Control - Controller Comparison")
    print("=" * 70)
    print(f"Samples: {num_samples}, Horizon: {horizon}, Lambda: {lambda_}")
    print(f"Target: {hover_position}, Initial: {initial_state[0:3]}")
    print(f"Control rate: {1 / dt:.0f} Hz, Steps: {num_steps}")
    print("=" * 70)

    results = {}

    # Run each controller
    for controller_type in ["mppi", "smppi", "kmppi"]:
        print(f"\nRunning {controller_type.upper()}...")

        states, actions, costs = run_controller(
            controller_type=controller_type,
            num_steps=num_steps,
            num_samples=num_samples,
            horizon=horizon,
            lambda_=lambda_,
            seed=seed,
            mass=mass,
            gravity=gravity,
            dt=dt,
            hover_position=hover_position,
            hover_quaternion=hover_quaternion,
            initial_state=initial_state,
            dynamics=dynamics,
            running_cost_fn=running_cost_fn,
            terminal_cost_fn=terminal_cost_fn,
            u_min=u_min,
            u_max=u_max,
            noise_sigma=noise_sigma,
        )

        # Compute metrics
        pos_errors = jnp.linalg.norm(states[:, 0:3] - hover_position, axis=1)
        vel_magnitudes = jnp.linalg.norm(states[:, 3:6], axis=1)

        # Settling criteria
        settled_mask = (pos_errors < 0.1) & (vel_magnitudes < 0.05)
        if jnp.any(settled_mask):
            settling_time = jnp.argmax(settled_mask) * dt
        else:
            settling_time = float("inf")

        # Control smoothness (action differences)
        action_diffs = jnp.diff(actions, axis=0)
        action_smoothness = jnp.mean(jnp.linalg.norm(action_diffs, axis=1))

        results[controller_type] = {
            "states": states,
            "actions": actions,
            "costs": costs,
            "pos_errors": pos_errors,
            "vel_magnitudes": vel_magnitudes,
            "settling_time": settling_time,
            "final_pos_error": pos_errors[-1],
            "final_vel": vel_magnitudes[-1],
            "total_cost": jnp.sum(costs),
            "avg_cost_last50": jnp.mean(costs[-50:]),
            "mean_pos_error": jnp.mean(pos_errors),
            "max_pos_error": jnp.max(pos_errors),
            "action_smoothness": action_smoothness,
        }

        print(f"  Settling time: {settling_time:.2f}s")
        print(
            f"  Final position error: {results[controller_type]['final_pos_error']:.4f}m"
        )
        print(
            f"  Final velocity: {results[controller_type]['final_vel']:.4f}m/s"
        )
        print(
            f"  Mean position error: {results[controller_type]['mean_pos_error']:.4f}m"
        )
        print(f"  Total cost: {results[controller_type]['total_cost']:.2f}")
        print(f"  Control smoothness: {action_smoothness:.4f}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"{'Metric':<30} {'MPPI':>12} {'SMPPI':>12} {'KMPPI':>12}")
    print("-" * 70)

    metrics = [
        ("Settling time (s)", "settling_time", ".2f"),
        ("Final pos error (m)", "final_pos_error", ".4f"),
        ("Mean pos error (m)", "mean_pos_error", ".4f"),
        ("Max pos error (m)", "max_pos_error", ".4f"),
        ("Final velocity (m/s)", "final_vel", ".4f"),
        ("Total cost", "total_cost", ".1f"),
        ("Avg cost (last 50)", "avg_cost_last50", ".2f"),
        ("Control smoothness", "action_smoothness", ".4f"),
    ]

    for name, key, fmt in metrics:
        values = [results[ct][key] for ct in ["mppi", "smppi", "kmppi"]]
        value_strs = [
            f"{v:{fmt}}" if v != float("inf") else "N/A" for v in values
        ]
        print(
            f"{name:<30} {value_strs[0]:>12} {value_strs[1]:>12} {value_strs[2]:>12}"
        )

    print("=" * 70)

    if visualize:
        try:
            import matplotlib.pyplot as plt

            time = jnp.arange(num_steps + 1) * dt
            time_actions = jnp.arange(num_steps) * dt

            fig = plt.figure(figsize=(16, 12))

            colors = {"mppi": "C0", "smppi": "C1", "kmppi": "C2"}
            labels = {"mppi": "MPPI", "smppi": "SMPPI", "kmppi": "KMPPI"}

            # Position tracking (X, Y, Z)
            for i, coord in enumerate(["X", "Y", "Z"]):
                ax = plt.subplot(3, 4, i * 4 + 1)
                for ct in ["mppi", "smppi", "kmppi"]:
                    ax.plot(
                        time,
                        results[ct]["states"][:, i],
                        label=labels[ct],
                        color=colors[ct],
                        alpha=0.8,
                    )
                ax.axhline(
                    hover_position[i],
                    color="k",
                    linestyle="--",
                    alpha=0.5,
                    label="Target",
                )
                ax.set_ylabel(f"{coord} Position (m)")
                if i == 0:
                    ax.legend(loc="best")
                ax.grid(True, alpha=0.3)
                if i == 2:
                    ax.set_xlabel("Time (s)")

            # Velocity magnitude
            ax = plt.subplot(3, 4, 2)
            for ct in ["mppi", "smppi", "kmppi"]:
                ax.plot(
                    time,
                    results[ct]["vel_magnitudes"],
                    label=labels[ct],
                    color=colors[ct],
                    alpha=0.8,
                )
            ax.axhline(0, color="k", linestyle="--", alpha=0.3)
            ax.set_ylabel("Velocity (m/s)")
            ax.set_title("Velocity Magnitude")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            # Position error
            ax = plt.subplot(3, 4, 6)
            for ct in ["mppi", "smppi", "kmppi"]:
                ax.plot(
                    time,
                    results[ct]["pos_errors"],
                    label=labels[ct],
                    color=colors[ct],
                    alpha=0.8,
                )
            ax.axhline(
                0.1,
                color="r",
                linestyle="--",
                alpha=0.5,
                label="Settling threshold",
            )
            ax.set_ylabel("Position Error (m)")
            ax.set_title("Tracking Error")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            # Cost
            ax = plt.subplot(3, 4, 10)
            for ct in ["mppi", "smppi", "kmppi"]:
                ax.plot(
                    time_actions,
                    results[ct]["costs"],
                    label=labels[ct],
                    color=colors[ct],
                    alpha=0.8,
                )
            ax.set_ylabel("Cost")
            ax.set_xlabel("Time (s)")
            ax.set_title("Running Cost")
            ax.set_yscale("log")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            # Thrust
            ax = plt.subplot(3, 4, 3)
            for ct in ["mppi", "smppi", "kmppi"]:
                ax.plot(
                    time_actions,
                    results[ct]["actions"][:, 0],
                    label=labels[ct],
                    color=colors[ct],
                    alpha=0.8,
                )
            ax.axhline(
                mass * gravity,
                color="k",
                linestyle="--",
                alpha=0.3,
                label="Hover",
            )
            ax.set_ylabel("Thrust (N)")
            ax.set_title("Control Input - Thrust")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            # Angular rate commands
            for i, axis in enumerate(["X", "Y", "Z"]):
                ax = plt.subplot(3, 4, (i + 1) * 4)
                for ct in ["mppi", "smppi", "kmppi"]:
                    ax.plot(
                        time_actions,
                        results[ct]["actions"][:, i + 1],
                        label=labels[ct],
                        color=colors[ct],
                        alpha=0.8,
                    )
                ax.axhline(0, color="k", linestyle="--", alpha=0.3)
                ax.set_ylabel(f"ω{axis} cmd (rad/s)")
                ax.set_title(f"Angular Rate {axis}")
                if i == 0:
                    ax.legend(loc="best")
                ax.grid(True, alpha=0.3)
                if i == 2:
                    ax.set_xlabel("Time (s)")

            plt.tight_layout()

            # Save to docs/media directory
            output_dir = Path(__file__).parent.parent.parent / "docs" / "media"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "quadrotor_hover_comparison.png"

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved to {output_path}")
            plt.show()

        except ImportError:
            print("\nMatplotlib not available for visualization")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare MPPI variants on quadrotor hover control"
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Number of control steps"
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of MPPI samples"
    )
    parser.add_argument(
        "--horizon", type=int, default=30, help="MPPI planning horizon"
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=1.0,
        dest="lambda_",
        help="MPPI temperature",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Plot results with matplotlib"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    results = run_quadrotor_hover_comparison(
        num_steps=args.steps,
        num_samples=args.samples,
        horizon=args.horizon,
        lambda_=args.lambda_,
        visualize=args.visualize,
        seed=args.seed,
    )
