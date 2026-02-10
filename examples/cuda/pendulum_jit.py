"""
Example: JIT-compiled MPPI for Pendulum Swing-Up

This example demonstrates how to use the JITMPPIController to define
custom dynamics and cost functions at runtime without recompiling the
C++ library.

The pendulum starts hanging down (theta = pi) and the controller
learns to swing it up to the upright position (theta = 0).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from jax_mppi import cuda_mppi

# Define pendulum dynamics in C++/CUDA
PENDULUM_DYNAMICS = """
struct UserDynamics {
    static constexpr float g = 9.81f;
    static constexpr float m = 1.0f;
    static constexpr float l = 1.0f;

    __device__ void step(const float* x, const float* u, float* x_next, float dt) {
        // x[0] = theta (angle from upright), x[1] = theta_dot (angular velocity)
        // u[0] = torque
        float theta = x[0];
        float theta_dot = x[1];
        float torque = u[0];

        // Pendulum dynamics: theta_ddot = (torque - m*g*l*sin(theta)) / (m*l^2)
        float I = m * l * l;  // moment of inertia
        float theta_ddot = (torque - m * g * l * sinf(theta)) / I;

        // Euler integration
        x_next[0] = theta + theta_dot * dt;
        x_next[1] = theta_dot + theta_ddot * dt;
    }
};
"""

# Define quadratic cost in C++/CUDA
PENDULUM_COST = """
struct UserCost {
    static constexpr float Q_theta = 100.0f;
    static constexpr float Q_theta_dot = 10.0f;
    static constexpr float R_torque = 0.1f;
    static constexpr float Q_terminal_theta = 100.0f;
    static constexpr float Q_terminal_theta_dot = 10.0f;

    __device__ float compute(const float* x, const float* u, int t) {
        // State cost: penalize deviation from upright (theta = 0)
        float theta = x[0];
        float theta_dot = x[1];
        float torque = u[0];

        float state_cost = Q_theta * theta * theta + Q_theta_dot * theta_dot * theta_dot;
        float control_cost = R_torque * torque * torque;

        return state_cost + control_cost;
    }

    __device__ float terminal_cost(const float* x) {
        float theta = x[0];
        float theta_dot = x[1];
        return Q_terminal_theta * theta * theta + Q_terminal_theta_dot * theta_dot * theta_dot;
    }
};
"""


def pendulum_dynamics_python(state, action, dt):
    """Python version for comparison/plotting"""
    g = 9.81
    m = 1.0
    l = 1.0

    theta = state[0]
    theta_dot = state[1]
    torque = action[0]

    I = m * l * l
    theta_ddot = (torque - m * g * l * np.sin(theta)) / I

    next_state = np.array([theta + theta_dot * dt, theta_dot + theta_ddot * dt])

    return next_state


def main():
    parser = argparse.ArgumentParser(
        description="JIT-Compiled MPPI Pendulum Swing-Up Example"
    )
    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Enable real-time pygame visualization",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("JIT-Compiled MPPI Pendulum Swing-Up Example")
    print("=" * 60)

    # Configuration
    config = cuda_mppi.MPPIConfig(
        num_samples=1000,  # Number of sampled trajectories
        horizon=50,  # Planning horizon (time steps)
        nx=2,  # State dimension [theta, theta_dot]
        nu=1,  # Control dimension [torque]
        lambda_=0.1,  # Temperature parameter
        dt=0.02,  # Time step (20ms)
        u_scale=5.0,  # Control scaling (max torque ~ 5 Nm)
        w_action_seq_cost=0.0,  # No smoothness penalty
        num_support_pts=10,  # For KMPPI (not used here)
    )

    print("\nMPPI Configuration:")
    print(f"  Samples: {config.num_samples}")
    print(f"  Horizon: {config.horizon}")
    print(f"  dt: {config.dt} s")
    print(f"  Control scale: {config.u_scale}")

    # Set up include paths for JIT compilation
    # The JIT compiler needs to find the CUDA MPPI headers
    include_paths = []

    # Try to find the include directory
    cuda_mppi_include = os.environ.get("CUDA_MPPI_INCLUDE_DIR")
    if cuda_mppi_include:
        include_paths.append(cuda_mppi_include)
    else:
        # Try relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_include = os.path.join(
            script_dir, "../third_party/cuda-mppi/include"
        )
        if os.path.exists(potential_include):
            include_paths.append(os.path.abspath(potential_include))
            os.environ["CUDA_MPPI_INCLUDE_DIR"] = os.path.abspath(
                potential_include
            )

    print("\nInclude paths for JIT compilation:")
    for path in include_paths:
        print(f"  {path}")

    # Create JIT controller
    print("\nCompiling JIT controller...")
    print("  (This happens once during initialization)")

    try:
        controller = cuda_mppi.JITMPPIController(
            config, PENDULUM_DYNAMICS, PENDULUM_COST, include_paths
        )
        print("  ✓ Compilation successful!")
    except Exception as e:
        print(f"  ✗ Compilation failed: {e}")
        print("\nTroubleshooting:")
        print(
            "  1. Set CUDA_MPPI_INCLUDE_DIR to point to third_party/cuda-mppi/include"
        )
        print("  2. Ensure CUDA toolkit is installed")
        print(
            f"     export CUDA_MPPI_INCLUDE_DIR={os.path.abspath('../third_party/cuda-mppi/include')}"
        )
        return

    # Initial state: pendulum hanging down
    state = np.array([np.pi, 0.0], dtype=np.float32)  # [theta, theta_dot]

    # Storage for plotting
    states = [state.copy()]
    actions = []
    times = [0.0]

    # Optional pygame visualization
    pygame = None
    screen = None
    clock = None
    if args.visualization:
        try:
            import pygame as _pygame  # type: ignore

            pygame = _pygame
            pygame.init()
            pygame.font.init()
            screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("CUDA MPPI Pendulum (Real-time)")
            clock = pygame.time.Clock()
            pygame.font.SysFont("Arial", 20)
        except Exception as e:
            print(
                f"  ⚠ Pygame unavailable, continuing without visualization: {e}"
            )
            pygame = None

    print("\nRunning simulation...")
    print(
        f"  Initial state: theta={state[0]:.3f} rad, theta_dot={state[1]:.3f} rad/s"
    )

    # Main control loop
    step = 0
    while True:
        if pygame is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    pygame = None
                    break

        # Compute optimal control
        controller.compute(state)
        action = controller.get_action()
        controller.shift()  # Prepare for next iteration

        # Apply control and simulate
        state = pendulum_dynamics_python(state, action, config.dt)

        # Store for plotting
        states.append(state.copy())
        actions.append(action.copy())
        times.append((step + 1) * config.dt)

        # Print progress
        if (step + 1) % 50 == 0:
            print(
                f"  Step {step + 1}: theta={state[0]:.3f} rad, torque={action[0]:.3f} Nm"
            )

        if pygame is not None:
            # Draw pendulum
            screen.fill((245, 245, 245))
            width, height = screen.get_size()
            origin = (width // 2, height // 2)
            length = int(min(width, height) * 0.35)
            theta = state[0]
            x = origin[0] + int(length * np.sin(theta))
            # Theta = 0 should be upright (upwards on screen).
            y = origin[1] - int(length * np.cos(theta))
            pygame.draw.line(screen, (30, 30, 30), origin, (x, y), 4)
            pygame.draw.circle(screen, (200, 60, 60), (x, y), 12)
            pygame.draw.circle(screen, (30, 30, 30), origin, 6)
            # Torque indicator: circle + arrow around origin
            torque = float(action[0])
            max_torque = float(config.u_scale)
            torque_norm = max(-1.0, min(1.0, torque / max_torque))
            circle_r = int(min(width, height) * 0.12)

            if abs(torque_norm) > 1e-3:
                # Arced arrow: positive torque -> CCW, negative -> CW
                # Start at top (12 o'clock = -π/2 in standard coords, but 3π/2 in pygame)
                base_angle = -np.pi / 2  # Top of circle
                max_sweep = np.pi * 1.2  # Maximum arc length
                sweep = torque_norm * max_sweep

                # For pygame's inverted y-axis, we need to flip the direction
                start_angle = base_angle
                end_angle = base_angle + sweep

                # Ensure angles are in correct order for pygame
                if sweep < 0:
                    start_angle, end_angle = end_angle, start_angle

                arc_rect = pygame.Rect(
                    origin[0] - circle_r,
                    origin[1] - circle_r,
                    circle_r * 2,
                    circle_r * 2,
                )

                pygame.draw.arc(
                    screen, (20, 120, 200), arc_rect, start_angle, end_angle, 3
                )

                # Arrow head at end of arc
                actual_end = base_angle + sweep
                ax = origin[0] + int(circle_r * np.cos(actual_end))
                ay = origin[1] + int(circle_r * np.sin(actual_end))

                # Tangent direction (perpendicular to radius)
                # Add π/2 for CCW (positive torque), subtract for CW (negative)
                tangent = (
                    actual_end + np.pi / 2
                    if sweep > 0
                    else actual_end - np.pi / 2
                )

                head_len = 12
                left = tangent + np.pi / 6
                right = tangent - np.pi / 6

                hx1 = ax + int(head_len * np.cos(left))
                hy1 = ay + int(head_len * np.sin(left))
                hx2 = ax + int(head_len * np.cos(right))
                hy2 = ay + int(head_len * np.sin(right))

                pygame.draw.line(
                    screen, (20, 120, 200), (ax, ay), (hx1, hy1), 3
                )
                pygame.draw.line(
                    screen, (20, 120, 200), (ax, ay), (hx2, hy2), 3
                )

            pygame.display.flip()
            if clock is not None:
                clock.tick(int(1.0 / config.dt))

        step += 1

    states = np.array(states)
    actions = np.array(actions)

    print(
        f"\n  Final state: theta={state[0]:.3f} rad, theta_dot={state[1]:.3f} rad/s"
    )
    print("  Target: theta=0.0 rad (upright)")

    # Plotting
    print("\nGenerating plots...")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Plot angle
    axes[0].plot(times, states[:, 0], "b-", linewidth=2)
    axes[0].axhline(y=0, color="r", linestyle="--", label="Target (upright)")
    axes[0].axhline(
        y=np.pi, color="g", linestyle="--", label="Initial (hanging)"
    )
    axes[0].set_ylabel("Angle (rad)")
    axes[0].set_title("Pendulum Swing-Up with JIT-Compiled MPPI")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot angular velocity
    axes[1].plot(times, states[:, 1], "b-", linewidth=2)
    axes[1].axhline(y=0, color="r", linestyle="--", label="Target")
    axes[1].set_ylabel("Angular Velocity (rad/s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot control torque
    axes[2].plot(times[1:], actions[:, 0], "b-", linewidth=2)
    axes[2].axhline(y=0, color="r", linestyle="--")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = "cuda_pendulum_jit_result.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  ✓ Plot saved to {output_file}")

    # Optionally show the plot
    # plt.show()

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
