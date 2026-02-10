import argparse
import os
import time

import numpy as np

try:
    import pygame
except ImportError:
    print("Pygame not installed. Skipping visualization.")
    pygame = None

from jax_mppi import k_mppi

# CUDA kernel code for pendulum dynamics and cost
cuda_code = r"""
extern "C" {
    // Constants
    static constexpr float g = 9.81f;
    static constexpr float m = 1.0f;
    static constexpr float L_pend = 1.0f;
    static constexpr float I = m * L_pend * L_pend;
    static constexpr float dt_sim = 0.05f;

    // Weights
    static constexpr float Q_theta = 20.0f;
    static constexpr float Q_th_dot = 2.0f;
    static constexpr float R_torque = 1.0f;
    static constexpr float Q_term_th = 100.0f;
    static constexpr float Q_term_th_dot = 10.0f;

    // Dynamics step
    __device__ void step(
        const float* x, const float* u, float* xn, float dt
    ) {
        // x[0] = theta (angle from upright), x[1] = theta_dot
        // u[0] = torque
        float theta = x[0];
        float theta_dot = x[1];
        float torque = u[0];

        // theta_ddot = (torque - m*g*L*sin(theta)) / I
        float theta_ddot = (torque - m * g * L_pend * sinf(theta)) / I;

        // Semi-implicit Euler
        float theta_dot_next = theta_dot + theta_ddot * dt;
        float theta_next = theta + theta_dot_next * dt;

        xn[0] = theta_next;
        xn[1] = theta_dot_next;
    }

    // Cost function
    __device__ float cost(
        const float* x, const float* u, float t, float dt
    ) {
        float theta = x[0];
        float theta_dot = x[1];
        float torque = u[0];

        float cost = Q_theta * theta * theta +
                     Q_th_dot * theta_dot * theta_dot;
        cost += R_torque * torque * torque;

        return cost;
    }

    // Terminal cost
    __device__ float terminal_cost(const float* x) {
        float theta = x[0];
        float theta_dot = x[1];
        return Q_term_th * theta * theta +
               Q_term_th_dot * theta_dot * theta_dot;
    }
};
"""


def main():
    parser = argparse.ArgumentParser(description="CUDA MPPI Pendulum")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    # Define dimensions
    nx = 2
    nu = 1

    # Initial state (hanging down)
    state = np.array([np.pi, 0.0], dtype=np.float32)

    # MPPI configuration
    config = {
        "num_samples": args.samples,
        "horizon": args.horizon,
        "lambda": 1.0,
        "noise_sigma": np.array([[1.0]], dtype=np.float32),
        "u_min": np.array([-2.0], dtype=np.float32),
        "u_max": np.array([2.0], dtype=np.float32),
        "dt": 0.05,
    }

    include_paths = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    potential_include = os.path.join(
        script_dir, "../third_party/cuda-mppi/include"
    )
    if os.path.exists(potential_include):
        include_paths.append(os.path.abspath(potential_include))

    # Initialize Controller
    try:
        controller = k_mppi.KMPPIController(
            cuda_code,
            config,
            nx=nx,
            nu=nu,
            include_paths=include_paths
        )
    except Exception as e:
        print(f"Failed to initialize CUDA controller: {e}")
        print("\nTroubleshooting:")
        print("  1. Set CUDA_MPPI_INCLUDE_DIR to third_party/cuda-mppi/include")
        print("  2. Ensure CUDA toolkit is installed")
        print(
            f"     export CUDA_MPPI_INCLUDE_DIR="
            f"{os.path.abspath('../third_party/cuda-mppi/include')}"
        )
        return

    # Visualization setup
    screen = None
    clock = None
    if args.visualize and pygame is not None:
        try:
            pygame.init()
            screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("CUDA MPPI Pendulum (Real-time)")
            clock = pygame.time.Clock()
        except Exception as e:
            print(f"Pygame init failed: {e}")

    print("\nRunning simulation...")
    print(
        f"  Init: theta={state[0]:.3f}, dtheta={state[1]:.3f}"
    )

    # Simulation loop
    dt = config["dt"]

    # For visualization mapping
    center = (400, 300)
    scale = 200

    start_time = time.time()

    for step in range(args.steps):
        if args.visualize and pygame is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

        # Compute control
        action = controller.command(state)

        # Step dynamics (CPU simulation for environment)
        g = 9.81
        m = 1.0
        L_pend = 1.0
        theta = state[0]
        theta_dot = state[1]
        torque = action[0]

        Inertia = m * L_pend * L_pend
        theta_ddot = (torque - m * g * L_pend * np.sin(theta)) / Inertia

        next_state = np.array(
            [theta + theta_dot * dt, theta_dot + theta_ddot * dt]
        )
        state = next_state

        if args.visualize and screen is not None:
            screen.fill((255, 255, 255))

            x = center[0] + scale * np.sin(state[0])
            y = center[1] - scale * np.cos(state[0])

            pygame.draw.circle(screen, (0, 0, 0), center, 5)
            pygame.draw.line(screen, (0, 0, 0), center, (x, y), 3)
            pygame.draw.circle(screen, (0, 0, 255), (int(x), int(y)), 20)

            torque_norm = np.clip(action[0] / 2.0, -1.0, 1.0)
            if abs(torque_norm) > 1e-3:
                sweep = -torque_norm * (np.pi * 1.2)
                t_end_x = center[0] + 50 * np.cos(-np.pi/2 + sweep)
                t_end_y = center[1] - 50 * np.sin(-np.pi/2 + sweep)
                pygame.draw.line(
                    screen, (255, 0, 0),
                    (center[0], center[1]-50), (t_end_x, t_end_y), 2
                )

            pygame.display.flip()
            clock.tick(60)

        if (step + 1) % 50 == 0:
            print(
                f"  Step {step + 1}: theta={state[0]:.3f}, u={action[0]:.3f}"
            )

    end_time = time.time()
    print(f"\nDone. Average FPS: {args.steps / (end_time - start_time):.1f}")
    print(
        f"\n  Final: theta={state[0]:.3f}, dtheta={state[1]:.3f}"
    )
    print("  Target: theta=0.0 rad (upright)")

if __name__ == "__main__":
    main()
