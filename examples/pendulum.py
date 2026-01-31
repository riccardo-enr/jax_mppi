"""Pendulum swing-up example using JAX-MPPI."""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_mppi import mppi


# 1. Define Dynamics
def pendulum_dynamics(state, action):
    """
    State: [theta, theta_dot]
    Action: [torque]
    """
    theta, theta_dot = state[0], state[1]
    torque = action[0]

    g = 9.81
    length = 1.0
    m = 1.0
    dt = 0.05

    # Torque limit (also handled by MPPI clipping, but good for physics)
    torque = jnp.clip(torque, -2.0, 2.0)

    # Pendulum dynamics:
    # theta_ddot = (torque - m*g*length*sin(theta)) / (m*length^2)
    theta_ddot = (torque - m * g * length * jnp.sin(theta)) / (
        m * length * length
    )

    new_theta_dot = theta_dot + theta_ddot * dt
    new_theta = theta + new_theta_dot * dt

    # Wrap theta to [-pi, pi]
    new_theta = jnp.arctan2(jnp.sin(new_theta), jnp.cos(new_theta))

    return jnp.stack([new_theta, new_theta_dot])


# 2. Define Cost
def pendulum_cost(state, action):
    """
    Goal: theta=0 (upright), theta_dot=0
    """
    theta, theta_dot = state[0], state[1]
    torque = action[0]

    # Target is 0 (upright)
    # Theta error: 1 - cos(theta) is robust
    # Or just theta^2 if wrapped
    theta_cost = 2.0 * (1.0 - jnp.cos(theta))
    theta_dot_cost = 0.1 * theta_dot**2
    control_cost = 0.001 * torque**2

    return theta_cost + theta_dot_cost + control_cost


def run_pendulum(render=False, steps=200):
    # 3. Configure MPPI
    config = mppi.MPPIConfig(
        dynamics_fn=pendulum_dynamics,
        cost_fn=pendulum_cost,
        nx=2,
        nu=1,
        num_samples=500,
        horizon=20,
        lambda_=0.1,  # Temperature
        noise_sigma=jnp.array([[0.5]]),  # Exploration noise
        u_min=jnp.array([-2.0]),
        u_max=jnp.array([2.0]),
        u_init=jnp.array([0.0]),
        step_method="mppi",
    )

    # 4. Initialize
    config, state = mppi.create(config, seed=42)

    # Initial state: hanging down [pi, 0]
    x0 = jnp.array([np.pi, 0.0])

    # 5. Run Loop
    print("Starting Pendulum Swing-up (Target: theta=0)")
    print(f"Initial State: theta={x0[0]:.2f}, theta_dot={x0[1]:.2f}")

    # Setup visualization if requested
    env = None
    if render:
        try:
            import gymnasium as gym

            # Use gymnasium's pendulum for rendering
            # Note: Gymnasium's pendulum has slightly different
            # dynamics/params
            # so visual might drift from internal model, but good for
            # sanity check
            env = gym.make("Pendulum-v1", render_mode="human")
            obs, _ = env.reset(seed=42)
            # Force environment to our initial state
            env.unwrapped.state = np.array([np.pi, 0.0])  # type: ignore
        except ImportError:
            print("Gymnasium not installed, skipping render.")

    total_cost = 0.0
    states = []
    costs_history = []

    times = []

    # JIT the step function
    jit_step = jax.jit(mppi.step)

    # Warmup JIT
    print("Compiling JAX graph...")
    jit_step(config, state, x0)
    print("Compilation complete.")

    sim_state = x0
    states.append(sim_state)

    try:
        for t in range(steps):
            iter_start = time.time()

            # Plan
            state, action, info = jit_step(config, state, sim_state)

            # Execute on internal dynamics (closed loop)
            # In a real robot, you would send 'action' to the hardware and get
            # new state
            next_sim_state = pendulum_dynamics(sim_state, action)

            # Record metrics
            step_cost = pendulum_cost(sim_state, action)
            total_cost += float(step_cost)
            costs_history.append(float(step_cost))
            times.append(time.time() - iter_start)

            # Log
            if t % 10 == 0:
                print(
                    f"Step {t}: Theta={sim_state[0]:.3f}, "
                    f"Action={action[0]:.3f}, Cost={step_cost:.3f}"
                )

            # Render
            if env is not None:
                # Update gymnasium environment state to match our JAX state
                # Gymnasium Pendulum-v1 state: [cos(theta), sin(theta),
                # theta_dot]
                theta, theta_dot = float(sim_state[0]), float(sim_state[1])
                # Accessing .state on unwrapped env is dynamic,
                # pyright doesn't know
                env.unwrapped.state = np.array([theta, theta_dot])  # type: ignore # noqa: E501
                env.render()
                time.sleep(0.05)  # Slow down for visibility

            sim_state = next_sim_state
            states.append(sim_state)

            # Check convergence
            if (
                jnp.abs(sim_state[0]) < 0.1
                and jnp.abs(sim_state[1]) < 0.1
                and t > 50
            ):
                print(f"Converged at step {t}!")
                # Don't break if rendering, to show stabilization
                if not render:
                    break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if env is not None:
            env.close()

    avg_freq = len(times) / (sum(times) + 1e-6)

    states = np.array(states)
    costs_history = jnp.array(costs_history)

    print("\n" + "=" * 40)
    print("Summary:")
    print(f"Total Steps: {len(times)}")
    print(f"Average Control Frequency: {avg_freq:.1f} Hz")
    print(
        f"\nFinal state: theta={states[-1, 0]:.3f}, "
        f"theta_dot={states[-1, 1]:.3f}"
    )
    print(f"Total cost: {jnp.sum(costs_history):.2f}")
    print("=" * 40)

    # Simple check for CI
    if not render:
        final_theta = states[-1, 0]
        # Allow some slack, but should be close to 0
        if abs(final_theta) < 0.5:
            print("SUCCESS: Pendulum swung up!")
        else:
            print("FAILURE: Pendulum did not swing up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render", action="store_true", help="Render simulation"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of control steps (default: 100, or infinite with "
        "--render)",
    )
    args = parser.parse_args()

    # Default steps logic
    if args.steps is None:
        steps = 1000 if args.render else 100
    else:
        steps = args.steps

    run_pendulum(render=args.render, steps=steps)
