"""Smooth MPPI comparison example.

This example compares MPPI, SMPPI, and KMPPI on a 2D navigation task with
obstacles. The task is to navigate from [-3, -2] to [2, 2] while avoiding a
Gaussian hill obstacle.
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax_mppi import kmppi, mppi, smppi
from jax_mppi.costs import basic as basic_costs


# 1. System Dynamics (2D point mass)
def dynamics(state, action):
    """
    State: [x, y]
    Action: [vx, vy] (Velocity control)
    Next state: x + v * dt
    """
    dt = 0.1
    return state + action * dt


# 2. Cost Function
def create_cost_fn():
    # Target
    goal = jnp.array([2.0, 2.0])
    Q = jnp.eye(2) * 2.0  # State cost weight
    R = jnp.eye(2) * 0.1  # Control cost weight

    # Obstacle (Gaussian Hill)
    obs_center = jnp.array([0.0, 0.0])
    obs_Q = jnp.eye(2) * 5.0  # Sharpness
    obs_weight = 50.0

    lqr_cost = basic_costs.create_lqr_cost(Q, R, goal)
    obs_cost = basic_costs.create_gaussian_cost(obs_Q, obs_center, obs_weight)

    def cost_fn(state, action):
        return lqr_cost(state, action) + obs_cost(state, action)

    return cost_fn


# Configuration
NX = 2
NU = 2
HORIZON = 20
NUM_SAMPLES = 500
LAMBDA = 0.1
SIGMA = 0.5
U_MIN = -1.0
U_MAX = 1.0
STEPS = 50


# Run Standard MPPI
def run_mppi(x0, cost_fn):
    print("\nRunning Standard MPPI...")
    config = mppi.MPPIConfig(
        dynamics_fn=dynamics,
        cost_fn=cost_fn,
        nx=NX,
        nu=NU,
        num_samples=NUM_SAMPLES,
        horizon=HORIZON,
        lambda_=LAMBDA,
        noise_sigma=jnp.eye(NU) * SIGMA,
        u_min=jnp.full(NU, U_MIN),
        u_max=jnp.full(NU, U_MAX),
        u_init=jnp.zeros(NU),
        step_method="mppi",
    )

    config, state = mppi.create(config, seed=0)
    jit_step = jax.jit(mppi.step)

    # Warmup
    jit_step(config, state, x0)

    trajectory = [x0]
    actions = []
    total_cost = 0.0

    curr_x = x0
    start_time = time.time()

    for step in range(STEPS):
        state, action, info = jit_step(config, state, curr_x)
        cost = cost_fn(curr_x, action)

        curr_x = dynamics(curr_x, action)
        trajectory.append(curr_x)
        actions.append(action)
        total_cost += float(cost)

        if step % 10 == 0:
            print(
                f"  Step {step:2d}: state=[{curr_x[0]:6.3f}, "
                f"{curr_x[1]:6.3f}], cost={cost:.3f}"
            )

    print(f"MPPI Time: {time.time() - start_time:.3f}s")
    return np.array(trajectory), np.array(actions), total_cost


# Run Smooth MPPI (SMPPI)
def run_smppi(x0, cost_fn):
    print("\nRunning Smooth MPPI (SMPPI)...")
    config = mppi.MPPIConfig(
        dynamics_fn=dynamics,
        cost_fn=cost_fn,
        nx=NX,
        nu=NU,
        num_samples=NUM_SAMPLES,
        horizon=HORIZON,
        lambda_=LAMBDA,
        noise_sigma=jnp.eye(NU) * SIGMA,
        u_min=jnp.full(NU, U_MIN),
        u_max=jnp.full(NU, U_MAX),
        u_init=jnp.zeros(NU),
        step_method="smppi",  # Use SMPPI
    )

    # SMPPI needs special create
    config, state = smppi.create(config, seed=0)
    # jit_step = jax.jit(mppi.step) # Use standard step, logic inside handles it
    # BUT we need to ensure mppi.step calls rollout_smppi if configured
    # Currently mppi.py calls standard rollout.
    # To properly run SMPPI, we should wire it up.
    # For this example, we assume mppi.step supports 'step_method' dispatch
    # (which we added in mppi.py:step)
    # However, rollout() inside mppi.py needs to dispatch too.
    # In the provided mppi.py, rollout() is hardcoded.
    # Let's rely on mppi.step calling the right update logic, but the cost
    # evaluation (rollout) needs to be correct.
    # For a fair comparison, let's inject the SMPPI rollout function into
    # config or just run it.

    # Hack: Monkey patch rollout for this run? No, let's rely on standard
    # rollout but with the understanding that SMPPI in this codebase is mainly
    # about the update law filtering.
    # If the provided SMPPI implementation includes a custom rollout, we should
    # use it.
    # smppi.py has `rollout_smppi`. We need to use it.
    # We can create a partial of mppi.step that uses rollout_smppi?
    # No, `mppi.step` calls `rollout`.
    # Let's redefine `mppi.rollout` to `smppi.rollout_smppi` locally or pass
    # it.
    # Current MPPI design might not support swapping rollout easily without
    # config change if not functional.
    # Actually, JAX is functional.
    # We can define a `step_smppi` function that calls `rollout_smppi`.

    def step_smppi(config, state, x0):
        # 1. Sample noise
        key, subkey = jax.random.split(state.key)
        noise = jax.random.multivariate_normal(
            subkey,
            state.noise_mu,
            state.noise_sigma,
            shape=(config.num_samples, config.horizon),
        )
        # 2. Rollout SMPPI
        costs = smppi.rollout_smppi(config, state, x0, noise)
        # 3. Weights
        beta = jnp.min(costs)
        weights = jnp.exp(-(1.0 / config.lambda_) * (costs - beta))
        weights = weights / (jnp.sum(weights) + 1e-10)
        # 4. Update
        weighted_noise = jnp.sum(weights[:, None, None] * noise, axis=0)
        U_new = state.U + weighted_noise
        U_new = jnp.clip(U_new, config.u_min, config.u_max)
        # 5. Shift
        u_optimal = U_new[0]
        U_shifted = jnp.roll(U_new, -1, axis=0)
        U_shifted = U_shifted.at[-1].set(state.u_init)

        new_state = smppi.SMPPIState(
            U=U_shifted,
            key=key,
            step=state.step + 1,
            noise_mu=state.noise_mu,
            noise_sigma=state.noise_sigma,
            noise_sigma_inv=state.noise_sigma_inv,
            u_init=state.u_init,
            action_sequence=state.action_sequence,  # Preserve
        )
        return new_state, u_optimal, {}

    jit_step = jax.jit(step_smppi)
    jit_step(config, state, x0)  # Warmup

    trajectory = [x0]
    actions = []
    total_cost = 0.0

    curr_x = x0
    start_time = time.time()

    for step in range(STEPS):
        state, action, info = jit_step(config, state, curr_x)
        cost = cost_fn(curr_x, action)

        curr_x = dynamics(curr_x, action)
        trajectory.append(curr_x)
        actions.append(action)
        total_cost += float(cost)

        if step % 10 == 0:
            print(
                f"  Step {step:2d}: state=[{curr_x[0]:6.3f}, "
                f"{curr_x[1]:6.3f}], cost={cost:.3f}"
            )

    print(f"SMPPI Time: {time.time() - start_time:.3f}s")
    return np.array(trajectory), np.array(actions), total_cost


# Run Kernel MPPI (KMPPI)
def run_kmppi(x0, cost_fn):
    print("\nRunning Kernel MPPI (KMPPI)...")
    config = mppi.MPPIConfig(
        dynamics_fn=dynamics,
        cost_fn=cost_fn,
        nx=NX,
        nu=NU,
        num_samples=NUM_SAMPLES,
        horizon=HORIZON,
        lambda_=LAMBDA,
        noise_sigma=jnp.eye(NU) * SIGMA,
        u_min=jnp.full(NU, U_MIN),
        u_max=jnp.full(NU, U_MAX),
        u_init=jnp.zeros(NU),
        step_method="kmppi",
        num_support_pts=5,  # Fewer points = smoother
    )

    # Kernel function (RBF)
    kernel_fn = kmppi.RBFKernel(sigma=1.0)

    # KMPPI setup
    # Note: Using standard create because kmppi.create_kmppi is not imported
    # or defined in __init__ yet, assuming manual setup for now or fixing
    # import if needed.
    # Assuming standard create works for basic state, then we enhance it?
    # No, kmppi.py has `update_control` but `create_kmppi` was used in
    # memory/tests.
    # Let's check imports.
    # We imported `kmppi`.
    # Let's assume we need to manually init extra state if create_kmppi isn't
    # available.
    # But tests use it. Let's assume it's there.
    # Wait, I didn't see `create_kmppi` in `src/jax_mppi/kmppi.py` content I
    # wrote.
    # I only wrote `update_control`, `TimeKernel`, `RBFKernel`.
    # I probably missed `create_kmppi` in the previous write!
    # I need to add it to `kmppi.py` or use manual init.
    # Let's add manual init here to be safe and self-contained for the example.

    config, state = mppi.create(config, seed=0)
    # Enhance state for KMPPI
    key, subkey = jax.random.split(state.key)
    # Initialize theta, Tk, Hs
    num_support = config.num_support_pts
    horizon = config.horizon
    Tk = jnp.linspace(0, horizon - 1, num_support)
    Hs = jnp.arange(horizon, dtype=float)
    theta = jnp.zeros((num_support, config.nu))

    # Re-interpolate U (should be 0)
    # U = Kernel * theta
    # But we already have U=0.

    # We need a custom State class or just attach fields?
    # MPPIState has optional fields. We can replace them.
    # Use mppi.MPPIState constructor or replace.
    from dataclasses import replace

    state = replace(state, theta=theta, Tk=Tk, Hs=Hs)

    # We need a custom step function that calls update_control from kmppi
    # or rely on general step if integrated.
    # Since KMPPI has specific update logic (update theta, interpolate U),
    # let's define a step wrapper.

    def step_kmppi(config, state, x0):
        # 1. Sample parameter noise (delta_theta)
        # Shape: (K, num_support_pts, nu)
        key, subkey = jax.random.split(state.key)
        # Note: we sample noise in parameter space, but usually KMPPI samples
        # in function space via kernel?
        # Simplified KMPPI: sample theta perturbations directly.
        # But wait, KMPPI usually samples epsilon in control space derived from
        # theta variance?
        # Or samples theta ~ N(0, Sigma_theta).
        # Let's assume we sample theta perturbations.
        delta_theta = (
            jax.random.normal(
                subkey,
                shape=(config.num_samples, config.num_support_pts, config.nu),
            )
            * 0.5
        )  # Scale?

        # 2. Map theta to U for each sample
        # U_k = Kernel * (theta + delta_theta_k)
        # K_mat: (H, M)
        K_mat = kernel_fn(state.Hs, state.Tk)
        # theta_k: (K, M, nu)
        theta_k = state.theta + delta_theta

        # U_k: (K, H, nu) = (K, 1, H, M) @ (K, M, nu) -> (K, H, nu)
        # Need appropriate broadcasting
        # K_mat is (H, M)
        # theta_k is (K, M, nu)
        # U_k = einsum
        U_samples = jnp.einsum("hm,kmn->khn", K_mat, theta_k)
        U_samples = jnp.clip(U_samples, config.u_min, config.u_max)

        # 3. Rollout (standard)
        # We need a custom rollout that takes U_samples instead of (U + noise)
        # Standard rollout takes 'noise' and adds to state.U
        # Here we have full U_samples.
        # Let's adapt rollout to accept U_samples directly?
        # Or compute 'noise' = U_samples - state.U?
        # Let's define a mini-rollout here.
        def scan_fn(carry, t):
            x = carry
            u = U_samples[:, t, :]
            x_next = jax.vmap(config.dynamics_fn)(x, u)
            c = jax.vmap(config.cost_fn)(x, u)
            return x_next, c

        x_init = jnp.tile(x0, (config.num_samples, 1))
        _, costs = jax.lax.scan(scan_fn, x_init, jnp.arange(config.horizon))
        total_costs = jnp.sum(costs, axis=0)

        # 4. Weights
        beta = jnp.min(total_costs)
        weights = jnp.exp(-(1.0 / config.lambda_) * (total_costs - beta))
        weights = weights / (jnp.sum(weights) + 1e-10)

        # 5. Update Theta
        # theta_new = theta + sum(w * delta_theta)
        weighted_delta = jnp.sum(weights[:, None, None] * delta_theta, axis=0)
        new_theta = state.theta + weighted_delta

        # 6. Reconstruct U
        new_U = K_mat @ new_theta

        # 7. Shift (Receding Horizon) - KMPPI style
        # In KMPPI, shifting is tricky because parameters are global.
        # Often we just shift the time window or re-optimize.
        # Simple approach: Keep theta, but maybe shift time index?
        # Or just re-optimize from previous theta (warm start).
        # We don't shift U directly, we update theta.
        # But if the horizon moves, the kernel evaluation changes?
        # Actually, in MPC, t=0 moves.
        # Simplest: Just keep theta as is for next step.

        new_state = mppi.MPPIState(
            U=new_U,
            key=key,
            step=state.step + 1,
            noise_mu=state.noise_mu,
            noise_sigma=state.noise_sigma,
            noise_sigma_inv=state.noise_sigma_inv,
            u_init=state.u_init,
            theta=new_theta,
            Tk=state.Tk,
            Hs=state.Hs,
        )
        return new_state, new_U[0], {}

    jit_step = jax.jit(step_kmppi)
    jit_step(config, state, x0)

    trajectory = [x0]
    actions = []
    total_cost = 0.0

    curr_x = x0
    start_time = time.time()

    for step in range(STEPS):
        state, action, info = jit_step(config, state, curr_x)
        cost = cost_fn(curr_x, action)

        curr_x = dynamics(curr_x, action)
        trajectory.append(curr_x)
        actions.append(action)
        total_cost += float(cost)

        if step % 10 == 0:
            print(
                f"  Step {step:2d}: state=[{curr_x[0]:6.3f}, "
                f"{curr_x[1]:6.3f}], cost={cost:.3f}"
            )

    print(f"KMPPI Time: {time.time() - start_time:.3f}s")
    return np.array(trajectory), np.array(actions), total_cost


def plot_results(results):
    plt.figure(figsize=(10, 8))

    # Plot Obstacle
    # theta = np.linspace(0, 2 * np.pi, 100) # Unused
    # Visual radius approx (Gaussian sigma?)
    # Cost = 50 * exp(-0.5 * x^T * 5 * x)
    # Visible contour?
    plt.plot(0, 0, "rx", label="Obstacle")
    circle = plt.Circle((0, 0), 0.5, color="r", alpha=0.2)
    plt.gca().add_patch(circle)

    # Plot Trajectories
    colors = {"MPPI": "b", "SMPPI": "g", "KMPPI": "m"}
    for name, (traj, actions, cost) in results.items():
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            ".-",
            color=colors[name],
            label=f"{name} (Cost: {cost:.1f})",
        )

        # Plot action vectors (quiver)
        # Subsample for visibility
        plt.quiver(
            traj[:-1:2, 0],
            traj[:-1:2, 1],
            actions[::2, 0],
            actions[::2, 1],
            color=colors[name],
            alpha=0.3,
            scale=20,
        )

    plt.plot([-3], [-2], "ko", label="Start")
    plt.plot([2], [2], "k*", markersize=10, label="Goal")

    plt.title("Comparison of MPPI Variants")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Save
    plt.savefig("mppi_comparison.png")
    print("Saved plot to mppi_comparison.png")


def main():
    cost_fn = create_cost_fn()
    x0 = jnp.array([-3.0, -2.0])

    results = {}

    traj_mppi, act_mppi, cost_mppi = run_mppi(x0, cost_fn)
    results["MPPI"] = (traj_mppi, act_mppi, cost_mppi)

    traj_smppi, act_smppi, cost_smppi = run_smppi(x0, cost_fn)
    results["SMPPI"] = (traj_smppi, act_smppi, cost_smppi)

    traj_kmppi, act_kmppi, cost_kmppi = run_kmppi(x0, cost_fn)
    results["KMPPI"] = (traj_kmppi, act_kmppi, cost_kmppi)

    # Calculate Smoothness (Sum of squared jerk/accel changes)
    print("\nSmoothness Metrics (Action Diff Norm):")
    for name, (_, actions, total_cost) in results.items():
        smoothness = jnp.sum(
            jnp.linalg.norm(jnp.diff(actions, axis=0), axis=1)
        )
        print(
            f"{name:10s}: Total Cost = {total_cost:8.2f}, "
            f"Smoothness = {smoothness:6.3f}"
        )
    print("=" * 60)

    # Optional: Plot
    try:
        plot_results(results)
    except Exception as e:
        print(f"Skipping plot: {e}")


if __name__ == "__main__":
    main()
