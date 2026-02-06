"""I-MPPI Simulation: Informative Model Predictive Path Integral.

This script implements the simulation described in docs/i_mppi.qmd.
It features:
- Layer 2: FSMI-driven trajectory generation (simulated by a state machine).
- Layer 3: Biased-MPPI control with mixture sampling.
- Environment: 2D corridor with walls and depletable information sources.

It can run MPPI, SMPPI, and KMPPI variants and compare runtime + smoothness.
"""

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jax_mppi import kmppi, mppi, smppi
from jax_mppi.i_mppi.environment import (
    GOAL_POS,
    INFO_ZONES,
    WALLS,
    augmented_dynamics,
    running_cost,
)
from jax_mppi.i_mppi.fsmi import FSMIConfig, FSMITrajectoryGenerator
from jax_mppi.i_mppi.planner import (
    biased_kmppi_command,
    biased_mppi_command,
    biased_smppi_command,
)

# --- Simulation Loop ---


def main():
    print("Setting up I-MPPI simulation...")

    # Config
    dt = 0.05
    nx = 16  # 13 (quad state) + 3 (info zones)
    nu = 4
    horizon = 40  # 2 seconds

    # Initial State
    # Quad: [0, 5, -2, ...] (Start at left middle)
    start_pos = jnp.array([1.0, 5.0, -2.0])
    # info levels = max (3 zones now)
    info_init = jnp.array([100.0, 100.0, 100.0])

    x0 = jnp.zeros(13)
    x0 = x0.at[:3].set(start_pos)
    x0 = x0.at[6].set(1.0)  # qw=1

    state = jnp.concatenate([x0, info_init])

    # Common MPPI Setup
    # Noise: Thrust ~ 5.0, Omega ~ 1.0
    noise_sigma = jnp.diag(jnp.array([2.0, 0.5, 0.5, 0.5]) ** 2)

    u_min = jnp.array([0.0, -10.0, -10.0, -10.0])
    u_max = jnp.array([4.0 * 9.81, 10.0, 10.0, 10.0])
    # Rate limits for SMPPI (control rates). Heuristic: allow full action range in ~0.5s.
    rate_time = 0.5
    action_range = u_max - u_min
    u_vel_max = action_range / rate_time
    u_vel_min = -u_vel_max
    u_init = jnp.array([9.81, 0.0, 0.0, 0.0])

    # Layer 2 Setup
    fsmi_config = FSMIConfig(info_threshold=20.0, goal_pos=GOAL_POS)
    fsmi_planner = FSMITrajectoryGenerator(fsmi_config, INFO_ZONES)

    # Run loop
    # Make the simulation be 60 seconds
    sim_duration = 60.0  # seconds
    control_hz = 50.0
    sim_steps = int(round(sim_duration * control_hz))
    fsmi_hz = 5.0
    fsmi_steps = int(round(control_hz / fsmi_hz))

    print("Starting simulation loop (JIT-compiled scan)...")

    # U_ref = Hover
    U_ref = jnp.tile(u_init, (horizon, 1))

    def block_until_ready(tree):
        for leaf in jax.tree_util.tree_leaves(tree):
            leaf.block_until_ready()

    def make_u_ref_from_traj(current_state, ref_traj):
        # Simple heuristic: bias towards the reference trajectory with small body-rate commands
        pos = current_state[:3]
        err = ref_traj - pos[None, :]

        # Thrust: hover + proportional on z error
        k_thrust = 3.0
        thrust = u_init[0] + k_thrust * err[:, 2]

        # Body rates: proportional to lateral error (clipped)
        k_omega = 0.6
        omega_x = k_omega * err[:, 1]
        omega_y = -k_omega * err[:, 0]
        omega_z = jnp.zeros_like(omega_x)

        u_ref = jnp.stack([thrust, omega_x, omega_y, omega_z], axis=1)
        u_ref = jnp.clip(u_ref, u_min, u_max)
        return u_ref

    def make_sim_fn(controller_name, config, ctrl_state, kernel_fn=None):
        def step_fn(carry, t):
            current_state, current_ctrl_state, ref_traj = carry

            # --- Layer 2: FSMI Logic ---
            current_info = current_state[13:]
            do_update = jnp.equal(jnp.mod(t, fsmi_steps), 0)
            ref_traj = jax.lax.cond(
                do_update,
                lambda _: fsmi_planner.get_reference_trajectory(
                    current_state, current_info, horizon, dt
                )[0],
                lambda _: ref_traj,
                operand=None,
            )

            # --- Layer 3: Biased MPPI ---
            cost_fn = partial(running_cost, target=ref_traj)
            U_ref_local = make_u_ref_from_traj(current_state, ref_traj)

            if controller_name == "mppi":
                action, next_ctrl_state = biased_mppi_command(
                    config,
                    current_ctrl_state,
                    current_state,
                    augmented_dynamics,
                    cost_fn,
                    U_ref_local,
                    bias_alpha=0.2,
                )
            elif controller_name == "smppi":
                action, next_ctrl_state = biased_smppi_command(
                    config,
                    current_ctrl_state,
                    current_state,
                    augmented_dynamics,
                    cost_fn,
                    U_ref_local,
                    bias_alpha=0.2,
                )
            elif controller_name == "kmppi":
                action, next_ctrl_state = biased_kmppi_command(
                    config,
                    current_ctrl_state,
                    current_state,
                    augmented_dynamics,
                    cost_fn,
                    kernel_fn,
                    U_ref_local,
                    bias_alpha=0.2,
                )
            else:
                raise ValueError(f"Unknown controller: {controller_name}")

            # Step Dynamics
            next_state = augmented_dynamics(current_state, action, dt=dt)

            return (next_state, next_ctrl_state, ref_traj), (
                next_state,
                current_info,
                ref_traj[0],
                action,
            )

        def sim_fn(initial_state, initial_ctrl_state):
            init_ref_traj, _ = fsmi_planner.get_reference_trajectory(
                initial_state, initial_state[13:], horizon, dt
            )
            (
                (final_state, final_ctrl_state, _),
                (
                    history_x,
                    history_info,
                    targets,
                    actions,
                ),
            ) = jax.lax.scan(
                step_fn,
                (initial_state, initial_ctrl_state, init_ref_traj),
                jnp.arange(sim_steps),
            )
            return (
                final_state,
                final_ctrl_state,
                history_x,
                history_info,
                targets,
                actions,
            )

        return jax.jit(sim_fn)

    def compute_smoothness(actions, positions, dt_local):
        action_jerk = jnp.diff(actions, n=2, axis=0) / (dt_local**2)
        action_jerk_mean = jnp.mean(jnp.linalg.norm(action_jerk, axis=1))

        pos = positions[:, :3]
        vel = jnp.diff(pos, axis=0) / dt_local
        acc = jnp.diff(vel, axis=0) / dt_local
        jerk = jnp.diff(acc, axis=0) / dt_local
        traj_jerk_mean = jnp.mean(jnp.linalg.norm(jerk, axis=1))

        return action_jerk_mean, traj_jerk_mean

    def run_controller(controller_name):
        if controller_name == "mppi":
            config, ctrl_state = mppi.create(
                nx=nx,
                nu=nu,
                noise_sigma=noise_sigma,
                num_samples=1000,  # reduced for speed
                horizon=horizon,
                lambda_=0.1,
                u_min=u_min,
                u_max=u_max,
                u_init=u_init,
                step_dependent_dynamics=True,
            )
            sim_fn = make_sim_fn(controller_name, config, ctrl_state)
        elif controller_name == "smppi":
            config, ctrl_state = smppi.create(
                nx=nx,
                nu=nu,
                noise_sigma=noise_sigma,
                num_samples=1000,
                horizon=horizon,
                lambda_=0.1,
                u_min=u_vel_min,
                u_max=u_vel_max,
                u_init=jnp.zeros_like(u_init),
                U_init=U_ref,
                action_min=u_min,
                action_max=u_max,
                step_dependent_dynamics=True,
                w_action_seq_cost=0.5,
                delta_t=dt,
            )
            sim_fn = make_sim_fn(controller_name, config, ctrl_state)
        elif controller_name == "kmppi":
            config, ctrl_state, kernel_fn = kmppi.create(
                nx=nx,
                nu=nu,
                noise_sigma=noise_sigma,
                num_samples=1000,
                horizon=horizon,
                lambda_=0.1,
                u_min=u_min,
                u_max=u_max,
                u_init=u_init,
                U_init=U_ref,
                step_dependent_dynamics=True,
            )
            sim_fn = make_sim_fn(
                controller_name, config, ctrl_state, kernel_fn=kernel_fn
            )
        else:
            raise ValueError(f"Unknown controller: {controller_name}")

        # Warm-up (compile)
        warm = sim_fn(state, ctrl_state)
        block_until_ready(warm)

        # Timed run
        start = time.perf_counter()
        (
            final_state,
            final_ctrl_state,
            history_x,
            history_info,
            targets,
            actions,
        ) = sim_fn(state, ctrl_state)
        block_until_ready(final_state)
        elapsed = time.perf_counter() - start

        action_jerk, traj_jerk = compute_smoothness(actions, history_x, dt)

        return {
            "config": config,
            "ctrl_state": final_ctrl_state,
            "history_x": history_x,
            "history_info": history_info,
            "targets": targets,
            "actions": actions,
            "runtime_s": elapsed,
            "action_jerk": action_jerk,
            "traj_jerk": traj_jerk,
            "final_state": final_state,
        }

    controllers = ["mppi", "smppi", "kmppi"]
    results = {name: run_controller(name) for name in controllers}

    # Print final status
    for name in controllers:
        final_state = results[name]["final_state"]
        print(
            f"{name.upper()} Final: Pos={final_state[:2]}, Info={final_state[13:]}"
        )

    print("\nComparison (runtime + smoothness):")
    print(
        f"{'controller':<10} {'runtime_ms':>12} {'act_jerk':>12} {'traj_jerk':>12}"
    )
    for name in controllers:
        runtime_ms = 1000.0 * results[name]["runtime_s"]
        action_jerk = float(results[name]["action_jerk"])
        traj_jerk = float(results[name]["traj_jerk"])
        print(
            f"{name:<10} {runtime_ms:>12.3f} {action_jerk:>12.4f} {traj_jerk:>12.4f}"
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("I-MPPI Simulation (2D)", "I-MPPI Simulation (3D)"),
    )

    # Plot Walls
    for w in WALLS:
        fig.add_shape(
            type="rect",
            x0=w[0],
            y0=w[1],
            x1=w[2],
            y1=w[3],
            fillcolor="gray",
            opacity=0.5,
            line_width=0,
            row=1,
            col=1,
        )

    # Plot Info Zones
    for i in range(len(INFO_ZONES)):
        cx, cy = INFO_ZONES[i, 0], INFO_ZONES[i, 1]
        w, h = INFO_ZONES[i, 2], INFO_ZONES[i, 3]
        fig.add_shape(
            type="rect",
            x0=cx - w / 2,
            y0=cy - h / 2,
            x1=cx + w / 2,
            y1=cy + h / 2,
            fillcolor="yellow",
            opacity=0.3,
            line_width=0,
            row=1,
            col=1,
        )

    # Legend entry for info zones
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="yellow"),
            name="Info Zone",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Plot Info Zones in 3D (as translucent planes at z=target altitude)
    zone_z = GOAL_POS[2]
    for i in range(len(INFO_ZONES)):
        cx, cy = INFO_ZONES[i, 0], INFO_ZONES[i, 1]
        w, h = INFO_ZONES[i, 2], INFO_ZONES[i, 3]
        x0, x1 = cx - w / 2, cx + w / 2
        y0, y1 = cy - h / 2, cy + h / 2
        xs = [x0, x1, x1, x0]
        ys = [y0, y0, y1, y1]
        zs = [zone_z, zone_z, zone_z, zone_z]
        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color="yellow",
                opacity=0.25,
                name="Info Zone",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Plot Goal
    fig.add_trace(
        go.Scatter(
            x=[GOAL_POS[0]],
            y=[GOAL_POS[1]],
            mode="markers",
            marker=dict(color="red", symbol="star", size=14),
            name="Goal",
        ),
        row=1,
        col=1,
    )

    # Plot Start
    fig.add_trace(
        go.Scatter(
            x=[start_pos[0]],
            y=[start_pos[1]],
            mode="markers",
            marker=dict(color="green", symbol="circle", size=10),
            name="Start",
        ),
        row=1,
        col=1,
    )

    colors = {"mppi": "blue", "smppi": "green", "kmppi": "orange"}
    for name in controllers:
        history_x = results[name]["history_x"]
        fig.add_trace(
            go.Scatter(
                x=history_x[:, 0],
                y=history_x[:, 1],
                mode="lines",
                line=dict(color=colors[name], width=2),
                name=f"{name.upper()} Trajectory",
            ),
            row=1,
            col=1,
        )

    # --- 3D View ---
    for name in controllers:
        history_x = results[name]["history_x"]
        targets = results[name]["targets"]
        fig.add_trace(
            go.Scatter3d(
                x=history_x[:, 0],
                y=history_x[:, 1],
                z=history_x[:, 2],
                mode="lines",
                line=dict(color=colors[name], width=2),
                name=f"{name.upper()} Trajectory",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter3d(
                x=targets[:, 0],
                y=targets[:, 1],
                z=targets[:, 2],
                mode="lines",
                line=dict(color=colors[name], width=2, dash="dash"),
                opacity=0.7,
                name=f"{name.upper()} Targets",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter3d(
            x=[start_pos[0]],
            y=[start_pos[1]],
            z=[start_pos[2]],
            mode="markers",
            marker=dict(color="green", size=5, symbol="circle"),
            name="Start",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[GOAL_POS[0]],
            y=[GOAL_POS[1]],
            z=[GOAL_POS[2]],
            mode="markers",
            marker=dict(color="red", size=7, symbol="diamond"),
            name="Goal",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(range=[-1, 14], row=1, col=1)
    fig.update_yaxes(
        range=[-1, 12],
        scaleanchor="x",
        scaleratio=1,
        row=1,
        col=1,
    )
    fig.update_scenes(
        xaxis=dict(range=[-1, 14]),
        yaxis=dict(range=[-1, 12]),
        zaxis=dict(range=[-4, 1]),
        row=1,
        col=2,
    )
    fig.update_layout(
        width=1200,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )

    output_path = "docs/_media/i_mppi_simulation.png"
    os.makedirs("docs/_media", exist_ok=True)
    fig.write_image(output_path)
    print(f"Simulation complete. Plot saved to {output_path}")


if __name__ == "__main__":
    main()
