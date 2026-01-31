"""Integration tests for autotune with MPPI variants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_mppi import autotune, mppi


class TestAutotuneMPPI:
    """Test autotuning with basic MPPI."""

    def test_autotune_lambda_improves_performance(self):
        """Verify that lambda tuning can find better parameters."""

        # Simple 1D dynamics: x' = x + u * dt
        def dynamics(state, action):
            return state + action * 0.1

        # Quadratic cost to origin
        def running_cost(state, action):
            return jnp.sum(state**2) + 0.01 * jnp.sum(action**2)

        def terminal_cost(state, action):
            return jnp.sum(state**2) * 10.0

        # Test multiple lambda values to show effect
        def evaluate_lambda(lambda_val):
            """Evaluate performance with a specific lambda value."""
            config, state = mppi.create(
                nx=2,
                nu=1,
                horizon=10,
                num_samples=50,
                lambda_=lambda_val,
                noise_sigma=jnp.eye(1) * 0.5,
            )

            # Run closed-loop rollout
            total_cost = 0.0
            obs = jnp.array([1.0, -0.5])
            rollout_state = state

            for _ in range(20):  # 20 timesteps
                action, rollout_state = mppi.command(
                    config,
                    rollout_state,
                    obs,
                    dynamics,
                    running_cost,
                    terminal_cost,
                )
                obs = dynamics(obs, action)
                total_cost += running_cost(obs, action)

            total_cost += terminal_cost(obs, jnp.zeros(1))
            return float(total_cost)

        # Compare different lambda values
        cost_high = evaluate_lambda(10.0)  # Too high
        cost_low = evaluate_lambda(0.5)  # Lower
        cost_mid = evaluate_lambda(2.0)  # Medium

        # At least one should be better than the highest
        assert min(cost_low, cost_mid) < cost_high, (
            "Lower lambda should improve cost"
        )

    def test_autotune_runs_successfully(self):
        """Verify that Autotune can run end-to-end with MPPI."""

        # Simple dynamics
        def dynamics(state, action):
            return state + action * 0.1

        def running_cost(state, action):
            return jnp.sum(state**2) + 0.01 * jnp.sum(action**2)

        def terminal_cost(state, action):
            return jnp.sum(state**2) * 5.0

        # Create MPPI with initial parameters
        config, state = mppi.create(
            nx=2,
            nu=1,
            horizon=8,
            num_samples=40,
            lambda_=5.0,
            noise_sigma=jnp.eye(1) * 0.8,
        )

        holder = autotune.ConfigStateHolder(config, state)

        # Evaluation function
        def evaluate():
            # Fresh state for each evaluation
            rollout_state = holder.state
            total_cost = 0.0
            obs = jnp.array([0.8, -0.3])

            for _ in range(15):
                action, rollout_state = mppi.command(
                    holder.config,
                    rollout_state,
                    obs,
                    dynamics,
                    running_cost,
                    terminal_cost,
                )
                obs = dynamics(obs, action)
                total_cost += running_cost(obs, action)

            total_cost += terminal_cost(obs, jnp.zeros(1))

            return autotune.EvaluationResult(
                mean_cost=float(total_cost),
                rollouts=jnp.zeros((1, 1, 2)),
                params={},
                iteration=0,
            )

        # Create autotuner
        tuner = autotune.Autotune(
            params_to_tune=[autotune.LambdaParameter(holder, min_value=0.1)],
            evaluate_fn=evaluate,
            optimizer=autotune.CMAESOpt(population=5, sigma=0.5),
        )

        # Run optimization
        best = tuner.optimize_all(iterations=5)

        # Verify it ran and produced results
        assert best.mean_cost > 0
        assert "lambda" in best.params
        assert best.params["lambda"][0] > 0.1
        assert best.iteration >= 0

    def test_autotune_multiple_parameters(self):
        """Tune lambda and noise_sigma together."""

        def dynamics(state, action):
            return state + action * 0.1

        def running_cost(state, action):
            return jnp.sum(state**2) + 0.01 * jnp.sum(action**2)

        def terminal_cost(state, action):
            return jnp.sum(state**2) * 5.0

        # Create MPPI
        config, state = mppi.create(
            nx=2,
            nu=2,
            horizon=8,
            num_samples=30,
            lambda_=3.0,
            noise_sigma=jnp.eye(2) * 1.0,
        )

        holder = autotune.ConfigStateHolder(config, state)

        def evaluate():
            rollout_state = holder.state
            total_cost = 0.0
            obs = jnp.array([0.5, -0.4])

            for _ in range(10):
                action, rollout_state = mppi.command(
                    holder.config,
                    rollout_state,
                    obs,
                    dynamics,
                    running_cost,
                    terminal_cost,
                )
                obs = dynamics(obs, action)
                total_cost += running_cost(obs, action)

            total_cost += terminal_cost(obs, jnp.zeros(2))

            return autotune.EvaluationResult(
                mean_cost=float(total_cost),
                rollouts=jnp.zeros((1, 1, 2)),
                params={},
                iteration=0,
            )

        # Tune both parameters
        tuner = autotune.Autotune(
            params_to_tune=[
                autotune.LambdaParameter(holder, min_value=0.1),
                autotune.NoiseSigmaParameter(holder, min_value=0.1),
            ],
            evaluate_fn=evaluate,
            optimizer=autotune.CMAESOpt(population=6, sigma=0.3),
        )

        best = tuner.optimize_all(iterations=5)

        # Verify both parameters were tuned
        assert "lambda" in best.params
        assert "noise_sigma" in best.params
        assert len(best.params["noise_sigma"]) == 2  # nu=2
        assert best.mean_cost > 0


class TestAutotuneHorizon:
    """Test horizon parameter tuning."""

    def test_horizon_parameter_application(self):
        """Verify that horizon parameter correctly updates config and state."""

        def dynamics(state, action):
            return state + action * 0.1

        def running_cost(state, action):
            return jnp.sum(state**2)

        def terminal_cost(state, action):
            return jnp.sum(state**2) * 5.0

        # Start with horizon=8
        config, state = mppi.create(
            nx=2,
            nu=1,
            horizon=8,
            num_samples=30,
            lambda_=1.0,
            noise_sigma=jnp.eye(1) * 0.3,
        )

        holder = autotune.ConfigStateHolder(config, state)
        param = autotune.HorizonParameter(holder, min_value=5, max_value=15)

        # Verify initial state
        assert holder.config.horizon == 8
        assert holder.state.U.shape[0] == 8

        # Apply new horizon
        param.apply_parameter_value(np.array([12.0]))

        # Verify changes
        assert holder.config.horizon == 12
        assert holder.state.U.shape[0] == 12

        # Apply smaller horizon
        param.apply_parameter_value(np.array([6.0]))
        assert holder.config.horizon == 6
        assert holder.state.U.shape[0] == 6
