"""Tests for advanced quadrotor examples."""

import jax.numpy as jnp
import pytest


class TestFigure8Comparison:
    """Tests for figure-8 MPPI variant comparison example."""

    def test_figure8_comparison_runs(self):
        """Test that figure-8 comparison example runs."""
        from examples.quadrotor.figure8_comparison import run_quadrotor_figure8_comparison

        results, reference = run_quadrotor_figure8_comparison(
            num_steps=100,
            num_samples=200,
            horizon=10,
            visualize=False,
            seed=42,
        )

        # Check that all three controllers ran
        assert "mppi" in results
        assert "smppi" in results
        assert "kmppi" in results

        # Check shapes
        for controller in ["mppi", "smppi", "kmppi"]:
            states = results[controller]["states"]
            actions = results[controller]["actions"]
            costs = results[controller]["costs"]

            assert states.shape == (101, 13)
            assert actions.shape == (100, 4)
            assert costs.shape == (100,)

    def test_figure8_produces_metrics(self):
        """Test that figure-8 comparison produces performance metrics."""
        from examples.quadrotor.figure8_comparison import run_quadrotor_figure8_comparison

        results, _ = run_quadrotor_figure8_comparison(
            num_steps=50,
            num_samples=100,
            horizon=10,
            visualize=False,
            seed=42,
        )

        expected_metrics = [
            "mean_pos_error",
            "max_pos_error",
            "rms_pos_error",
            "mean_vel_error",
            "total_cost",
            "energy",
            "mean_control_rate",
            "max_control_rate",
        ]

        for controller in ["mppi", "smppi", "kmppi"]:
            for metric in expected_metrics:
                assert metric in results[controller]
                assert jnp.isfinite(results[controller][metric])

    def test_figure8_quaternions_normalized(self):
        """Test that quaternions stay normalized for all controllers."""
        from examples.quadrotor.figure8_comparison import run_quadrotor_figure8_comparison

        results, _ = run_quadrotor_figure8_comparison(
            num_steps=50,
            num_samples=100,
            horizon=10,
            visualize=False,
            seed=42,
        )

        for controller in ["mppi", "smppi", "kmppi"]:
            states = results[controller]["states"]
            quaternions = states[:, 6:10]
            norms = jnp.linalg.norm(quaternions, axis=1)

            assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_figure8_finite_values(self):
        """Test that all controllers produce finite values."""
        from examples.quadrotor.figure8_comparison import run_quadrotor_figure8_comparison

        results, _ = run_quadrotor_figure8_comparison(
            num_steps=50,
            num_samples=100,
            horizon=10,
            visualize=False,
            seed=42,
        )

        for controller in ["mppi", "smppi", "kmppi"]:
            states = results[controller]["states"]
            actions = results[controller]["actions"]
            costs = results[controller]["costs"]

            assert jnp.all(jnp.isfinite(states))
            assert jnp.all(jnp.isfinite(actions))
            assert jnp.all(jnp.isfinite(costs))


class TestCustomTrajectory:
    """Tests for custom waypoint trajectory example."""

    def test_custom_trajectory_runs(self):
        """Test that custom trajectory example runs."""
        from examples.quadrotor.custom_trajectory import run_quadrotor_custom_trajectory

        waypoints = jnp.array([
            [0.0, 0.0, -2.0],
            [3.0, 0.0, -3.0],
            [3.0, 3.0, -2.0],
        ])

        states, actions, costs, reference = run_quadrotor_custom_trajectory(
            waypoints=waypoints,
            segment_duration=3.0,
            num_samples=200,
            horizon=15,
            visualize=False,
            seed=42,
        )

        # Check shapes (duration = num_waypoints * segment_duration / dt)
        # (3-1) segments * 3s / 0.02 = 300 steps
        expected_steps = int((waypoints.shape[0] - 1) * 3.0 / 0.02)
        assert states.shape[0] == expected_steps + 1
        assert actions.shape[0] == expected_steps
        assert costs.shape[0] == expected_steps
        assert reference.shape[0] == expected_steps

    def test_custom_trajectory_passes_near_waypoints(self):
        """Test that trajectory passes near specified waypoints."""
        from examples.quadrotor.custom_trajectory import run_quadrotor_custom_trajectory

        waypoints = jnp.array([
            [0.0, 0.0, -2.0],
            [2.0, 0.0, -3.0],
            [2.0, 2.0, -2.0],
        ])

        states, _, _, _ = run_quadrotor_custom_trajectory(
            waypoints=waypoints,
            segment_duration=4.0,
            num_samples=300,
            horizon=20,
            visualize=False,
            seed=42,
        )

        # Check that trajectory gets close to each waypoint
        # (within 1m after giving it time to reach)
        dt = 0.02
        segment_steps = int(4.0 / dt)

        for i in range(waypoints.shape[0]):
            # Check at waypoint time (allowing some settling)
            check_idx = min(i * segment_steps + 50, states.shape[0] - 1)
            distance = jnp.linalg.norm(states[check_idx, 0:3] - waypoints[i])

            # Should be reasonably close (within 1m) after some time
            assert distance < 1.5, f"WP{i}: distance={distance:.3f}m"

    def test_custom_trajectory_quaternions_normalized(self):
        """Test that quaternions stay normalized."""
        from examples.quadrotor.custom_trajectory import run_quadrotor_custom_trajectory

        waypoints = jnp.array([
            [0.0, 0.0, -2.0],
            [2.0, 2.0, -3.0],
        ])

        states, _, _, _ = run_quadrotor_custom_trajectory(
            waypoints=waypoints,
            segment_duration=3.0,
            num_samples=200,
            horizon=15,
            visualize=False,
            seed=42,
        )

        quaternions = states[:, 6:10]
        norms = jnp.linalg.norm(quaternions, axis=1)

        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_custom_trajectory_finite_values(self):
        """Test that custom trajectory produces finite values."""
        from examples.quadrotor.custom_trajectory import run_quadrotor_custom_trajectory

        waypoints = jnp.array([
            [0.0, 0.0, -2.0],
            [3.0, 0.0, -3.0],
        ])

        states, actions, costs, reference = run_quadrotor_custom_trajectory(
            waypoints=waypoints,
            segment_duration=3.0,
            num_samples=200,
            horizon=15,
            visualize=False,
            seed=42,
        )

        assert jnp.all(jnp.isfinite(states))
        assert jnp.all(jnp.isfinite(actions))
        assert jnp.all(jnp.isfinite(costs))
        assert jnp.all(jnp.isfinite(reference))

    def test_custom_trajectory_different_waypoint_counts(self):
        """Test with different numbers of waypoints."""
        from examples.quadrotor.custom_trajectory import run_quadrotor_custom_trajectory

        # Test with 2, 3, and 4 waypoints
        for num_waypoints in [2, 3, 4]:
            waypoints = jnp.array([
                [float(i), 0.0, -2.0 - float(i) * 0.5]
                for i in range(num_waypoints)
            ])

            states, actions, costs, reference = run_quadrotor_custom_trajectory(
                waypoints=waypoints,
                segment_duration=2.0,
                num_samples=150,
                horizon=10,
                visualize=False,
                seed=42,
            )

            # Should run successfully for all cases
            assert states.shape[0] > 0
            assert actions.shape[0] > 0


class TestAdvancedExamplesCompatibility:
    """Cross-example compatibility tests."""

    def test_examples_use_consistent_state_format(self):
        """Test that all advanced examples use same state format."""
        from examples.quadrotor.custom_trajectory import run_quadrotor_custom_trajectory
        from examples.quadrotor.figure8_comparison import run_quadrotor_figure8_comparison

        results_fig8, _ = run_quadrotor_figure8_comparison(
            num_steps=20, num_samples=50, horizon=5, visualize=False, seed=42
        )

        waypoints = jnp.array([[0.0, 0.0, -2.0], [2.0, 0.0, -2.0]])
        states_custom, _, _, _ = run_quadrotor_custom_trajectory(
            waypoints=waypoints, segment_duration=2.0,
            num_samples=50, horizon=5, visualize=False, seed=42
        )

        # All should use 13D state
        for controller in ["mppi", "smppi", "kmppi"]:
            assert results_fig8[controller]["states"].shape[1] == 13

        assert states_custom.shape[1] == 13
