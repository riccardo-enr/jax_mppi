"""Integration tests for quadrotor examples."""

import jax.numpy as jnp
import pytest


class TestQuadrotorHoverExample:
    """Tests for quadrotor hover control example."""

    def test_hover_example_runs(self):
        """Test that hover example runs without errors."""
        from examples.quadrotor_hover import run_quadrotor_hover

        states, actions, costs = run_quadrotor_hover(
            num_steps=50,
            num_samples=100,
            horizon=10,
            visualize=False,
            seed=42,
        )

        # Check shapes
        assert states.shape == (51, 13)  # num_steps + 1
        assert actions.shape == (50, 4)
        assert costs.shape == (50,)

    def test_hover_converges_to_setpoint(self):
        """Test that hover controller converges to setpoint."""
        from examples.quadrotor_hover import run_quadrotor_hover

        states, actions, costs = run_quadrotor_hover(
            num_steps=300,
            num_samples=500,
            horizon=20,
            visualize=False,
            seed=42,
        )

        # Target hover position
        hover_position = jnp.array([0.0, 0.0, -5.0])

        # Check final position error
        final_pos = states[-1, 0:3]
        pos_error = jnp.linalg.norm(final_pos - hover_position)

        # Should be within 0.5m of target after 300 steps
        assert pos_error < 0.5

        # Check final velocity is small
        final_vel = states[-1, 3:6]
        vel_magnitude = jnp.linalg.norm(final_vel)

        # Velocity should be small (< 0.5 m/s)
        assert vel_magnitude < 0.5

    def test_hover_quaternion_remains_normalized(self):
        """Test that quaternion stays normalized during hover."""
        from examples.quadrotor_hover import run_quadrotor_hover

        states, _, _ = run_quadrotor_hover(
            num_steps=100,
            num_samples=200,
            horizon=15,
            visualize=False,
            seed=42,
        )

        # Check quaternion norms
        quaternions = states[:, 6:10]
        norms = jnp.linalg.norm(quaternions, axis=1)

        # All quaternions should have unit norm
        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_hover_cost_decreases(self):
        """Test that cost generally decreases over time."""
        from examples.quadrotor_hover import run_quadrotor_hover

        _, _, costs = run_quadrotor_hover(
            num_steps=200,
            num_samples=500,
            horizon=20,
            visualize=False,
            seed=42,
        )

        # Cost at start should be higher than at end
        initial_avg_cost = jnp.mean(costs[0:20])
        final_avg_cost = jnp.mean(costs[-20:])

        assert final_avg_cost < initial_avg_cost


class TestQuadrotorCircleExample:
    """Tests for quadrotor circle tracking example."""

    def test_circle_example_runs(self):
        """Test that circle example runs without errors."""
        from examples.quadrotor_circle import run_quadrotor_circle

        states, actions, costs, reference = run_quadrotor_circle(
            num_steps=50,
            num_samples=100,
            horizon=10,
            visualize=False,
            seed=42,
        )

        # Check shapes
        assert states.shape == (51, 13)
        assert actions.shape == (50, 4)
        assert costs.shape == (50,)
        assert reference.shape == (50, 6)

    def test_circle_tracks_reference(self):
        """Test that circle controller tracks reference trajectory."""
        from examples.quadrotor_circle import run_quadrotor_circle

        states, _, _, reference = run_quadrotor_circle(
            num_steps=500,
            num_samples=500,
            horizon=20,
            radius=3.0,
            period=15.0,
            visualize=False,
            seed=42,
        )

        # Compute tracking error (skip first 100 steps for transient)
        pos_errors = jnp.linalg.norm(
            states[100:-1, 0:3] - reference[100:, 0:3], axis=1
        )

        # Mean tracking error should be reasonable (< 1m after settling)
        mean_error = jnp.mean(pos_errors)
        assert mean_error < 1.0

    def test_circle_maintains_altitude(self):
        """Test that circle tracking maintains approximately constant altitude."""
        from examples.quadrotor_circle import run_quadrotor_circle

        states, _, _, reference = run_quadrotor_circle(
            num_steps=300,
            num_samples=300,
            horizon=15,
            visualize=False,
            seed=42,
        )

        # Check altitude variation (skip transient)
        z_positions = states[50:, 2]
        z_std = jnp.std(z_positions)

        # Altitude should be relatively constant (std < 0.5m)
        assert z_std < 0.5

    def test_circle_quaternion_normalized(self):
        """Test that quaternion stays normalized during circle tracking."""
        from examples.quadrotor_circle import run_quadrotor_circle

        states, _, _, _ = run_quadrotor_circle(
            num_steps=100,
            num_samples=200,
            horizon=15,
            visualize=False,
            seed=42,
        )

        # Check quaternion norms
        quaternions = states[:, 6:10]
        norms = jnp.linalg.norm(quaternions, axis=1)

        # All quaternions should have unit norm
        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_circle_different_parameters(self):
        """Test that circle example works with different parameters."""
        from examples.quadrotor_circle import run_quadrotor_circle

        # Test with smaller radius and faster period
        states, actions, costs, reference = run_quadrotor_circle(
            num_steps=100,
            num_samples=300,
            horizon=15,
            radius=2.0,
            period=8.0,
            visualize=False,
            seed=42,
        )

        # Should still run successfully
        assert states.shape == (101, 13)
        assert actions.shape == (100, 4)
        assert costs.shape == (100,)


class TestQuadrotorExamplesCompatibility:
    """Tests for compatibility and consistency across examples."""

    def test_examples_use_same_state_format(self):
        """Test that both examples use consistent state format."""
        from examples.quadrotor_circle import run_quadrotor_circle
        from examples.quadrotor_hover import run_quadrotor_hover

        states_hover, _, _ = run_quadrotor_hover(
            num_steps=10, num_samples=50, horizon=5, visualize=False, seed=42
        )

        states_circle, _, _, _ = run_quadrotor_circle(
            num_steps=10, num_samples=50, horizon=5, visualize=False, seed=42
        )

        # Both should have same state dimension
        assert states_hover.shape[1] == states_circle.shape[1] == 13

    def test_examples_produce_finite_values(self):
        """Test that examples don't produce NaN or Inf values."""
        from examples.quadrotor_circle import run_quadrotor_circle
        from examples.quadrotor_hover import run_quadrotor_hover

        states_hover, actions_hover, costs_hover = run_quadrotor_hover(
            num_steps=50, num_samples=100, horizon=10, visualize=False, seed=42
        )

        states_circle, actions_circle, costs_circle, _ = run_quadrotor_circle(
            num_steps=50, num_samples=100, horizon=10, visualize=False, seed=42
        )

        # Check hover
        assert jnp.all(jnp.isfinite(states_hover))
        assert jnp.all(jnp.isfinite(actions_hover))
        assert jnp.all(jnp.isfinite(costs_hover))

        # Check circle
        assert jnp.all(jnp.isfinite(states_circle))
        assert jnp.all(jnp.isfinite(actions_circle))
        assert jnp.all(jnp.isfinite(costs_circle))
