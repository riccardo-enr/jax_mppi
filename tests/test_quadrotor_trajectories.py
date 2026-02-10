"""Unit tests for quadrotor trajectory generators."""

import jax.numpy as jnp
import pytest

from examples.quadrotor.trajectories import (
    compute_trajectory_metrics,
    generate_circle_trajectory,
    generate_helix_trajectory,
    generate_hover_setpoint,
    generate_lemniscate_trajectory,
    generate_waypoint_trajectory,
)


class TestHoverSetpoint:
    """Tests for hover setpoint generation."""

    def test_hover_creates_constant_position(self):
        """Test that hover setpoint has constant position."""
        position = jnp.array([1.0, 2.0, -5.0])
        duration = 5.0
        dt = 0.01

        trajectory = generate_hover_setpoint(position, duration, dt)

        # All positions should be the same
        positions = trajectory[:, 0:3]
        for i in range(positions.shape[0]):
            assert jnp.allclose(positions[i], position, atol=1e-6)

    def test_hover_has_zero_velocity(self):
        """Test that hover setpoint has zero velocity."""
        position = jnp.array([0.0, 0.0, -3.0])
        duration = 10.0
        dt = 0.01

        trajectory = generate_hover_setpoint(position, duration, dt)

        # All velocities should be zero
        velocities = trajectory[:, 3:6]
        assert jnp.allclose(velocities, 0.0, atol=1e-6)

    def test_hover_duration(self):
        """Test that hover trajectory has correct duration."""
        position = jnp.array([0.0, 0.0, 0.0])
        duration = 5.0
        dt = 0.01

        trajectory = generate_hover_setpoint(position, duration, dt)

        expected_steps = int(jnp.ceil(duration / dt))
        assert trajectory.shape[0] == expected_steps

    def test_hover_shape(self):
        """Test that hover trajectory has correct shape."""
        position = jnp.array([1.0, 2.0, 3.0])
        trajectory = generate_hover_setpoint(position, 1.0, 0.01)

        assert trajectory.shape[1] == 6  # [px, py, pz, vx, vy, vz]


class TestCircleTrajectory:
    """Tests for circular trajectory generation."""

    def test_circle_starts_at_correct_position(self):
        """Test that circle starts at radius from center."""
        radius = 3.0
        height = -5.0
        center = jnp.array([1.0, 2.0])

        trajectory = generate_circle_trajectory(
            radius, height, period=10.0, duration=10.0, dt=0.01, center=center
        )

        # First position should be at [center[0] + radius, center[1], height]
        start_pos = trajectory[0, 0:3]
        expected = jnp.array([center[0] + radius, center[1], height])
        assert jnp.allclose(start_pos, expected, atol=1e-6)

    def test_circle_constant_altitude(self):
        """Test that circular trajectory maintains constant altitude."""
        radius = 2.0
        height = -10.0

        trajectory = generate_circle_trajectory(
            radius, height, period=5.0, duration=10.0, dt=0.01
        )

        # All z positions should be constant
        z_positions = trajectory[:, 2]
        assert jnp.allclose(z_positions, height, atol=1e-6)

    def test_circle_maintains_radius(self):
        """Test that trajectory stays at constant radius from center."""
        radius = 4.0
        height = -5.0
        center = jnp.array([0.0, 0.0])

        trajectory = generate_circle_trajectory(
            radius, height, period=10.0, duration=10.0, dt=0.01
        )

        # Compute distance from center for all points
        positions = trajectory[:, 0:2]  # xy positions
        distances = jnp.linalg.norm(positions - center, axis=1)

        assert jnp.allclose(distances, radius, atol=0.01)

    def test_circle_periodicity(self):
        """Test that trajectory repeats after one period."""
        radius = 3.0
        height = -5.0
        period = 5.0

        trajectory = generate_circle_trajectory(
            radius, height, period=period, duration=period * 2, dt=0.01
        )

        steps_per_period = int(period / 0.01)

        # First point should match point after one period
        start = trajectory[0]
        after_period = trajectory[steps_per_period]

        assert jnp.allclose(start, after_period, atol=0.1)

    def test_circle_velocity_tangent(self):
        """Test that velocity is tangent to circle."""
        radius = 2.0
        height = -5.0

        trajectory = generate_circle_trajectory(
            radius, height, period=10.0, duration=5.0, dt=0.01
        )

        # At several points, check velocity is perpendicular to position
        for i in [0, 50, 100, 200]:
            pos = trajectory[i, 0:2]  # xy position
            vel = trajectory[i, 3:5]  # xy velocity

            # Dot product should be near zero (perpendicular)
            dot_product = jnp.dot(pos, vel)
            assert jnp.abs(dot_product) < 0.5

    def test_circle_phase_shift(self):
        """Test that phase parameter shifts starting position."""
        radius = 3.0
        height = -5.0

        traj1 = generate_circle_trajectory(
            radius, height, period=10.0, duration=1.0, dt=0.01, phase=0.0
        )
        traj2 = generate_circle_trajectory(
            radius, height, period=10.0, duration=1.0, dt=0.01, phase=jnp.pi / 2
        )

        # Starting positions should be different
        assert not jnp.allclose(traj1[0, 0:3], traj2[0, 0:3], atol=0.1)


class TestLemniscateTrajectory:
    """Tests for figure-8 (lemniscate) trajectory generation."""

    def test_lemniscate_passes_through_origin(self):
        """Test that lemniscate passes through center."""
        scale = 4.0
        height = -5.0
        center = jnp.array([0.0, 0.0])

        trajectory = generate_lemniscate_trajectory(
            scale, height, period=10.0, duration=10.0, dt=0.01, center=center
        )

        # Find point closest to center
        positions = trajectory[:, 0:2]
        distances = jnp.linalg.norm(positions - center, axis=1)
        min_distance = jnp.min(distances)

        # Should pass through or very close to center
        assert min_distance < 0.1

    def test_lemniscate_constant_altitude_xy(self):
        """Test that horizontal lemniscate maintains constant altitude."""
        scale = 3.0
        height = -8.0

        trajectory = generate_lemniscate_trajectory(
            scale, height, period=15.0, duration=15.0, dt=0.01, axis="xy"
        )

        # All z positions should be constant
        z_positions = trajectory[:, 2]
        assert jnp.allclose(z_positions, height, atol=1e-6)

    def test_lemniscate_vertical_varies_altitude(self):
        """Test that vertical lemniscate varies altitude."""
        scale = 3.0
        height = -5.0

        trajectory = generate_lemniscate_trajectory(
            scale, height, period=15.0, duration=15.0, dt=0.01, axis="xz"
        )

        # Z positions should vary
        z_positions = trajectory[:, 2]
        z_range = jnp.max(z_positions) - jnp.min(z_positions)
        assert z_range > 1.0  # Should vary by at least 1m

    def test_lemniscate_periodicity(self):
        """Test that lemniscate repeats after one period."""
        scale = 4.0
        height = -5.0
        period = 8.0

        trajectory = generate_lemniscate_trajectory(
            scale, height, period=period, duration=period * 2, dt=0.01
        )

        steps_per_period = int(period / 0.01)

        # Compare start and after one period
        start = trajectory[0]
        after_period = trajectory[steps_per_period]

        assert jnp.allclose(start, after_period, atol=0.1)

    def test_lemniscate_symmetry(self):
        """Test that lemniscate is symmetric about center."""
        scale = 3.0
        height = -5.0
        period = 10.0

        trajectory = generate_lemniscate_trajectory(
            scale, height, period=period, duration=period, dt=0.01
        )

        # At quarter period, should be at maximum x
        quarter_period_idx = int((period / 4) / 0.01)
        pos_quarter = trajectory[quarter_period_idx, 0]  # x position

        # Should be near maximum scale value
        assert jnp.abs(pos_quarter) > scale * 0.9  # Within 10% of scale

    def test_lemniscate_invalid_axis_raises_error(self):
        """Test that invalid axis raises ValueError."""
        with pytest.raises(ValueError):
            generate_lemniscate_trajectory(
                3.0, -5.0, period=10.0, duration=10.0, dt=0.01, axis="invalid"
            )


class TestHelixTrajectory:
    """Tests for helical trajectory generation."""

    def test_helix_maintains_radius(self):
        """Test that helix maintains constant radius."""
        radius = 2.5
        height_rate = -0.5
        center = jnp.array([0.0, 0.0])

        trajectory = generate_helix_trajectory(
            radius,
            height_rate,
            period=10.0,
            duration=10.0,
            dt=0.01,
            center=center
        )

        # Check horizontal distance from center
        positions = trajectory[:, 0:2]
        distances = jnp.linalg.norm(positions - center, axis=1)

        assert jnp.allclose(distances, radius, atol=0.01)

    def test_helix_vertical_motion(self):
        """Test that helix changes altitude at correct rate."""
        radius = 2.0
        height_rate = -0.3  # Climbing at 0.3 m/s
        start_height = -2.0
        duration = 10.0

        trajectory = generate_helix_trajectory(
            radius, height_rate, period=5.0, duration=duration, dt=0.01,
            start_height=start_height
        )

        # Check altitude change
        start_z = trajectory[0, 2]
        end_z = trajectory[-1, 2]
        expected_change = height_rate * duration

        assert jnp.allclose(end_z - start_z, expected_change, atol=0.1)

    def test_helix_constant_vertical_velocity(self):
        """Test that helix has constant vertical velocity."""
        radius = 3.0
        height_rate = 0.5

        trajectory = generate_helix_trajectory(
            radius, height_rate, period=10.0, duration=10.0, dt=0.01
        )

        # All vertical velocities should be constant
        vz = trajectory[:, 5]
        assert jnp.allclose(vz, height_rate, atol=1e-6)

    def test_helix_circular_in_xy(self):
        """Test that helix projection on xy is circular."""
        radius = 3.0
        height_rate = -0.2

        trajectory = generate_helix_trajectory(
            radius, height_rate, period=8.0, duration=8.0, dt=0.01
        )

        # xy positions should form a circle
        positions = trajectory[:, 0:2]
        distances = jnp.linalg.norm(positions, axis=1)

        assert jnp.allclose(distances, radius, atol=0.01)


class TestWaypointTrajectory:
    """Tests for waypoint trajectory generation."""

    def test_waypoint_passes_through_waypoints(self):
        """Test that trajectory passes through specified waypoints."""
        waypoints = jnp.array([
            [0.0, 0.0, -2.0],
            [5.0, 0.0, -5.0],
            [5.0, 5.0, -5.0],
        ])

        trajectory = generate_waypoint_trajectory(
            waypoints, segment_duration=5.0, dt=0.01
        )

        # Check start and end points
        assert jnp.allclose(trajectory[0, 0:3], waypoints[0], atol=0.01)
        assert jnp.allclose(trajectory[-1, 0:3], waypoints[-1], atol=0.1)

    def test_waypoint_smooth_transitions(self):
        """Test that waypoint trajectory has continuous velocity."""
        waypoints = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

        trajectory = generate_waypoint_trajectory(
            waypoints, segment_duration=2.0, dt=0.01
        )

        # Check velocity continuity (no large jumps)
        velocities = trajectory[:, 3:6]
        vel_diff = jnp.diff(velocities, axis=0)
        max_vel_change = jnp.max(jnp.linalg.norm(vel_diff, axis=1))

        # Velocity should change smoothly
        assert max_vel_change < 1.0  # No large jumps

    def test_waypoint_requires_minimum_waypoints(self):
        """Test that fewer than 2 waypoints raises error."""
        waypoints = jnp.array([[0.0, 0.0, 0.0]])

        with pytest.raises(ValueError):
            generate_waypoint_trajectory(
                waypoints, segment_duration=5.0, dt=0.01
            )

    def test_waypoint_with_specified_velocities(self):
        """Test waypoint trajectory with specified velocities."""
        waypoints = jnp.array([
            [0.0, 0.0, -2.0],
            [5.0, 0.0, -2.0],
        ])
        velocities = jnp.array([
            [0.0, 0.0, 0.0],  # Start from rest
            [0.0, 0.0, 0.0],  # End at rest
        ])

        trajectory = generate_waypoint_trajectory(
            waypoints, velocities=velocities, segment_duration=5.0, dt=0.01
        )

        # Start and end velocities should be near zero
        assert jnp.allclose(trajectory[0, 3:6], 0.0, atol=0.1)
        assert jnp.allclose(trajectory[-1, 3:6], 0.0, atol=0.1)


class TestTrajectoryMetrics:
    """Tests for trajectory metrics computation."""

    def test_metrics_circle_distance(self):
        """Test that circle trajectory distance is approximately correct."""
        radius = 3.0
        period = 10.0

        trajectory = generate_circle_trajectory(
            radius, -5.0, period, duration=period, dt=0.01
        )

        metrics = compute_trajectory_metrics(trajectory, dt=0.01)

        # One complete circle: distance ≈ 2πr
        expected_distance = 2 * jnp.pi * radius
        assert jnp.abs(metrics["total_distance"] - expected_distance) < 0.5

    def test_metrics_hover_zero_velocity(self):
        """Test that hover has zero velocities."""
        position = jnp.array([0.0, 0.0, -5.0])
        trajectory = generate_hover_setpoint(position, duration=5.0, dt=0.01)

        metrics = compute_trajectory_metrics(trajectory, dt=0.01)

        assert metrics["max_velocity"] < 1e-6
        assert metrics["avg_velocity"] < 1e-6

    def test_metrics_returns_expected_keys(self):
        """Test that metrics dictionary has all expected keys."""
        trajectory = generate_circle_trajectory(
            2.0, -5.0, 10.0, duration=5.0, dt=0.01
        )

        metrics = compute_trajectory_metrics(trajectory, dt=0.01)

        expected_keys = [
            "total_distance",
            "max_velocity",
            "avg_velocity",
            "max_acceleration",
            "avg_acceleration",
        ]

        for key in expected_keys:
            assert key in metrics

    def test_metrics_positive_values(self):
        """Test that all metrics are non-negative."""
        trajectory = generate_lemniscate_trajectory(
            3.0, -5.0, 10.0, duration=10.0, dt=0.01
        )

        metrics = compute_trajectory_metrics(trajectory, dt=0.01)

        for key, value in metrics.items():
            assert value >= 0.0
