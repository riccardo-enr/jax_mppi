"""Unit tests for quadrotor cost functions."""

import jax
import jax.numpy as jnp

from jax_mppi.costs.quadrotor import (
    create_hover_cost,
    create_terminal_cost,
    create_time_indexed_trajectory_cost,
    create_trajectory_tracking_cost,
    quaternion_distance,
)


class TestQuaternionDistance:
    """Tests for quaternion distance metric."""

    def test_identical_quaternions_zero_distance(self):
        """Test that identical quaternions have zero distance."""
        q1 = jnp.array([1.0, 0.0, 0.0, 0.0])
        q2 = jnp.array([1.0, 0.0, 0.0, 0.0])

        dist = quaternion_distance(q1, q2)
        assert jnp.allclose(dist, 0.0, atol=1e-6)

    def test_opposite_quaternions_max_distance(self):
        """Test that opposite quaternions have maximum distance."""
        q1 = jnp.array([1.0, 0.0, 0.0, 0.0])
        q2 = jnp.array([-1.0, 0.0, 0.0, 0.0])

        dist = quaternion_distance(q1, q2)
        # Opposite quaternions represent same rotation, so distance should be 0
        # Due to abs in the distance metric
        assert jnp.allclose(dist, 0.0, atol=1e-6)

    def test_orthogonal_quaternions(self):
        """Test distance for orthogonal quaternions."""
        q1 = jnp.array([1.0, 0.0, 0.0, 0.0])
        q2 = jnp.array([0.0, 1.0, 0.0, 0.0])

        dist = quaternion_distance(q1, q2)
        # Orthogonal quaternions: dot product = 0, distance = 1
        assert jnp.allclose(dist, 1.0, atol=1e-6)

    def test_distance_is_symmetric(self):
        """Test that quaternion distance is symmetric."""
        q1 = jnp.array([0.5, 0.5, 0.5, 0.5])
        q2 = jnp.array([0.7, 0.3, 0.3, 0.3])

        # Normalize
        q1 = q1 / jnp.linalg.norm(q1)
        q2 = q2 / jnp.linalg.norm(q2)

        dist1 = quaternion_distance(q1, q2)
        dist2 = quaternion_distance(q2, q1)

        assert jnp.allclose(dist1, dist2, atol=1e-6)

    def test_distance_in_valid_range(self):
        """Test that distance is always in [0, 1]."""
        # Random quaternions
        key = jax.random.PRNGKey(42)
        for _ in range(10):
            key, k1, k2 = jax.random.split(key, 3)
            q1 = jax.random.normal(k1, (4,))
            q2 = jax.random.normal(k2, (4,))

            # Normalize
            q1 = q1 / jnp.linalg.norm(q1)
            q2 = q2 / jnp.linalg.norm(q2)

            dist = quaternion_distance(q1, q2)
            assert 0.0 <= dist <= 1.0


class TestTrajectoryTrackingCost:
    """Tests for trajectory tracking cost function."""

    def test_zero_cost_at_reference(self):
        """Test that cost is zero when state matches reference."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        R = jnp.eye(4) * 0.01

        # Reference at origin with zero velocity
        reference = jnp.zeros((1, 6))

        cost_fn = create_trajectory_tracking_cost(Q_pos, Q_vel, R, reference)

        # State at origin with zero velocity, identity quaternion
        state = jnp.zeros(13)
        state = state.at[6].set(1.0)

        # Zero action
        action = jnp.zeros(4)

        cost = cost_fn(state, action)
        assert jnp.allclose(cost, 0.0, atol=1e-6)

    def test_position_error_increases_cost(self):
        """Test that position error increases cost."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        R = jnp.eye(4) * 0.01

        reference = jnp.zeros((1, 6))

        cost_fn = create_trajectory_tracking_cost(Q_pos, Q_vel, R, reference)

        # State at origin
        state1 = jnp.zeros(13)
        state1 = state1.at[6].set(1.0)

        # State away from origin
        state2 = jnp.zeros(13)
        state2 = state2.at[0].set(1.0)  # px = 1.0
        state2 = state2.at[6].set(1.0)

        action = jnp.zeros(4)

        cost1 = cost_fn(state1, action)
        cost2 = cost_fn(state2, action)

        assert cost2 > cost1

    def test_velocity_error_increases_cost(self):
        """Test that velocity error increases cost."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        R = jnp.eye(4) * 0.01

        reference = jnp.zeros((1, 6))

        cost_fn = create_trajectory_tracking_cost(Q_pos, Q_vel, R, reference)

        # State with zero velocity
        state1 = jnp.zeros(13)
        state1 = state1.at[6].set(1.0)

        # State with non-zero velocity
        state2 = jnp.zeros(13)
        state2 = state2.at[3].set(1.0)  # vx = 1.0
        state2 = state2.at[6].set(1.0)

        action = jnp.zeros(4)

        cost1 = cost_fn(state1, action)
        cost2 = cost_fn(state2, action)

        assert cost2 > cost1

    def test_control_effort_increases_cost(self):
        """Test that control effort increases cost."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        R = jnp.eye(4) * 0.01

        cost_fn = create_trajectory_tracking_cost(Q_pos, Q_vel, R, None)

        state = jnp.zeros(13)
        state = state.at[6].set(1.0)

        action1 = jnp.zeros(4)
        action2 = jnp.array([10.0, 1.0, 1.0, 1.0])

        cost1 = cost_fn(state, action1)
        cost2 = cost_fn(state, action2)

        assert cost2 > cost1

    def test_cost_without_action(self):
        """Test that cost can be computed without action (terminal cost)."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        R = jnp.eye(4) * 0.01

        cost_fn = create_trajectory_tracking_cost(Q_pos, Q_vel, R, None)

        state = jnp.zeros(13)
        state = state.at[0].set(1.0)  # Some position error
        state = state.at[6].set(1.0)

        # Cost without action should only include position/velocity terms
        cost = cost_fn(state, None)
        assert cost > 0.0

    def test_cost_is_differentiable(self):
        """Test that cost is differentiable w.r.t. state and action."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        R = jnp.eye(4) * 0.01

        cost_fn = create_trajectory_tracking_cost(Q_pos, Q_vel, R, None)

        state = jnp.zeros(13)
        state = state.at[6].set(1.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])

        # Gradient w.r.t. state
        grad_state = jax.grad(lambda s: cost_fn(s, action))(state)
        assert grad_state.shape == (13,)

        # Gradient w.r.t. action
        grad_action = jax.grad(lambda a: cost_fn(state, a))(action)
        assert grad_action.shape == (4,)


class TestTimeIndexedTrajectoryCost:
    """Tests for time-indexed trajectory cost function."""

    def test_follows_time_indexed_reference(self):
        """Test that cost uses correct reference at each time step."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        R = jnp.eye(4) * 0.01

        # Create trajectory: moving in x direction
        T = 100
        trajectory = jnp.zeros((T, 6))
        trajectory = trajectory.at[:, 0].set(jnp.arange(T) * 0.1)  # px increases

        cost_fn = create_time_indexed_trajectory_cost(
            Q_pos, Q_vel, R, trajectory
        )

        # State at origin
        state = jnp.zeros(13)
        state = state.at[6].set(1.0)
        action = jnp.zeros(4)

        # Cost at t=0 (reference is origin) should be low
        cost_t0 = cost_fn(state, action, t=0)

        # Cost at t=10 (reference is at x=1.0) should be higher
        cost_t10 = cost_fn(state, action, t=10)

        assert cost_t10 > cost_t0

    def test_bounds_checking(self):
        """Test that time index is bounded to valid range."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        R = jnp.eye(4) * 0.01

        T = 10
        trajectory = jnp.zeros((T, 6))

        cost_fn = create_time_indexed_trajectory_cost(
            Q_pos, Q_vel, R, trajectory
        )

        state = jnp.zeros(13)
        state = state.at[6].set(1.0)
        action = jnp.zeros(4)

        # Should not crash with out-of-bounds index
        cost_negative = cost_fn(state, action, t=-5)
        cost_large = cost_fn(state, action, t=1000)

        # Should return valid costs
        assert jnp.isfinite(cost_negative)
        assert jnp.isfinite(cost_large)


class TestHoverCost:
    """Tests for hover control cost function."""

    def test_zero_cost_at_hover_point(self):
        """Test that cost is zero at hover point with zero velocity."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        Q_att = jnp.eye(4) * 1.0
        R = jnp.eye(4) * 0.01

        hover_position = jnp.array([1.0, 2.0, -5.0])
        hover_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

        cost_fn = create_hover_cost(
            Q_pos, Q_vel, Q_att, R, hover_position, hover_quaternion
        )

        # State at hover position with hover attitude
        state = jnp.zeros(13)
        state = state.at[0:3].set(hover_position)
        state = state.at[6:10].set(hover_quaternion)

        action = jnp.zeros(4)

        cost = cost_fn(state, action)
        assert jnp.allclose(cost, 0.0, atol=1e-6)

    def test_position_error_increases_cost(self):
        """Test that deviation from hover position increases cost."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        Q_att = jnp.eye(4) * 1.0
        R = jnp.eye(4) * 0.01

        hover_position = jnp.zeros(3)

        cost_fn = create_hover_cost(Q_pos, Q_vel, Q_att, R, hover_position)

        # State at hover position
        state1 = jnp.zeros(13)
        state1 = state1.at[6].set(1.0)

        # State away from hover position
        state2 = jnp.zeros(13)
        state2 = state2.at[0].set(2.0)  # px = 2.0
        state2 = state2.at[6].set(1.0)

        action = jnp.zeros(4)

        cost1 = cost_fn(state1, action)
        cost2 = cost_fn(state2, action)

        assert cost2 > cost1

    def test_velocity_increases_cost(self):
        """Test that non-zero velocity increases cost."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        Q_att = jnp.eye(4) * 1.0
        R = jnp.eye(4) * 0.01

        hover_position = jnp.zeros(3)

        cost_fn = create_hover_cost(Q_pos, Q_vel, Q_att, R, hover_position)

        # State with zero velocity
        state1 = jnp.zeros(13)
        state1 = state1.at[6].set(1.0)

        # State with velocity
        state2 = jnp.zeros(13)
        state2 = state2.at[3].set(1.0)  # vx = 1.0
        state2 = state2.at[6].set(1.0)

        action = jnp.zeros(4)

        cost1 = cost_fn(state1, action)
        cost2 = cost_fn(state2, action)

        assert cost2 > cost1

    def test_attitude_error_increases_cost(self):
        """Test that attitude deviation increases cost."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        Q_att = jnp.eye(4) * 10.0
        R = jnp.eye(4) * 0.01

        hover_position = jnp.zeros(3)
        hover_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

        cost_fn = create_hover_cost(
            Q_pos, Q_vel, Q_att, R, hover_position, hover_quaternion
        )

        # State with hover attitude
        state1 = jnp.zeros(13)
        state1 = state1.at[6:10].set(hover_quaternion)

        # State with different attitude (90 deg rotation around x)
        state2 = jnp.zeros(13)
        rotated_quat = jnp.array([0.7071, 0.7071, 0.0, 0.0])  # 90 deg around x
        state2 = state2.at[6:10].set(rotated_quat)

        action = jnp.zeros(4)

        cost1 = cost_fn(state1, action)
        cost2 = cost_fn(state2, action)

        assert cost2 > cost1


class TestTerminalCost:
    """Tests for terminal cost function."""

    def test_zero_cost_at_goal(self):
        """Test that cost is zero at goal state."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        Q_att = jnp.eye(4) * 1.0

        goal_position = jnp.array([5.0, 3.0, -10.0])
        goal_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

        terminal_cost = create_terminal_cost(
            Q_pos, Q_vel, Q_att, goal_position, goal_quaternion
        )

        # State at goal
        state = jnp.zeros(13)
        state = state.at[0:3].set(goal_position)
        state = state.at[6:10].set(goal_quaternion)

        cost = terminal_cost(state, None)
        assert jnp.allclose(cost, 0.0, atol=1e-6)

    def test_position_error_increases_terminal_cost(self):
        """Test that position error at terminal state increases cost."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        Q_att = jnp.eye(4) * 1.0

        goal_position = jnp.zeros(3)

        terminal_cost = create_terminal_cost(Q_pos, Q_vel, Q_att, goal_position)

        # State at goal
        state1 = jnp.zeros(13)
        state1 = state1.at[6].set(1.0)

        # State away from goal
        state2 = jnp.zeros(13)
        state2 = state2.at[0].set(5.0)
        state2 = state2.at[6].set(1.0)

        cost1 = terminal_cost(state1, None)
        cost2 = terminal_cost(state2, None)

        assert cost2 > cost1

    def test_velocity_at_goal_penalized(self):
        """Test that arriving at goal with velocity is penalized."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 10.0  # High velocity penalty
        Q_att = jnp.eye(4) * 1.0

        goal_position = jnp.zeros(3)

        terminal_cost = create_terminal_cost(Q_pos, Q_vel, Q_att, goal_position)

        # State at goal with zero velocity
        state1 = jnp.zeros(13)
        state1 = state1.at[6].set(1.0)

        # State at goal with velocity
        state2 = jnp.zeros(13)
        state2 = state2.at[3].set(2.0)  # vx = 2.0
        state2 = state2.at[6].set(1.0)

        cost1 = terminal_cost(state1, None)
        cost2 = terminal_cost(state2, None)

        assert cost2 > cost1

    def test_terminal_cost_is_jit_compatible(self):
        """Test that terminal cost can be JIT compiled."""
        Q_pos = jnp.eye(3) * 10.0
        Q_vel = jnp.eye(3) * 1.0
        Q_att = jnp.eye(4) * 1.0

        goal_position = jnp.zeros(3)

        terminal_cost = create_terminal_cost(Q_pos, Q_vel, Q_att, goal_position)
        terminal_cost_jit = jax.jit(terminal_cost)

        state = jnp.zeros(13)
        state = state.at[6].set(1.0)

        cost = terminal_cost_jit(state, None)
        assert jnp.isfinite(cost)
