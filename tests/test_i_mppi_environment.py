"""Tests for the I-MPPI environment module (FOV, LOS, dynamics, costs)."""

import jax
import jax.numpy as jnp

from jax_mppi.i_mppi.environment import (
    DEPLETION_ALPHA,
    GOAL_POS,
    INFO_ZONES,
    _fov_coverage,
    _fov_coverage_with_los,
    _line_of_sight_grid,
    augmented_dynamics,
    augmented_dynamics_with_grid,
    dist_rect,
    informative_running_cost,
    quat_to_yaw,
    running_cost,
)
from jax_mppi.i_mppi.map import rasterize_environment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_quad_state(x=0.0, y=0.0, z=-2.0):
    """13D quadrotor state at (x,y,z) with identity quat, zero vel/omega."""
    s = jnp.zeros(13)
    s = s.at[0].set(x)
    s = s.at[1].set(y)
    s = s.at[2].set(z)
    s = s.at[6].set(1.0)  # qw = 1 (identity quaternion)
    return s


def _make_augmented_state(x, y, z=-2.0, yaw=0.0, info_levels=None):
    """16D augmented state: 13 quad + 3 info levels."""
    quad = _identity_quad_state(x, y, z)
    # Set yaw via quaternion: qw=cos(yaw/2), qz=sin(yaw/2)
    quad = quad.at[6].set(jnp.cos(yaw / 2))
    quad = quad.at[9].set(jnp.sin(yaw / 2))
    if info_levels is None:
        info_levels = jnp.array([100.0, 100.0, 100.0])
    return jnp.concatenate([quad, info_levels])


def _make_test_grid():
    """10x10 grid at 0.5m (5m x 5m) with a wall column at x=2.5.

    Non-wall cells are set to 0.5 (unknown) so that entropy-weighted
    FOV coverage gives non-zero values for visible unknown areas.
    """
    origin = jnp.array([0.0, 0.0])
    grid = 0.5 * jnp.ones((10, 10))
    # Wall column at col=5 (x=2.5m)
    grid = grid.at[:, 5].set(1.0)
    return grid, origin, 0.5


# ---------------------------------------------------------------------------
# TestDistRect (existing + additions)
# ---------------------------------------------------------------------------

def test_dist_rect():
    center = jnp.array([0.0, 0.0])
    size = jnp.array([2.0, 2.0])

    # Inside
    assert dist_rect(jnp.array([0.0, 0.0]), center, size) == 0.0
    assert dist_rect(jnp.array([0.5, 0.5]), center, size) == 0.0
    assert dist_rect(jnp.array([0.9, 0.9]), center, size) == 0.0

    # Boundary
    assert jnp.isclose(
        dist_rect(jnp.array([1.0, 0.0]), center, size), 0.0, atol=1e-6
    )

    # Outside
    assert jnp.isclose(dist_rect(jnp.array([2.0, 0.0]), center, size), 1.0)
    assert jnp.isclose(dist_rect(jnp.array([0.0, 2.0]), center, size), 1.0)

    # Corner
    assert jnp.isclose(
        dist_rect(jnp.array([2.0, 2.0]), center, size), jnp.sqrt(2.0)
    )


def test_dist_rect_asymmetric():
    center = jnp.array([3.0, 4.0])
    size = jnp.array([4.0, 2.0])  # half: 2.0 x 1.0

    # Inside center
    assert dist_rect(jnp.array([3.0, 4.0]), center, size) == 0.0
    # Outside x-axis
    d = dist_rect(jnp.array([6.0, 4.0]), center, size)
    assert jnp.isclose(d, 1.0)  # |6-3| - 2 = 1
    # Outside y-axis
    d = dist_rect(jnp.array([3.0, 6.0]), center, size)
    assert jnp.isclose(d, 1.0)  # |6-4| - 1 = 1


# ---------------------------------------------------------------------------
# TestQuatToYaw
# ---------------------------------------------------------------------------

class TestQuatToYaw:
    def test_identity(self):
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        assert jnp.isclose(quat_to_yaw(q), 0.0, atol=1e-6)

    def test_90_deg(self):
        # 90° about z: qw=cos(45°), qz=sin(45°)
        q = jnp.array([jnp.cos(jnp.pi / 4), 0.0, 0.0, jnp.sin(jnp.pi / 4)])
        assert jnp.isclose(quat_to_yaw(q), jnp.pi / 2, atol=1e-5)

    def test_negative_yaw(self):
        # -45° about z
        angle = -jnp.pi / 4
        q = jnp.array([jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)])
        assert jnp.isclose(quat_to_yaw(q), angle, atol=1e-5)

    def test_small_pitch_no_yaw(self):
        # 30° pitch (about y): qw=cos(15°), qy=sin(15°) — no yaw
        angle = jnp.pi / 6
        q = jnp.array([jnp.cos(angle / 2), 0.0, jnp.sin(angle / 2), 0.0])
        assert jnp.isclose(quat_to_yaw(q), 0.0, atol=1e-5)

    def test_pure_roll_no_yaw(self):
        # 90° roll (about x): qw=cos(45°), qx=sin(45°)
        q = jnp.array([jnp.cos(jnp.pi / 4), jnp.sin(jnp.pi / 4), 0.0, 0.0])
        assert jnp.isclose(quat_to_yaw(q), 0.0, atol=1e-5)

    def test_jit(self):
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        assert jnp.isclose(jax.jit(quat_to_yaw)(q), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# TestFOVCoverage
# ---------------------------------------------------------------------------

class TestFOVCoverage:
    def test_facing_nearby_zone(self):
        # UAV 1m away, facing right toward a small zone
        pos = jnp.array([0.0, 0.0])
        yaw = 0.0
        zone_center = jnp.array([1.0, 0.0])
        zone_size = jnp.array([0.5, 0.5])
        cov = _fov_coverage(pos, yaw, zone_center, zone_size)
        assert cov > 0.5

    def test_facing_away(self):
        pos = jnp.array([0.0, 0.0])
        yaw = jnp.pi  # facing left, zone is to the right
        zone_center = jnp.array([1.0, 0.0])
        zone_size = jnp.array([0.5, 0.5])
        cov = _fov_coverage(pos, yaw, zone_center, zone_size)
        assert cov < 0.1

    def test_zone_beyond_range(self):
        pos = jnp.array([0.0, 0.0])
        yaw = 0.0
        zone_center = jnp.array([5.0, 0.0])  # > SENSOR_MAX_RANGE
        zone_size = jnp.array([0.5, 0.5])
        cov = _fov_coverage(pos, yaw, zone_center, zone_size)
        assert jnp.isclose(cov, 0.0, atol=1e-5)

    def test_large_zone_partial(self):
        pos = jnp.array([0.0, 0.0])
        yaw = 0.0
        zone_center = jnp.array([2.0, 0.0])
        zone_size = jnp.array([4.0, 4.0])  # extends past range and FOV
        cov = _fov_coverage(pos, yaw, zone_center, zone_size)
        assert 0.0 < float(cov) < 1.0

    def test_tiny_zone_in_front(self):
        pos = jnp.array([0.0, 0.0])
        yaw = 0.0
        zone_center = jnp.array([0.5, 0.0])
        zone_size = jnp.array([0.1, 0.1])
        cov = _fov_coverage(pos, yaw, zone_center, zone_size)
        assert cov > 0.8

    def test_range_01(self):
        pos = jnp.array([0.0, 0.0])
        zone_center = jnp.array([1.0, 0.5])
        zone_size = jnp.array([1.0, 1.0])
        for yaw in [0.0, jnp.pi / 2, jnp.pi, -jnp.pi / 2]:
            cov = _fov_coverage(pos, yaw, zone_center, zone_size)
            assert 0.0 <= float(cov) <= 1.0


# ---------------------------------------------------------------------------
# TestLineOfSight
# ---------------------------------------------------------------------------

class TestLineOfSight:
    def test_clear_path(self):
        grid = jnp.zeros((10, 10))
        origin = jnp.array([0.0, 0.0])
        p = jnp.array([1.0, 1.0])
        q = jnp.array([3.0, 3.0])
        assert _line_of_sight_grid(p, q, grid, origin, 0.5) == 1.0

    def test_blocked_by_wall(self):
        grid, origin, res = _make_test_grid()
        # Wall at col 5 (x=2.5), from (1,1) to (4,1) crosses it
        p = jnp.array([1.0, 1.0])
        q = jnp.array([4.0, 1.0])
        assert _line_of_sight_grid(p, q, grid, origin, res) == 0.0

    def test_wall_not_on_path(self):
        grid, origin, res = _make_test_grid()
        # Both points on same side of wall (left side)
        p = jnp.array([0.5, 0.5])
        q = jnp.array([2.0, 0.5])
        assert _line_of_sight_grid(p, q, grid, origin, res) == 1.0

    def test_same_point(self):
        grid = jnp.zeros((10, 10))
        origin = jnp.array([0.0, 0.0])
        p = jnp.array([1.0, 1.0])
        assert _line_of_sight_grid(p, p, grid, origin, 0.5) == 1.0

    def test_threshold_blocked(self):
        grid = jnp.zeros((10, 10)).at[2, 4].set(0.7)
        origin = jnp.array([0.0, 0.0])
        # Ray from (0.25, 1.25) going right through cell (col=4, row=2)
        p = jnp.array([0.25, 1.25])
        q = jnp.array([4.0, 1.25])
        assert _line_of_sight_grid(p, q, grid, origin, 0.5) == 0.0

    def test_below_threshold(self):
        grid = jnp.zeros((10, 10)).at[2, 4].set(0.69)
        origin = jnp.array([0.0, 0.0])
        p = jnp.array([0.25, 1.25])
        q = jnp.array([4.0, 1.25])
        assert _line_of_sight_grid(p, q, grid, origin, 0.5) == 1.0

    def test_jit(self):
        grid = jnp.zeros((10, 10))
        origin = jnp.array([0.0, 0.0])
        p = jnp.array([1.0, 1.0])
        q = jnp.array([3.0, 3.0])
        val = jax.jit(
            lambda p, q: _line_of_sight_grid(p, q, grid, origin, 0.5)
        )(p, q)
        assert val == 1.0


# ---------------------------------------------------------------------------
# TestFOVCoverageWithLOS
# ---------------------------------------------------------------------------

class TestFOVCoverageWithLOS:
    def test_no_wall_matches_basic(self):
        # Unknown cells (p=0.5) → entropy weight = 1.0, should match basic
        grid = 0.5 * jnp.ones((10, 10))
        origin = jnp.array([0.0, 0.0])
        pos = jnp.array([0.5, 0.5])
        yaw = 0.0
        zone_center = jnp.array([1.5, 0.5])
        zone_size = jnp.array([0.5, 0.5])
        cov_basic = _fov_coverage(pos, yaw, zone_center, zone_size)
        cov_los = _fov_coverage_with_los(
            pos, yaw, zone_center, zone_size, grid, origin, 0.5
        )
        assert jnp.isclose(cov_basic, cov_los, atol=0.05)

    def test_known_cells_reduce_coverage(self):
        """Known-free cells produce less coverage than unknown."""
        origin = jnp.array([0.0, 0.0])
        pos = jnp.array([0.5, 0.5])
        yaw = 0.0
        zone_center = jnp.array([1.5, 0.5])
        zone_size = jnp.array([0.5, 0.5])

        grid_unknown = 0.5 * jnp.ones((10, 10))
        cov_unknown = _fov_coverage_with_los(
            pos, yaw, zone_center, zone_size, grid_unknown, origin, 0.5
        )

        grid_known = 0.2 * jnp.ones((10, 10))
        cov_known = _fov_coverage_with_los(
            pos, yaw, zone_center, zone_size, grid_known, origin, 0.5
        )

        assert cov_unknown > 0
        assert cov_known < cov_unknown

    def test_wall_blocks(self):
        grid, origin, res = _make_test_grid()
        # UAV on left side of wall, zone on right side
        pos = jnp.array([1.0, 2.5])
        yaw = 0.0  # facing right
        zone_center = jnp.array([3.5, 2.5])
        zone_size = jnp.array([1.0, 1.0])
        cov_basic = _fov_coverage(pos, yaw, zone_center, zone_size)
        cov_los = _fov_coverage_with_los(
            pos, yaw, zone_center, zone_size, grid, origin, res
        )
        assert cov_basic > 0  # FOV sees the zone ignoring walls
        assert cov_los < cov_basic  # LOS blocks it

    def test_clear_los_allows_coverage(self):
        # Zone on same side as UAV, no wall between
        grid, origin, res = _make_test_grid()
        pos = jnp.array([0.5, 2.5])
        yaw = 0.0
        zone_center = jnp.array([1.5, 2.5])
        zone_size = jnp.array([0.5, 0.5])
        cov = _fov_coverage_with_los(
            pos, yaw, zone_center, zone_size, grid, origin, res
        )
        assert cov > 0


# ---------------------------------------------------------------------------
# TestAugmentedDynamics
# ---------------------------------------------------------------------------

class TestAugmentedDynamics:
    def test_output_shape(self):
        state = _make_augmented_state(5.0, 5.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        ns = augmented_dynamics(state, action, dt=0.05)
        assert ns.shape == (16,)

    def test_info_depletes_when_facing_zone(self):
        # Zone 0 is at (2.5, 6.0). Place UAV nearby facing it.
        state = _make_augmented_state(2.5, 5.0, yaw=jnp.pi / 2)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        ns = augmented_dynamics(state, action, dt=0.05)
        # Zone 0 info should decrease (UAV is within range, facing it)
        assert ns[13] < 100.0

    def test_info_constant_when_far(self):
        # Place UAV far from all zones, in free space
        state = _make_augmented_state(7.0, 5.0, yaw=0.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        ns = augmented_dynamics(state, action, dt=0.05)
        # All zones should remain at 100 (too far to see)
        assert jnp.allclose(ns[13:], jnp.array([100.0, 100.0, 100.0]), atol=0.1)

    def test_depletion_rate_over_many_steps(self):
        # Simulate 50 steps with full coverage → info ~ 100 * 0.98^50 ≈ 36.4
        # Use a synthetic scenario: UAV very close and facing zone 0
        state = _make_augmented_state(2.5, 6.0, yaw=0.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])

        # Get the coverage for this position (to estimate expected depletion)
        zone_center = INFO_ZONES[0, :2]
        zone_size = INFO_ZONES[0, 2:4]
        cov = _fov_coverage(state[:2], 0.0, zone_center, zone_size)

        # Run 50 steps
        s = state
        for _ in range(50):
            s = augmented_dynamics(s, action, dt=0.05)

        # Expected: info * (1 - DEPLETION_ALPHA * cov)^50
        expected = 100.0 * (1.0 - DEPLETION_ALPHA * float(cov)) ** 50
        # Allow generous tolerance since position shifts each step
        assert jnp.isclose(s[13], expected, atol=15.0)

    def test_jit(self):
        state = _make_augmented_state(5.0, 5.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        ns = jax.jit(augmented_dynamics, static_argnames=["dt"])(
            state, action, dt=0.05
        )
        assert ns.shape == (16,)


# ---------------------------------------------------------------------------
# TestAugmentedDynamicsWithGrid
# ---------------------------------------------------------------------------

class TestAugmentedDynamicsWithGrid:
    def _grid_and_params(self):
        walls = jnp.array([
            [0.0, 2.0, 4.0, 2.0],
            [0.0, 8.0, 4.0, 8.0],
        ])
        info_zones = INFO_ZONES
        origin = jnp.array([0.0, 0.0])
        gm = rasterize_environment(walls, info_zones, origin, 30, 24, 0.5)
        return gm

    def test_output_shape(self):
        gm = self._grid_and_params()
        state = _make_augmented_state(5.0, 5.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        ns = augmented_dynamics_with_grid(
            state, action, dt=0.05,
            grid=gm.grid, grid_origin=gm.origin,
            grid_resolution=gm.resolution,
        )
        assert ns.shape == (16,)

    def test_wall_blocks_depletion(self):
        gm = self._grid_and_params()
        # Zone 0 at (2.5, 6.0). Wall at y=8. Place UAV above wall.
        # The wall should block LOS to the zone from above.
        state = _make_augmented_state(2.5, 9.0, yaw=-jnp.pi / 2)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        ns = augmented_dynamics_with_grid(
            state, action, dt=0.05,
            grid=gm.grid, grid_origin=gm.origin,
            grid_resolution=gm.resolution,
        )
        # Info should be mostly unchanged (wall blocks view)
        assert jnp.isclose(ns[13], 100.0, atol=1.0)

    def test_no_wall_allows_depletion(self):
        gm = self._grid_and_params()
        # UAV inside zone 0, facing it directly (no wall between)
        state = _make_augmented_state(2.5, 6.0, yaw=0.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        ns = augmented_dynamics_with_grid(
            state, action, dt=0.05,
            grid=gm.grid, grid_origin=gm.origin,
            grid_resolution=gm.resolution,
        )
        assert ns[13] < 100.0

    def test_matches_basic_on_unknown_grid(self):
        # Unknown grid (p=0.5) with no walls → LOS never blocked,
        # entropy weight = 1.0 → same depletion as augmented_dynamics
        grid = 0.5 * jnp.ones((20, 20))
        origin = jnp.array([0.0, 0.0])
        state = _make_augmented_state(2.5, 6.0, yaw=0.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        ns_basic = augmented_dynamics(state, action, dt=0.05)
        ns_grid = augmented_dynamics_with_grid(
            state, action, dt=0.05,
            grid=grid, grid_origin=origin, grid_resolution=0.5,
        )
        assert jnp.allclose(ns_basic[13:], ns_grid[13:], atol=0.05)

    def test_known_grid_no_depletion(self):
        # Known-free grid (p=0.0) → entropy weight = 0 → no depletion
        grid = jnp.zeros((20, 20))
        origin = jnp.array([0.0, 0.0])
        state = _make_augmented_state(2.5, 6.0, yaw=0.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        ns = augmented_dynamics_with_grid(
            state, action, dt=0.05,
            grid=grid, grid_origin=origin, grid_resolution=0.5,
        )
        assert jnp.allclose(ns[13:], state[13:], atol=1e-5)


# ---------------------------------------------------------------------------
# TestRunningCost
# ---------------------------------------------------------------------------

class TestRunningCost:
    def test_at_target_finite(self):
        state = _make_augmented_state(
            float(GOAL_POS[0]), float(GOAL_POS[1]), float(GOAL_POS[2])
        )
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        cost = running_cost(state, action, 0, GOAL_POS)
        assert jnp.isfinite(cost)

    def test_wall_collision_high(self):
        # Place UAV on a wall segment: WALLS[0] = [0, 2, 4, 2]
        state = _make_augmented_state(2.0, 2.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        cost = running_cost(state, action, 0, GOAL_POS)
        assert cost >= 1000.0

    def test_out_of_bounds(self):
        state = _make_augmented_state(-2.0, 5.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        cost = running_cost(state, action, 0, GOAL_POS)
        assert cost >= 1000.0

    def test_height_deviation(self):
        state_good = _make_augmented_state(5.0, 5.0, z=-2.0)
        state_bad = _make_augmented_state(5.0, 5.0, z=0.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        cost_good = running_cost(state_good, action, 0, GOAL_POS)
        cost_bad = running_cost(state_bad, action, 0, GOAL_POS)
        assert cost_bad > cost_good

    def test_info_reward(self):
        # High info near zone → lower cost than depleted info
        state_high = _make_augmented_state(
            2.5, 6.0, yaw=0.0, info_levels=jnp.array([100.0, 100.0, 100.0])
        )
        state_low = _make_augmented_state(
            2.5, 6.0, yaw=0.0, info_levels=jnp.array([0.0, 0.0, 0.0])
        )
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        cost_high = running_cost(state_high, action, 0, GOAL_POS)
        cost_low = running_cost(state_low, action, 0, GOAL_POS)
        # High info → more reward → lower cost
        assert cost_high < cost_low

    def test_target_as_trajectory(self):
        state = _make_augmented_state(5.0, 5.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        target_traj = jnp.tile(GOAL_POS, (10, 1))  # (10, 3)
        cost = running_cost(state, action, 0, target_traj)
        assert jnp.isfinite(cost)


# ---------------------------------------------------------------------------
# TestInformativeRunningCost
# ---------------------------------------------------------------------------

class TestInformativeRunningCost:
    def _dummy_fsmi_fn(self, grid_map, pos, yaw):
        """Simple mock: high gain in unknown regions, zero elsewhere."""
        return 1.0

    def test_scalar_output(self):
        state = _make_augmented_state(5.0, 5.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        grid = jnp.zeros((20, 20))
        cost = informative_running_cost(
            state, action, 0, GOAL_POS,
            grid_map=grid,
            uniform_fsmi_fn=self._dummy_fsmi_fn,
        )
        assert cost.shape == ()

    def test_grid_obstacle_penalty(self):
        state = _make_augmented_state(2.5, 2.5)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        # Grid with obstacle at UAV position
        grid = jnp.zeros((10, 10)).at[5, 5].set(1.0)
        origin = jnp.array([0.0, 0.0])
        cost = informative_running_cost(
            state, action, 0, GOAL_POS,
            grid_map=grid,
            uniform_fsmi_fn=self._dummy_fsmi_fn,
            grid_origin=origin,
            grid_resolution=0.5,
        )
        assert cost >= 1000.0

    def test_info_weight_scaling(self):
        state = _make_augmented_state(5.0, 5.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        grid = jnp.zeros((20, 20))
        cost_low = informative_running_cost(
            state, action, 0, GOAL_POS,
            grid_map=grid,
            uniform_fsmi_fn=self._dummy_fsmi_fn,
            info_weight=0.0,
        )
        cost_high = informative_running_cost(
            state, action, 0, GOAL_POS,
            grid_map=grid,
            uniform_fsmi_fn=self._dummy_fsmi_fn,
            info_weight=10.0,
        )
        assert cost_low != cost_high

    def test_no_grid_origin(self):
        """grid_origin=None should skip grid obstacle cost."""
        state = _make_augmented_state(5.0, 5.0)
        action = jnp.array([9.81, 0.0, 0.0, 0.0])
        grid = jnp.zeros((20, 20))
        cost = informative_running_cost(
            state, action, 0, GOAL_POS,
            grid_map=grid,
            uniform_fsmi_fn=self._dummy_fsmi_fn,
            grid_origin=None,
            grid_resolution=None,
        )
        assert jnp.isfinite(cost)
