"""Tests for the FSMI module (fsmi.py) and map utilities (map.py)."""

import jax
import jax.numpy as jnp

from jax_mppi.i_mppi.fsmi import (
    FSMIConfig,
    FSMIModule,
    FSMITrajectoryGenerator,
    UniformFSMI,
    UniformFSMIConfig,
    _entropy_proxy,
    _fov_cell_masks,
    _yaws_from_trajectory,
    cast_ray_fsmi,
    compute_fsmi_gain,
    fsmi_trajectory_direct,
    fsmi_trajectory_discounted,
    fsmi_trajectory_filtered,
)
from jax_mppi.i_mppi.map import (
    GridMap,
    grid_to_world,
    rasterize_environment,
    world_to_grid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_grid():
    """20x20 grid at 0.25m res (5m x 5m).  Wall at x=2.5, zone at (1, 2.5)."""
    walls = jnp.array([[2.5, 0.0, 2.5, 5.0]])
    info_zones = jnp.array([[1.0, 2.5, 1.0, 1.0, 100.0]])
    origin = jnp.array([0.0, 0.0])
    return rasterize_environment(walls, info_zones, origin, 20, 20, 0.25)


def _default_fsmi_module():
    cfg = FSMIConfig(num_beams=8, max_range=3.0, ray_step=0.1, fov_rad=1.57)
    return FSMIModule(cfg, jnp.array([0.0, 0.0]), 0.25)


def _default_uniform_fsmi():
    cfg = UniformFSMIConfig(num_beams=4, max_range=2.0, ray_step=0.2)
    return UniformFSMI(cfg, jnp.array([0.0, 0.0]), 0.25)


# ---------------------------------------------------------------------------
# Existing tests (kept as-is)
# ---------------------------------------------------------------------------

def test_fsmi_target_selector():
    walls = jnp.array([[5.0, 0.0, 5.0, 10.0]])
    info_zones = jnp.array([[2.0, 5.0, 2.0, 2.0, 100.0]])
    goal_pos = jnp.array([9.0, 5.0, -2.0])
    origin = jnp.array([0.0, 0.0])
    res = 0.5
    w, h = 20, 20

    grid_map = rasterize_environment(walls, info_zones, origin, w, h, res)
    config = FSMIConfig(goal_pos=goal_pos, min_gain_threshold=1.0)
    gen = FSMITrajectoryGenerator(config, info_zones, grid_map)

    # Case 1: High Info
    info_levels_high = jnp.array([100.0])
    gen = gen.update_map(info_levels_high)
    current_pos = jnp.array([5.0, 2.0, -2.0])
    target, mode = gen.select_target(current_pos)

    assert mode == 1
    assert jnp.allclose(target[:2], info_zones[0, :2])

    # Case 2: Low Info
    info_levels_low = jnp.array([0.0])
    gen = gen.update_map(info_levels_low)
    target, mode = gen.select_target(current_pos)

    assert mode == 0
    assert jnp.allclose(target, goal_pos)


def test_fsmi_gain():
    walls = jnp.array([[5.0, 0.0, 5.0, 10.0]])
    info_zones = jnp.array([[8.0, 5.0, 2.0, 2.0, 100.0]])
    origin = jnp.array([0.0, 0.0])
    res = 0.5
    w, h = 20, 20
    grid_map = rasterize_environment(walls, info_zones, origin, w, h, res)

    gain_blocked = compute_fsmi_gain(
        jnp.array([2.0, 5.0]),
        grid_map.grid,
        grid_map.origin,
        grid_map.resolution,
    )
    gain_clear = compute_fsmi_gain(
        jnp.array([6.0, 5.0]),
        grid_map.grid,
        grid_map.origin,
        grid_map.resolution,
    )

    assert gain_clear > gain_blocked


# ---------------------------------------------------------------------------
# TestEntropyProxy
# ---------------------------------------------------------------------------

class TestEntropyProxy:
    def test_at_half(self):
        assert jnp.isclose(_entropy_proxy(jnp.float32(0.5)), 1.0)

    def test_at_zero(self):
        assert jnp.isclose(_entropy_proxy(jnp.float32(0.0)), 0.0)

    def test_at_one(self):
        assert jnp.isclose(_entropy_proxy(jnp.float32(1.0)), 0.0)

    def test_symmetry(self):
        for p in [0.1, 0.2, 0.3, 0.4]:
            p = jnp.float32(p)
            assert jnp.isclose(_entropy_proxy(p), _entropy_proxy(1.0 - p))

    def test_range(self):
        ps = jnp.linspace(0, 1, 50)
        vals = jax.vmap(_entropy_proxy)(ps)
        assert jnp.all(vals >= 0.0)
        assert jnp.all(vals <= 1.0)


# ---------------------------------------------------------------------------
# TestFScore
# ---------------------------------------------------------------------------

class TestFScore:
    def test_positive_for_uncertain(self):
        mod = _default_fsmi_module()
        r = jnp.array(1.0)  # p=0.5
        delta = jnp.exp(jnp.float32(mod.cfg.inv_sensor_model_occ))
        f = mod._get_f_score(r, delta)
        assert f > 0

    def test_near_zero_for_known_free(self):
        mod = _default_fsmi_module()
        r = jnp.array(1e-3)  # p ≈ 0
        delta = jnp.exp(jnp.float32(mod.cfg.inv_sensor_model_occ))
        f = mod._get_f_score(r, delta)
        assert f < 0.05

    def test_near_zero_for_known_occupied(self):
        mod = _default_fsmi_module()
        r = jnp.array(1e3)  # p ≈ 1
        delta = jnp.exp(jnp.float32(mod.cfg.inv_sensor_model_occ))
        f = mod._get_f_score(r, delta)
        assert f < 0.05

    def test_monotonic_in_uncertainty(self):
        mod = _default_fsmi_module()
        delta = jnp.exp(jnp.float32(mod.cfg.inv_sensor_model_occ))
        f_uncertain = mod._get_f_score(jnp.array(1.0), delta)
        f_less = mod._get_f_score(jnp.array(0.1), delta)
        assert f_uncertain > f_less


# ---------------------------------------------------------------------------
# TestBeamFSMI
# ---------------------------------------------------------------------------

class TestBeamFSMI:
    def _make_dists(self, n=20):
        return jnp.arange(n) * 0.1

    def test_all_free(self):
        mod = _default_fsmi_module()
        probs = jnp.zeros(20)
        mi = mod._compute_beam_fsmi(probs, self._make_dists())
        assert jnp.isclose(mi, 0.0, atol=1e-4)

    def test_all_unknown(self):
        mod = _default_fsmi_module()
        probs = jnp.full(20, 0.5)
        mi = mod._compute_beam_fsmi(probs, self._make_dists())
        assert mi > 0

    def test_unknown_higher_than_free(self):
        mod = _default_fsmi_module()
        dists = self._make_dists()
        mi_free = mod._compute_beam_fsmi(jnp.zeros(20), dists)
        mi_unknown = mod._compute_beam_fsmi(jnp.full(20, 0.5), dists)
        assert mi_unknown > mi_free

    def test_wall_blocks(self):
        mod = _default_fsmi_module()
        dists = self._make_dists()
        full_unknown = jnp.full(20, 0.5)
        # Wall at index 5, unknown after
        blocked = jnp.concatenate([
            jnp.zeros(5), jnp.ones(1), jnp.full(14, 0.5)
        ])
        mi_full = mod._compute_beam_fsmi(full_unknown, dists)
        mi_blocked = mod._compute_beam_fsmi(blocked, dists)
        assert mi_full > mi_blocked

    def test_nonnegative(self):
        mod = _default_fsmi_module()
        dists = self._make_dists()
        for probs in [jnp.zeros(20), jnp.full(20, 0.5), jnp.ones(20)]:
            mi = mod._compute_beam_fsmi(probs, dists)
            assert mi >= -1e-6


# ---------------------------------------------------------------------------
# TestComputeFSMI
# ---------------------------------------------------------------------------

class TestComputeFSMI:
    def test_unknown_region_high(self):
        gm = _make_simple_grid()
        mod = _default_fsmi_module()
        # Pos inside the unknown zone at (1.0, 2.5)
        mi = mod.compute_fsmi(gm.grid, jnp.array([1.0, 2.5]), 0.0)
        assert mi > 0

    def test_all_free_grid_low(self):
        # On a large fully known-free grid, FSMI should be near zero
        # Use large grid so rays don't go OOB (OOB treated as 0.5 unknown)
        mod = _default_fsmi_module()
        free_grid = jnp.zeros((80, 80))  # 20m x 20m at 0.25m res
        pos = jnp.array([10.0, 10.0])
        mi_free = mod.compute_fsmi(free_grid, pos, 0.0)
        unknown_grid = jnp.full((80, 80), 0.5)
        mi_unknown = mod.compute_fsmi(unknown_grid, pos, 0.0)
        assert mi_unknown > mi_free

    def test_fov_directionality(self):
        gm = _make_simple_grid()
        mod = _default_fsmi_module()
        pos = jnp.array([0.5, 2.5])
        # Facing right (toward zone) vs facing left (away)
        mi_toward = mod.compute_fsmi(gm.grid, pos, 0.0)
        mi_away = mod.compute_fsmi(gm.grid, pos, jnp.pi)
        assert mi_toward != mi_away

    def test_jit_compatible(self):
        gm = _make_simple_grid()
        mod = _default_fsmi_module()
        pos = jnp.array([1.0, 2.5])
        mi_eager = mod.compute_fsmi(gm.grid, pos, 0.0)
        mi_jit = jax.jit(mod.compute_fsmi)(gm.grid, pos, 0.0)
        assert jnp.isclose(mi_eager, mi_jit, atol=1e-4)


# ---------------------------------------------------------------------------
# TestUniformFSMI
# ---------------------------------------------------------------------------

class TestUniformFSMI:
    def test_beam_all_free(self):
        ufsmi = _default_uniform_fsmi()
        probs = jnp.zeros(10)
        mi = ufsmi._compute_beam_uniform_fsmi(probs)
        assert jnp.isclose(mi, 0.0, atol=1e-4)

    def test_beam_all_unknown(self):
        ufsmi = _default_uniform_fsmi()
        probs = jnp.full(10, 0.5)
        mi = ufsmi._compute_beam_uniform_fsmi(probs)
        assert mi > 0

    def test_beam_nonneg(self):
        ufsmi = _default_uniform_fsmi()
        for probs in [jnp.zeros(10), jnp.full(10, 0.5), jnp.ones(10)]:
            mi = ufsmi._compute_beam_uniform_fsmi(probs)
            assert mi >= -1e-6

    def test_compute_scalar(self):
        gm = _make_simple_grid()
        ufsmi = _default_uniform_fsmi()
        val = ufsmi.compute(gm.grid, jnp.array([1.0, 2.5]), 0.0)
        assert val.shape == ()

    def test_compute_unknown_vs_known(self):
        gm = _make_simple_grid()
        ufsmi = _default_uniform_fsmi()
        mi_zone = ufsmi.compute(gm.grid, jnp.array([1.0, 2.5]), 0.0)
        mi_free = ufsmi.compute(gm.grid, jnp.array([4.0, 4.0]), 0.0)
        assert mi_zone > mi_free

    def test_batch_shape(self):
        gm = _make_simple_grid()
        ufsmi = _default_uniform_fsmi()
        positions = jnp.array([[1.0, 2.5], [4.0, 4.0], [0.5, 0.5], [2.0, 2.0]])
        yaws = jnp.array([0.0, 0.0, 0.0, 0.0])
        vals = ufsmi.compute_batch(gm.grid, positions, yaws)
        assert vals.shape == (4,)

    def test_batch_matches_individual(self):
        gm = _make_simple_grid()
        ufsmi = _default_uniform_fsmi()
        positions = jnp.array([[1.0, 2.5], [4.0, 4.0], [0.5, 0.5]])
        yaws = jnp.array([0.0, 1.0, -0.5])
        batch_vals = ufsmi.compute_batch(gm.grid, positions, yaws)
        for i in range(3):
            individual = ufsmi.compute(gm.grid, positions[i], yaws[i])
            assert jnp.isclose(batch_vals[i], individual, atol=1e-5)


# ---------------------------------------------------------------------------
# TestCastRayFSMI
# ---------------------------------------------------------------------------

class TestCastRayFSMI:
    def test_free_space(self):
        grid = jnp.zeros((10, 10))
        origin = jnp.array([0.0, 0.0])
        gain = cast_ray_fsmi(jnp.array([2.5, 2.5]), 0.0, grid, origin, 0.5)
        assert jnp.isclose(gain, 0.0, atol=1e-5)

    def test_into_unknown(self):
        grid = jnp.zeros((10, 10)).at[4:6, 6:8].set(0.5)
        origin = jnp.array([0.0, 0.0])
        # Ray pointing right from (2.5, 2.25) toward unknown cells
        gain = cast_ray_fsmi(jnp.array([2.5, 2.25]), 0.0, grid, origin, 0.5)
        assert gain > 0

    def test_wall_blocks(self):
        origin = jnp.array([0.0, 0.0])
        # Grid with wall then unknown
        grid_blocked = jnp.zeros((10, 10)).at[5, 4].set(1.0).at[5, 6:8].set(0.5)
        grid_clear = jnp.zeros((10, 10)).at[5, 6:8].set(0.5)
        gain_blocked = cast_ray_fsmi(
            jnp.array([0.25, 2.75]), 0.0, grid_blocked, origin, 0.5
        )
        gain_clear = cast_ray_fsmi(
            jnp.array([0.25, 2.75]), 0.0, grid_clear, origin, 0.5
        )
        assert gain_clear >= gain_blocked


# ---------------------------------------------------------------------------
# TestComputeFSMIGain
# ---------------------------------------------------------------------------

class TestComputeFSMIGain:
    def test_in_unknown_region(self):
        gm = _make_simple_grid()
        gain = compute_fsmi_gain(
            jnp.array([1.0, 2.5]), gm.grid, gm.origin, gm.resolution
        )
        assert gain > 0

    def test_in_free_region(self):
        grid = jnp.zeros((10, 10))
        origin = jnp.array([0.0, 0.0])
        gain = compute_fsmi_gain(jnp.array([2.5, 2.5]), grid, origin, 0.5)
        assert jnp.isclose(gain, 0.0, atol=1e-4)

    def test_near_vs_far(self):
        gm = _make_simple_grid()
        gain_near = compute_fsmi_gain(
            jnp.array([1.0, 2.5]), gm.grid, gm.origin, gm.resolution
        )
        gain_far = compute_fsmi_gain(
            jnp.array([4.0, 4.0]), gm.grid, gm.origin, gm.resolution
        )
        assert gain_near > gain_far


# ---------------------------------------------------------------------------
# TestFSMITrajectoryGenerator
# ---------------------------------------------------------------------------

class TestFSMITrajectoryGenerator:
    def _make_gen(self):
        walls = jnp.array([[5.0, 0.0, 5.0, 10.0]])
        info_zones = jnp.array([
            [2.0, 5.0, 2.0, 2.0, 100.0],
            [8.0, 5.0, 2.0, 2.0, 100.0],
        ])
        goal_pos = jnp.array([9.0, 5.0, -2.0])
        origin = jnp.array([0.0, 0.0])
        gm = rasterize_environment(walls, info_zones, origin, 20, 20, 0.5)
        cfg = FSMIConfig(goal_pos=goal_pos, min_gain_threshold=1.0)
        gen = FSMITrajectoryGenerator(cfg, info_zones, gm)
        return gen, info_zones, goal_pos

    def test_select_target_prefers_high_info(self):
        gen, zones, goal = self._make_gen()
        # Zone 0 high, zone 1 depleted
        info = jnp.array([100.0, 0.0])
        target, mode = gen.select_target(jnp.array([5.0, 5.0, -2.0]), info)
        assert mode == 1
        assert jnp.allclose(target[:2], zones[0, :2])

    def test_select_target_equidistant_prefers_higher(self):
        gen, zones, goal = self._make_gen()
        # Equidistant from both zones, zone 1 has more info
        pos = jnp.array([5.0, 5.0, -2.0])
        info = jnp.array([30.0, 80.0])
        target, mode = gen.select_target(pos, info)
        assert mode == 1
        assert jnp.allclose(target[:2], zones[1, :2])

    def test_select_target_all_depleted_goes_to_goal(self):
        gen, zones, goal = self._make_gen()
        info = jnp.array([0.0, 0.0])
        target, mode = gen.select_target(jnp.array([5.0, 5.0, -2.0]), info)
        assert mode == 0
        assert jnp.allclose(target, goal)

    def test_make_ref_traj_shape(self):
        gen, _, _ = self._make_gen()
        pos0 = jnp.array([1.0, 1.0, -2.0])
        target = jnp.array([5.0, 5.0, -2.0])
        traj = gen._make_ref_traj(pos0, target, horizon=20, dt=0.1)
        assert traj.shape == (20, 3)

    def test_make_ref_traj_starts_at_pos0(self):
        gen, _, _ = self._make_gen()
        pos0 = jnp.array([1.0, 1.0, -2.0])
        target = jnp.array([5.0, 5.0, -2.0])
        traj = gen._make_ref_traj(pos0, target, horizon=20, dt=0.1)
        assert jnp.allclose(traj[0], pos0, atol=1e-5)

    def test_make_ref_traj_clamped(self):
        gen, _, _ = self._make_gen()
        pos0 = jnp.array([1.0, 1.0, -2.0])
        target = jnp.array([1.5, 1.5, -2.0])  # very close
        traj = gen._make_ref_traj(pos0, target, horizon=50, dt=0.1)
        # Last points should all be at target
        assert jnp.allclose(traj[-1], target, atol=1e-3)

    def test_get_reference_trajectory(self):
        gen, zones, goal = self._make_gen()
        state = jnp.zeros(16 + 3)  # 13 quad + 2 info + need 16+3?
        # Actually: state is 13 + N_zones. With 2 zones: 15D
        quad = jnp.zeros(13)
        quad = quad.at[:3].set(jnp.array([5.0, 5.0, -2.0]))
        quad = quad.at[6].set(1.0)
        state = jnp.concatenate([quad, jnp.array([100.0, 100.0])])
        info_data = (gen.grid_map.grid, jnp.array([100.0, 100.0]))
        traj, mode = gen.get_reference_trajectory(state, info_data, 20, 0.1)
        assert traj.shape == (20, 3)

    def test_pytree_roundtrip(self):
        gen, _, _ = self._make_gen()
        children, aux = gen.tree_flatten()
        gen2 = FSMITrajectoryGenerator.tree_unflatten(aux, children)
        assert jnp.allclose(gen2.info_zones, gen.info_zones)
        assert jnp.allclose(gen2.grid_map.grid, gen.grid_map.grid)


# ---------------------------------------------------------------------------
# TestMapModule
# ---------------------------------------------------------------------------

class TestMapModule:
    def test_gridmap_pytree_roundtrip(self):
        gm = _make_simple_grid()
        children, aux = gm.tree_flatten()
        gm2 = GridMap.tree_unflatten(aux, children)
        assert jnp.allclose(gm2.grid, gm.grid)
        assert jnp.allclose(gm2.origin, gm.origin)
        assert gm2.width == gm.width
        assert gm2.height == gm.height

    def test_rasterize_wall_cells(self):
        gm = _make_simple_grid()
        # Wall at x=2.5 → at 0.25m res, cells near col=10 should be 1.0
        assert jnp.any(gm.grid >= 0.99)

    def test_rasterize_zone_cells(self):
        gm = _make_simple_grid()
        # Zone at (1.0, 2.5), 1x1m → should have some 0.5 cells
        assert jnp.any(jnp.isclose(gm.grid, 0.5))

    def test_rasterize_free_cells(self):
        gm = _make_simple_grid()
        # Most cells should be free
        assert jnp.any(gm.grid == 0.0)

    def test_rasterize_wall_priority(self):
        """Wall overlapping zone → wall (1.0) wins."""
        # Horizontal wall through zone center
        walls = jnp.array([[0.5, 2.5, 1.5, 2.5]])
        info_zones = jnp.array([[1.0, 2.5, 1.0, 1.0, 100.0]])
        origin = jnp.array([0.0, 0.0])
        gm = rasterize_environment(walls, info_zones, origin, 20, 20, 0.25)
        # Cell at the wall segment should be 1.0 (wall wins over zone's 0.5)
        col = int((1.0 - 0.0) / 0.25)
        row = int((2.5 - 0.0) / 0.25)
        assert gm.grid[row, col] == 1.0

    def test_world_to_grid_origin(self):
        origin = jnp.array([0.0, 0.0])
        idx = world_to_grid(origin, origin, 0.5)
        assert jnp.allclose(idx, jnp.array([0.0, 0.0]))

    def test_world_to_grid_offset(self):
        origin = jnp.array([0.0, 0.0])
        idx = world_to_grid(jnp.array([0.5, 0.0]), origin, 0.5)
        assert jnp.isclose(idx[0], 1.0)

    def test_grid_to_world_center(self):
        origin = jnp.array([0.0, 0.0])
        world = grid_to_world(jnp.array([0.0, 0.0]), origin, 0.5)
        assert jnp.allclose(world, jnp.array([0.25, 0.25]))

    def test_world_grid_roundtrip(self):
        origin = jnp.array([1.0, 2.0])
        res = 0.5
        p = jnp.array([3.7, 5.3])
        grid_idx = world_to_grid(p, origin, res)
        floored = jnp.floor(grid_idx)
        p_back = grid_to_world(floored, origin, res)
        # Should be within one cell
        assert jnp.all(jnp.abs(p - p_back) < res)


# ---------------------------------------------------------------------------
# TestComputeFSMIBatch
# ---------------------------------------------------------------------------

class TestComputeFSMIBatch:
    def test_shape(self):
        gm = _make_simple_grid()
        mod = _default_fsmi_module()
        positions = jnp.array([[1.0, 2.5], [4.0, 4.0], [0.5, 0.5]])
        yaws = jnp.array([0.0, 1.0, -0.5])
        vals = mod.compute_fsmi_batch(gm.grid, positions, yaws)
        assert vals.shape == (3,)

    def test_matches_individual(self):
        gm = _make_simple_grid()
        mod = _default_fsmi_module()
        positions = jnp.array([[1.0, 2.5], [4.0, 4.0], [0.5, 0.5]])
        yaws = jnp.array([0.0, 1.0, -0.5])
        batch_vals = mod.compute_fsmi_batch(gm.grid, positions, yaws)
        for i in range(3):
            individual = mod.compute_fsmi(gm.grid, positions[i], yaws[i])
            assert jnp.isclose(batch_vals[i], individual, atol=1e-5)

    def test_jit_compatible(self):
        gm = _make_simple_grid()
        mod = _default_fsmi_module()
        positions = jnp.array([[1.0, 2.5], [4.0, 4.0]])
        yaws = jnp.array([0.0, 1.0])
        eager = mod.compute_fsmi_batch(gm.grid, positions, yaws)
        jitted = jax.jit(mod.compute_fsmi_batch)(gm.grid, positions, yaws)
        assert jnp.allclose(eager, jitted, atol=1e-4)


# ---------------------------------------------------------------------------
# TestYawsFromTrajectory
# ---------------------------------------------------------------------------

class TestYawsFromTrajectory:
    def test_straight_right(self):
        traj = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        yaws = _yaws_from_trajectory(traj)
        assert jnp.allclose(yaws, 0.0, atol=1e-5)

    def test_straight_up(self):
        traj = jnp.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        yaws = _yaws_from_trajectory(traj)
        assert jnp.allclose(yaws, jnp.pi / 2, atol=1e-5)

    def test_last_reuses_previous(self):
        traj = jnp.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])
        yaws = _yaws_from_trajectory(traj)
        assert jnp.isclose(yaws[-1], yaws[-2], atol=1e-5)

    def test_shape(self):
        traj = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
        yaws = _yaws_from_trajectory(traj)
        assert yaws.shape == (4,)


# ---------------------------------------------------------------------------
# TestFovCellMasks
# ---------------------------------------------------------------------------

class TestFovCellMasks:
    def test_shape(self):
        positions = jnp.array([[2.5, 2.5], [1.0, 1.0]])
        yaws = jnp.array([0.0, jnp.pi / 2])
        masks = _fov_cell_masks(
            positions, yaws,
            fov_rad=1.57, max_range=2.0,
            grid_shape=(20, 20),
            grid_origin=jnp.array([0.0, 0.0]),
            grid_resolution=0.25,
        )
        assert masks.shape == (2, 20, 20)
        assert masks.dtype == jnp.bool_

    def test_empty_fov(self):
        """Pose looking away from all cells should see very few."""
        # Position at corner, looking outward
        positions = jnp.array([[0.0, 0.0]])
        yaws = jnp.array([jnp.pi])  # looking left (outside grid)
        masks = _fov_cell_masks(
            positions, yaws,
            fov_rad=0.5, max_range=2.0,
            grid_shape=(20, 20),
            grid_origin=jnp.array([0.0, 0.0]),
            grid_resolution=0.25,
        )
        # Should see very few cells (maybe edge cells at most)
        assert jnp.sum(masks[0]) < 20


# ---------------------------------------------------------------------------
# TestTrajectoryFSMI
# ---------------------------------------------------------------------------

def _make_trajectory_test_setup():
    """Helper: grid with unknown zone + straight-line trajectory through it."""
    walls = jnp.array([[5.0, 0.0, 5.0, 5.0]])
    info_zones = jnp.array([[2.0, 2.5, 2.0, 2.0, 100.0]])
    origin = jnp.array([0.0, 0.0])
    gm = rasterize_environment(walls, info_zones, origin, 20, 20, 0.25)
    cfg = FSMIConfig(num_beams=8, max_range=3.0, ray_step=0.1, fov_rad=1.57)
    mod = FSMIModule(cfg, origin, 0.25)

    # Trajectory passing through the unknown zone
    traj = jnp.column_stack([
        jnp.linspace(0.5, 3.5, 20),
        jnp.full(20, 2.5),
        jnp.full(20, -2.0),
    ])
    return mod, gm, traj


class TestTrajectoryFSMIDirect:
    def test_positive_for_unknown(self):
        mod, gm, traj = _make_trajectory_test_setup()
        mi = fsmi_trajectory_direct(mod, traj, gm.grid, subsample_rate=5, dt=0.1)
        assert mi > 0

    def test_zero_for_free(self):
        """Large free grid → MI should be near zero."""
        cfg = FSMIConfig(num_beams=8, max_range=3.0, ray_step=0.1, fov_rad=1.57)
        origin = jnp.array([0.0, 0.0])
        mod = FSMIModule(cfg, origin, 0.25)
        free_grid = jnp.zeros((80, 80))
        traj = jnp.column_stack([
            jnp.linspace(5.0, 15.0, 20),
            jnp.full(20, 10.0),
            jnp.full(20, -2.0),
        ])
        mi = fsmi_trajectory_direct(mod, traj, free_grid, subsample_rate=5, dt=0.1)
        # Should be very small (not exactly 0 due to boundary effects)
        assert mi < 0.1

    def test_jit_compatible(self):
        mod, gm, traj = _make_trajectory_test_setup()
        fn = lambda g, t: fsmi_trajectory_direct(mod, t, g, subsample_rate=5, dt=0.1)
        eager = fn(gm.grid, traj)
        jitted = jax.jit(fn)(gm.grid, traj)
        assert jnp.isclose(eager, jitted, atol=1e-4)


class TestTrajectoryFSMIDiscount:
    def test_leq_direct(self):
        """Discounted MI should be <= direct MI for overlapping trajectory."""
        mod, gm, traj = _make_trajectory_test_setup()
        mi_direct = fsmi_trajectory_direct(mod, traj, gm.grid, subsample_rate=2, dt=0.1)
        mi_discount = fsmi_trajectory_discounted(
            mod, traj, gm.grid, subsample_rate=2, dt=0.1, decay=0.7
        )
        assert mi_discount <= mi_direct + 1e-6

    def test_zero_decay_equals_direct(self):
        """With decay=0, discount weights are all 1 → same as direct."""
        mod, gm, traj = _make_trajectory_test_setup()
        mi_direct = fsmi_trajectory_direct(mod, traj, gm.grid, subsample_rate=5, dt=0.1)
        mi_discount = fsmi_trajectory_discounted(
            mod, traj, gm.grid, subsample_rate=5, dt=0.1, decay=0.0
        )
        assert jnp.isclose(mi_direct, mi_discount, atol=1e-4)

    def test_jit_compatible(self):
        mod, gm, traj = _make_trajectory_test_setup()
        fn = lambda g, t: fsmi_trajectory_discounted(
            mod, t, g, subsample_rate=5, dt=0.1, decay=0.7
        )
        eager = fn(gm.grid, traj)
        jitted = jax.jit(fn)(gm.grid, traj)
        assert jnp.isclose(eager, jitted, atol=1e-4)


class TestTrajectoryFSMIFiltered:
    def test_leq_direct(self):
        """Filtered MI should be <= direct MI."""
        mod, gm, traj = _make_trajectory_test_setup()
        mi_direct = fsmi_trajectory_direct(mod, traj, gm.grid, subsample_rate=2, dt=0.1)
        mi_filtered = fsmi_trajectory_filtered(
            mod, traj, gm.grid, subsample_rate=2, dt=0.1
        )
        assert mi_filtered <= mi_direct + 1e-6

    def test_equals_direct_for_single_pose(self):
        """With only one pose, filtered == direct (no overlap possible)."""
        mod, gm, traj = _make_trajectory_test_setup()
        # subsample_rate large enough to get just 1 pose
        mi_direct = fsmi_trajectory_direct(
            mod, traj, gm.grid, subsample_rate=100, dt=0.1
        )
        mi_filtered = fsmi_trajectory_filtered(
            mod, traj, gm.grid, subsample_rate=100, dt=0.1
        )
        assert jnp.isclose(mi_direct, mi_filtered, atol=1e-4)

    def test_jit_compatible(self):
        mod, gm, traj = _make_trajectory_test_setup()
        fn = lambda g, t: fsmi_trajectory_filtered(
            mod, t, g, subsample_rate=5, dt=0.1
        )
        eager = fn(gm.grid, traj)
        jitted = jax.jit(fn)(gm.grid, traj)
        assert jnp.isclose(eager, jitted, atol=1e-4)


# ---------------------------------------------------------------------------
# TestInfoGainGridDispatch
# ---------------------------------------------------------------------------

class TestInfoGainGridDispatch:
    """Test that _info_gain_grid dispatches to the correct method."""

    def _make_gen_with_method(self, method="direct"):
        walls = jnp.array([[5.0, 0.0, 5.0, 5.0]])
        info_zones = jnp.array([[2.0, 2.5, 2.0, 2.0, 100.0]])
        origin = jnp.array([0.0, 0.0])
        gm = rasterize_environment(walls, info_zones, origin, 20, 20, 0.25)
        cfg = FSMIConfig(
            num_beams=8,
            max_range=3.0,
            ray_step=0.1,
            fov_rad=1.57,
            trajectory_subsample_rate=5,
            trajectory_ig_method=method,
        )
        return FSMITrajectoryGenerator(cfg, info_zones, gm), gm

    def test_direct_positive(self):
        gen, gm = self._make_gen_with_method("direct")
        traj = jnp.column_stack([
            jnp.linspace(0.5, 3.5, 20),
            jnp.full(20, 2.5),
            jnp.full(20, -2.0),
        ])
        mi = gen._info_gain_grid(traj, jnp.array([1.0, 0.0]), gm.grid, 0.1)
        assert mi > 0

    def test_discount_positive(self):
        gen, gm = self._make_gen_with_method("discount")
        traj = jnp.column_stack([
            jnp.linspace(0.5, 3.5, 20),
            jnp.full(20, 2.5),
            jnp.full(20, -2.0),
        ])
        mi = gen._info_gain_grid(traj, jnp.array([1.0, 0.0]), gm.grid, 0.1)
        assert mi > 0

    def test_filtered_positive(self):
        gen, gm = self._make_gen_with_method("filtered")
        traj = jnp.column_stack([
            jnp.linspace(0.5, 3.5, 20),
            jnp.full(20, 2.5),
            jnp.full(20, -2.0),
        ])
        mi = gen._info_gain_grid(traj, jnp.array([1.0, 0.0]), gm.grid, 0.1)
        assert mi > 0
