import jax.numpy as jnp

from jax_mppi.i_mppi.fsmi import (
    FSMIConfig,
    FSMITrajectoryGenerator,
    compute_fsmi_gain,
)
from jax_mppi.i_mppi.map import rasterize_environment


def test_fsmi_target_selector():
    # 10x10 map
    walls = jnp.array([[5.0, 0.0, 5.0, 10.0]])  # Wall at x=5
    info_zones = jnp.array([[2.0, 5.0, 2.0, 2.0, 100.0]])  # Zone at (2,5)
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
