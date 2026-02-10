"""C4: Benchmarks for Parallel I-MPPI components.

Measures wall-clock times for:
  1. Trajectory FSMI methods (direct, discounted, filtered)
  2. compute_info_field
  3. Full parallel I-MPPI step (mppi.command with parallel cost)

Uses pytest-benchmark. Run with:
  uv run pytest tests/test_benchmarks.py -m slow --benchmark-columns=min,max,mean,median
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from jax_mppi import mppi
from jax_mppi.i_mppi.environment import (
    INFO_ZONES,
    WALLS,
    augmented_dynamics_with_grid,
    parallel_imppi_running_cost,
)
from jax_mppi.i_mppi.fsmi import (
    FSMIConfig,
    FSMIModule,
    InfoFieldConfig,
    UniformFSMI,
    UniformFSMIConfig,
    compute_info_field,
    fsmi_trajectory_direct,
    fsmi_trajectory_discounted,
    fsmi_trajectory_filtered,
)
from jax_mppi.i_mppi.map import rasterize_environment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jax_benchmark(benchmark, fn, *args):
    """Run a JAX benchmark: warmup JIT, then let pytest-benchmark measure."""
    # Warmup (JIT compile + first execution)
    result = fn(*args)
    jax.block_until_ready(result)

    def run():
        r = fn(*args)
        jax.block_until_ready(r)
        return r

    result = benchmark(run)
    return result


def _make_grid_40x40():
    """40x40 grid at 0.25m res with walls and an unknown zone."""
    origin = jnp.array([0.0, 0.0])
    walls = jnp.array(
        [
            [5.0, 0.0, 5.0, 10.0],
            [0.0, 5.0, 5.0, 5.0],
        ]
    )
    info_zones = jnp.array([[2.5, 2.5, 2.0, 2.0, 100.0]])
    return rasterize_environment(walls, info_zones, origin, 40, 40, 0.25)


def _make_trajectory(n_waypoints=50):
    """Straight-line trajectory for FSMI benchmarks."""
    xs = jnp.linspace(1.0, 8.0, n_waypoints)
    ys = jnp.full(n_waypoints, 5.0)
    zs = jnp.full(n_waypoints, -2.0)
    return jnp.stack([xs, ys, zs], axis=1)


# ---------------------------------------------------------------------------
# 1. Trajectory FSMI methods
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTrajectoryFSMIBenchmark:
    """Benchmark direct, discounted, and filtered trajectory FSMI."""

    def _setup(self):
        gm = _make_grid_40x40()
        mod = FSMIModule(
            FSMIConfig(num_beams=16, max_range=5.0, ray_step=0.1, fov_rad=1.57),
            gm.origin,
            gm.resolution,
        )
        traj = _make_trajectory(50)
        return mod, gm.grid, traj

    def test_trajectory_direct(self, benchmark):
        mod, grid, traj = self._setup()
        fn = jax.jit(lambda g, t: fsmi_trajectory_direct(mod, t, g, 5, 0.05))
        _jax_benchmark(benchmark, fn, grid, traj)

    def test_trajectory_discounted(self, benchmark):
        mod, grid, traj = self._setup()
        fn = jax.jit(
            lambda g, t: fsmi_trajectory_discounted(mod, t, g, 5, 0.05)
        )
        _jax_benchmark(benchmark, fn, grid, traj)

    def test_trajectory_filtered(self, benchmark):
        mod, grid, traj = self._setup()
        fn = jax.jit(lambda g, t: fsmi_trajectory_filtered(mod, t, g, 5, 0.05))
        _jax_benchmark(benchmark, fn, grid, traj)


# ---------------------------------------------------------------------------
# 2. compute_info_field
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestInfoFieldBenchmark:
    """Benchmark compute_info_field on a 20x20 field grid."""

    def test_compute_info_field(self, benchmark):
        gm = _make_grid_40x40()
        mod = FSMIModule(
            FSMIConfig(num_beams=8, max_range=3.0, ray_step=0.1, fov_rad=1.57),
            gm.origin,
            gm.resolution,
        )
        cfg = InfoFieldConfig(field_res=0.5, field_extent=2.0, n_yaw=8)
        pos = jnp.array([5.0, 5.0])

        fn = jax.jit(lambda g, p: compute_info_field(mod, g, p, cfg))
        _jax_benchmark(benchmark, fn, gm.grid, pos)


# ---------------------------------------------------------------------------
# 3. Full parallel I-MPPI step
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestParallelImppiStepBenchmark:
    """Benchmark a single mppi.command call with parallel_imppi_running_cost."""

    def test_mppi_step(self, benchmark):
        # Use real WALLS/INFO_ZONES since dynamics hard-codes them
        origin = jnp.array([0.0, 0.0])
        resolution = 0.5
        gm = rasterize_environment(
            WALLS, INFO_ZONES, origin, 26, 22, resolution
        )

        uniform = UniformFSMI(
            UniformFSMIConfig(num_beams=4, max_range=1.5, ray_step=0.2),
            origin,
            resolution,
        )
        mod = FSMIModule(
            FSMIConfig(num_beams=8, max_range=3.0, ray_step=0.1, fov_rad=1.57),
            origin,
            resolution,
        )
        cfg = InfoFieldConfig(field_res=0.5, field_extent=2.0, n_yaw=4)
        pos_xy = jnp.array([5.0, 5.0])
        info_field, field_origin = compute_info_field(mod, gm.grid, pos_xy, cfg)

        # MPPI setup — 3 info zones -> NX=16
        noise_sigma = jnp.diag(jnp.array([2.0, 0.5, 0.5, 0.5]) ** 2)
        u_min = jnp.array([0.0, -10.0, -10.0, -10.0])
        u_max = jnp.array([4.0 * 9.81, 10.0, 10.0, 10.0])
        u_init = jnp.array([9.81, 0.0, 0.0, 0.0])

        nx = 16  # 13 quad + 3 info zones
        mppi_config, mppi_state = mppi.create(
            nx=nx,
            nu=4,
            noise_sigma=noise_sigma,
            num_samples=256,
            horizon=15,
            u_min=u_min,
            u_max=u_max,
            u_init=u_init,
            lambda_=1.0,
            step_dependent_dynamics=True,
        )

        # State: quad at (5, 5, -2) + 3 info zones
        quad = jnp.zeros(13)
        quad = quad.at[:3].set(jnp.array([5.0, 5.0, -2.0]))
        quad = quad.at[6].set(1.0)
        state = jnp.concatenate([quad, jnp.array([100.0, 100.0, 100.0])])

        cost_fn = partial(
            parallel_imppi_running_cost,
            grid_map=gm.grid,
            grid_origin=origin,
            grid_resolution=resolution,
            info_field=info_field,
            field_origin=field_origin,
            field_res=cfg.field_res,
            uniform_fsmi_fn=uniform.compute,
        )
        dynamics_fn = partial(
            augmented_dynamics_with_grid,
            dt=0.05,
            grid=gm.grid,
            grid_origin=origin,
            grid_resolution=resolution,
        )

        @partial(jax.jit, static_argnums=(0,))
        def step(cfg, ctrl, s):
            return mppi.command(cfg, ctrl, s, dynamics_fn, cost_fn)

        _jax_benchmark(benchmark, step, mppi_config, mppi_state, state)
