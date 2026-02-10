# JAX-MPPI Examples

This directory contains examples demonstrating various features of the `jax-mppi` library.

## Directory Structure

### [`basic/`](basic/) - Getting Started
Simple introductory examples.

- **`pendulum.py`** - Classic inverted pendulum swing-up with MPPI

### [`quadrotor/`](quadrotor/) - Quadrotor Control
Quadrotor trajectory tracking with MPPI variants.

- **`hover.py`** - Basic hover stabilization
- **`circle.py`** - Circular trajectory tracking
- **`custom_trajectory.py`** - Waypoint-based trajectory following
- **`hover_comparison.py`** - Compare MPPI/SMPPI/KMPPI for hover
- **`figure8_comparison.py`** - Compare controllers on figure-8 trajectory
- **`trajectories.py`** - Shared trajectory generation utilities

### [`i_mppi/`](i_mppi/) - Informative MPPI
I-MPPI (Informative MPPI) with two-layer architecture (Zhang et al. 2020).

- **Layer 2 (~5 Hz):** Full FSMI with O(n^2) computation for reference trajectory
- **Layer 3 (~50 Hz):** Uniform-FSMI with O(n) for local reactivity

Files:
- **`simulation.py`** - Main I-MPPI simulation with two-layer architecture
- **`fsmi_grid_demo.py`** - Standalone FSMI demonstration

### [`autotuning/`](autotuning/) - Hyperparameter Tuning
Automatic hyperparameter optimization using evolutionary strategies.

- **`basic.py`** - Minimal autotuning example
- **`pendulum.py`** - Tune MPPI for pendulum swing-up
- **`evosax_comparison.py`** - Compare different ES algorithms

### [`cuda/`](cuda/) - CUDA Acceleration
CUDA-accelerated MPPI kernels.

- **`pendulum_jit.py`** - CUDA JIT-compiled MPPI example
- **`test_cuda_config.py`** - Test CUDA configuration
- **`test_cuda_bindings.py`** - Test CUDA bindings
- **`test_cuda_mppi.py`** - Test CUDA MPPI controller

### [`benchmarks/`](benchmarks/) - Performance Comparisons
Benchmarks and performance analysis.

- **`smooth_comparison.py`** - Compare MPPI variants on 2D navigation

## Quick Start

```bash
# Pendulum swing-up
uv run python examples/basic/pendulum.py

# Quadrotor hover
uv run python examples/quadrotor/hover.py

# I-MPPI simulation
uv run python examples/i_mppi/simulation.py

# Quadrotor comparison
uv run python examples/quadrotor/hover_comparison.py --visualize
```

## Output

Most examples save visualizations to `docs/_media/` (organized by topic subdirectory).

## Dependencies

All examples use the dependencies specified in `pyproject.toml`. Install with:
```bash
uv sync
```

For CUDA examples, you need a CUDA toolkit and GPU.

## Further Reading

- [I-MPPI Documentation](../docs/i_mppi.qmd)
- [FSMI Usage Guide](../docs/fsmi_usage.md)
