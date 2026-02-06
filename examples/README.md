# JAX-MPPI Examples

This directory contains examples demonstrating various features of the `jax-mppi` library.

## Directory Structure

### [`i_mppi/`](i_mppi/) - Informative MPPI
Examples of I-MPPI (Informative Model Predictive Path Integral) control with information-driven exploration.

**Two-Layer Architecture (Zhang et al. 2020):**
- **Layer 2 (FSMI Analyzer, ~5 Hz):** Full FSMI with O(n^2) computation for reference trajectory
- **Layer 3 (I-MPPI Controller, ~50 Hz):** Uniform-FSMI with O(n) for local reactivity

The key insight: Layer 3 must include an informative term (Uniform-FSMI), otherwise you're doing
trajectory tracking, not informative control. The Uniform-FSMI ensures reactive viewpoint
maintenance during disturbances and handles occlusions between Layer 2 updates.

- **`i_mppi_simulation.py`** - Main I-MPPI simulation with **two-layer architecture**
  - Layer 2: Full FSMI using occupancy grids for reference trajectory
  - Layer 3: Uniform-FSMI in cost function for local reactivity
  - Biased MPPI with mixture sampling (MPPI, SMPPI, KMPPI)
  - Cost: `J = Tracking(ref) + Obstacles - lambda * Uniform_FSMI(local)`
  - Run: `uv run python examples/i_mppi/i_mppi_simulation.py`

- **`fsmi_grid_demo.py`** - Standalone FSMI demonstration
  - Shows FSMI computation on synthetic occupancy grids
  - Visualizes information gain heatmaps
  - Compares grid-based vs legacy geometric methods
  - Run: `uv run python examples/i_mppi/fsmi_grid_demo.py`

- **`i_mppi_simulation_legacy.py`** - Legacy geometric zone implementation
  - Original heuristic approach using rectangular zones
  - Kept for backward compatibility and comparison

### [`quadrotor/`](quadrotor/) - Quadrotor Control
Examples of quadrotor trajectory tracking with MPPI variants.

- **`quadrotor_hover.py`** - Basic hover control
- **`quadrotor_circle.py`** - Circular trajectory tracking
- **`quadrotor_custom_trajectory.py`** - Custom trajectory following
- **`quadrotor_hover_comparison.py`** - Compare MPPI/SMPPI/KMPPI for hover
- **`quadrotor_figure8_comparison.py`** - Compare controllers on figure-8 trajectory

### [`autotune/`](autotune/) - Hyperparameter Tuning
Examples of automatic hyperparameter optimization using evolutionary strategies.

- **`autotune_basic.py`** - Basic autotuning example
- **`autotune_pendulum.py`** - Tune MPPI for pendulum swing-up
- **`autotune_evosax_comparison.py`** - Compare different ES algorithms

### [`cuda/`](cuda/) - CUDA Acceleration
Examples using CUDA-accelerated MPPI kernels.

- **`cuda_pendulum_jit.py`** - CUDA kernel example
- **`test_cuda_mppi.py`** - CUDA MPPI tests
- **`test_cuda_bindings.py`** - Test CUDA bindings
- **`test_cuda_config.py`** - Test CUDA configuration

### [`basic/`](basic/) - Basic Examples
Simple examples for getting started.

- **`pendulum.py`** - Classic inverted pendulum swing-up
- **`smooth_comparison.py`** - Compare smoothness metrics across MPPI variants

## Quick Start

### Run the main I-MPPI simulation (Grid-based FSMI)
```bash
uv run python examples/i_mppi/i_mppi_simulation.py
```

### Run a basic quadrotor example
```bash
uv run python examples/quadrotor/quadrotor_hover.py
```

### Try the pendulum swing-up
```bash
uv run python examples/basic/pendulum.py
```

## Output

Most examples save visualizations to `docs/_media/`:
- `i_mppi_simulation.py` → `docs/_media/i_mppi_simulation.png`
- `fsmi_grid_demo.py` → `docs/_media/fsmi_grid_demo.png`, `fsmi_beam_demo.png`

## Dependencies

All examples use the dependencies specified in `pyproject.toml`. Install with:
```bash
uv sync
```

For CUDA examples, you need:
- CUDA toolkit installed
- CUDA-enabled GPU
- Built CUDA bindings (see `cuda_mppi/` directory)

## Further Reading

- [I-MPPI Documentation](../docs/i_mppi.qmd)
- [FSMI Usage Guide](../docs/fsmi_usage.md)
- [FSMI Implementation Notes](../docs/plan/completed/fsmi_grid_implementation.md)
