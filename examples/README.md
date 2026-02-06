# JAX-MPPI Examples

This directory contains examples demonstrating various features of the `jax-mppi` library, organized by category and complexity.

## Directory Structure

### [`basic/`](basic/) - Basic Examples
Simple introductory examples for getting started.
- `pendulum.py`: Classic inverted pendulum swing-up and stabilization.

### [`quadrotor/`](quadrotor/) - Quadrotor Control
Trajectory tracking for quadrotors using various MPPI variants.
- `hover.py`: Basic hover stabilization.
- `circle.py`: Circular trajectory tracking.
- `figure8.py`: Aggressive figure-8 trajectory tracking with MPPI/SMPPI/KMPPI comparison.
- `custom_trajectory.py`: Following user-defined waypoint trajectories.
- `hover_comparison.py`: Comparing MPPI variants on a hover task.

### [`i_mppi/`](i_mppi/) - Informative MPPI
Informative Model Predictive Path Integral control for exploration.
- `simulation.py`: Main I-MPPI simulation with two-layer architecture.
- `fsmi_grid_demo.py`: Standalone demonstration of Fast Shannon Mutual Information (FSMI) on grids.

### [`autotuning/`](autotuning/) - Hyperparameter Tuning
Automatic optimization of MPPI parameters.
- `basic.py`: Minimal example of MPPI autotuning.
- `pendulum.py`: Tuning MPPI for the pendulum swing-up task.
- `evosax_comparison.py`: Comparison of different Evolutionary Strategies.

### [`cuda/`](cuda/) - CUDA Acceleration
Examples utilizing CUDA-accelerated MPPI kernels.
- `pendulum_jit.py`: Runtime JIT compilation of CUDA kernels for custom dynamics.
- `test_cuda_mppi.py`: Testing CUDA MPPI implementation.

### [`benchmarks/`](benchmarks/) - Performance & Comparison
Benchmarks and comparative analysis.
- `smooth_comparison.py`: Comparison of smoothness and efficiency across MPPI variants.

## Quick Start

### Run a basic pendulum example
```bash
uv run python examples/basic/pendulum.py
```

### Run a quadrotor hover example
```bash
uv run python examples/quadrotor/hover.py
```

### Run the main I-MPPI simulation
```bash
uv run python examples/i_mppi/simulation.py
```

## Output

Most examples save visualizations to `docs/_media/`. Ensure this directory exists or check the script output for saved paths.

## Dependencies

Install dependencies using `uv`:
```bash
uv sync
```

For autotuning examples:
```bash
uv pip install ".[autotuning]"
```

For CUDA examples, a CUDA-capable GPU and the CUDA toolkit are required.

## Further Reading

- [I-MPPI Documentation](../docs/i_mppi.qmd)
- [Autotuning Guide](../docs/autotuning.md)
