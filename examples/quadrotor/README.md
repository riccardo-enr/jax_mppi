# Quadrotor Trajectory Following Examples

This directory contains examples for quadrotor control using JAX-MPPI. The examples demonstrate how to use MPPI to control a 13D nonlinear quadrotor model to follow various reference trajectories.

## Structure

- **`trajectories.py`**: Utilities for generating reference trajectories (hover, circle, figure-8, etc.).
- **`../quadrotor_hover.py`**: Basic hover control example.
- **`../quadrotor_circle.py`**: Circular trajectory tracking example.
- **`../quadrotor_figure8_comparison.py`**: Comparison of MPPI variants (Standard, Smooth, Kernel) on a figure-8 trajectory.
- **`../quadrotor_custom_trajectory.py`**: Trajectory tracking with user-defined waypoints.

## Usage

Each example can be run directly from the root directory:

```bash
# Run hover example
python examples/quadrotor_hover.py --visualize

# Run circle tracking example
python examples/quadrotor_circle.py --visualize

# Run figure-8 comparison
python examples/quadrotor_figure8_comparison.py --visualize

# Run custom trajectory
python examples/quadrotor_custom_trajectory.py --visualize
```

## Visualization

The examples use `matplotlib` for visualization. Use the `--visualize` flag to enable plotting. Plots are saved to `docs/media/`.

## Theory

For detailed theoretical background and implementation details, please refer to the [documentation](../../docs/examples/quadrotor.md).
