# jax_mppi

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)
[![Build](https://github.com/riccardo-enr/jax_mppi/actions/workflows/test.yml/badge.svg)](https://github.com/riccardo-enr/jax_mppi/actions/workflows/test.yml)
[![Publish to PyPI](https://github.com/riccardo-enr/jax_mppi/actions/workflows/publish.yml/badge.svg)](https://github.com/riccardo-enr/jax_mppi/actions/workflows/publish.yml)

**jax_mppi** is a functional, JIT-compilable port of the [pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi) library to JAX. It implements Model Predictive Path Integral (MPPI) control with a focus on performance and composability.

## Design Philosophy

This library embraces JAX's functional paradigm:

- **Pure Functions**: Core logic is implemented as pure functions `command(state, mppi_state) -> (action, mppi_state)`.
- **Dataclass State**: State is held in `jax.tree_util.register_dataclass` containers, allowing easy integration with `jit`, `vmap`, and `grad`.
- **No Side Effects**: Unlike the PyTorch version, there is no mutable `self`. State transitions are explicit.

## Key Features

- **Core MPPI**: Robust implementation of the standard MPPI algorithm.
- **Smooth MPPI (SMPPI)**: Maintains action sequences and smoothness costs for better trajectory generation.
- **Kernel MPPI (KMPPI)**: Uses kernel interpolation for control points, reducing the parameter space.
- **I-MPPI (Informative MPPI)**: Two-layer hierarchical architecture for autonomous exploration:
  - **Layer 2 (FSMI)**: Fast Shannon Mutual Information trajectory generation (~5 Hz)
  - **Layer 3 (Biased MPPI)**: Information-aware tracking control (~50 Hz)
  - Occupancy grid-based exploration with GPU acceleration
  - Interactive Colab notebook for real-time parameter tuning
- **Autotuning**: Built-in hyperparameter optimization with multiple backends:
  - **CMA-ES** (via \`cma\` library) - Classic evolution strategy
  - **CMA-ES, Sep-CMA-ES, OpenES** (via \`evosax\`) - JAX-native, GPU-accelerated ⚡
  - **Ray Tune** - Distributed hyperparameter search
  - **CMA-ME** (via \`ribs\`) - Quality diversity optimization
- **CUDA/C++ Backend**: High-performance implementations of all controllers in CUDA/C++17, exposed to Python via \`nanobind\`. Ideal for deployments needing maximum throughput.
- **JAX Integration**:
  - `jax.vmap` for efficient batch processing.
  - `jax.lax.scan` for fast horizon loops.
  - Fully compatible with JIT compilation for high-performance control loops.

## Installation

```bash
# Install from PyPI
pip install jax-mppi

# Or with optional dependencies
pip install jax-mppi[dev]              # Development tools
pip install jax-mppi[docs]             # Documentation
pip install jax-mppi[autotuning]       # Autotuning (cma + evosax)
pip install jax-mppi[autotuning-extra] # Ray Tune, Hyperopt, Ribs
```

### Development Installation

For contributors who want to work on the package (requires Python 3.12+):

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/riccardo-enr/jax_mppi.git
cd jax_mppi

# Or if already cloned without --recursive
git submodule update --init --recursive

# Install in development mode
pip install -e .
```

**Note:** The CUDA backend lives in a separate repository ([cuda_mppi](https://github.com/riccardo-enr/cuda-mppi)) integrated as a git submodule at `third_party/cuda-mppi`. You need to initialize submodules to build the CUDA components.

## Versioning

This project uses **Semantic Versioning** following the `major.minor.patch` scheme:

- **Major**: Breaking changes to the API or significant feature additions.
- **Minor**: New features or enhancements that are backward compatible.
- **Patch**: Bug fixes and minor updates.

See [CHANGELOG](./CHANGELOG.md) for detailed version history.

## Usage

```python
import jax
import jax.numpy as jnp
from jax_mppi import mppi

# Define dynamics and cost functions
def dynamics(state, action):
    # Your dynamics model here
    return state + action

def running_cost(state, action):
    # Your cost function here
    return jnp.sum(state**2) + jnp.sum(action**2)

# Create configuration and initial state
config, mppi_state = mppi.create(
    nx=4, nu=2,
    noise_sigma=jnp.eye(2) * 0.1,
    horizon=20,
    lambda_=1.0
)

# Control loop
key = jax.random.PRNGKey(0)
current_obs = jnp.zeros(4)

# JIT compile the command function for performance
jitted_command = jax.jit(mppi.command, static_argnames=['dynamics', 'running_cost'])

for _ in range(100):
    key, subkey = jax.random.split(key)
    action, mppi_state = jitted_command(
        config,
        mppi_state,
        current_obs,
        dynamics=dynamics,
        running_cost=running_cost
    )
    # Apply action to environment...
```

## Autotuning

JAX-MPPI includes powerful hyperparameter optimization capabilities. You can automatically tune MPPI parameters like `lambda_`, `noise_sigma`, and `horizon` using multiple optimization backends.

### Quick Example

```python
from jax_mppi import autotune, mppi

# Create MPPI configuration
config, state = mppi.create(nx=4, nu=2, horizon=20)
holder = autotune.ConfigStateHolder(config, state)

# Define what to tune
params_to_tune = [
    autotune.LambdaParameter(holder, min_value=0.1),
    autotune.NoiseSigmaParameter(holder, min_value=0.01),
]

# Define evaluation function
def evaluate():
    # Run MPPI, return cost
    # ... your evaluation logic ...
    return autotune.EvaluationResult(mean_cost=cost, ...)

# Choose an optimizer
from jax_mppi import autotune_evosax  # JAX-native, GPU-accelerated
optimizer = autotune_evosax.CMAESOpt(population=10, sigma=0.1)

# Or use classic CMA-ES
# optimizer = autotune.CMAESOpt(population=10, sigma=0.1)

# Run optimization
tuner = autotune.Autotune(
    params_to_tune=params_to_tune,
    evaluate_fn=evaluate,
    optimizer=optimizer,
)
best = tuner.optimize_all(iterations=50)
```

### Available Optimizers

| Optimizer                      | Backend       | GPU Support | Best For                           |
| ------------------------------ | ------------- | ----------- | ---------------------------------- |
| `autotune.CMAESOpt`            | `cma` library | ❌          | Classic CMA-ES, stable             |
| `autotune_evosax.CMAESOpt`     | evosax        | ✅          | JAX-native, 5-10x faster on GPU    |
| `autotune_evosax.SepCMAESOpt`  | evosax        | ✅          | High-dimensional problems          |
| `autotune_evosax.OpenESOpt`    | evosax        | ✅          | Large populations, parallelization |
| `autotune_global.RayOptimizer` | Ray Tune      | ✅          | Distributed search                 |
| `autotune_qd.CMAMEOpt`         | ribs          | ❌          | Quality diversity                  |

### Evosax vs CMA Library

**Migrating from `cma` to `evosax`:**

```python
# Before (cma library)
from jax_mppi.autotune import CMAESOpt
optimizer = CMAESOpt(population=10, sigma=0.1)

# After (evosax - JAX-native)
from jax_mppi.autotune_evosax import CMAESOpt
optimizer = CMAESOpt(population=10, sigma=0.1)
```

**Benefits of evosax:**

- ⚡ **5-10x faster** on GPU due to JIT compilation
- 🔧 **Multiple strategies** (CMA-ES, Sep-CMA-ES, OpenES, SNES, xNES)
- 🎯 **JAX-native** - seamless integration with JAX code
- 📦 **Pure Python** - no external C++ dependencies

See `examples/autotuning/evosax_comparison.py` for a detailed performance comparison.

## I-MPPI: Informative Path Planning

I-MPPI extends the MPPI framework with information-theoretic path planning for autonomous exploration. The system uses a hierarchical architecture that combines global strategic planning with reactive control:

### Architecture

- **Layer 2 (FSMI Planner)**: Generates information-rich reference trajectories using Fast Shannon Mutual Information
- **Layer 3 (Biased MPPI)**: Tracks references while gathering local information via Uniform-FSMI
- **Occupancy Grid**: Represents environment uncertainty and enables information gain computation

### Key Capabilities

- **Autonomous Exploration**: Seeks high-information regions while avoiding obstacles
- **Real-time Performance**: ~5 Hz for global planning, ~50 Hz for local control
- **GPU Accelerated**: Full JAX implementation for efficient computation
- **Interactive Tuning**: Jupyter notebook with widgets for parameter exploration

### Getting Started with I-MPPI

```python
from jax_mppi.i_mppi import FSMIConfig, create_fsmi_state

# Configure information-driven planner
config = FSMIConfig(
    grid_resolution=0.1,  # 10cm grid cells
    sensor_range=5.0,     # 5m sensing range
    info_weight=1.0       # Information gain weight
)

# Run I-MPPI simulation
# See examples/i_mppi/simulation.py for complete example
```

For detailed theory and implementation, see the [I-MPPI documentation](docs/src/i_mppi.qmd).

## Quadrotor Examples

JAX-MPPI includes comprehensive quadrotor control examples demonstrating trajectory tracking with nonlinear 6-DOF dynamics:

### Features

- **6-DOF Dynamics**: Full quaternion-based attitude representation with NED/FRD frame conventions
- **Multiple Trajectories**: Hover, circle, figure-8, and custom waypoint-based paths
- **MPPI Variant Comparison**: Side-by-side performance analysis of MPPI, SMPPI, and KMPPI
- **Real-time Performance**: 50 Hz control loops with JIT compilation
- **Rich Visualizations**: Trajectory plots, tracking errors, control inputs, and performance metrics

### Available Examples

```python
# Basic stabilization
python examples/quadrotor/hover.py

# Trajectory tracking
python examples/quadrotor/circle.py

# Waypoint navigation
python examples/quadrotor/custom_trajectory.py

# Compare MPPI variants
python examples/quadrotor/figure8_comparison.py
```

**Key Results**: SMPPI achieves 30-40% smoother control (lower jerk) compared to standard MPPI while maintaining similar tracking accuracy (<0.1m RMS error).

See [`examples/quadrotor/`](examples/quadrotor/) for more details.

## Project Structure

```text
jax_mppi/
├── src/jax_mppi/
│   ├── mppi.py              # Core MPPI implementation
│   ├── smppi.py             # Smooth MPPI variant
│   ├── kmppi.py             # Kernel MPPI variant
│   ├── types.py             # Type definitions
│   ├── autotune.py          # Autotuning core & CMA-ES (cma lib)
│   ├── autotune_evosax.py   # JAX-native optimizers (evosax)
│   ├── autotune_global.py   # Ray Tune integration
│   └── autotune_qd.py       # Quality Diversity optimization
├── examples/
│   ├── basic/               # Introductory examples (pendulum)
│   ├── quadrotor/           # Quadrotor control & comparisons
│   ├── i_mppi/              # Informative MPPI simulation
│   ├── autotuning/          # Hyperparameter optimization
│   ├── cuda/                # CUDA acceleration examples
│   └── benchmarks/          # Performance comparisons
└── tests/                   # Unit and integration tests
```

## Roadmap

The development is structured in phases:

1. **Core MPPI**: Basic implementation with JAX parity.
2. **Integration**: Pendulum example and verification.
3. **Smooth MPPI**: Implementation of smoothness constraints.
4. **Kernel MPPI**: Kernel-based control parameterization.
5. **Comparisons**: Benchmarking and visual comparisons.
6. **Autotuning**: Parameter optimization using CMA-ES, Ray Tune, and QD.

## Credits

This project is a direct port of [pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi). We aim to maintain parity with the original implementation while leveraging JAX's unique features for performance and flexibility.
