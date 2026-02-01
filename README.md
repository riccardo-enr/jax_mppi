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
- **Autotuning**: Built-in hyperparameter optimization with multiple backends:
  - **CMA-ES** (via `cma` library) - Classic evolution strategy
  - **CMA-ES, Sep-CMA-ES, OpenES** (via `evosax`) - JAX-native, GPU-accelerated ⚡
  - **Ray Tune** - Distributed hyperparameter search
  - **CMA-ME** (via `ribs`) - Quality diversity optimization
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
# Clone the repository
git clone https://github.com/riccardo-enr/jax_mppi.git
cd jax_mppi

# Install in development mode
pip install -e .
```

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

See `examples/autotune_evosax_comparison.py` for a detailed performance comparison.

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
│   ├── pendulum.py                    # Pendulum environment example
│   ├── autotune_basic.py              # Basic autotuning example
│   ├── autotune_pendulum.py           # Autotuning pendulum
│   ├── autotune_evosax_comparison.py  # Evosax vs cma performance
│   └── smooth_comparison.py           # Comparison of MPPI variants
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
