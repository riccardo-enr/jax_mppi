# jax_mppi

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/pypi/pyversions/jax-mppi)
![Status](https://img.shields.io/badge/status-alpha-orange)
[![Build](https://github.com/riccardo-enr/jax_mppi/actions/workflows/test.yml/badge.svg)](https://github.com/riccardo-enr/jax_mppi/actions/workflows/test.yml)
[![Publish to PyPI](https://github.com/riccardo-enr/jax_mppi/actions/workflows/publish.yml/badge.svg)](https://github.com/riccardo-enr/jax_mppi/actions/workflows/publish.yml)

**jax_mppi** is a functional, JIT-compilable port of the [pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi) library to JAX. It implements Model Predictive Path Integral (MPPI) control with a focus on performance and composability.

## Design Philosophy

This library embraces JAX's functional paradigm:

-   **Pure Functions**: Core logic is implemented as pure functions `command(state, mppi_state) -> (action, mppi_state)`.
-   **Dataclass State**: State is held in `jax.tree_util.register_dataclass` containers, allowing easy integration with `jit`, `vmap`, and `grad`.
-   **No Side Effects**: Unlike the PyTorch version, there is no mutable `self`. State transitions are explicit.

## Key Features

-   **Core MPPI**: Robust implementation of the standard MPPI algorithm.
-   **Smooth MPPI (SMPPI)**: Maintains action sequences and smoothness costs for better trajectory generation.
-   **Kernel MPPI (KMPPI)**: Uses kernel interpolation for control points, reducing the parameter space.
-   **Autotuning**: Built-in hyperparameter optimization using CMA-ES, Ray Tune, and Quality Diversity.
-   **JAX Integration**:
    -   `jax.vmap` for efficient batch processing.
    -   `jax.lax.scan` for fast horizon loops.
    -   Fully compatible with JIT compilation for high-performance control loops.

## Installation

```bash
# Install from PyPI
pip install jax-mppi

# Or with optional dependencies
pip install jax-mppi[dev]  # Development tools
pip install jax-mppi[docs]  # Documentation
pip install jax-mppi[autotuning]  # Autotuning features
```

### Development Installation

For contributors who want to work on the package:

```bash
# Clone the repository
git clone https://github.com/riccardo-enr/jax_mppi.git
cd jax_mppi

# Install in development mode
pip install -e .
```

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

## Project Structure

```
jax_mppi/
├── src/jax_mppi/
│   ├── mppi.py              # Core MPPI implementation
│   ├── smppi.py             # Smooth MPPI variant
│   ├── kmppi.py             # Kernel MPPI variant
│   ├── types.py             # Type definitions
│   ├── autotune.py          # Autotuning core & CMA-ES
│   ├── autotune_global.py   # Ray Tune integration
│   └── autotune_qd.py       # Quality Diversity optimization
├── examples/
│   ├── pendulum.py          # Pendulum environment example
│   ├── autotune_basic.py    # Basic autotuning example
│   ├── autotune_pendulum.py # Autotuning pendulum
│   └── smooth_comparison.py # Comparison of MPPI variants
└── tests/                   # Unit and integration tests
```

## Roadmap

The development is structured in phases:

1.  **Core MPPI**: Basic implementation with JAX parity.
2.  **Integration**: Pendulum example and verification.
3.  **Smooth MPPI**: Implementation of smoothness constraints.
4.  **Kernel MPPI**: Kernel-based control parameterization.
5.  **Comparisons**: Benchmarking and visual comparisons.
6.  **Autotuning**: Parameter optimization using CMA-ES, Ray Tune, and QD.

## Credits

This project is a direct port of [pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi). We aim to maintain parity with the original implementation while leveraging JAX's unique features for performance and flexibility.
