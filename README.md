# jax_mppi

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![JAX](https://img.shields.io/badge/backend-JAX-blue)
![Status](https://img.shields.io/badge/status-planning-yellow)

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
-   **JAX Integration**:
    -   `jax.vmap` for efficient batch processing.
    -   `jax.lax.scan` for fast horizon loops.
    -   Fully compatible with JIT compilation for high-performance control loops.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jax_mppi.git
cd jax_mppi

# Install dependencies
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
│   └── autotune.py          # Autotuning utilities
├── examples/
│   ├── pendulum.py          # Pendulum environment example
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
6.  **Autotuning**: Parameter optimization using CMA-ES.

## Credits

This project is a direct port of [pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi). We aim to maintain parity with the original implementation while leveraging JAX's unique features for performance and flexibility.
