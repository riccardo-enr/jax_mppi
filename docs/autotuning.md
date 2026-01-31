# Autotuning Guide

JAX-MPPI includes a robust autotuning framework to optimize MPPI hyperparameters (like temperature $\lambda$, noise covariance $\Sigma$, and planning horizon). The framework supports multiple optimization strategies, including CMA-ES, Ray Tune, and Quality Diversity (QD) methods.

## Overview

The autotuning process involves three main components:

1.  **Tunable Parameters**: Parameters you want to optimize (e.g., `LambdaParameter`, `NoiseSigmaParameter`).
2.  **Evaluation Function**: A function that runs MPPI with a specific configuration and returns a cost (and optionally other metrics).
3.  **Optimizer**: The algorithm used to search for the best parameters (e.g., `CMAESOpt`).

## Basic Usage (CMA-ES)

The `autotune` module provides a simple interface for CMA-ES optimization.

```python
import jax.numpy as jnp
from jax_mppi import mppi, autotune

# 1. Setup MPPI
config, state = mppi.create(...)
holder = autotune.ConfigStateHolder(config, state)

# 2. Define evaluation
def evaluate():
    # Run simulation with holder.config and holder.state
    # Calculate performance cost
    return autotune.EvaluationResult(mean_cost=cost, ...)

# 3. Create Tuner
tuner = autotune.Autotune(
    params_to_tune=[
        autotune.LambdaParameter(holder, min_value=0.1),
        autotune.NoiseSigmaParameter(holder, min_value=0.1),
    ],
    evaluate_fn=evaluate,
    optimizer=autotune.CMAESOpt(population=10),
)

# 4. Optimize
best_result = tuner.optimize_all(iterations=30)
print(f"Best parameters: {best_result.params}")
```

See `examples/autotune_basic.py` and `examples/autotune_pendulum.py` for complete running examples.

## Advanced Usage

### Global Optimization with Ray Tune

For more complex search spaces or when you want to use advanced schedulers and search algorithms (like HyperOpt or Bayesian Optimization), use `autotune_global`.

> **Note**: Requires `ray[tune]`, `hyperopt`, and `bayesian-optimization`.

```python
from ray import tune
from jax_mppi import autotune_global as autog

# Define search space using Ray Tune's API
params = [
    autog.GlobalLambdaParameter(holder, search_space=tune.loguniform(0.1, 10.0)),
    autog.GlobalNoiseSigmaParameter(holder, search_space=tune.uniform(0.1, 2.0)),
]

tuner = autog.AutotuneGlobal(
    params_to_tune=params,
    evaluate_fn=evaluate,
    optimizer=autog.RayOptimizer(),
)

best = tuner.optimize_all(iterations=100)
```

### Quality Diversity (QD)

To find a diverse set of high-performing parameters (e.g., finding parameters that work well for different environments or behavioral descriptors), use `autotune_qd`.

```python
from jax_mppi import autotune_qd

tuner = autotune.Autotune(
    params_to_tune=[...],
    evaluate_fn=evaluate,
    optimizer=autotune_qd.CMAMEOpt(population=20, bins=10),
)
```

## Tunable Parameters

The framework supports tuning the following parameters out-of-the-box:

-   `LambdaParameter`: MPPI temperature ($\lambda$).
-   `NoiseSigmaParameter`: Exploration noise covariance diagonal.
-   `MuParameter`: Exploration noise mean.
-   `HorizonParameter`: Planning horizon length (resizes internal buffers automatically).

You can also define custom parameters by subclassing `TunableParameter`.
