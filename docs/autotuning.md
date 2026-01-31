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

## Theoretical Background

### CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

CMA-ES is a derivative-free optimization algorithm that adapts a multivariate normal distribution to sample candidate solutions. It is particularly effective for non-convex optimization problems with continuous parameters.

The algorithm maintains a mean vector $\mathbf{m}$ and a covariance matrix $\mathbf{C}$. In each generation $g$:

1.  **Sampling**: Generate $\lambda$ offspring:
    \[
    \mathbf{x}_i \sim \mathcal{N}(\mathbf{m}^{(g)}, \sigma^{(g)2} \mathbf{C}^{(g)})
    \]
2.  **Selection**: Evaluate the offspring and select the best $\mu$ candidates.
3.  **Update**: Update $\mathbf{m}^{(g+1)}$ as a weighted mean of the selected candidates. Update $\mathbf{C}^{(g+1)}$ and step size $\sigma^{(g+1)}$ to increase the likelihood of successful steps.

This allows CMA-ES to learn the local landscape of the objective function (e.g., MPPI performance) and scale the search distribution accordingly.

### Quality Diversity (QD) and MAP-Elites

Quality Diversity algorithms aim to find a set of solutions that are both high-performing and diverse. The `jax_mppi` autotuning framework uses the MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) algorithm.

MAP-Elites maintains an archive of high-performing solutions, discretized by a user-defined feature space (Behavioral Descriptors).

1.  **Mapping**: Each candidate solution $\mathbf{x}$ is mapped to a feature descriptor $\mathbf{b}(\mathbf{x})$.
2.  **Archive**: The feature space is divided into cells (bins). Each cell stores the best solution found so far with that descriptor.
3.  **Selection and Variation**: Solutions are selected from the archive and perturbed to generate new candidates.
4.  **Replacement**: A new candidate replaces the occupant of its corresponding cell if it has a higher fitness.

This approach is useful for finding MPPI parameters that work well across different regimes (e.g., aggressive vs. conservative behavior) or environment conditions.

### Global Optimization (Ray Tune)

`autotune_global` leverages Ray Tune to perform global search over hyperparameters. This allows for defining complex search spaces (e.g., log-uniform distributions) and using advanced schedulers like ASHA or Population Based Training (PBT).

The objective is to find:
\[
\theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}[J(\text{MPPI}(\theta))]
\]
where $\Theta$ is the hyperparameter search space and $J$ is the cumulative cost of the control task.
