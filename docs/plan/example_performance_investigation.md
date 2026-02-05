# Performance Investigation: Quadrotor Examples

This document details the findings regarding performance differences between the quadrotor control examples: `quadrotor_hover.py`, `quadrotor_circle.py`, and `quadrotor_figure8_comparison.py`.

## Summary

- **`quadrotor_hover.py`**: Fast (~0.2s/step simulated).
- **`quadrotor_circle.py`**: Previously slow (~45x slower). Optimized to use `jax.lax.scan` for maximum performance.
- **`quadrotor_figure8_comparison.py`**: Previously slowest. Optimized to use `jax.lax.scan` for maximum performance.

## Root Cause Analysis

The performance disparity was primarily due to the usage of JAX's Just-In-Time (JIT) compilation and how cost functions are handled in the control loop.

### 1. `quadrotor_hover.py` (Fast)

In this example, the MPPI command function is explicitly JIT-compiled by the user, and the cost function is effectively constant (closed over static parameters).

### 2. `quadrotor_circle.py` (Slow -> Optimized)

Originally, this example re-created the cost function at every time step to update the reference target. This prevented JIT compilation of the main MPPI loop, forcing it to run in eager execution mode (or incurring massive re-compilation costs).

**Optimization Implemented:**
The implementation has been refactored to:

1. Use `step_dependent_dynamics=True` to allow passing the time step `t` to the cost function.
2. Use **`jax.lax.scan`** to wrap the entire simulation loop into a single JIT-compiled kernel. This eliminates Python loop dispatch overhead completely.
3. Pass the entire reference trajectory to the scan function and use `jax.lax.dynamic_slice` inside the loop to extract the current horizon's reference. This allows the cost function to close over dynamic data efficiently without recompilation.

**Parameter Tuning:**
To improve tracking performance, the following parameters were tuned:

- `num_samples`: Increased from 1000 to 2000.
- `horizon`: Increased from 30 to 50.
- `lambda`: Decreased from 1.0 to 0.1 (sharper selection).
- Cost weights: Significantly increased position and velocity weights.

### 3. `quadrotor_figure8_comparison.py` (Slowest -> Optimized)

This example shared the same issue as `quadrotor_circle.py` but for three different controllers (`mppi`, `smppi`, `kmppi`).

**Optimization Implemented:**
Similar to `quadrotor_circle.py`, the controllers have been updated to use `jax.lax.scan` for the simulation loop. This required adapting the update logic for all three variants to be compatible with `scan` and dynamic reference slicing.

**Parameter Tuning:**
Parameters were similarly tuned to handle the aggressive figure-8 trajectory (samples=2000, horizon=50, lambda=0.1).

## Recommendation (For Future Reference)

When implementing tracking controllers with JAX MPPI:

1. **Use `jax.lax.scan`**: For simulation loops, wrapping the entire loop in `scan` provides the best performance by minimizing Python overhead.
2. **Parametrize the Cost Function**: Avoid capturing changing concrete values (like current target) in closures if they prevent JIT.
3. **Use Data Dependencies**: Pass changing targets as arguments (Tracers) to the JIT-compiled function.
4. **Step-Dependent Dynamics**: Use `step_dependent_dynamics=True` to utilize the relative time index `t` for looking up references in a passed trajectory slice.
