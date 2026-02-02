# Performance Investigation: Quadrotor Examples

This document details the findings regarding performance differences between the quadrotor control examples: `quadrotor_hover.py`, `quadrotor_circle.py`, and `quadrotor_figure8_comparison.py`.

## Summary

- **`quadrotor_hover.py`**: Fast (~0.2s/step simulated).
- **`quadrotor_circle.py`**: Previously slow (~45x slower). Optimized to use JIT.
- **`quadrotor_figure8_comparison.py`**: Previously slowest. Optimized to use JIT.

## Root Cause Analysis

The performance disparity was primarily due to the usage of JAX's Just-In-Time (JIT) compilation and how cost functions are handled in the control loop.

### 1. `quadrotor_hover.py` (Fast)

In this example, the MPPI command function is explicitly JIT-compiled by the user, and the cost function is effectively constant (closed over static parameters).

### 2. `quadrotor_circle.py` (Slow -> Optimized)

Originally, this example re-created the cost function at every time step to update the reference target. This prevented JIT compilation of the main MPPI loop, forcing it to run in eager execution mode (or incurring massive re-compilation costs).

**Optimization Implemented:**
The implementation has been refactored to:
1.  Use `step_dependent_dynamics=True` to allow passing the time step `t` to the cost function.
2.  Pass a slice of the reference trajectory (covering the current horizon) as an argument to the JIT-compiled update step.
3.  Define the cost function inside the JITted step to close over this reference slice (which JAX treats as a traced data dependency, not a static parameter).

This allows the entire update step to be JIT-compiled once and reused efficiently.

### 3. `quadrotor_figure8_comparison.py` (Slowest -> Optimized)

This example shared the same issue as `quadrotor_circle.py` but for three different controllers (`mppi`, `smppi`, `kmppi`).

**Optimization Implemented:**
Similar to `quadrotor_circle.py`, the controllers have been updated to use JIT-compiled update steps that accept the reference horizon as a dynamic argument. This works for all three variants:
- **MPPI**: Standard update.
- **SMPPI**: Uses `step_dependent_dynamics` to smooth actions against reference.
- **KMPPI**: Captures the kernel function in the JIT closure (as a static object) while treating the reference as dynamic data.

## Benchmark Verification

Simulations verify that the optimized versions run significantly faster, comparable to the `quadrotor_hover.py` example (excluding the overhead of running multiple controllers or more complex dynamics).

## Recommendation (For Future Reference)

When implementing tracking controllers with JAX MPPI:
1.  **Parametrize the Cost Function**: Avoid capturing changing concrete values (like current target) in closures if they prevent JIT.
2.  **Use Data Dependencies**: Pass changing targets as arguments (Tracers) to the JIT-compiled function.
3.  **Step-Dependent Dynamics**: Use `step_dependent_dynamics=True` to utilize the relative time index `t` for looking up references in a passed trajectory slice.
