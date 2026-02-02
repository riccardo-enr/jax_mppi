# Performance Investigation: Quadrotor Examples

This document details the findings regarding performance differences between the quadrotor control examples: `quadrotor_hover.py`, `quadrotor_circle.py`, and `quadrotor_figure8_comparison.py`.

## Summary

- **`quadrotor_hover.py`**: Fast (~0.2s/step simulated).
- **`quadrotor_circle.py`**: Slow (~45x slower).
- **`quadrotor_figure8_comparison.py`**: Slowest (performs 3x the work of circle).

## Root Cause Analysis

The performance disparity is primarily due to the usage of JAX's Just-In-Time (JIT) compilation and how cost functions are handled in the control loop.

### 1. `quadrotor_hover.py` (Fast)

In this example, the MPPI command function is explicitly JIT-compiled by the user, and the cost function is effectively constant (closed over static parameters).

```python
# JIT compile the command function for speed
command_fn = jax.jit(
    lambda mppi_state, obs: mppi.command(
        config=config,
        mppi_state=mppi_state,
        current_obs=obs,
        dynamics=dynamics,
        running_cost=running_cost_fn,
        terminal_cost=terminal_cost_fn,
        shift=True,
    )
)
```

Because `command_fn` is JIT-compiled once, the entire MPPI iteration (sampling, rollout, cost evaluation, weight computation, update) runs as a highly optimized XLA kernel.

### 2. `quadrotor_circle.py` (Slow)

In this example, the MPPI command function is **not** JIT-compiled in the loop, and the cost function is re-created at every time step to update the reference target.

```python
# Control loop
for step in range(num_steps):
    # Create cost function for current reference point
    ref_pos = reference[step, 0:3]
    ref_vel = reference[step, 3:6]
    # New python function object created here:
    running_cost_fn = create_tracking_cost_at_time(
        Q_pos, Q_vel, R, ref_pos, ref_vel
    )

    # Compute optimal action (Eager execution / Re-tracing)
    action, mppi_state = mppi.command(
        config=config,
        mppi_state=mppi_state,
        current_obs=state,
        dynamics=dynamics,
        running_cost=running_cost_fn,
        terminal_cost=terminal_cost_fn,
        shift=True,
    )
```

**Issues:**
1.  **Lack of JIT**: `mppi.command` is executed in Python, dispatching JAX primitives (like `scan` and `vmap`) at every step. This incurs significant Python overhead.
2.  **Re-creation of Cost Function**: Even if one were to JIT `mppi.command`, passing a new `running_cost_fn` (a new Python callable closure) at every step would force JAX to re-compile the function at every iteration, because callables are treated as static arguments. Re-compilation is extremely expensive (often slower than eager execution).

### 3. `quadrotor_figure8_comparison.py` (Slowest)

This example shares the same implementation pattern as `quadrotor_circle.py` (re-creating cost functions in the loop without JIT). Additionally, it runs **three** separate controllers (`mppi`, `smppi`, `kmppi`) sequentially for each step, tripling the workload and overhead.

```python
    # Run each controller
    results = {}
    for controller in ["mppi", "smppi", "kmppi"]:
        states, actions, costs = run_controller(...)
```

## Benchmark Verification

A reproduction script demonstrated the magnitude of these differences for a 100-step simulation:

- **JIT Loop (Hover pattern)**: 0.24s
- **No JIT + Recreated Cost (Circle pattern)**: 10.85s (**~45x slower**)
- **JIT + Recreated Cost (Re-compilation)**: ~1.0s per 10 steps -> ~10s per 100 steps (Similar to No JIT)

## Recommendation (For Future Reference)

To fix the performance in tracking examples (`circle` and `figure8`), the implementation should be refactored to:

1.  **Parametrize the Cost Function**: Instead of capturing `ref_pos` in a closure, the cost function should accept the reference as an argument: `cost_fn(state, action, t, reference)`.
2.  **Pass Reference to Command**: Update `mppi.command` (or wrap it) to accept dynamic arguments (like `reference`) that are passed down to the cost function.
3.  **JIT Compile**: Wrap the update step in `jax.jit`.

Alternatively, for simple tracking, one can pass the entire reference trajectory to the cost function (if it fits in memory, which it does) and use the time index `t` (available in step-dependent dynamics/costs) to look up the current reference, allowing the cost function to be static and JIT-compatible.
