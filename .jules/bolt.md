# BOLT'S JOURNAL

## 2026-02-05 - JAX Performance Optimizations

**Learning:** `jax.nn.softmax` performs redundant `max` reduction for stability. When we already know the maximum (e.g., when subtracting `min_cost` from `costs` to get negative values where max is 0), manual implementation (`exp(x)/sum(exp(x))`) saves one reduction.
**Action:** Use manual softmax when numerical stability is already handled by domain logic.

**Learning:** Nested `jax.vmap` for batched quadratic forms (e.g., `x^T M x` over batch dimensions) incurs tracing overhead and may not map to optimal kernels.
**Action:** Replace `vmap(vmap(lambda x: x @ M @ x))` with explicit vectorized operations (`sum((x @ M) * x, axis=-1)`) using `jnp.dot` and `jnp.sum`.

**Learning:** Python control flow (loops, if-else) prevents full JIT compilation of simulation loops.
**Action:** Use `jax.lax.scan` for time-stepping loops and `jax.lax.select`/`jax.lax.cond` for state-dependent logic to enable end-to-end JIT compilation, yielding massive speedups (e.g., 20x).
