## 2026-02-05 - JAX Performance Optimizations
**Learning:** `jax.nn.softmax` performs redundant `max` reduction for stability. When we already know the maximum (e.g., when subtracting `min_cost` from `costs` to get negative values where max is 0), manual implementation (`exp(x)/sum(exp(x))`) saves one reduction.
**Action:** Use manual softmax when numerical stability is already handled by domain logic.

**Learning:** Nested `jax.vmap` for batched quadratic forms (e.g., `x^T M x` over batch dimensions) incurs tracing overhead and may not map to optimal kernels.
**Action:** Replace `vmap(vmap(lambda x: x @ M @ x))` with explicit vectorized operations (`sum((x @ M) * x, axis=-1)`) using `jnp.dot` and `jnp.sum`.
