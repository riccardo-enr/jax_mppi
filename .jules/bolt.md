## 2026-02-07 - [JAX Einsum Performance]
**Learning:** `jnp.einsum` can be significantly slower than `jnp.dot` + `jnp.sum` for batched quadratic forms like $x^T A x$ in JAX.
**Action:** Replace `einsum` with explicit `dot` and `sum` for performance-critical inner loops, especially when shapes align well for matrix multiplication.
