import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
noise = jax.random.multivariate_normal(subkey, mean=jnp.zeros(1), cov=jnp.eye(1), shape=(100, 15))
print(f"Mean of first step noise: {jnp.mean(noise[:, 0, 0])}")
