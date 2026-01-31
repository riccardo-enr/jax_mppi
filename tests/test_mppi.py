import jax.numpy as jnp
from jax_mppi import mppi
import jax

def test_mppi_create():
    nx = 4
    nu = 2
    noise_sigma = jnp.eye(nu) * 0.1
    horizon = 20

    config, state = mppi.create(
        nx=nx,
        nu=nu,
        noise_sigma=noise_sigma,
        horizon=horizon
    )

    assert config.nx == nx
    assert config.nu == nu
    assert config.horizon == horizon
    assert state.U.shape == (horizon, nu)
    assert state.noise_sigma.shape == (nu, nu)

def test_mppi_reset():
    nx = 4
    nu = 2
    noise_sigma = jnp.eye(nu) * 0.1
    horizon = 20

    config, state = mppi.create(
        nx=nx,
        nu=nu,
        noise_sigma=noise_sigma,
        horizon=horizon,
        U_init=jnp.ones((horizon, nu))
    )

    assert jnp.all(state.U == 1.0)

    key = jax.random.PRNGKey(1)
    state = mppi.reset(config, state, key)

    # reset should set U to u_init (default zeros)
    assert jnp.all(state.U == 0.0)
