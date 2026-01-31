import jax
import jax.numpy as jnp
from jax_mppi import mppi

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

def test_mppi_command_shapes_and_bounds():
    nx = 1
    nu = 1
    noise_sigma = jnp.eye(nu) * 0.2
    horizon = 12

    def dynamics(state, action):
        return state + action

    def running_cost(state, action):
        return jnp.sum(state ** 2) + 0.01 * jnp.sum(action ** 2)

    config, state = mppi.create(
        nx=nx,
        nu=nu,
        noise_sigma=noise_sigma,
        horizon=horizon,
        num_samples=64,
        u_min=jnp.array([-0.2]),
        u_max=jnp.array([0.2]),
        u_per_command=2,
    )

    action, new_state = mppi.command(
        config,
        state,
        current_obs=jnp.array([1.0]),
        dynamics=dynamics,
        running_cost=running_cost,
    )

    assert action.shape == (2, 1)
    assert jnp.all(action <= 0.2 + 1e-6)
    assert jnp.all(action >= -0.2 - 1e-6)
    assert new_state.U.shape == (horizon, nu)

def test_mppi_get_rollouts_shapes():
    nx = 2
    nu = 2
    noise_sigma = jnp.eye(nu) * 0.1
    horizon = 5

    def dynamics(state, action):
        return state + action

    config, state = mppi.create(
        nx=nx,
        nu=nu,
        noise_sigma=noise_sigma,
        horizon=horizon,
        num_samples=32,
    )

    rollouts = mppi.get_rollouts(
        config,
        state,
        current_obs=jnp.zeros(nx),
        dynamics=dynamics,
        num_rollouts=3,
    )
    assert rollouts.shape == (3, horizon + 1, nx)

    batched_rollouts = mppi.get_rollouts(
        config,
        state,
        current_obs=jnp.zeros((2, nx)),
        dynamics=dynamics,
        num_rollouts=2,
    )
    assert batched_rollouts.shape == (2, 2, horizon + 1, nx)
