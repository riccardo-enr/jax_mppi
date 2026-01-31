import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from dataclasses import dataclass, replace
from typing import Optional, Tuple
from .types import DynamicsFn, RunningCostFn, TerminalCostFn

@dataclass(frozen=True)
class MPPIConfig:
    # Static config (not traced through JAX)
    num_samples: int       # K
    horizon: int           # T
    nx: int
    nu: int
    lambda_: float
    u_scale: float
    u_per_command: int
    step_dependent_dynamics: bool
    rollout_samples: int   # M
    rollout_var_cost: float
    rollout_var_discount: float
    sample_null_action: bool
    noise_abs_cost: bool

@register_pytree_node_class
@dataclass
class MPPIState:
    # Dynamic state (carried through JAX transforms)
    U: jax.Array           # (T, nu) nominal trajectory
    u_init: jax.Array      # (nu,) default action for shift
    noise_mu: jax.Array    # (nu,)
    noise_sigma: jax.Array # (nu, nu)
    noise_sigma_inv: jax.Array
    u_min: Optional[jax.Array]
    u_max: Optional[jax.Array]
    key: jax.Array         # PRNG key

    def tree_flatten(self):
        return (
            (self.U, self.u_init, self.noise_mu, self.noise_sigma, self.noise_sigma_inv, self.u_min, self.u_max, self.key),
            None
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

def create(
    nx: int,
    nu: int,
    noise_sigma: jax.Array,
    num_samples: int = 100,
    horizon: int = 15,
    lambda_: float = 1.0,
    noise_mu: Optional[jax.Array] = None,
    u_min: Optional[jax.Array] = None,
    u_max: Optional[jax.Array] = None,
    u_init: Optional[jax.Array] = None,
    U_init: Optional[jax.Array] = None,
    u_scale: float = 1.0,
    u_per_command: int = 1,
    step_dependent_dynamics: bool = False,
    rollout_samples: int = 1,
    rollout_var_cost: float = 0.0,
    rollout_var_discount: float = 0.95,
    sample_null_action: bool = False,
    noise_abs_cost: bool = False,
    key: Optional[jax.Array] = None,
) -> Tuple[MPPIConfig, MPPIState]:
    """Factory: create config + initial state."""
    if key is None:
        key = jax.random.PRNGKey(0)

    config = MPPIConfig(
        num_samples=num_samples,
        horizon=horizon,
        nx=nx,
        nu=nu,
        lambda_=lambda_,
        u_scale=u_scale,
        u_per_command=u_per_command,
        step_dependent_dynamics=step_dependent_dynamics,
        rollout_samples=rollout_samples,
        rollout_var_cost=rollout_var_cost,
        rollout_var_discount=rollout_var_discount,
        sample_null_action=sample_null_action,
        noise_abs_cost=noise_abs_cost,
    )

    # Initialize state variables
    if noise_mu is None:
        noise_mu = jnp.zeros(nu)

    # Ensure noise_sigma is 2D
    if noise_sigma.ndim == 1:
        noise_sigma = jnp.diag(noise_sigma)

    noise_sigma_inv = jnp.linalg.inv(noise_sigma)

    if u_init is None:
        u_init = jnp.zeros(nu)

    if U_init is None:
        U_init = jnp.tile(u_init, (horizon, 1))

    mppi_state = MPPIState(
        U=U_init,
        u_init=u_init,
        noise_mu=noise_mu,
        noise_sigma=noise_sigma,
        noise_sigma_inv=noise_sigma_inv,
        u_min=u_min,
        u_max=u_max,
        key=key
    )

    return config, mppi_state

def command(
    config: MPPIConfig,
    mppi_state: MPPIState,
    current_obs: jax.Array,
    dynamics: DynamicsFn,
    running_cost: RunningCostFn,
    terminal_cost: Optional[TerminalCostFn] = None,
    shift: bool = True,
) -> Tuple[jax.Array, MPPIState]:
    """Compute optimal action and return updated state."""
    # Placeholder
    return mppi_state.U[0], mppi_state

def reset(config: MPPIConfig, mppi_state: MPPIState, key: jax.Array) -> MPPIState:
    """Reset nominal trajectory."""
    U_new = jnp.tile(mppi_state.u_init, (config.horizon, 1))
    return replace(mppi_state, U=U_new, key=key)

def get_rollouts(
    config: MPPIConfig, mppi_state: MPPIState,
    current_obs: jax.Array, dynamics: DynamicsFn,
    num_rollouts: int = 1,
) -> jax.Array:
    """Forward-simulate trajectories for visualization."""
    # Placeholder
    return jnp.zeros((num_rollouts, config.horizon + 1, config.nx))
