from dataclasses import dataclass, replace
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .types import DynamicsFn, RunningCostFn, TerminalCostFn


@dataclass(frozen=True)
class MPPIConfig:
    # Static config (not traced through JAX)
    num_samples: int  # K
    horizon: int  # T
    nx: int
    nu: int
    lambda_: float
    u_scale: float
    u_per_command: int
    step_dependent_dynamics: bool
    rollout_samples: int  # M
    rollout_var_cost: float
    rollout_var_discount: float
    sample_null_action: bool
    noise_abs_cost: bool


@register_pytree_node_class
@dataclass
class MPPIState:
    # Dynamic state (carried through JAX transforms)
    U: jax.Array  # (T, nu) nominal trajectory
    u_init: jax.Array  # (nu,) default action for shift
    noise_mu: jax.Array  # (nu,)
    noise_sigma: jax.Array  # (nu, nu)
    noise_sigma_inv: jax.Array
    u_min: Optional[jax.Array]
    u_max: Optional[jax.Array]
    key: jax.Array  # PRNG key

    def tree_flatten(self):
        return (
            (
                self.U,
                self.u_init,
                self.noise_mu,
                self.noise_sigma,
                self.noise_sigma_inv,
                self.u_min,
                self.u_max,
                self.key,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _bound_action(
    action: jax.Array, u_min: Optional[jax.Array], u_max: Optional[jax.Array]
) -> jax.Array:
    if u_min is None and u_max is None:
        return action
    if u_min is None:
        assert u_max is not None
        return jnp.minimum(action, u_max)
    if u_max is None:
        return jnp.maximum(action, u_min)
    return jnp.clip(action, u_min, u_max)


def _scaled_bounds(
    u_min: Optional[jax.Array],
    u_max: Optional[jax.Array],
    u_scale: float,
) -> Tuple[Optional[jax.Array], Optional[jax.Array]]:
    if u_scale == 1.0 or u_scale == 0.0:
        return u_min, u_max
    u_min_scaled = None if u_min is None else (u_min / u_scale)
    u_max_scaled = None if u_max is None else (u_max / u_scale)
    return u_min_scaled, u_max_scaled


def _shift_nominal(mppi_state: MPPIState, shift_steps: int) -> MPPIState:
    if shift_steps <= 0:
        return mppi_state
    horizon = mppi_state.U.shape[0]
    shift_steps = int(min(shift_steps, horizon))
    u_control = jnp.roll(mppi_state.U, -shift_steps, axis=0)
    fill = jnp.tile(mppi_state.u_init, (shift_steps, 1))
    u_control = u_control.at[-shift_steps:].set(fill)
    return replace(mppi_state, U=u_control)


def _sample_noise(
    key: jax.Array,
    num_samples: int,
    horizon: int,
    noise_mu: jax.Array,
    noise_sigma: jax.Array,
    sample_null_action: bool,
) -> Tuple[jax.Array, jax.Array]:
    key, subkey = jax.random.split(key)
    noise = jax.random.multivariate_normal(
        subkey,
        mean=noise_mu,
        cov=noise_sigma,
        shape=(num_samples, horizon),
    )
    if sample_null_action:
        noise = noise.at[0].set(jnp.zeros((horizon, noise_mu.shape[0])))
    return noise, key


def _state_for_cost(state: jax.Array, nx: int) -> jax.Array:
    if state.shape[-1] <= nx:
        return state
    return state[..., :nx]


def _call_dynamics(
    dynamics: DynamicsFn,
    state: jax.Array,
    action: jax.Array,
    t: int,
    step_dependent: bool,
) -> jax.Array:
    if step_dependent:
        return dynamics(state, action, t)
    return dynamics(state, action)


def _call_running_cost(
    running_cost: RunningCostFn,
    state: jax.Array,
    action: jax.Array,
    t: int,
    step_dependent: bool,
) -> jax.Array:
    if step_dependent:
        return running_cost(state, action, t)
    return running_cost(state, action)


def _single_rollout_costs(
    config: MPPIConfig,
    current_obs: jax.Array,
    actions: jax.Array,
    dynamics: DynamicsFn,
    running_cost: RunningCostFn,
    terminal_cost: Optional[TerminalCostFn],
) -> Tuple[jax.Array, jax.Array]:
    def step_fn(state, inputs):
        t, action = inputs
        next_state = _call_dynamics(
            dynamics, state, action, t, config.step_dependent_dynamics
        )
        cost_state = _state_for_cost(state, config.nx)
        step_cost = _call_running_cost(
            running_cost, cost_state, action, t, config.step_dependent_dynamics
        )
        return next_state, step_cost

    ts = jnp.arange(config.horizon)
    final_state, step_costs = jax.lax.scan(step_fn, current_obs, (ts, actions))
    if terminal_cost is None:
        terminal = jnp.array(0.0)
    else:
        terminal_state = _state_for_cost(final_state, config.nx)
        terminal = terminal_cost(terminal_state, actions[-1])
    return step_costs, terminal


def _compute_rollout_costs(
    config: MPPIConfig,
    current_obs: jax.Array,
    actions: jax.Array,
    dynamics: DynamicsFn,
    running_cost: RunningCostFn,
    terminal_cost: Optional[TerminalCostFn],
) -> jax.Array:
    per_step_costs, terminal_costs = jax.vmap(
        lambda a: _single_rollout_costs(
            config, current_obs, a, dynamics, running_cost, terminal_cost
        )
    )(actions)

    mean_step_costs = per_step_costs
    var_step_costs = jnp.zeros_like(per_step_costs)

    if config.rollout_samples > 1:
        # Placeholder: allow variance penalty without explicit stochastic rollouts.
        mean_step_costs = per_step_costs
        var_step_costs = jnp.zeros_like(per_step_costs)

    var_discount = config.rollout_var_discount ** jnp.arange(config.horizon)
    var_penalty = config.rollout_var_cost * jnp.sum(
        var_step_costs * var_discount, axis=1
    )
    return jnp.sum(mean_step_costs, axis=1) + terminal_costs + var_penalty


def _compute_noise_cost(
    noise: jax.Array,
    noise_sigma_inv: jax.Array,
    noise_abs_cost: bool,
) -> jax.Array:
    if noise_abs_cost:
        abs_noise = jnp.abs(noise)
        # Optimized batched quadratic cost: x^T * M * x
        # This is faster than einsum on most backends
        weighted_noise = jnp.dot(abs_noise, jnp.abs(noise_sigma_inv))
        quad = jnp.sum(weighted_noise * abs_noise, axis=-1)
    else:
        weighted_noise = jnp.dot(noise, noise_sigma_inv)
        quad = jnp.sum(weighted_noise * noise, axis=-1)
    return 0.5 * jnp.sum(quad, axis=1)


def _compute_weights(costs: jax.Array, lambda_: float) -> jax.Array:
    min_cost = jnp.min(costs)
    scaled = -(costs - min_cost) / lambda_
    return jax.nn.softmax(scaled)


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
        u_min=None if u_min is None else jnp.array(u_min),
        u_max=None if u_max is None else jnp.array(u_max),
        key=key,
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
    noise, key = _sample_noise(
        mppi_state.key,
        config.num_samples,
        config.horizon,
        mppi_state.noise_mu,
        mppi_state.noise_sigma,
        config.sample_null_action,
    )

    perturbed_actions = mppi_state.U[None, :, :] + noise
    scaled_actions = perturbed_actions * config.u_scale
    scaled_actions = _bound_action(
        scaled_actions, mppi_state.u_min, mppi_state.u_max
    )

    rollout_costs = _compute_rollout_costs(
        config,
        current_obs,
        scaled_actions,
        dynamics,
        running_cost,
        terminal_cost,
    )
    noise_costs = _compute_noise_cost(
        noise, mppi_state.noise_sigma_inv, config.noise_abs_cost
    )
    total_costs = rollout_costs + noise_costs

    weights = _compute_weights(total_costs, config.lambda_)
    delta_U = jnp.tensordot(weights, noise, axes=1)
    U_new = mppi_state.U + delta_U

    u_min_scaled, u_max_scaled = _scaled_bounds(
        mppi_state.u_min, mppi_state.u_max, config.u_scale
    )
    U_new = _bound_action(U_new, u_min_scaled, u_max_scaled)

    action_seq = U_new[: config.u_per_command]
    scaled_action_seq = _bound_action(
        action_seq * config.u_scale, mppi_state.u_min, mppi_state.u_max
    )
    action = (
        scaled_action_seq[0] if config.u_per_command == 1 else scaled_action_seq
    )

    new_state = replace(mppi_state, U=U_new, key=key)
    if shift:
        new_state = _shift_nominal(new_state, config.u_per_command)

    return action, new_state


def reset(
    config: MPPIConfig, mppi_state: MPPIState, key: jax.Array
) -> MPPIState:
    """Reset nominal trajectory."""
    U_new = jnp.tile(mppi_state.u_init, (config.horizon, 1))
    return replace(mppi_state, U=U_new, key=key)


def get_rollouts(
    config: MPPIConfig,
    mppi_state: MPPIState,
    current_obs: jax.Array,
    dynamics: DynamicsFn,
    num_rollouts: int = 1,
) -> jax.Array:
    """Forward-simulate trajectories for visualization."""
    noise, key = _sample_noise(
        mppi_state.key,
        num_rollouts,
        config.horizon,
        mppi_state.noise_mu,
        mppi_state.noise_sigma,
        sample_null_action=False,
    )
    perturbed_actions = mppi_state.U[None, :, :] + noise
    scaled_actions = perturbed_actions * config.u_scale
    scaled_actions = _bound_action(
        scaled_actions, mppi_state.u_min, mppi_state.u_max
    )

    def rollout_single(actions, obs):
        def step_fn(state, inputs):
            t, action = inputs
            next_state = _call_dynamics(
                dynamics, state, action, t, config.step_dependent_dynamics
            )
            return next_state, _state_for_cost(next_state, config.nx)

        ts = jnp.arange(config.horizon)
        init_state = obs
        _, states = jax.lax.scan(step_fn, init_state, (ts, actions))
        init_out = _state_for_cost(init_state, config.nx)
        return jnp.concatenate([init_out[None, :], states], axis=0)

    if current_obs.ndim == 1:
        rollouts = jax.vmap(lambda a: rollout_single(a, current_obs))(
            scaled_actions
        )
    else:
        rollouts = jax.vmap(
            lambda obs: jax.vmap(lambda a: rollout_single(a, obs))(
                scaled_actions
            )
        )(current_obs)

    return rollouts
