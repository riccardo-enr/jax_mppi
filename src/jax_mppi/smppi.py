"""Smooth MPPI (SMPPI) implementation in JAX.

SMPPI operates in a lifted control space where the nominal trajectory U represents
velocity/acceleration commands rather than direct actions. The actual action sequence
is computed through numerical integration, with smoothness penalties on action differences.

Reference: Based on pytorch_mppi SMPPI implementation
"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .types import DynamicsFn, RunningCostFn, TerminalCostFn


@dataclass(frozen=True)
class SMPPIConfig:
    """Configuration for Smooth MPPI.

    Extends base MPPI with smoothness-specific parameters.
    """

    # Base MPPI parameters
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

    # SMPPI-specific parameters
    w_action_seq_cost: float  # Weight on smoothness penalty
    delta_t: float  # Integration timestep


@register_pytree_node_class
@dataclass
class SMPPIState:
    """State for Smooth MPPI.

    Includes both velocity controls (U) and integrated actions (action_sequence).
    """

    # Base MPPI state
    U: jax.Array  # (T, nu) velocity/acceleration commands
    u_init: jax.Array  # (nu,) default velocity for shift
    noise_mu: jax.Array  # (nu,)
    noise_sigma: jax.Array  # (nu, nu)
    noise_sigma_inv: jax.Array
    u_min: Optional[jax.Array]  # Control velocity bounds
    u_max: Optional[jax.Array]
    key: jax.Array  # PRNG key

    # SMPPI-specific state
    action_sequence: jax.Array  # (T, nu) integrated actions
    action_min: Optional[jax.Array]  # Action bounds
    action_max: Optional[jax.Array]

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
                self.action_sequence,
                self.action_min,
                self.action_max,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _bound_control(
    control: jax.Array, u_min: Optional[jax.Array], u_max: Optional[jax.Array]
) -> jax.Array:
    """Bound control velocity (U space)."""
    if u_min is None and u_max is None:
        return control
    if u_min is None:
        assert u_max is not None
        return jnp.minimum(control, u_max)
    if u_max is None:
        return jnp.maximum(control, u_min)
    return jnp.clip(control, u_min, u_max)


def _bound_action(
    action: jax.Array,
    action_min: Optional[jax.Array],
    action_max: Optional[jax.Array],
) -> jax.Array:
    """Bound final action (action_sequence space)."""
    if action_min is None and action_max is None:
        return action
    if action_min is None:
        assert action_max is not None
        return jnp.minimum(action, action_max)
    if action_max is None:
        return jnp.maximum(action, action_min)
    return jnp.clip(action, action_min, action_max)


def _scaled_bounds(
    bounds: Optional[jax.Array],
    u_scale: float,
) -> Optional[jax.Array]:
    """Scale bounds by u_scale factor."""
    if bounds is None:
        return None
    return bounds / u_scale


def _shift_nominal(smppi_state: SMPPIState, shift_steps: int) -> SMPPIState:
    """Shift nominal trajectory for SMPPI.

    Shifts both U (velocity) and action_sequence, maintaining continuity.
    """
    # Shift control velocity: roll left, fill end with u_init
    u_shifted = jnp.roll(smppi_state.U, -shift_steps, axis=0)
    u_shifted = u_shifted.at[-shift_steps:].set(smppi_state.u_init)

    # Shift action sequence: roll left, hold at last value (not reset!)
    action_shifted = jnp.roll(smppi_state.action_sequence, -shift_steps, axis=0)
    # Fill with the last valid action (preserves continuity)
    last_action = smppi_state.action_sequence[-shift_steps - 1]
    action_shifted = action_shifted.at[-shift_steps:].set(last_action)

    return replace(smppi_state, U=u_shifted, action_sequence=action_shifted)


def _sample_noise(
    key: jax.Array,
    K: int,
    T: int,
    noise_mu: jax.Array,
    noise_sigma: jax.Array,
    sample_null_action: bool,
) -> Tuple[jax.Array, jax.Array]:
    """Sample noise for perturbations in velocity space."""
    if sample_null_action:
        # Reserve first sample for null action (U with no perturbation)
        key, subkey = jax.random.split(key)
        noise = jax.random.multivariate_normal(
            subkey, noise_mu, noise_sigma, shape=(K - 1, T)
        )
        # Prepend zeros for null action
        null_noise = jnp.zeros((1, T, noise_mu.shape[0]))
        noise = jnp.concatenate([null_noise, noise], axis=0)
    else:
        key, subkey = jax.random.split(key)
        noise = jax.random.multivariate_normal(
            subkey, noise_mu, noise_sigma, shape=(K, T)
        )

    return noise, key


def _compute_perturbed_actions_and_noise(
    config: SMPPIConfig,
    smppi_state: SMPPIState,
    noise: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Compute perturbed actions by integrating perturbed control velocities.

    Args:
        config: SMPPI configuration
        smppi_state: SMPPI state
        noise: (K, T, nu) noise samples in velocity space

    Returns:
        perturbed_actions: (K, T, nu) integrated perturbed actions
        effective_noise: (K, T, nu) noise after bounding adjustments
    """
    # Perturb control velocity
    perturbed_control = smppi_state.U[None, :, :] + noise  # (K, T, nu)

    # Bound control velocity
    perturbed_control = _bound_control(
        perturbed_control, smppi_state.u_min, smppi_state.u_max
    )

    # Integrate to action space
    perturbed_actions = (
        smppi_state.action_sequence[None, :, :]
        + perturbed_control * config.delta_t
    )

    # Bound actions
    perturbed_actions = _bound_action(
        perturbed_actions, smppi_state.action_min, smppi_state.action_max
    )

    # Recompute effective noise in velocity space after bounding
    # This ensures the noise cost reflects actual perturbations
    effective_noise = (
        perturbed_actions - smppi_state.action_sequence[None, :, :]
    ) / config.delta_t - smppi_state.U[None, :, :]

    return perturbed_actions, effective_noise


def _compute_smoothness_cost(
    perturbed_actions: jax.Array, config: SMPPIConfig
) -> jax.Array:
    """Compute smoothness cost from action differences.

    Args:
        perturbed_actions: (K, T, nu) perturbed action sequences
        config: SMPPI configuration

    Returns:
        smoothness_costs: (K,) smoothness cost per sample
    """
    # Temporal differences between consecutive actions
    # diff along time dimension: (K, T-1, nu)
    action_diff = jnp.diff(perturbed_actions, axis=1) * config.u_scale

    # L2 penalty on action changes: sum over time and action dims
    smoothness_costs = jnp.sum(action_diff**2, axis=(1, 2))

    # Weight the smoothness cost
    smoothness_costs *= config.w_action_seq_cost

    return smoothness_costs


# Import these from base mppi module (we'll reuse them)
from .mppi import (
    _call_dynamics,
    _call_running_cost,
    _state_for_cost,
)


def _single_rollout_costs(
    config: SMPPIConfig,
    current_obs: jax.Array,
    actions: jax.Array,
    dynamics: DynamicsFn,
    running_cost: RunningCostFn,
    terminal_cost: Optional[TerminalCostFn],
) -> Tuple[jax.Array, jax.Array]:
    """Compute costs for a single action sequence rollout."""

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
    config: SMPPIConfig,
    current_obs: jax.Array,
    actions: jax.Array,
    dynamics: DynamicsFn,
    running_cost: RunningCostFn,
    terminal_cost: Optional[TerminalCostFn],
) -> jax.Array:
    """Compute rollout costs for all perturbed action sequences.

    Args:
        config: SMPPI configuration
        current_obs: (nx,) current state
        actions: (K, T, nu) perturbed action sequences
        dynamics: dynamics function
        running_cost: running cost function
        terminal_cost: optional terminal cost function

    Returns:
        total_costs: (K,) total cost per sample
    """
    per_step_costs, terminal_costs = jax.vmap(
        lambda a: _single_rollout_costs(
            config, current_obs, a, dynamics, running_cost, terminal_cost
        )
    )(actions)

    # Sum running costs over horizon
    running_costs_sum = jnp.sum(per_step_costs, axis=1)

    # Total rollout cost
    return running_costs_sum + terminal_costs


def _compute_noise_cost(
    noise: jax.Array,
    noise_sigma_inv: jax.Array,
    noise_abs_cost: bool,
) -> jax.Array:
    """Compute cost of noise (perturbations in velocity space).

    Args:
        noise: (K, T, nu) effective noise samples
        noise_sigma_inv: (nu, nu) inverse covariance
        noise_abs_cost: whether to use absolute value cost

    Returns:
        noise_costs: (K,) noise cost per sample
    """
    if noise_abs_cost:
        # Absolute value cost: sum |noise|
        return jnp.sum(jnp.abs(noise), axis=(1, 2))
    else:
        # Quadratic cost: noise^T Sigma^-1 noise
        # For each sample and timestep: nu^T Sigma^-1 nu
        costs_per_timestep = jax.vmap(
            jax.vmap(lambda n: n @ noise_sigma_inv @ n)
        )(noise)
        return jnp.sum(costs_per_timestep, axis=1)


def _compute_weights(costs: jax.Array, lambda_: float) -> jax.Array:
    """Compute importance weights from costs using softmax."""
    # Numerical stability: subtract min cost
    costs_normalized = costs - jnp.min(costs)
    weights = jnp.exp(-costs_normalized / lambda_)
    weights = weights / jnp.sum(weights)
    return weights


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
    action_min: Optional[jax.Array] = None,
    action_max: Optional[jax.Array] = None,
    u_scale: float = 1.0,
    u_per_command: int = 1,
    step_dependent_dynamics: bool = False,
    rollout_samples: int = 1,
    rollout_var_cost: float = 0.0,
    rollout_var_discount: float = 0.95,
    sample_null_action: bool = False,
    noise_abs_cost: bool = False,
    w_action_seq_cost: float = 1.0,
    delta_t: float = 1.0,
    key: Optional[jax.Array] = None,
) -> Tuple[SMPPIConfig, SMPPIState]:
    """Create SMPPI configuration and initial state.

    Args:
        nx: State dimension
        nu: Action dimension
        noise_sigma: (nu, nu) noise covariance matrix
        num_samples: Number of MPPI samples (K)
        horizon: Planning horizon (T)
        lambda_: Temperature parameter for importance weighting
        noise_mu: (nu,) noise mean (default: zeros)
        u_min: (nu,) lower bounds on control velocity
        u_max: (nu,) upper bounds on control velocity
        u_init: (nu,) default control velocity for shift (default: zeros)
        U_init: (T, nu) initial control velocity trajectory (default: zeros)
        action_min: (nu,) lower bounds on actions
        action_max: (nu,) upper bounds on actions
        u_scale: Scale factor for control
        u_per_command: Number of control steps per command
        step_dependent_dynamics: Whether dynamics depend on timestep
        rollout_samples: Number of rollout samples for stochastic dynamics
        rollout_var_cost: Variance cost weight
        rollout_var_discount: Discount factor for variance cost
        sample_null_action: Whether to include null action in samples
        noise_abs_cost: Use absolute value cost for noise
        w_action_seq_cost: Weight on smoothness penalty
        delta_t: Integration timestep
        key: PRNG key (default: create new)

    Returns:
        config: SMPPI configuration
        state: SMPPI initial state
    """
    # Initialize defaults
    if noise_mu is None:
        noise_mu = jnp.zeros(nu)
    if u_init is None:
        u_init = jnp.zeros(nu)
    if key is None:
        key = jax.random.PRNGKey(0)

    # Scale bounds
    u_min_scaled = _scaled_bounds(u_min, u_scale)
    u_max_scaled = _scaled_bounds(u_max, u_scale)
    action_min_scaled = _scaled_bounds(action_min, u_scale)
    action_max_scaled = _scaled_bounds(action_max, u_scale)

    # Symmetric bounds inference
    if action_min_scaled is not None and action_max_scaled is None:
        action_max_scaled = -action_min_scaled
    if action_max_scaled is not None and action_min_scaled is None:
        action_min_scaled = -action_max_scaled

    # Initialize control velocity trajectory (U starts at zeros for SMPPI)
    if U_init is None:
        u_control = jnp.zeros((horizon, nu))
        action_sequence = jnp.zeros((horizon, nu))
    else:
        u_control = jnp.zeros_like(U_init)  # Start with zero velocity
        action_sequence = (
            U_init.copy()
        )  # U_init is interpreted as initial actions

    # Compute noise covariance inverse
    noise_sigma_inv = jnp.linalg.inv(noise_sigma)

    # Create config
    config = SMPPIConfig(
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
        w_action_seq_cost=w_action_seq_cost,
        delta_t=delta_t,
    )

    # Create state
    state = SMPPIState(
        U=u_control,
        u_init=u_init,
        noise_mu=noise_mu,
        noise_sigma=noise_sigma,
        noise_sigma_inv=noise_sigma_inv,
        u_min=u_min_scaled,
        u_max=u_max_scaled,
        key=key,
        action_sequence=action_sequence,
        action_min=action_min_scaled,
        action_max=action_max_scaled,
    )

    return config, state


def command(
    config: SMPPIConfig,
    smppi_state: SMPPIState,
    current_obs: jax.Array,
    dynamics: DynamicsFn,
    running_cost: RunningCostFn,
    terminal_cost: Optional[TerminalCostFn] = None,
    shift: bool = True,
) -> Tuple[jax.Array, SMPPIState]:
    """Compute optimal action using Smooth MPPI.

    Args:
        config: SMPPI configuration
        smppi_state: Current SMPPI state
        current_obs: (nx,) current observation/state
        dynamics: Dynamics function
        running_cost: Running cost function
        terminal_cost: Optional terminal cost function
        shift: Whether to shift nominal trajectory after computing action

    Returns:
        action: (u_per_command * nu,) or (nu,) optimal action
        new_state: Updated SMPPI state
    """
    # Sample noise in velocity space
    noise, new_key = _sample_noise(
        smppi_state.key,
        config.num_samples,
        config.horizon,
        smppi_state.noise_mu,
        smppi_state.noise_sigma,
        config.sample_null_action,
    )

    # Compute perturbed actions and effective noise
    perturbed_actions, effective_noise = _compute_perturbed_actions_and_noise(
        config, smppi_state, noise
    )

    # Compute rollout costs
    rollout_costs = _compute_rollout_costs(
        config,
        current_obs,
        perturbed_actions,
        dynamics,
        running_cost,
        terminal_cost,
    )

    # Compute noise cost (in velocity space)
    noise_costs = _compute_noise_cost(
        effective_noise,
        smppi_state.noise_sigma_inv,
        config.noise_abs_cost,
    )

    # Compute smoothness cost
    smoothness_costs = _compute_smoothness_cost(perturbed_actions, config)

    # Total cost combines all three components
    total_costs = rollout_costs + noise_costs + smoothness_costs

    # Compute importance weights
    weights = _compute_weights(total_costs, config.lambda_)

    # Weighted update to control velocity
    delta_U = jnp.sum(weights[:, None, None] * effective_noise, axis=0)
    new_U = smppi_state.U + delta_U

    # Integrate to update action sequence
    new_action_sequence = smppi_state.action_sequence + new_U * config.delta_t

    # Update state
    new_state = replace(
        smppi_state,
        U=new_U,
        action_sequence=new_action_sequence,
        key=new_key,
    )

    # Shift nominal trajectory if requested
    if shift:
        new_state = _shift_nominal(new_state, shift_steps=config.u_per_command)

    # Extract action to return
    if config.u_per_command == 1:
        action = new_action_sequence[0] * config.u_scale
    else:
        action = (
            new_action_sequence[: config.u_per_command].reshape(-1)
            * config.u_scale
        )

    return action, new_state


def reset(
    config: SMPPIConfig, smppi_state: SMPPIState, key: jax.Array
) -> SMPPIState:
    """Reset SMPPI state with new random key."""
    # Reset both U and action_sequence to zeros
    return replace(
        smppi_state,
        U=jnp.zeros_like(smppi_state.U),
        action_sequence=jnp.zeros_like(smppi_state.action_sequence),
        key=key,
    )


def get_rollouts(
    config: SMPPIConfig,
    smppi_state: SMPPIState,
    current_obs: jax.Array,
    dynamics: DynamicsFn,
    num_rollouts: int = 1,
) -> jax.Array:
    """Generate rollout trajectories using current action sequence.

    Args:
        config: SMPPI configuration
        smppi_state: Current SMPPI state
        current_obs: (nx,) or (batch, nx) current state
        dynamics: Dynamics function
        num_rollouts: Number of rollout samples

    Returns:
        rollouts: (num_rollouts, horizon+1, nx) state trajectories
    """

    def single_rollout(carry, _):
        state = carry

        def step_fn(s, inputs):
            t, action = inputs
            next_s = _call_dynamics(
                dynamics, s, action, t, config.step_dependent_dynamics
            )
            next_s_trimmed = _state_for_cost(next_s, config.nx)
            return next_s, next_s_trimmed

        ts = jnp.arange(config.horizon)
        _, trajectory = jax.lax.scan(
            step_fn, state, (ts, smppi_state.action_sequence)
        )

        # Prepend initial state
        initial_state = _state_for_cost(state, config.nx)
        full_trajectory = jnp.concatenate(
            [initial_state[None, :], trajectory], axis=0
        )

        return state, full_trajectory

    # Handle batched or single state
    if current_obs.ndim == 1:
        obs_batch = current_obs[None, :]
    else:
        obs_batch = current_obs

    # Generate multiple rollouts
    _, rollouts = jax.lax.scan(
        single_rollout, obs_batch[0], jnp.arange(num_rollouts)
    )

    return rollouts
