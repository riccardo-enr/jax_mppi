from dataclasses import replace
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_mppi.mppi import (
    MPPIConfig,
    MPPIState,
    _bound_action,
    _compute_noise_cost,
    _compute_rollout_costs,
    _compute_weights,
    _scaled_bounds,
    _shift_nominal,
)


def biased_mppi_command(
    config: MPPIConfig,
    mppi_state: MPPIState,
    current_obs: jax.Array,
    dynamics: Callable,
    running_cost: Callable,
    U_ref: jax.Array,  # The reference trajectory to bias towards (T, nu)
    bias_alpha: float = 0.5,  # Mixture weight for biased samples
    terminal_cost: Optional[Callable] = None,
    shift: bool = True,
) -> Tuple[jax.Array, MPPIState]:
    """Biased MPPI command with mixture sampling."""

    key, subkey1, subkey2 = jax.random.split(mppi_state.key, 3)

    K = config.num_samples
    K_biased = int(K * bias_alpha)
    K_nominal = K - K_biased

    # Sample nominal noise
    noise_nominal = jax.random.multivariate_normal(
        subkey1,
        mean=mppi_state.noise_mu,
        cov=mppi_state.noise_sigma,
        shape=(K_nominal, config.horizon),
    )

    # Sample biased noise
    noise_biased_base = jax.random.multivariate_normal(
        subkey2,
        mean=mppi_state.noise_mu,
        cov=mppi_state.noise_sigma,
        shape=(K_biased, config.horizon),
    )

    # Difference between reference and current nominal
    delta_ref = U_ref - mppi_state.U
    noise_biased = noise_biased_base + delta_ref[None, :, :]

    # Combine noise
    noise = jnp.concatenate([noise_nominal, noise_biased], axis=0)

    perturbed_actions = mppi_state.U[None, :, :] + noise
    scaled_actions = perturbed_actions * config.u_scale
    scaled_actions = _bound_action(
        scaled_actions, mppi_state.u_min, mppi_state.u_max
    )

    # Compute Costs
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

    # Update U
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
