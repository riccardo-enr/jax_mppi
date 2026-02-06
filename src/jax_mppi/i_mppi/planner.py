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
from jax_mppi.smppi import (
    SMPPIConfig,
    SMPPIState,
    _compute_perturbed_actions_and_noise,
    _compute_smoothness_cost,
    _sample_noise as _sample_noise_smppi,
    _shift_nominal as _shift_nominal_smppi,
)
from jax_mppi.smppi import (
    _compute_rollout_costs as _compute_rollout_costs_smppi,
)
from jax_mppi.smppi import (
    _compute_noise_cost as _compute_noise_cost_smppi,
)
from jax_mppi.smppi import (
    _compute_weights as _compute_weights_smppi,
)
from jax_mppi.kmppi import (
    KMPPIConfig,
    KMPPIState,
    TimeKernel,
    _kernel_interpolate,
    _sample_noise as _sample_noise_kmppi,
    _shift_control_points,
)
from jax_mppi.kmppi import (
    _bound_action as _bound_action_kmppi,
)
from jax_mppi.kmppi import (
    _compute_rollout_costs as _compute_rollout_costs_kmppi,
)
from jax_mppi.kmppi import (
    _compute_noise_cost as _compute_noise_cost_kmppi,
)
from jax_mppi.kmppi import (
    _compute_weights as _compute_weights_kmppi,
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


def biased_smppi_command(
    config: SMPPIConfig,
    smppi_state: SMPPIState,
    current_obs: jax.Array,
    dynamics: Callable,
    running_cost: Callable,
    U_ref: jax.Array,  # The reference action trajectory to bias towards (T, nu)
    bias_alpha: float = 0.5,  # Mixture weight for biased samples
    terminal_cost: Optional[Callable] = None,
    shift: bool = True,
) -> Tuple[jax.Array, SMPPIState]:
    """Biased SMPPI command with mixture sampling.

    Similar to biased_mppi_command but for SMPPI (Smooth MPPI).
    SMPPI operates in a lifted control space with smoothness penalties.
    """
    key, subkey1, subkey2 = jax.random.split(smppi_state.key, 3)

    K = config.num_samples
    K_biased = int(K * bias_alpha)
    K_nominal = K - K_biased

    # Sample nominal noise in velocity space
    noise_nominal, _ = _sample_noise_smppi(
        subkey1,
        K_nominal,
        config.horizon,
        smppi_state.noise_mu,
        smppi_state.noise_sigma,
        sample_null_action=False,
    )

    # Sample biased noise
    noise_biased_base, _ = _sample_noise_smppi(
        subkey2,
        K_biased,
        config.horizon,
        smppi_state.noise_mu,
        smppi_state.noise_sigma,
        sample_null_action=False,
    )

    # Compute reference velocity from action difference
    # delta_ref is the difference in actions needed to match reference
    delta_ref_action = U_ref - smppi_state.action_sequence
    # Convert to velocity space
    delta_ref_velocity = delta_ref_action / config.delta_t
    noise_biased = noise_biased_base + delta_ref_velocity[None, :, :]

    # Combine noise
    noise = jnp.concatenate([noise_nominal, noise_biased], axis=0)

    # Compute perturbed actions and effective noise
    perturbed_actions, effective_noise = _compute_perturbed_actions_and_noise(
        config, smppi_state, noise
    )

    # Compute rollout costs
    rollout_costs = _compute_rollout_costs_smppi(
        config,
        current_obs,
        perturbed_actions,
        dynamics,
        running_cost,
        terminal_cost,
    )

    # Compute noise cost (in velocity space)
    noise_costs = _compute_noise_cost_smppi(
        effective_noise,
        smppi_state.noise_sigma_inv,
        config.noise_abs_cost,
    )

    # Compute smoothness cost
    smoothness_costs = _compute_smoothness_cost(perturbed_actions, config)

    # Total cost
    total_costs = rollout_costs + noise_costs + smoothness_costs

    # Compute weights
    weights = _compute_weights_smppi(total_costs, config.lambda_)

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
        key=key,
    )

    # Shift nominal trajectory if requested
    if shift:
        new_state = _shift_nominal_smppi(
            new_state, shift_steps=config.u_per_command
        )

    # Extract action to return
    if config.u_per_command == 1:
        action = new_action_sequence[0] * config.u_scale
    else:
        action = (
            new_action_sequence[: config.u_per_command].reshape(-1)
            * config.u_scale
        )

    return action, new_state


def biased_kmppi_command(
    config: KMPPIConfig,
    kmppi_state: KMPPIState,
    current_obs: jax.Array,
    dynamics: Callable,
    running_cost: Callable,
    kernel_fn: TimeKernel,
    U_ref: jax.Array,  # The reference trajectory to bias towards (T, nu)
    bias_alpha: float = 0.5,  # Mixture weight for biased samples
    terminal_cost: Optional[Callable] = None,
    shift: bool = True,
) -> Tuple[jax.Array, KMPPIState]:
    """Biased KMPPI command with mixture sampling.

    Similar to biased_mppi_command but for KMPPI (Kernel MPPI).
    KMPPI uses kernel interpolation over control points.
    """
    key, subkey1, subkey2 = jax.random.split(kmppi_state.key, 3)

    K = config.num_samples
    K_biased = int(K * bias_alpha)
    K_nominal = K - K_biased

    # Sample nominal noise in control point space
    noise_nominal, _ = _sample_noise_kmppi(
        subkey1,
        K_nominal,
        config.num_support_pts,
        kmppi_state.noise_mu,
        kmppi_state.noise_sigma,
        sample_null_action=False,
    )

    # Sample biased noise
    noise_biased_base, _ = _sample_noise_kmppi(
        subkey2,
        K_biased,
        config.num_support_pts,
        kmppi_state.noise_mu,
        kmppi_state.noise_sigma,
        sample_null_action=False,
    )

    # For KMPPI, we need to project the reference trajectory to control point space
    # Compute theta_ref (control points for reference trajectory)
    # We can use the kernel interpolation matrix to solve for control points
    # that best approximate U_ref
    K_matrix = kernel_fn(kmppi_state.Hs, kmppi_state.Tk)  # (T, num_support_pts)
    Ktktk = kernel_fn(
        kmppi_state.Tk, kmppi_state.Tk
    )  # (num_support_pts, num_support_pts)

    # Solve for theta_ref: minimize ||K @ theta_ref - U_ref||^2
    # Solution: theta_ref = (K^T K)^-1 K^T U_ref
    KtK = K_matrix.T @ K_matrix
    KtU_ref = K_matrix.T @ U_ref
    theta_ref = jax.scipy.linalg.solve(
        KtK + 1e-6 * jnp.eye(config.num_support_pts), KtU_ref, assume_a="pos"
    )

    # Difference between reference and current control points
    delta_theta_ref = theta_ref - kmppi_state.theta
    noise_biased = noise_biased_base + delta_theta_ref[None, :, :]

    # Combine noise
    noise_theta = jnp.concatenate([noise_nominal, noise_biased], axis=0)

    # Perturb control points
    perturbed_theta = kmppi_state.theta[None, :, :] + noise_theta
    perturbed_theta = _bound_action_kmppi(
        perturbed_theta, kmppi_state.u_min, kmppi_state.u_max
    )

    # Effective noise after bounding
    effective_noise_theta = perturbed_theta - kmppi_state.theta[None, :, :]

    # Interpolate perturbed control points to full trajectories
    def interpolate_single(theta_single):
        U_interp, _ = _kernel_interpolate(
            kmppi_state.Hs, kmppi_state.Tk, theta_single, kernel_fn
        )
        return U_interp

    perturbed_actions = jax.vmap(interpolate_single)(
        perturbed_theta
    )  # (K, T, nu)
    perturbed_actions = _bound_action_kmppi(
        perturbed_actions, kmppi_state.u_min, kmppi_state.u_max
    )

    # Compute rollout costs
    rollout_costs = _compute_rollout_costs_kmppi(
        config,
        current_obs,
        perturbed_actions,
        dynamics,
        running_cost,
        terminal_cost,
    )

    # Compute noise cost (in control point space)
    noise_costs = _compute_noise_cost_kmppi(
        effective_noise_theta,
        kmppi_state.noise_sigma_inv,
        config.noise_abs_cost,
    )

    # Total cost
    total_costs = rollout_costs + noise_costs

    # Compute importance weights
    weights = _compute_weights_kmppi(total_costs, config.lambda_)

    # Update control points (optimization in control point space)
    delta_theta = jnp.sum(
        weights[:, None, None] * effective_noise_theta, axis=0
    )
    new_theta = kmppi_state.theta + delta_theta

    # Interpolate updated control points to get full trajectory
    new_U, _ = _kernel_interpolate(
        kmppi_state.Hs, kmppi_state.Tk, new_theta, kernel_fn
    )

    # Update state
    new_state = replace(
        kmppi_state,
        U=new_U,
        theta=new_theta,
        key=key,
    )

    # Shift nominal trajectory if requested
    if shift:
        shifted_theta = _shift_control_points(
            new_state.theta,
            new_state.Tk,
            new_state.u_init,
            config.u_per_command,
            kernel_fn,
        )
        # Also shift U (via interpolation)
        shifted_U, _ = _kernel_interpolate(
            new_state.Hs, new_state.Tk, shifted_theta, kernel_fn
        )
        new_state = replace(new_state, U=shifted_U, theta=shifted_theta)

    # Extract action to return
    if config.u_per_command == 1:
        action = new_state.U[0] * config.u_scale
    else:
        action = (
            new_state.U[: config.u_per_command].reshape(-1) * config.u_scale
        )

    return action, new_state
