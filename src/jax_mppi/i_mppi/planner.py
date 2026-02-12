from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jax_mppi import mppi
from jax_mppi.kmppi import KMPPIState, TimeKernel
from jax_mppi.mppi import MPPIConfig, MPPIState, _sample_noise


def interpolate_trajectory(
    kmppi_state: KMPPIState,
    kernel_fn: TimeKernel,
) -> Float[Array, "T nu"]:
    """Interpolate the full trajectory from control points (theta)."""
    # In KMPPI implementation (kmppi.py), interpolation is U = K @ inv(K_tk_tk) @ theta
    # But usually this is pre-solved or cached.
    # If we assume kmppi_state.U is already the interpolated trajectory:
    return kmppi_state.U


def fit_trajectory_to_control_points(
    trajectory: Float[Array, "T nu"],
    num_support_pts: int,
    horizon: int,
    kernel_fn: TimeKernel,
    reg: float = 1e-6,
) -> Float[Array, "M nu"]:
    """Fit optimal control points theta to approximate a target trajectory.

    Solves: theta* = argmin || U_target - W * theta ||^2
    Solution: theta* = (W^T W + reg*I)^-1 W^T U_target
    where W is the interpolation matrix.
    """
    Tk = jnp.linspace(0, horizon - 1, num_support_pts)
    Hs = jnp.linspace(0, horizon - 1, horizon)

    # W = K_t_tk @ inv(K_tk_tk)
    K_t_tk = kernel_fn(Hs, Tk)  # (T, M)
    K_tk_tk = kernel_fn(Tk, Tk)  # (M, M)

    # Solve for weights matrix: W = (K_tk_tk.T \ K_t_tk.T).T
    # Or just use the pre-computed weights logic from kmppi
    # For fitting, we treat W as the design matrix.
    # W = K_t_tk @ jnp.linalg.inv(K_tk_tk + reg*I)
    W = K_t_tk @ jnp.linalg.inv(K_tk_tk + jnp.eye(num_support_pts) * reg)

    # Least squares: (W^T W)^-1 W^T U
    # Using QR or solve for stability
    theta = jnp.linalg.lstsq(W, trajectory, rcond=None)[0]

    return theta


def biased_mppi_command(
    config: MPPIConfig,
    mppi_state: MPPIState,
    current_obs: jax.Array,
    dynamics: Callable,
    running_cost: Callable,
    reference_actions: jax.Array,
    bias_weight: float = 0.5,
    terminal_cost: Optional[Callable] = None,
    shift: bool = True,
) -> Tuple[jax.Array, MPPIState]:
    """Execute MPPI command step with mixture sampling (biased towards reference).

    This modifies the standard MPPI sampling to include a component biased towards
    a reference action sequence (e.g. from Layer 2 trajectory).

    Args:
        config: MPPI configuration
        mppi_state: Current MPPI state
        current_obs: Current observation
        dynamics: Dynamics function
        running_cost: Cost function
        reference_actions: (T, nu) reference action sequence
        bias_weight: Probability of sampling from reference-biased distribution
        terminal_cost: Optional terminal cost
        shift: Whether to shift the nominal trajectory

    Returns:
        action: Optimal action
        new_state: Updated MPPI state
    """
    # 1. Sample noise
    # We need to manually construct mixture noise
    key, subkey1, subkey2, subkey3 = jax.random.split(mppi_state.key, 4)

    num_biased = int(config.num_samples * bias_weight)
    num_nominal = config.num_samples - num_biased

    # Nominal samples: centered on mppi_state.U
    noise_nominal = jax.random.multivariate_normal(
        subkey1,
        mean=jnp.zeros(config.nu),
        cov=mppi_state.noise_sigma,
        shape=(num_nominal, config.horizon),
    )

    # Biased samples: centered on reference_actions
    # Noise required to get from U to Ref is (Ref - U)
    # So we sample around that offset
    bias_offset = reference_actions - mppi_state.U

    noise_biased_center = jax.random.multivariate_normal(
        subkey2,
        mean=jnp.zeros(config.nu),
        cov=mppi_state.noise_sigma,
        shape=(num_biased, config.horizon),
    )
    noise_biased = noise_biased_center + bias_offset[None, :, :]

    # Combine noise
    noise = jnp.concatenate([noise_nominal, noise_biased], axis=0)

    # Note: We need to use internal functions or copy logic because
    # mppi.command doesn't accept pre-sampled noise directly.
    # However, mppi.command calls _sample_noise.
    # To implement this cleanly without modifying core mppi.py,
    # we have to duplicate the command logic here or monkey-patch.
    # Given the constraints, duplicating the core logic (which is short) is safer.

    # --- Re-implementation of mppi.command logic with custom noise ---

    perturbed_actions = mppi_state.U[None, :, :] + noise
    scaled_actions = perturbed_actions * config.u_scale
    scaled_actions = mppi._bound_action(
        scaled_actions, mppi_state.u_min, mppi_state.u_max
    )

    rollout_costs = mppi._compute_rollout_costs(
        config,
        current_obs,
        scaled_actions,
        dynamics,
        running_cost,
        terminal_cost,
    )
    noise_costs = mppi._compute_noise_cost(
        noise, mppi_state.noise_sigma_inv, config.noise_abs_cost
    )
    total_costs = rollout_costs + noise_costs

    weights = mppi._compute_weights(total_costs, config.lambda_)
    delta_U = jnp.tensordot(weights, noise, axes=1)
    U_new = mppi_state.U + delta_U

    u_min_scaled, u_max_scaled = mppi._scaled_bounds(
        mppi_state.u_min, mppi_state.u_max, config.u_scale
    )
    U_new = mppi._bound_action(U_new, u_min_scaled, u_max_scaled)

    action_seq = U_new[: config.u_per_command]
    scaled_action_seq = mppi._bound_action(
        action_seq * config.u_scale, mppi_state.u_min, mppi_state.u_max
    )
    action = (
        scaled_action_seq[0] if config.u_per_command == 1 else scaled_action_seq
    )

    new_state = mppi.replace(mppi_state, U=U_new, key=key)
    if shift:
        new_state = mppi._shift_nominal(new_state, config.u_per_command)

    return action, new_state
