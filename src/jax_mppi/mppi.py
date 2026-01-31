# ruff: noqa: F722, F821
"""Core MPPI implementation in JAX.

This module provides the main MPPI algorithm logic, including:
- State definition (MPPIState)
- Configuration (MPPIConfig)
- Initialization
- Rollout generation
- Control update
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float

from .types import CostFn, DynamicsFn


@dataclass(frozen=True)
class MPPIConfig:
    """Configuration for MPPI controller.

    Attributes:
        dynamics_fn: Function to step dynamics forward (x, u) -> x_next
        cost_fn: Function to compute step cost (x, u) -> cost
        num_samples: Number of trajectories to sample per step
        horizon: Planning horizon length
        lambda_: Temperature parameter (inverse sensitivity)
        noise_sigma: Exploration noise covariance (vector or matrix)
        u_min: Minimum control value (for clipping)
        u_max: Maximum control value (for clipping)
        u_init: Initial guess for control (default: 0)
        step_method: 'mppi', 'smppi', or 'kmppi' (internal flag)
        num_support_pts: For KMPPI, number of support points
        discount: Discount factor for cost (gamma)
        rollout_samples: Number of rollouts per sample for stochastic
            dynamics (default: 1)
    """

    dynamics_fn: DynamicsFn
    cost_fn: CostFn
    num_samples: int
    horizon: int
    lambda_: float
    noise_sigma: Float[Array, "nu nu"]
    u_min: Float[Array, "nu"]
    u_max: Float[Array, "nu"]
    u_init: Float[Array, "nu"]
    nx: int
    nu: int
    step_method: str = "mppi"
    num_support_pts: int = 0  # For KMPPI
    discount: float = 1.0
    rollout_samples: int = 1  # For stochastic dynamics


@register_pytree_node_class
class MPPIState:
    """State of the MPPI controller.

    Contains the current control distribution parameters and other
    internal state variables.

    Attributes:
        U: Current mean control trajectory (horizon, nu)
        key: JAX PRNG key
        step: Current time step
        # Optional fields for variants:
        action_sequence: For SMPPI (horizon, nu)
        theta: For KMPPI (num_support_pts, nu)
        Tk: For KMPPI (num_support_pts,)
        Hs: For KMPPI (horizon,)
        noise_mu: Mean of noise distribution (nu,)
        noise_sigma: Covariance of noise (nu, nu)
        noise_sigma_inv: Inverse covariance (nu, nu)
    """

    def __init__(
        self,
        U: Float[Array, "H nu"],
        key: Array,
        step: int,
        noise_mu: Optional[Float[Array, "nu"]] = None,
        noise_sigma: Optional[Float[Array, "nu nu"]] = None,
        noise_sigma_inv: Optional[Float[Array, "nu nu"]] = None,
        u_init: Optional[Float[Array, "nu"]] = None,
        # Variant specific
        action_sequence: Optional[Float[Array, "H nu"]] = None,
        theta: Optional[Float[Array, "K nu"]] = None,
        Tk: Optional[Float[Array, "K"]] = None,
        Hs: Optional[Float[Array, "H"]] = None,
    ):
        self.U = U
        self.key = key
        self.step = step
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.noise_sigma_inv = noise_sigma_inv
        self.u_init = u_init
        self.action_sequence = action_sequence
        self.theta = theta
        self.Tk = Tk
        self.Hs = Hs

    def tree_flatten(self):
        children = (
            self.U,
            self.key,
            self.noise_mu,
            self.noise_sigma,
            self.noise_sigma_inv,
            self.u_init,
            self.action_sequence,
            self.theta,
            self.Tk,
            self.Hs,
        )
        aux_data = (self.step,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            U=children[0],
            key=children[1],
            step=aux_data[0],
            noise_mu=children[2],
            noise_sigma=children[3],
            noise_sigma_inv=children[4],
            u_init=children[5],
            action_sequence=children[6],
            theta=children[7],
            Tk=children[8],
            Hs=children[9],
        )


def create(
    config: MPPIConfig,
    seed: int = 0,
) -> Tuple[MPPIConfig, MPPIState]:
    """Initialize MPPI controller state.

    Args:
        config: MPPI configuration
        seed: Random seed

    Returns:
        Initial config and state
    """
    key = jax.random.PRNGKey(seed)

    # Precompute noise matrices
    noise_sigma = config.noise_sigma
    noise_sigma_inv = jnp.linalg.inv(noise_sigma)
    noise_mu = jnp.zeros(config.nu)

    # Initialize control trajectory with u_init
    U = jnp.tile(config.u_init, (config.horizon, 1))

    state = MPPIState(
        U=U,
        key=key,
        step=0,
        noise_mu=noise_mu,
        noise_sigma=noise_sigma,
        noise_sigma_inv=noise_sigma_inv,
        u_init=config.u_init,
    )

    return config, state


def rollout(
    config: MPPIConfig,
    state: MPPIState,
    x0: Float[Array, "nx"],
    noise: Float[Array, "K H nu"],
) -> Float[Array, "K"]:
    """Perform trajectory rollouts and compute costs.

    Args:
        config: MPPI config
        state: MPPI state
        x0: Initial state
        noise: Sampled noise perturbations (K, H, nu)

    Returns:
        Trajectory costs (K,)
    """
    # Perturb controls: U_k = U + noise_k
    # Note: noise is already scaled by sigma in sampling step if needed,
    # but here we assume 'noise' is epsilon ~ N(0, Sigma) or similar.
    # Actually, standard MPPI samples epsilon ~ N(0, Sigma) and adds it.
    # Let's assume 'noise' passed in is epsilon.

    # U shape: (H, nu)
    # noise shape: (K, H, nu)
    # U_perturbed: (K, H, nu)
    U_perturbed = state.U + noise

    # Clip controls
    U_perturbed = jnp.clip(
        U_perturbed,
        config.u_min,
        config.u_max,
    )

    def scan_fn(carry, t):
        x = carry
        u = U_perturbed[:, t, :]  # (K, nu)

        # Step dynamics (vmap over samples)
        # We need to vmap dynamics_fn and cost_fn over the K samples
        # x is (K, nx)
        x_next = jax.vmap(config.dynamics_fn)(x, u)
        c = jax.vmap(config.cost_fn)(x, u)

        return x_next, c

    # Initial state repeated for all samples
    x_init = jnp.tile(x0, (config.num_samples, 1))

    # Run rollout
    _, costs = jax.lax.scan(scan_fn, x_init, jnp.arange(config.horizon))

    # Sum costs over horizon (with optional discount)
    if config.discount < 1.0:
        discounts = config.discount ** jnp.arange(config.horizon)
        total_costs = jnp.sum(costs * discounts[:, None], axis=0)
    else:
        total_costs = jnp.sum(costs, axis=0)

    # Add terminal cost if defined? (Not in minimal config yet)

    # Add control cost: lambda * u^T Sigma^-1 noise
    # This is part of the MPPI derivation (importance sampling weight)
    # The term is usually: gamma * u_nominal^T Sigma^-1 epsilon
    # But often simplified or included in the cost function.
    # In standard MPPI:
    # S(tau) = cost(tau) + sum(0.5 * u^T Sigma^-1 u) ?
    # Let's stick to the cost_fn provided by user plus the control cost term
    # required for the update law if not implicit.
    # The weight w = exp(-1/lambda * (S - rho))
    # where S includes the task cost + control cost.

    # Control effort cost for MPPI update:
    # The cost function usually includes a term for control effort.
    # If not, we might need to add it here.
    # Standard MPPI usually formulates minimizing:
    # J = phi(x_T) + sum(q(x) + 0.5 * u^T R u)
    # And the update uses the total cost.

    # For now, return the computed path costs.
    # User-provided cost_fn is assumed to cover everything.

    return total_costs


def step(
    config: MPPIConfig,
    state: MPPIState,
    x0: Float[Array, "nx"],
) -> Tuple[MPPIState, Float[Array, "nu"], dict]:
    """Execute one MPPI optimization step.

    Args:
        config: MPPI configuration
        state: Current MPPI state
        x0: Current system state

    Returns:
        Updated state, optimal control u0, and info dict
    """
    # 1. Sample noise
    # epsilon ~ N(0, Sigma)
    # Shape: (K, H, nu)
    key, subkey = jax.random.split(state.key)
    noise = jax.random.multivariate_normal(
        subkey,
        state.noise_mu,  # type: ignore
        state.noise_sigma,  # type: ignore
        shape=(config.num_samples, config.horizon),
    )

    # 2. Rollout and evaluate costs
    # costs: (K,)
    costs = rollout(config, state, x0, noise)

    # 3. Compute weights
    # w ~ exp(-1/lambda * (S - min(S)))
    beta = jnp.min(costs)
    weights = jnp.exp(-(1.0 / config.lambda_) * (costs - beta))
    weights = weights / (jnp.sum(weights) + 1e-10)  # Normalize

    # 4. Update control trajectory
    # U_new = U + sum(w * epsilon)
    # noise: (K, H, nu)
    # weights: (K,)
    # delta: (H, nu)
    weighted_noise = jnp.sum(weights[:, None, None] * noise, axis=0)

    # Apply smoothing or other update logic depending on method
    if config.step_method == "kmppi":
        # KMPPI update logic handled externally or here
        # For now, standard MPPI update
        U_new = state.U + weighted_noise
    elif config.step_method == "smppi":
        # SMPPI update
        U_new = state.U + weighted_noise
    else:
        # Standard MPPI
        U_new = state.U + weighted_noise

    # Clip controls
    U_new = jnp.clip(U_new, config.u_min, config.u_max)

    # 5. Shift trajectory (receding horizon)
    # u0 = U_new[0]
    # U_shifted = [U_new[1:], u_init]
    u_optimal = U_new[0]
    U_shifted = jnp.roll(U_new, -1, axis=0)
    U_shifted = U_shifted.at[-1].set(state.u_init)  # type: ignore

    # Update state
    new_state = MPPIState(
        U=U_shifted,
        key=key,
        step=state.step + 1,
        noise_mu=state.noise_mu,
        noise_sigma=state.noise_sigma,
        noise_sigma_inv=state.noise_sigma_inv,
        u_init=state.u_init,
        # Preserve other fields
        action_sequence=state.action_sequence,
        theta=state.theta,
        Tk=state.Tk,
        Hs=state.Hs,
    )

    info = {
        "costs": costs,
        "weights": weights,
        "best_cost": beta,
    }

    return new_state, u_optimal, info
