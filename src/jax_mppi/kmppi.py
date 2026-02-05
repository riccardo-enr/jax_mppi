"""Kernel MPPI (KMPPI) implementation in JAX.

KMPPI uses kernel interpolation to smooth control trajectories by working with
a reduced set of control points (theta) rather than the full trajectory. This
allows smoother actions with fewer parameters to optimize.

Reference: Based on pytorch_mppi KMPPI implementation
"""

from dataclasses import dataclass, replace
from typing import Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .types import DynamicsFn, RunningCostFn, TerminalCostFn


class TimeKernel(Protocol):
    """Protocol for time-domain kernels used in trajectory interpolation."""

    def __call__(self, t: jax.Array, tk: jax.Array) -> jax.Array:
        """Evaluate kernel between time points.

        Args:
            t: Query time points, shape (T,) or (T, 1)
            tk: Control point times, shape (num_support_pts,) or (num_support_pts, 1)

        Returns:
            K: Kernel matrix, shape (T, num_support_pts)
        """
        ...


class RBFKernel:
    """Radial Basis Function kernel for time-domain interpolation."""

    def __init__(self, sigma: float = 1.0):
        """Initialize RBF kernel.

        Args:
            sigma: Bandwidth parameter (controls kernel width)
        """
        self.sigma = sigma

    def __call__(self, t: jax.Array, tk: jax.Array) -> jax.Array:
        """Evaluate RBF kernel: k(t, tk) = exp(-||t - tk||^2 / (2*sigma^2))

        Args:
            t: Query times, shape (T,) or (T, 1)
            tk: Control point times, shape (num_support_pts,) or (num_support_pts, 1)

        Returns:
            K: kernel matrix, shape (T, num_support_pts)
        """
        # Ensure proper shapes for broadcasting
        if t.ndim == 1:
            t = t[:, None]  # (T, 1)
        if tk.ndim == 1:
            tk = tk[None, :]  # (1, num_support_pts)

        # Squared Euclidean distance in 1D time space
        # t[:, None] - tk creates (T, num_support_pts) difference matrix
        d = (t - tk) ** 2

        # RBF formula
        k = jnp.exp(-d / (2 * self.sigma**2 + 1e-8))

        return k


@dataclass(frozen=True)
class KMPPIConfig:
    """Configuration for Kernel MPPI."""

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

    # KMPPI-specific parameters
    num_support_pts: int  # Number of control points for interpolation


@register_pytree_node_class
@dataclass
class KMPPIState:
    """State for Kernel MPPI."""

    # Base parameters
    U: jax.Array  # (T, nu) full trajectory (interpolated from theta)
    u_init: jax.Array  # (nu,) default action for shift
    noise_mu: jax.Array  # (nu,)
    noise_sigma: jax.Array  # (nu, nu)
    noise_sigma_inv: jax.Array
    u_min: Optional[jax.Array]
    u_max: Optional[jax.Array]
    key: jax.Array  # PRNG key

    # KMPPI-specific state
    theta: jax.Array  # (num_support_pts, nu) control points
    Tk: jax.Array  # (num_support_pts,) control point times
    Hs: jax.Array  # (T,) full trajectory times

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
                self.theta,
                self.Tk,
                self.Hs,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _bound_action(
    action: jax.Array, u_min: Optional[jax.Array], u_max: Optional[jax.Array]
) -> jax.Array:
    """Bound action values."""
    if u_min is None and u_max is None:
        return action
    if u_min is None:
        assert u_max is not None
        return jnp.minimum(action, u_max)
    if u_max is None:
        return jnp.maximum(action, u_min)
    return jnp.clip(action, u_min, u_max)


def _scaled_bounds(
    bounds: Optional[jax.Array],
    u_scale: float,
) -> Optional[jax.Array]:
    """Scale bounds by u_scale factor."""
    if bounds is None:
        return None
    return bounds / u_scale


def _kernel_interpolate(
    t: jax.Array,
    tk: jax.Array,
    control_points: jax.Array,
    kernel_fn: TimeKernel,
) -> Tuple[jax.Array, jax.Array]:
    """Interpolate control points to full trajectory using kernel.

    Args:
        t: Query times, shape (T,)
        tk: Control point times, shape (num_support_pts,)
        control_points: Control point values, shape (num_support_pts, nu)
        kernel_fn: Kernel function

    Returns:
        interpolated: Interpolated trajectory, shape (T, nu)
        K: Interpolation matrix, shape (T, num_support_pts)
    """
    # Compute kernel matrices
    K = kernel_fn(t, tk)  # (T, num_support_pts)
    Ktktk = kernel_fn(tk, tk)  # (num_support_pts, num_support_pts)

    # Solve: Ktktk @ weights.T = K.T for weights
    # This gives weights = K @ inv(Ktktk)
    weights = jax.scipy.linalg.solve(
        Ktktk, K.T, assume_a="pos"
    ).T  # (T, num_support_pts)

    # Interpolate: U(t) = weights @ control_points
    interpolated = weights @ control_points  # (T, nu)

    return interpolated, K


def _sample_noise(
    key: jax.Array,
    K: int,
    num_support_pts: int,
    noise_mu: jax.Array,
    noise_sigma: jax.Array,
    sample_null_action: bool,
) -> Tuple[jax.Array, jax.Array]:
    """Sample noise in control point space.

    Args:
        key: PRNG key
        K: Number of samples
        num_support_pts: Number of control points
        noise_mu: Noise mean (nu,)
        noise_sigma: Noise covariance (nu, nu)
        sample_null_action: Include null action in samples

    Returns:
        noise: Noise samples, shape (K, num_support_pts, nu)
        new_key: Updated PRNG key
    """
    if sample_null_action:
        # Reserve first sample for null action
        key, subkey = jax.random.split(key)
        noise = jax.random.multivariate_normal(
            subkey, noise_mu, noise_sigma, shape=(K - 1, num_support_pts)
        )
        null_noise = jnp.zeros((1, num_support_pts, noise_mu.shape[0]))
        noise = jnp.concatenate([null_noise, noise], axis=0)
    else:
        key, subkey = jax.random.split(key)
        noise = jax.random.multivariate_normal(
            subkey, noise_mu, noise_sigma, shape=(K, num_support_pts)
        )

    return noise, key


def _shift_control_points(
    theta: jax.Array,
    Tk: jax.Array,
    u_init: jax.Array,
    shift_steps: int,
    kernel_fn: TimeKernel,
) -> jax.Array:
    """Shift control points forward in time via interpolation.

    Args:
        theta: Current control points, shape (num_support_pts, nu)
        Tk: Current control point times, shape (num_support_pts,)
        u_init: Default action for extrapolation
        shift_steps: Number of steps to shift
        kernel_fn: Kernel function

    Returns:
        shifted_theta: Shifted control points, shape (num_support_pts, nu)
    """
    # Shift time grid forward
    Tk_shifted = Tk + shift_steps

    # Interpolate theta values at the new time points
    shifted_theta, _ = _kernel_interpolate(Tk_shifted, Tk, theta, kernel_fn)

    return shifted_theta


# Import these from base mppi module
from .mppi import (
    _call_dynamics,
    _call_running_cost,
    _state_for_cost,
)


def _single_rollout_costs(
    config: KMPPIConfig,
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
    config: KMPPIConfig,
    current_obs: jax.Array,
    actions: jax.Array,
    dynamics: DynamicsFn,
    running_cost: RunningCostFn,
    terminal_cost: Optional[TerminalCostFn],
) -> jax.Array:
    """Compute rollout costs for all perturbed action sequences."""
    per_step_costs, terminal_costs = jax.vmap(
        lambda a: _single_rollout_costs(
            config, current_obs, a, dynamics, running_cost, terminal_cost
        )
    )(actions)

    running_costs_sum = jnp.sum(per_step_costs, axis=1)
    return running_costs_sum + terminal_costs


def _compute_noise_cost(
    noise: jax.Array,
    noise_sigma_inv: jax.Array,
    noise_abs_cost: bool,
) -> jax.Array:
    """Compute cost of noise in control point space."""
    if noise_abs_cost:
        return jnp.sum(jnp.abs(noise), axis=(1, 2))
    else:
        # Quadratic cost in control point space
        # Optimized: use dot product instead of nested vmap
        term = jnp.dot(noise, noise_sigma_inv)
        costs_per_point = jnp.sum(term * noise, axis=-1)
        return jnp.sum(costs_per_point, axis=1)


def _compute_weights(costs: jax.Array, lambda_: float) -> jax.Array:
    """Compute importance weights from costs using softmax."""
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
    num_support_pts: Optional[int] = None,
    kernel: Optional[TimeKernel] = None,
    u_scale: float = 1.0,
    u_per_command: int = 1,
    step_dependent_dynamics: bool = False,
    rollout_samples: int = 1,
    rollout_var_cost: float = 0.0,
    rollout_var_discount: float = 0.95,
    sample_null_action: bool = False,
    noise_abs_cost: bool = False,
    key: Optional[jax.Array] = None,
) -> Tuple[KMPPIConfig, KMPPIState, TimeKernel]:
    """Create KMPPI configuration, state, and kernel.

    Args:
        nx: State dimension
        nu: Action dimension
        noise_sigma: (nu, nu) noise covariance matrix
        num_samples: Number of MPPI samples (K)
        horizon: Planning horizon (T)
        lambda_: Temperature parameter for importance weighting
        noise_mu: (nu,) noise mean (default: zeros)
        u_min: (nu,) lower bounds on actions
        u_max: (nu,) upper bounds on actions
        u_init: (nu,) default action for shift (default: zeros)
        U_init: (T, nu) initial trajectory (default: zeros)
        num_support_pts: Number of control points (default: horizon // 2)
        kernel: TimeKernel instance (default: RBFKernel(sigma=1.0))
        u_scale: Scale factor for control
        u_per_command: Number of control steps per command
        step_dependent_dynamics: Whether dynamics depend on timestep
        rollout_samples: Number of rollout samples for stochastic dynamics
        rollout_var_cost: Variance cost weight
        rollout_var_discount: Discount factor for variance cost
        sample_null_action: Whether to include null action in samples
        noise_abs_cost: Use absolute value cost for noise
        key: PRNG key (default: create new)

    Returns:
        config: KMPPI configuration
        state: KMPPI initial state
        kernel_fn: Kernel function instance
    """
    # Initialize defaults
    if noise_mu is None:
        noise_mu = jnp.zeros(nu)
    if u_init is None:
        u_init = jnp.zeros(nu)
    if key is None:
        key = jax.random.PRNGKey(0)
    if num_support_pts is None:
        num_support_pts = max(horizon // 2, 2)  # At least 2 support points
    if kernel is None:
        kernel = RBFKernel(sigma=1.0)

    # Scale bounds
    u_min_scaled = _scaled_bounds(u_min, u_scale)
    u_max_scaled = _scaled_bounds(u_max, u_scale)

    # Initialize control points (theta) and time grids
    if U_init is None:
        theta = jnp.zeros((num_support_pts, nu))
    else:
        # Sample initial control points from U_init
        indices = jnp.linspace(0, horizon - 1, num_support_pts, dtype=jnp.int32)
        theta = U_init[indices]

    # Time grids
    Tk = jnp.linspace(0, horizon - 1, num_support_pts)  # Control point times
    Hs = jnp.linspace(0, horizon - 1, horizon)  # Full trajectory times

    # Interpolate theta to get full trajectory U
    u_control, _ = _kernel_interpolate(Hs, Tk, theta, kernel)

    # Compute noise covariance inverse
    noise_sigma_inv = jnp.linalg.inv(noise_sigma)

    # Create config
    config = KMPPIConfig(
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
        num_support_pts=num_support_pts,
    )

    # Create state
    state = KMPPIState(
        U=u_control,
        u_init=u_init,
        noise_mu=noise_mu,
        noise_sigma=noise_sigma,
        noise_sigma_inv=noise_sigma_inv,
        u_min=u_min_scaled,
        u_max=u_max_scaled,
        key=key,
        theta=theta,
        Tk=Tk,
        Hs=Hs,
    )

    return config, state, kernel


def command(
    config: KMPPIConfig,
    kmppi_state: KMPPIState,
    current_obs: jax.Array,
    dynamics: DynamicsFn,
    running_cost: RunningCostFn,
    kernel_fn: TimeKernel,
    terminal_cost: Optional[TerminalCostFn] = None,
    shift: bool = True,
) -> Tuple[jax.Array, KMPPIState]:
    """Compute optimal action using Kernel MPPI.

    Args:
        config: KMPPI configuration
        kmppi_state: Current KMPPI state
        current_obs: (nx,) current observation/state
        dynamics: Dynamics function
        running_cost: Running cost function
        kernel_fn: Kernel function for interpolation
        terminal_cost: Optional terminal cost function
        shift: Whether to shift nominal trajectory after computing action

    Returns:
        action: (u_per_command * nu,) or (nu,) optimal action
        new_state: Updated KMPPI state
    """
    # Sample noise in control point space
    noise_theta, new_key = _sample_noise(
        kmppi_state.key,
        config.num_samples,
        config.num_support_pts,
        kmppi_state.noise_mu,
        kmppi_state.noise_sigma,
        config.sample_null_action,
    )

    # Perturb control points
    perturbed_theta = (
        kmppi_state.theta[None, :, :] + noise_theta
    )  # (K, num_support_pts, nu)
    perturbed_theta = _bound_action(
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
    perturbed_actions = _bound_action(
        perturbed_actions, kmppi_state.u_min, kmppi_state.u_max
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

    # Compute noise cost (in control point space)
    noise_costs = _compute_noise_cost(
        effective_noise_theta,
        kmppi_state.noise_sigma_inv,
        config.noise_abs_cost,
    )

    # Total cost
    total_costs = rollout_costs + noise_costs

    # Compute importance weights
    weights = _compute_weights(total_costs, config.lambda_)

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
        key=new_key,
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


def reset(
    config: KMPPIConfig,
    kmppi_state: KMPPIState,
    kernel_fn: TimeKernel,
    key: jax.Array,
) -> KMPPIState:
    """Reset KMPPI state with new random key."""
    # Reset control points to zeros
    theta_reset = jnp.zeros_like(kmppi_state.theta)

    # Interpolate to get U
    U_reset, _ = _kernel_interpolate(
        kmppi_state.Hs, kmppi_state.Tk, theta_reset, kernel_fn
    )

    return replace(
        kmppi_state,
        U=U_reset,
        theta=theta_reset,
        key=key,
    )


def get_rollouts(
    config: KMPPIConfig,
    kmppi_state: KMPPIState,
    current_obs: jax.Array,
    dynamics: DynamicsFn,
    num_rollouts: int = 1,
) -> jax.Array:
    """Generate rollout trajectories using current control sequence.

    Args:
        config: KMPPI configuration
        kmppi_state: Current KMPPI state
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
        _, trajectory = jax.lax.scan(step_fn, state, (ts, kmppi_state.U))

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
