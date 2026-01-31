# ruff: noqa: F722, F821
"""Kernel-based MPPI (KMPPI) implementation.

KMPPI parameterizes the control sequence using basis functions (kernels)
rather than optimizing individual control steps independently.
This implicitly enforces smoothness.
"""

from typing import Protocol

import jax.numpy as jnp
from jaxtyping import Array, Float

from .mppi import MPPIState


class TimeKernel(Protocol):
    """Protocol for time-based kernels."""

    def __call__(
        self, t: Float[Array, "T"], tk: Float[Array, "K"]
    ) -> Float[Array, "T K"]:
        """Compute kernel matrix between query times t and support times tk.

        Args:
            t: Query time points, shape (T,) or (T, 1)
            tk: Control point times, shape (num_support_pts,) or
                (num_support_pts, 1)

        Returns:
            Kernel matrix, shape (T, num_support_pts)
        """
        ...


class RBFKernel:
    """Radial Basis Function (Gaussian) kernel."""

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(
        self, t: Float[Array, "T"], tk: Float[Array, "K"]
    ) -> Float[Array, "T K"]:
        """Compute RBF kernel.

        k(t, t') = exp(-||t - t'||^2 / (2 * sigma^2))

        Args:
            t: Query times, shape (T,) or (T, 1)
            tk: Control point times, shape (num_support_pts,) or
                (num_support_pts, 1)

        Returns:
            Kernel matrix
        """
        # Ensure correct shapes for broadcasting
        t = jnp.atleast_2d(t)
        if t.shape[0] == 1 and t.shape[1] > 1:
            t = t.T  # (T, 1)

        tk = jnp.atleast_2d(tk)
        if tk.shape[0] == 1 and tk.shape[1] > 1:
            tk = tk.T  # (K, 1)

        # Distances
        dists = jnp.abs(t - tk.T)  # (T, K)
        return jnp.exp(-(dists**2) / (2 * self.sigma**2))


def update_control(
    state: MPPIState,
    weights: Float[Array, "K"],
    noise: Float[Array, "K H nu"],
    kernel_fn: TimeKernel,
) -> MPPIState:
    """Update control parameters (theta) for KMPPI.

    Args:
        state: Current MPPI state
        weights: Sample weights (K,)
        noise: Perturbations applied (K, num_support_pts, nu)
        kernel_fn: Kernel function

    Returns:
        Updated MPPI state
    """
    # Verify we have KMPPI state components
    if not hasattr(state, "theta") or not hasattr(state, "Tk"):
        raise ValueError("State missing KMPPI components (theta, Tk)")

    # Weighted average of noise in parameter space
    # (num_support_pts, nu)
    delta_theta = jnp.sum(
        weights[:, None, None] * noise, axis=0
    )  # pyright: ignore

    # Update parameters
    new_theta = state.theta + delta_theta  # pyright: ignore

    # Re-interpolate control trajectory U from new theta
    # U = Kernel * theta
    # Kernel matrix: (horizon, num_support_pts)
    K_mat = kernel_fn(state.Hs, state.Tk)  # pyright: ignore
    new_U = K_mat @ new_theta

    return replace(state, theta=new_theta, U=new_U)  # pyright: ignore


# Helper to replace dataclass fields
def replace(obj, **kwargs):
    from dataclasses import replace as dataclass_replace

    return dataclass_replace(obj, **kwargs)
