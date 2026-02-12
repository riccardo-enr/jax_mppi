
import jax.numpy as jnp
from jaxtyping import Array, Float

from jax_mppi.kmppi import KMPPIState, TimeKernel


def interpolate_trajectory(
    kmppi_state: KMPPIState,
    kernel_fn: TimeKernel,
) -> Float[Array, "T nu"]:
    """Interpolate the full trajectory from control points (theta)."""
    # K_matrix = kernel_fn(kmppi_state.Hs, kmppi_state.Tk) # Unused
    # Ktktk = kernel_fn(kmppi_state.Tk, kmppi_state.Tk) # Unused

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
