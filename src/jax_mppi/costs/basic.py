from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

# Type alias for cost function
CostFn = Callable[[Float[Array, "nx"], Float[Array, "nu"]], Float[Array, ""]]


def create_lqr_cost(
    Q: Float[Array, "nx nx"],
    R: Float[Array, "nu nu"],
    goal: Float[Array, "nx"],
) -> CostFn:
    """Create LQR cost function: (goal - state)^T Q (goal - state) + action^T R
    action

    Args:
        Q: State cost matrix (nx x nx)
        R: Control cost matrix (nu x nu)
        goal: Target state (nx,)

    Returns:
        Cost function taking (state, action) -> scalar cost
    """

    def cost_fn(state, action):
        err = state - goal
        return err.T @ Q @ err + action.T @ R @ action

    return cost_fn


def create_obstacle_cost(
    center: Float[Array, "nx"],
    radius: float,
    cost_value: float = 1000.0,
) -> CostFn:
    """Create obstacle cost function.

    Args:
        center: Center of the obstacle (nx,)
        radius: Radius of the obstacle
        cost_value: Cost to apply when inside obstacle

    Returns:
        Cost function taking (state, action) -> scalar cost
    """

    def cost_fn(state, action):
        dist = jnp.linalg.norm(state - center)
        # Smooth obstacle cost using sigmoid or simply step
        # For MPPI, smoother is often better
        return jnp.where(dist < radius, cost_value, 0.0)

    return cost_fn


def create_gaussian_cost(
    Q: Float[Array, "nx nx"],
    center: Float[Array, "nx"],
    cost_at_center: float = 1.0,
) -> CostFn:
    """Create Gaussian cost function (inverted Gaussian).

    Cost = cost_at_center * exp(-0.5 * (x - center)^T Q (x - center))

    Args:
        Q: Shape matrix for the Gaussian (nx x nx). Higher values create
            sharper peaks.
        center: Center of the Gaussian hill (nx,)
        cost_at_center: Maximum cost value at the center (default: 1.0)

    Returns:
        Cost function taking (state, action) -> scalar cost
    """

    def cost_fn(state, action):
        diff = state - center
        exponent = -0.5 * diff.T @ Q @ diff
        return cost_at_center * jnp.exp(exponent)

    return cost_fn
