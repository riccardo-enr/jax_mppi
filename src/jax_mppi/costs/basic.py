"""Basic cost functions for JAX MPPI."""

from typing import Callable, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

CostFn = Callable[[Float[Array, "nx"], Optional[Float[Array, "nu"]]], Float[Array, ""]]


def create_lqr_cost(
    Q: Float[Array, "nx nx"],
    R: Float[Array, "nu nu"],
    goal: Float[Array, "nx"],
) -> CostFn:
    """Create LQR cost function: (goal - state)^T Q (goal - state) + action^T R action

    Args:
        Q: State cost matrix (nx x nx)
        R: Action cost matrix (nu x nu)
        goal: Goal state (nx,)

    Returns:
        Cost function that computes LQR cost

    Example:
        >>> Q = jnp.eye(2)
        >>> R = jnp.eye(2) * 0.01
        >>> goal = jnp.array([2.0, 2.0])
        >>> cost_fn = create_lqr_cost(Q, R, goal)
        >>> state = jnp.array([0.0, 0.0])
        >>> action = jnp.array([1.0, 1.0])
        >>> cost = cost_fn(state, action)
    """

    def cost_fn(
        state: Float[Array, "nx"], action: Optional[Float[Array, "nu"]] = None
    ) -> Float[Array, ""]:
        dx = goal - state
        # Quadratic form: dx^T Q dx
        c = dx @ Q @ dx

        if action is not None:
            # Add action cost: action^T R action
            c = c + action @ R @ action

        return c

    return cost_fn


def create_hill_cost(
    Q: Float[Array, "nx nx"],
    center: Float[Array, "nx"],
    cost_at_center: float = 1.0,
) -> CostFn:
    """Create Gaussian hill cost (obstacle avoidance).

    Creates a cost landscape with a Gaussian peak at the specified center.
    This is useful for obstacle avoidance where the cost increases as the
    state gets closer to the obstacle center.

    Cost = cost_at_center * exp(-(center - state)^T Q (center - state))

    Args:
        Q: Shape matrix for the Gaussian (nx x nx). Higher values create sharper peaks.
        center: Center of the Gaussian hill (nx,)
        cost_at_center: Maximum cost value at the center (default: 1.0)

    Returns:
        Cost function that computes Gaussian hill cost

    Example:
        >>> Q = jnp.eye(2) * 0.5
        >>> center = jnp.array([-0.5, -1.0])
        >>> cost_fn = create_hill_cost(Q, center, cost_at_center=200.0)
        >>> state = jnp.array([-0.5, -1.0])  # At center
        >>> cost = cost_fn(state)  # Returns 200.0
    """

    def cost_fn(
        state: Float[Array, "nx"], action: Optional[Float[Array, "nu"]] = None
    ) -> Float[Array, ""]:
        dx = center - state
        # Mahalanobis distance: dx^T Q dx
        d = dx @ Q @ dx
        # Gaussian: exp(-d)
        c = cost_at_center * jnp.exp(-d)
        return c

    return cost_fn
