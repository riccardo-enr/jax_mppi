"""Linear dynamics models for JAX MPPI."""

from typing import Protocol

from jaxtyping import Array, Float


class Dynamics(Protocol):
    """Protocol for dynamics functions."""

    def __call__(
        self, state: Float[Array, "nx"], action: Float[Array, "nu"]
    ) -> Float[Array, "nx"]:
        """Compute next state given current state and action."""
        ...


def create_linear_delta_dynamics(B: Float[Array, "nx nu"]) -> Dynamics:
    """Create linear delta dynamics: next_state = state + action @ B.T

    This is a simple linear dynamics model where the state update is:
        x_{t+1} = x_t + u_t @ B^T

    Args:
        B: Linear transformation matrix (nx x nu)

    Returns:
        Dynamics function that applies the linear transformation

    Example:
        >>> B = jnp.array([[0.5, 0.0], [0.0, -0.5]])
        >>> dynamics = create_linear_delta_dynamics(B)
        >>> state = jnp.array([-3.0, -2.0])
        >>> action = jnp.array([1.0, 1.0])
        >>> next_state = dynamics(state, action)
    """

    def dynamics(
        state: Float[Array, "nx"], action: Float[Array, "nu"]
    ) -> Float[Array, "nx"]:
        # action @ B.T gives the delta to add to state
        delta = action @ B.T
        return state + delta

    return dynamics
