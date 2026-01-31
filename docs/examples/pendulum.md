# Pendulum Swing-Up

This example demonstrates how to use `jax_mppi` to control an inverted pendulum. The goal is to swing the pendulum up from a hanging position and stabilize it at the top.

## Code

The full example code is available in `examples/pendulum.py`.

## Dynamics

The pendulum dynamics are defined as a pure function:

```python
def pendulum_dynamics(state: jax.Array, action: jax.Array) -> jax.Array:
    """Pendulum dynamics.

    State: [theta, theta_dot]
        theta: angle from upright (0 = upright, pi = hanging down)
        theta_dot: angular velocity
    Action: [torque]
        torque: applied torque (control input)
    """
    g = 10.0  # gravity
    m = 1.0  # mass
    l = 1.0  # length
    dt = 0.05  # timestep

    theta, theta_dot = state[0], state[1]
    torque = action[0]

    # Clip torque to reasonable bounds
    torque = jnp.clip(torque, -2.0, 2.0)

    # Pendulum dynamics: theta_ddot = (torque - m*g*l*sin(theta)) / (m*l^2)
    theta_ddot = (torque - m * g * l * jnp.sin(theta)) / (m * l * l)

    # Euler integration
    theta_dot_next = theta_dot + theta_ddot * dt
    theta_next = theta + theta_dot_next * dt

    # Normalize angle to [-pi, pi]
    theta_next = ((theta_next + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    return jnp.array([theta_next, theta_dot_next])
```

## Cost Function

The running cost penalizes deviation from the upright position and high control effort:

```python
def pendulum_cost(state: jax.Array, action: jax.Array) -> jax.Array:
    theta, theta_dot = state[0], state[1]
    torque = action[0]

    # Cost for being away from upright (theta=0)
    angle_cost = theta**2

    # Cost for high angular velocity
    velocity_cost = 0.1 * theta_dot**2

    # Cost for using torque
    control_cost = 0.01 * torque**2

    return angle_cost + velocity_cost + control_cost
```

## Running the Example

You can run the example using:

```bash
python examples/pendulum.py --visualize
```
