# Quadrotor Trajectory Following with MPPI

**Status**: In Progress
**Branch**: `feat/quadrotor-traj-foll-example`
**Created**: 2026-02-01

## Objective

Implement a comprehensive set of examples demonstrating quadrotor trajectory following using MPPI control. The goal is to showcase the JAX-MPPI library's capabilities on a realistic robotic system with nonlinear dynamics and provide a reference implementation for users.

## Background

The current JAX-MPPI library includes:

- Three MPPI variants: standard MPPI, SMPPI (smooth), and KMPPI (kernel-based)
- Examples: inverted pendulum, 2D navigation with obstacles
- Well-structured functional API with JIT compilation support
- Autotuning infrastructure for hyperparameter optimization

A quadrotor trajectory following example will:

- Demonstrate MPPI on a high-dimensional nonlinear system (13D state space)
- Showcase reference tracking capabilities
- Provide a realistic robotics benchmark
- Enable comparison between MPPI variants for trajectory smoothness

## Theoretical Background

The quadrotor is modeled as a rigid body with 6 degrees of freedom. The state space is 13-dimensional, representing position, velocity, orientation (quaternion), and angular velocity.

### State and Control

The state vector $\mathbf{x} \in \mathbb{R}^{13}$ is defined as:

   \[
   \mathbf{x} = [\mathbf{p}^T, \mathbf{v}^T, \mathbf{q}^T, \boldsymbol{\omega}^T]^T
   \]

where:

- $\mathbf{p} = [p_x, p_y, p_z]^T$ is the position in the world frame.
- $\mathbf{v} = [v_x, v_y, v_z]^T$ is the linear velocity in the world frame.
- $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$ is the unit quaternion representing orientation (body to world).
- $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$ is the angular velocity in the body frame.

The control input $\mathbf{u} \in \mathbb{R}^{4}$ consists of the total thrust and body angular rates:

   \[
   \mathbf{u} = [T, \omega_{x,cmd}, \omega_{y,cmd}, \omega_{z,cmd}]^T
   \]

### Dynamics

The system dynamics are governed by the following equations:

#### Translational Kinematics

\[
\dot{\mathbf{p}} = \mathbf{v}
\]

#### Translational Dynamics

\[
\dot{\mathbf{v}} = \mathbf{g} + \frac{1}{m} R(\mathbf{q}) \begin{bmatrix} 0 \\ 0 \\ T \end{bmatrix}
\]

where $\mathbf{g} = [0, 0, -g]^T$ is the gravity vector, $m$ is the mass, and $R(\mathbf{q})$ is the rotation matrix derived from quaternion $\mathbf{q}$.

#### Rotational Kinematics

The time derivative of the quaternion is given by:

\[
\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix}
\]

where $\otimes$ denotes quaternion multiplication. In matrix form involving the skew-symmetric matrix

\[
\dot{\mathbf{q}} = \frac{1}{2} \Omega(\boldsymbol{\omega}) \mathbf{q}
\]

*Note: The implementation must ensure $\|\mathbf{q}\| = 1$, typically by normalization after integration.*

#### Rotational Dynamics (First-order actuator model)

\[
\dot{\boldsymbol{\omega}} = \frac{1}{\tau_\omega} (\boldsymbol{\omega}_{cmd} - \boldsymbol{\omega})
\]

where $\tau_\omega$ is the time constant for the angular velocity tracking.

### Cost Function

The MPPI controller optimizes a cost function $J$ over a horizon $H$. The instantaneous cost $C(\mathbf{x}_t, \mathbf{u}_t)$ is defined as

\[
C(\mathbf{x}_t, \mathbf{u}_t) = \|\mathbf{p}_t - \mathbf{p}_{ref,t}\|_{Q_{pos}}^2 + \|\mathbf{v}_t - \mathbf{v}_{ref,t}\|_{Q_{vel}}^2 + \|\mathbf{u}_t\|_{R}^2
\]

where $\|\mathbf{z}\|_W^2 = \mathbf{z}^T W \mathbf{z}$.

## Requirements

### Functional Requirements

#### Quadrotor Dynamics Model

- 6-DOF rigid body dynamics
- State representation: position (3D), velocity (3D), orientation (quaternion), angular velocity (3D) = 13D
- Control inputs: body thrust + body rates (roll, pitch, yaw rates)
- Physical parameters: mass, inertia matrix, arm length
- Full nonlinear dynamics with quaternion-based attitude representation

#### Trajectory Generation

- Multiple reference trajectory types:
  - Circular/helical trajectories
  - Lemniscate (figure-8) trajectories
  - Minimum snap polynomial trajectories
  - Waypoint-based trajectories
- Time-parameterized trajectories with position, velocity, acceleration references

#### Cost Functions

- Position tracking error (weighted L2 norm)
- Velocity tracking error
- Attitude tracking error (quaternion distance: 1 - |q^T q_ref|)
- Control effort penalty (R matrix on actions)
- Trajectory smoothness penalty (action rate limiting)
- Terminal cost for goal convergence

#### Examples to Implement

- **Example 1**: Basic hover control (stabilization around setpoint)
- **Example 2**: Circular trajectory following
- **Example 3**: Figure-8 trajectory with MPPI/SMPPI/KMPPI comparison
- **Example 4**: Minimum snap trajectory following
- **Example 5**: Obstacle avoidance during trajectory following (stretch goal)

#### Visualization

- 3D trajectory plots (reference vs actual)
- Tracking error over time
- Control inputs over time
- Energy consumption
- Optional: animated 3D quadrotor visualization

### Non-Functional Requirements

- **Performance**: JIT-compiled control loops running at >100 Hz on CPU
- **Code Quality**: Follow existing examples pattern (pendulum.py structure)
- **Documentation**: Clear docstrings, inline comments for dynamics equations
- **Testing**: Unit tests for dynamics, cost functions, and integration tests

## Technical Design

### State Representation

#### 13D State with Quaternion Representation

```python
state = [
    px, py, pz,        # position (3)
    vx, vy, vz,        # velocity (3)
    qw, qx, qy, qz,    # quaternion (4) - unit norm constraint
    wx, wy, wz         # angular velocity in body frame (3)
]
```

#### Rationale

- Quaternions avoid gimbal lock singularities present in Euler angle representations
- More numerically stable for aggressive maneuvers
- Standard representation in modern quadrotor control literature
- Unit norm constraint: ||q|| = 1 (enforced after integration)

### Control Input Representation

#### 4D Control: Body Thrust + Body Rates

```python
action = [
    T,              # total thrust in body z-axis (N) - [0, max_thrust]
    wx_cmd,         # roll rate command (rad/s) - body x-axis
    wy_cmd,         # pitch rate command (rad/s) - body y-axis
    wz_cmd          # yaw rate command (rad/s) - body z-axis
]
```

#### Rationale

- Direct control of thrust and angular velocities
- Easier to enforce control bounds than motor-level commands
- More intuitive for trajectory tracking
- Standard in many quadrotor control frameworks

### Dynamics Model

Implement a modular dynamics function following the library's pattern:

```python
def quadrotor_dynamics(
    state: jax.Array,
    action: jax.Array,
    dt: float = 0.01,
    mass: float = 1.0,
    inertia: jax.Array = jnp.eye(3) * 0.1,
    gravity: float = 9.81,
    tau_omega: float = 0.05  # angular velocity time constant
) -> jax.Array:
    """
    6-DOF quadrotor dynamics with quaternion representation.

    State: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz] (13D)
    Action: [T, wx_cmd, wy_cmd, wz_cmd] (4D body thrust + body rates)

    Returns: next_state after dt using RK4 integration
    """
    # Extract state components
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]  # [qw, qx, qy, qz]
    omega = state[10:13]  # angular velocity in body frame

    # Extract control
    thrust = action[0]
    omega_cmd = action[1:4]

    # Rotation matrix from body to world frame
    R = quaternion_to_rotation_matrix(quat)

    # Translational dynamics (world frame)
    f_gravity = jnp.array([0, 0, -mass * gravity])
    f_thrust = R @ jnp.array([0, 0, thrust])  # thrust in body +z
    accel = (f_gravity + f_thrust) / mass

    # Rotational dynamics (body frame, first-order model)
    # For more realism, can use: omega_dot = inv(I) @ (torque - omega x (I @ omega))
    omega_dot = (omega_cmd - omega) / tau_omega

    # Quaternion kinematics: q_dot = 0.5 * Omega(omega) @ q
    # where Omega(omega) is the skew-symmetric matrix
    q_dot = 0.5 * jnp.array([
        -omega[0]*quat[1] - omega[1]*quat[2] - omega[2]*quat[3],  # qw_dot
         omega[0]*quat[0] + omega[2]*quat[2] - omega[1]*quat[3],  # qx_dot
         omega[1]*quat[0] - omega[2]*quat[1] + omega[0]*quat[3],  # qy_dot
         omega[2]*quat[0] + omega[1]*quat[1] - omega[0]*quat[2]   # qz_dot
    ])

    # State derivative
    state_dot = jnp.concatenate([vel, accel, q_dot, omega_dot])

    # Integration (can use RK4 for better accuracy)
    next_state = state + dt * state_dot

    # Normalize quaternion to maintain unit norm
    next_quat = next_state[6:10]
    next_quat = next_quat / jnp.linalg.norm(next_quat)
    next_state = next_state.at[6:10].set(next_quat)

    return next_state
```

#### Key Implementation Notes

- Quaternion normalization after integration is critical
- RK4 integration recommended for better accuracy
- First-order model for angular velocity (can be extended to full Euler dynamics)
- Body frame thrust assumed along +z axis

### Cost Function Design

#### Running Cost

```python
def trajectory_running_cost(
    state: jax.Array,
    action: jax.Array,
    reference: jax.Array,
    t: int,
    Q_pos: jax.Array,
    Q_vel: jax.Array,
    Q_att: jax.Array,
    R: jax.Array
) -> float:
    """
    Trajectory tracking cost with control penalty.

    reference: [px_ref, py_ref, pz_ref, vx_ref, vy_ref, vz_ref, ...]
    """
    # Extract reference for current time step
    ref_t = reference[t]  # or interpolate

    # Position tracking error
    pos_error = state[0:3] - ref_t[0:3]
    cost_pos = pos_error.T @ Q_pos @ pos_error

    # Velocity tracking error
    vel_error = state[3:6] - ref_t[3:6]
    cost_vel = vel_error.T @ Q_vel @ vel_error

    # Attitude tracking (optional)
    # att_error = ...
    # cost_att = att_error.T @ Q_att @ att_error

    # Control effort
    cost_control = action.T @ R @ action

    return cost_pos + cost_vel + cost_control
```

#### Terminal Cost

```python
def trajectory_terminal_cost(
    state: jax.Array,
    last_action: jax.Array,
    goal: jax.Array,
    Q_terminal: jax.Array
) -> float:
    """Terminal cost for reaching goal state."""
    error = state - goal
    return error.T @ Q_terminal @ error
```

### Trajectory Generators

Implement modular trajectory generators:

```python
def generate_circle_trajectory(
    radius: float,
    height: float,
    period: float,
    num_steps: int,
    dt: float
) -> jax.Array:
    """Generate circular trajectory in xy plane."""
    t = jnp.arange(num_steps) * dt
    omega = 2 * jnp.pi / period

    x = radius * jnp.cos(omega * t)
    y = radius * jnp.sin(omega * t)
    z = jnp.ones_like(t) * height

    vx = -radius * omega * jnp.sin(omega * t)
    vy = radius * omega * jnp.cos(omega * t)
    vz = jnp.zeros_like(t)

    # Stack into trajectory array
    trajectory = jnp.stack([x, y, z, vx, vy, vz], axis=1)
    return trajectory


def generate_lemniscate_trajectory(
    scale: float,
    height: float,
    period: float,
    num_steps: int,
    dt: float
) -> jax.Array:
    """Generate figure-8 (lemniscate) trajectory."""
    t = jnp.arange(num_steps) * dt
    omega = 2 * jnp.pi / period

    # Lemniscate of Gerono
    x = scale * jnp.sin(omega * t)
    y = scale * jnp.sin(omega * t) * jnp.cos(omega * t)
    z = jnp.ones_like(t) * height

    # Velocities (derivatives)
    vx = scale * omega * jnp.cos(omega * t)
    vy = scale * omega * (jnp.cos(omega * t)**2 - jnp.sin(omega * t)**2)
    vz = jnp.zeros_like(t)

    trajectory = jnp.stack([x, y, z, vx, vy, vz], axis=1)
    return trajectory
```

## Implementation Plan

### Phase 1: Core Components

- [x] Explore existing codebase
- [x] Create feature branch `feat/quadrotor-traj-foll-example`
- [x] Draft implementation plan
- [ ] Implement quadrotor dynamics module (`src/jax_mppi/dynamics/quadrotor.py`)
  - [ ] Quaternion utilities (to rotation matrix, normalization, etc.)
  - [ ] Quaternion kinematics
  - [ ] 6-DOF dynamics with RK4 integration
  - [ ] Unit tests for dynamics (quaternion norm preservation, energy conservation)
- [ ] Implement trajectory cost functions (`src/jax_mppi/costs/quadrotor.py`)
  - [ ] Position/velocity tracking cost
  - [ ] Quaternion-based attitude tracking cost
  - [ ] Terminal cost
  - [ ] Unit tests for costs

### Phase 2: Trajectory Generators

- [ ] Create trajectory generation utilities (`examples/quadrotor/trajectories.py`)
  - [ ] Circular trajectory
  - [ ] Figure-8 (lemniscate) trajectory
  - [ ] Hover setpoint
  - [ ] Minimum snap trajectory (stretch goal)
  - [ ] Waypoint interpolation (stretch goal)
- [ ] Unit tests for trajectory generators

### Phase 3: Basic Examples

- [ ] Example 1: Hover control (`examples/quadrotor_hover.py`)
  - [ ] Stabilization around fixed setpoint
  - [ ] Visualization of state vs time
  - [ ] Performance metrics (settling time, overshoot)
- [ ] Example 2: Circle following (`examples/quadrotor_circle.py`)
  - [ ] Circular trajectory tracking
  - [ ] Tracking error visualization
  - [ ] Control input visualization

### Phase 4: Advanced Examples

- [ ] Example 3: Figure-8 comparison (`examples/quadrotor_figure8_comparison.py`)
  - [ ] MPPI vs SMPPI vs KMPPI comparison
  - [ ] Smoothness metrics
  - [ ] Energy consumption comparison
  - [ ] Side-by-side trajectory plots
- [ ] Example 4: Custom trajectory (`examples/quadrotor_custom_trajectory.py`)
  - [ ] Minimum snap or waypoint-based
  - [ ] User-defined reference trajectories

### Phase 5: Documentation and Polish

- [ ] Add comprehensive docstrings
- [ ] Create README for quadrotor examples (`examples/quadrotor/README.md`)
- [ ] Add theory documentation (`docs/examples/quadrotor.md`)
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Update main README with quadrotor examples

### Phase 6: Stretch Goals (Optional)

- [ ] Obstacle avoidance during trajectory following
- [ ] Full Euler dynamics for rotational motion (torque-based control)
- [ ] Motor-level control (PWM to thrust mapping)
- [ ] Wind disturbance modeling
- [ ] Autotuning example for quadrotor MPPI hyperparameters
- [ ] Real-time visualization with animation
- [ ] ROS integration example

## File Structure

```python
jax_mppi/
├── src/jax_mppi/
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── linear.py
│   │   └── quadrotor.py         # NEW: Quadrotor dynamics
│   ├── costs/
│   │   ├── __init__.py
│   │   ├── basic.py
│   │   └── quadrotor.py         # NEW: Quadrotor-specific costs
│   └── ...
├── examples/
│   ├── pendulum.py
│   ├── smooth_comparison.py
│   ├── quadrotor/               # NEW: Quadrotor examples directory
│   │   ├── __init__.py
│   │   ├── trajectories.py      # NEW: Trajectory generators
│   │   ├── plotting.py          # NEW: Visualization utilities
│   │   └── README.md            # NEW: Quadrotor examples guide
│   ├── quadrotor_hover.py       # NEW: Example 1
│   ├── quadrotor_circle.py      # NEW: Example 2
│   ├── quadrotor_figure8_comparison.py  # NEW: Example 3
│   └── quadrotor_custom_trajectory.py   # NEW: Example 4
├── tests/
│   ├── test_quadrotor_dynamics.py  # NEW
│   ├── test_quadrotor_costs.py     # NEW
│   └── test_quadrotor_examples.py  # NEW
└── docs/
    ├── plan/
    │   └── quadrotor_trajectory_following.md  # This file
    └── examples/
        └── quadrotor.md         # NEW: Theory and usage guide
```

## Success Criteria

1. **Functionality**: All examples run without errors and demonstrate trajectory following
2. **Performance**: Control loops run at >100 Hz on CPU (JIT-compiled)
3. **Accuracy**: Tracking error <5% of trajectory scale for well-tuned parameters
4. **Code Quality**: Follows existing code style, comprehensive tests (>80% coverage)
5. **Documentation**: Clear README, docstrings, and theory documentation
6. **Usability**: New users can run examples out-of-the-box with minimal setup

## Testing Strategy

### Unit Tests

- Dynamics model: verify state evolution, energy conservation
- Cost functions: verify gradient correctness, cost bounds
- Trajectory generators: verify continuity, derivative correctness

### Integration Tests

- End-to-end MPPI control loop with quadrotor
- JIT compilation compatibility
- Batch rollout generation for visualization

### Performance Tests

- Benchmark control loop frequency
- Memory usage profiling
- Comparison with baseline implementations

## Risk Mitigation

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Dynamics too complex for real-time control | High | Profile performance early, optimize JIT compilation |
| Quaternion norm drift during integration | Medium | Normalize after each integration step |
| Poor tracking performance | Medium | Implement autotuning example, provide tuning guidelines |
| Integration complexity | Low | Follow existing example patterns closely |

## Dependencies

- JAX (already required)
- matplotlib (for visualization, already used in examples)
- scipy (optional, for minimum snap trajectories)
- All dependencies should be compatible with existing `pyproject.toml`

## References

- [1] Williams, G., et al. "Information theoretic MPC for model-based reinforcement learning." ICRA 2017.
- [2] Williams, G., et al. "Model predictive path integral control using covariance variable importance sampling." arXiv:1509.01149, 2015.
- [3] Beard, R. W., & McLain, T. W. "Small Unmanned Aircraft: Theory and Practice." Princeton University Press, 2012.
- [4] Mellinger, D., & Kumar, V. "Minimum snap trajectory generation and control for quadrotors." ICRA 2011.
- [5] [pytorch_mppi original implementation](https://github.com/UM-ARM-Lab/pytorch_mppi)

## Notes

- This plan should be updated as implementation progresses
- Move to `docs/plan/completed/` when all phases are finished
- Link any related issues or PRs here

---

**Last Updated**: 2026-02-01
**Author**: riccardo-enr
