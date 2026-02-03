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

**Frame Conventions:**
- **NED (North-East-Down)**: Global/world frame where Z-axis points down (gravity is positive Z)
- **FRD (Forward-Right-Down)**: Body frame where X-axis points forward, Y-axis points right, Z-axis points down

### State and Control

The state vector $\mathbf{x} \in \mathbb{R}^{13}$ is defined as:

   \[
   \mathbf{x} = [\mathbf{p}^T, \mathbf{v}^T, \mathbf{q}^T, \boldsymbol{\omega}^T]^T
   \]

where:

- $\mathbf{p} = [p_x, p_y, p_z]^T$ is the position in the NED world frame.
- $\mathbf{v} = [v_x, v_y, v_z]^T$ is the linear velocity in the NED world frame.
- $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$ is the unit quaternion representing orientation (body FRD to world NED).
- $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$ is the angular velocity in the FRD body frame.

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
\dot{\mathbf{v}} = \mathbf{g} + \frac{1}{m} R(\mathbf{q}) \begin{bmatrix} 0 \\ 0 \\ -T \end{bmatrix}
\]

where $\mathbf{g} = [0, 0, g]^T$ is the gravity vector in NED frame (positive down), $m$ is the mass, $T$ is the thrust magnitude (positive), and $R(\mathbf{q})$ is the rotation matrix from FRD body frame to NED world frame. The thrust vector in body frame is $[0, 0, -T]^T$ (upward thrust is negative Z in FRD).

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
    T,              # total thrust magnitude (N) - [0, max_thrust]
                    # Acts in -Z direction of FRD body frame (upward)
    wx_cmd,         # roll rate command (rad/s) - body X-axis (FRD forward)
    wy_cmd,         # pitch rate command (rad/s) - body Y-axis (FRD right)
    wz_cmd          # yaw rate command (rad/s) - body Z-axis (FRD down)
]
```

#### Rationale

- Direct control of thrust and angular velocities
- Easier to enforce control bounds than motor-level commands
- More intuitive for trajectory tracking
- Standard in many quadrotor control frameworks
- FRD body frame convention: thrust acts in -Z direction (upward)

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
    Frame conventions: NED (world), FRD (body)

    State: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz] (13D)
    - Position/velocity in NED world frame
    - Quaternion: body FRD to world NED
    - Angular velocity in FRD body frame

    Action: [T, wx_cmd, wy_cmd, wz_cmd] (4D)
    - T: thrust magnitude (positive, acts in -Z body direction)
    - w_cmd: angular rate commands in FRD body frame

    Returns: next_state after dt using RK4 integration
    """
    # Extract state components
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]  # [qw, qx, qy, qz]
    omega = state[10:13]  # angular velocity in FRD body frame

    # Extract control
    thrust = action[0]  # positive magnitude
    omega_cmd = action[1:4]

    # Rotation matrix from FRD body to NED world frame
    R = quaternion_to_rotation_matrix(quat)

    # Translational dynamics (NED world frame)
    # Gravity: positive Z in NED (downward)
    f_gravity = jnp.array([0, 0, mass * gravity])
    # Thrust in body frame: [0, 0, -T] (upward in FRD)
    # Transform to world frame
    f_thrust = R @ jnp.array([0, 0, -thrust])
    accel = (f_gravity + f_thrust) / mass

    # Rotational dynamics (FRD body frame, first-order model)
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

- **Frame Conventions**: NED world frame, FRD body frame
- **Gravity**: Acts in +Z direction in NED (down is positive)
- **Thrust**: Magnitude T (positive) acts in -Z direction in FRD (upward)
- Quaternion normalization after integration is critical
- RK4 integration recommended for better accuracy
- First-order model for angular velocity (can be extended to full Euler dynamics)

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
    """
    Generate circular trajectory in NED frame.
    
    Args:
        radius: Circle radius in xy plane (m)
        height: Altitude in NED frame (positive down, e.g., -5.0 for 5m above ground)
        period: Period of one revolution (s)
        num_steps: Number of trajectory points
        dt: Time step (s)
    """
    t = jnp.arange(num_steps) * dt
    omega = 2 * jnp.pi / period

    x = radius * jnp.cos(omega * t)
    y = radius * jnp.sin(omega * t)
    z = jnp.ones_like(t) * height  # NED: positive down

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
    """
    Generate figure-8 (lemniscate) trajectory in NED frame.
    
    Args:
        scale: Size of the figure-8 (m)
        height: Altitude in NED frame (positive down, e.g., -5.0 for 5m above ground)
        period: Period of one complete figure-8 (s)
        num_steps: Number of trajectory points
        dt: Time step (s)
    """
    t = jnp.arange(num_steps) * dt
    omega = 2 * jnp.pi / period

    # Lemniscate of Gerono
    x = scale * jnp.sin(omega * t)
    y = scale * jnp.sin(omega * t) * jnp.cos(omega * t)
    z = jnp.ones_like(t) * height  # NED: positive down

    # Velocities (derivatives)
    vx = scale * omega * jnp.cos(omega * t)
    vy = scale * omega * (jnp.cos(omega * t)**2 - jnp.sin(omega * t)**2)
    vz = jnp.zeros_like(t)

    trajectory = jnp.stack([x, y, z, vx, vy, vz], axis=1)
    return trajectory
```

## Implementation Plan

### Phase 1: Core Components ✓

- [x] Explore existing codebase
- [x] Create feature branch `feat/quadrotor-traj-foll-example`
- [x] Draft implementation plan
- [x] Implement quadrotor dynamics module (`src/jax_mppi/dynamics/quadrotor.py`)
  - [x] Quaternion utilities (to rotation matrix, normalization, etc.)
  - [x] Quaternion kinematics
  - [x] 6-DOF dynamics with RK4 integration
  - [x] Unit tests for dynamics (quaternion norm preservation, energy conservation)
- [x] Implement trajectory cost functions (`src/jax_mppi/costs/quadrotor.py`)
  - [x] Position/velocity tracking cost
  - [x] Quaternion-based attitude tracking cost
  - [x] Terminal cost
  - [x] Unit tests for costs

### Phase 2: Trajectory Generators ✓

- [x] Create trajectory generation utilities (`examples/quadrotor/trajectories.py`)
  - [x] Circular trajectory
  - [x] Figure-8 (lemniscate) trajectory
  - [x] Hover setpoint
  - [x] Helix trajectory (bonus)
  - [x] Waypoint interpolation with cubic Hermite splines
  - [x] Trajectory metrics computation
- [x] Unit tests for trajectory generators (28 tests, all passing)

### Phase 3: Basic Examples ✓

- [x] Example 1: Hover control (`examples/quadrotor_hover.py`)
  - [x] Stabilization around fixed setpoint
  - [x] Visualization of state vs time
  - [x] Performance metrics (settling time, overshoot)
- [x] Example 2: Circle following (`examples/quadrotor_circle.py`)
  - [x] Circular trajectory tracking
  - [x] Tracking error visualization
  - [x] Control input visualization
  - [x] 3D trajectory plotting
- [x] Integration tests (11 tests covering both examples)

### Phase 4: Advanced Examples ✓

- [x] Example 3: Figure-8 comparison (`examples/quadrotor_figure8_comparison.py`)
  - [x] MPPI vs SMPPI vs KMPPI comparison
  - [x] Smoothness metrics (control rate, jerk)
  - [x] Energy consumption comparison
  - [x] Side-by-side trajectory plots (6 subplots)
  - [x] Comprehensive performance comparison table
- [x] Example 4: Custom trajectory (`examples/quadrotor_custom_trajectory.py`)
  - [x] Waypoint-based trajectories with cubic Hermite interpolation
  - [x] User-defined reference trajectories
  - [x] Waypoint passage verification
  - [x] Command-line waypoint parsing
- [x] Integration tests (13 tests for advanced examples)

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

### Frame Convention Summary

**NED-FRD Convention:**
- **NED (North-East-Down)**: World/global frame
  - X: North, Y: East, Z: Down (positive downward)
  - Gravity: g = [0, 0, +9.81] m/s² (positive Z direction)
  
- **FRD (Forward-Right-Down)**: Body frame
  - X: Forward, Y: Right, Z: Down (positive downward)
  - Thrust: T acts in -Z direction (upward thrust)
  - Angular rates: [ωx, ωy, ωz] about [Forward, Right, Down] axes

**Important Implementation Details:**
- Altitude: Negative values indicate height above ground (e.g., z = -5.0 means 5m altitude)
- Thrust: Positive magnitude T, applied as [0, 0, -T] in body frame
- Rotation matrix R(q): transforms from FRD body to NED world

---

## Progress Log

### 2026-02-02: Phase 1 Complete ✓

**Completed:**
- Implemented `src/jax_mppi/dynamics/quadrotor.py` with full 6-DOF quadrotor dynamics
  - Quaternion utilities: rotation matrix conversion, normalization, multiplication
  - RK4 integration for accurate numerical integration
  - First-order angular velocity tracking model
  - NED-FRD frame conventions properly implemented
  - Control bounds enforcement

- Implemented `src/jax_mppi/costs/quadrotor.py` with comprehensive cost functions
  - Trajectory tracking cost (position + velocity)
  - Time-indexed trajectory cost
  - Hover control cost (with attitude tracking)
  - Terminal cost for goal reaching
  - Quaternion distance metric

- Comprehensive test coverage (40 tests, all passing)
  - `tests/test_quadrotor_dynamics.py` (19 tests)
  - `tests/test_quadrotor_costs.py` (21 tests)
  - Tests verify: quaternion math, dynamics correctness, JIT compatibility, gradients

**Key Features:**
- All functions are JIT-compatible for high performance
- Gradients work correctly through all dynamics and cost functions
- Quaternion norm preservation verified during integration
- Physical behaviors validated (gravity, thrust, angular tracking)

**Next Steps:**
- Phase 2: Trajectory generators (circle, figure-8, hover setpoint)

### 2026-02-02: Phase 2 Complete ✓

**Completed:**
- Implemented `examples/quadrotor/trajectories.py` with comprehensive trajectory generators
  - `generate_hover_setpoint()` - Constant position stabilization
  - `generate_circle_trajectory()` - Circular paths with configurable center and phase
  - `generate_lemniscate_trajectory()` - Figure-8 patterns (horizontal or vertical)
  - `generate_helix_trajectory()` - Spiral paths with vertical motion
  - `generate_waypoint_trajectory()` - Smooth cubic Hermite interpolation through waypoints
  - `compute_trajectory_metrics()` - Analyze distance, velocity, acceleration

- Comprehensive test coverage (28 tests, all passing)
  - Tests verify: trajectory shapes, periodicity, continuity
  - Validates velocity/position relationships
  - Checks metric computation accuracy

**Key Features:**
- All trajectories follow NED frame convention
- Analytical derivatives for velocity (no numerical differentiation)
- Configurable parameters (center, phase, duration, dt)
- Support for both horizontal and vertical figure-8 patterns

**Next Steps:**
- Phase 3: Basic examples (hover control, circle following)

### 2026-02-02: Phase 3 Complete ✓

**Completed:**
- Implemented `examples/quadrotor_hover.py` - Hover control stabilization
  - MPPI-based hover controller with position and attitude tracking
  - Performance metrics (settling time, position/velocity error)
  - Comprehensive visualization (9 subplots: position, velocity, angular velocity, control inputs, errors, cost)
  - Command-line interface with configurable parameters

- Implemented `examples/quadrotor_circle.py` - Circular trajectory tracking
  - Time-varying reference tracking using trajectory generators
  - 3D trajectory visualization with top-view projection
  - Tracking error analysis and metrics
  - Configurable circle parameters (radius, period)

- Integration tests (11 tests)
  - Example execution tests
  - Convergence validation
  - Quaternion norm preservation
  - Cost decrease verification
  - Cross-example compatibility checks

**Key Features:**
- Both examples run at 50 Hz control rate (JIT-compiled)
- Detailed visualizations saved to `docs/media/`
- Proper NED frame convention throughout
- Performance metrics automatically computed and reported

**Next Steps:**
- Phase 4: Advanced examples (figure-8 comparison, custom trajectories)

### 2026-02-02: Phase 4 Complete ✓

**Completed:**
- Implemented `examples/quadrotor_figure8_comparison.py` - MPPI variant comparison
  - Side-by-side comparison of MPPI, SMPPI, and KMPPI on aggressive figure-8
  - Comprehensive metrics: tracking accuracy, control smoothness, energy consumption
  - Smoothness metrics: control rate (acceleration) and jerk analysis
  - 6-subplot visualization comparing all three variants
  - Performance comparison table with 7 key metrics
  - Demonstrates trade-offs between tracking accuracy and control smoothness

- Implemented `examples/quadrotor_custom_trajectory.py` - Waypoint following
  - User-defined waypoint trajectories with smooth interpolation
  - Cubic Hermite splines for C1 continuity
  - Waypoint passage verification and error reporting
  - Command-line interface for custom waypoint specification
  - 6-subplot visualization including waypoint markers
  - Default square pattern demonstration

- Integration tests (13 tests)
  - Figure-8 comparison execution and metrics validation
  - Custom trajectory with various waypoint configurations
  - Quaternion normalization across all controllers
  - Finite value checks for all outputs

**Key Features:**
- Figure-8 example shows SMPPI produces smoother control (lower jerk)
- Custom trajectory allows arbitrary waypoint sequences
- All examples maintain 50 Hz control rate
- Publication-quality comparison visualizations

**Next Steps:**
- Phase 5: Documentation and polish

---

**Last Updated**: 2026-02-02
**Author**: riccardo-enr
