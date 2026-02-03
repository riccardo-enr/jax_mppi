# Quadrotor Trajectory Following

This example demonstrates how to use `jax_mppi` to control a 13-dimensional quadrotor system to follow complex 3D trajectories. It covers nonlinear dynamics, quaternion-based attitude control, and multiple MPPI variants.

## Overview

The quadrotor is modeled as a 6-DOF rigid body with a 13-dimensional state space and 4-dimensional control input. The examples showcase:

- **Hover Control**: Stabilizing at a fixed setpoint.
- **Circular Tracking**: Following a dynamic reference trajectory.
- **Figure-8 Tracking**: Aggressive maneuvering comparison between MPPI, SMPPI, and KMPPI.
- **Custom Waypoints**: Tracking a user-defined path interpolated with splines.

## Theoretical Background

### State and Control

The state vector $\mathbf{x} \in \mathbb{R}^{13}$ is defined as:

$$
\mathbf{x} = [\mathbf{p}^T, \mathbf{v}^T, \mathbf{q}^T, \boldsymbol{\omega}^T]^T
$$

where:

- $\mathbf{p} = [p_x, p_y, p_z]^T$ is the position in the NED (North-East-Down) world frame.
- $\mathbf{v} = [v_x, v_y, v_z]^T$ is the linear velocity in the NED world frame.
- $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$ is the unit quaternion representing orientation (body FRD to world NED).
- $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$ is the angular velocity in the FRD (Forward-Right-Down) body frame.

The control input $\mathbf{u} \in \mathbb{R}^{4}$ consists of the total thrust and body angular rates:

$$
\mathbf{u} = [T, \omega_{x,cmd}, \omega_{y,cmd}, \omega_{z,cmd}]^T
$$

### Dynamics

The system follows standard rigid body dynamics with a first-order actuator model for angular rates.

1. **Translational**: $\dot{\mathbf{v}} = \mathbf{g} + \frac{1}{m} R(\mathbf{q}) \begin{bmatrix} 0 \\ 0 \\ -T \end{bmatrix}$
2. **Rotational**: $\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix}$
3. **Actuators**: $\dot{\boldsymbol{\omega}} = \frac{1}{\tau_\omega} (\boldsymbol{\omega}_{cmd} - \boldsymbol{\omega})$

### Cost Function

The MPPI controller minimizes a cost function that penalizes tracking error and control effort:

$$
C(\mathbf{x}_t, \mathbf{u}_t) = \|\mathbf{p}_t - \mathbf{p}_{ref,t}\|_{Q_{pos}}^2 + \|\mathbf{v}_t - \mathbf{v}_{ref,t}\|_{Q_{vel}}^2 + \|\mathbf{u}_t\|_{R}^2
$$

## Running the Examples

All examples are located in `examples/`.

### 1. Hover Control

Stabilizes the quadrotor at a fixed altitude and position.

```bash
python examples/quadrotor_hover.py --visualize
```

### 2. Circular Trajectory

Tracks a circle in the horizontal plane.

```bash
python examples/quadrotor_circle.py --visualize
```

### 3. Figure-8 Comparison

Compares Standard MPPI, Smooth MPPI (SMPPI), and Kernel MPPI (KMPPI) on a figure-8 trajectory. This example produces a detailed performance report comparing tracking error and control smoothness (jerk).

```bash
python examples/quadrotor_figure8_comparison.py --visualize
```

### 4. Custom Trajectory

Allows you to define custom waypoints via command line arguments.

```bash
# Square pattern
python examples/quadrotor_custom_trajectory.py --visualize
```

## Results

The examples produce plots in `docs/media/` showing the 3D trajectory, tracking errors, and control inputs.

![Hover Control](../media/quadrotor_hover_mppi.png)
![Circle Tracking](../media/quadrotor_circle_mppi.png)
