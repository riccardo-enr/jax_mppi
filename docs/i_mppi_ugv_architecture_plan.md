# Parallel I-MPPI Architecture for UGV: Design Plan

**Date**: 2026-02-12
**Status**: Draft
**Target**: JAX/Python Implementation

---

## 1. Executive Summary

This document outlines the design and implementation plan for adapting the existing I-MPPI (Informative Model Predictive Path Integral) architecture to work with Unmanned Ground Vehicles (UGVs). The implementation will leverage the existing JAX-based codebase and extend the two-layer hierarchical architecture currently used for quadrotors.

### Key Objectives
1. **UGV Dynamics Model**: Implement ground vehicle kinematics/dynamics models
2. **Terrain-Aware Planning**: Extend grid maps to include elevation and traversability
3. **Ground-Based Sensing**: Adapt FSMI for horizontal sensor models (LiDAR, cameras)
4. **Parallel Architecture**: Maintain two-layer hierarchy (Layer 2: FSMI planner, Layer 3: MPPI controller)
5. **Leverage Existing Code**: Reuse core MPPI, FSMI, and map infrastructure

---

## 2. Architecture Overview

### 2.1 Two-Layer Hierarchical Control

```
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Information-Theoretic Planner (~5-10 Hz)     │
│                                                         │
│  ┌─────────────┐      ┌──────────────────┐            │
│  │ FSMI        │──────>│ Trajectory       │            │
│  │ Analyzer    │      │ Generator        │            │
│  └─────────────┘      └──────────────────┘            │
│         │                      │                        │
│         v                      v                        │
│  Information Field      Reference Trajectory           │
│  (High MI regions)      (x_ref, u_ref)                │
└─────────────────────────────────────────────────────────┘
                            │
                            │ x_ref(t), u_ref(t)
                            v
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Biased MPPI Controller (~50-100 Hz)          │
│                                                         │
│  ┌─────────────┐      ┌──────────────────┐            │
│  │ Sampling    │──────>│ Trajectory       │            │
│  │ (Biased)    │      │ Rollouts         │            │
│  └─────────────┘      └──────────────────┘            │
│         │                      │                        │
│         v                      v                        │
│  K samples             Cost Evaluation                 │
│  (50% ref, 50% noise)  (tracking + info + obstacles)  │
└─────────────────────────────────────────────────────────┘
                            │
                            v
                        u(t) → UGV
```

### 2.2 Key Components

| Component | Frequency | Purpose | JAX Pattern |
|-----------|-----------|---------|-------------|
| FSMI Planner | 5-10 Hz | Generate information-maximizing reference trajectories | `jax.vmap` over trajectory samples |
| Biased MPPI | 50-100 Hz | Track reference + local obstacle avoidance | `jax.vmap` over K samples, `jax.lax.scan` over T horizon |
| UGV Dynamics | Per-step | Forward integration of vehicle model | `jax.jit` compiled RK4/Euler |
| Grid Map | Static/Update | Occupancy + traversability + information | Pytree-registered |
| Sensor Model | Per-pose | LiDAR/Camera FOV and range | `jax.vmap` over beams |

---

## 3. UGV Dynamics Models

### 3.1 Model Selection

We'll implement three UGV models with increasing complexity:

#### Model 1: Unicycle Kinematics (Simplest)
**State** (3D): `[x, y, θ]`
**Control** (2D): `[v, ω]` (linear velocity, angular velocity)

```python
def unicycle_dynamics(state, action, dt):
    x, y, theta = state
    v, omega = action

    x_next = x + v * jnp.cos(theta) * dt
    y_next = y + v * jnp.sin(theta) * dt
    theta_next = theta + omega * dt

    return jnp.array([x_next, y_next, theta_next])
```

**Pros**: Simple, fast, good for initial testing
**Cons**: No dynamics, instantaneous velocity changes

---

#### Model 2: Differential Drive Dynamics (Recommended Start)
**State** (5D): `[x, y, θ, v, ω]`
**Control** (2D): `[a, α]` (linear acceleration, angular acceleration)

```python
def diffdrive_dynamics(state, action, dt, max_v, max_omega, drag_coeff):
    x, y, theta, v, omega = state
    a, alpha = action

    # Velocity dynamics with drag
    v_next = v + (a - drag_coeff * v) * dt
    omega_next = omega + (alpha - drag_coeff * omega) * dt

    # Clip to limits
    v_next = jnp.clip(v_next, 0.0, max_v)
    omega_next = jnp.clip(omega_next, -max_omega, max_omega)

    # Position kinematics (RK4 or Euler)
    x_next = x + v * jnp.cos(theta) * dt
    y_next = y + v * jnp.sin(theta) * dt
    theta_next = theta + omega * dt

    return jnp.array([x_next, y_next, theta_next, v_next, omega_next])
```

**Pros**: Captures inertia, realistic for most UGVs
**Cons**: Still simplified dynamics
**Use Case**: Urban exploration, warehouse robots

---

#### Model 3: Ackermann Steering (Car-like)
**State** (5D): `[x, y, θ, v, δ]`
**Control** (2D): `[a, δ_dot]` (acceleration, steering rate)

```python
def ackermann_dynamics(state, action, dt, L_wheelbase, max_v, max_delta):
    x, y, theta, v, delta = state
    a, delta_dot = action

    # Steering dynamics
    delta_next = jnp.clip(delta + delta_dot * dt, -max_delta, max_delta)

    # Velocity dynamics
    v_next = jnp.clip(v + a * dt, 0.0, max_v)

    # Kinematic bicycle model
    x_next = x + v * jnp.cos(theta) * dt
    y_next = y + v * jnp.sin(theta) * dt
    theta_next = theta + (v / L_wheelbase) * jnp.tan(delta) * dt

    return jnp.array([x_next, y_next, theta_next, v_next, delta_next])
```

**Pros**: Realistic for car-like vehicles
**Cons**: More complex, turning radius constraints
**Use Case**: Autonomous cars, large outdoor robots

---

### 3.2 Factory Functions (Following Existing Pattern)

```python
# In src/jax_mppi/dynamics/ugv.py

def create_unicycle_dynamics(dt: float, v_max: float, omega_max: float):
    """Factory for unicycle kinematics"""
    def dynamics(state: jax.Array, action: jax.Array) -> jax.Array:
        # Implementation
        pass

    config = {
        'dt': dt,
        'nx': 3,
        'nu': 2,
        'u_min': jnp.array([0.0, -omega_max]),
        'u_max': jnp.array([v_max, omega_max])
    }

    return dynamics, config


def create_diffdrive_dynamics(
    dt: float = 0.05,
    max_v: float = 2.0,
    max_omega: float = 2.0,
    max_accel: float = 1.0,
    max_alpha: float = 2.0,
    drag_coeff: float = 0.1
):
    """Factory for differential drive dynamics (RECOMMENDED)"""
    # Implementation
    pass


def create_ackermann_dynamics(
    dt: float = 0.05,
    wheelbase: float = 0.5,
    max_v: float = 5.0,
    max_steering_angle: float = 0.6,  # ~35 degrees
    max_steering_rate: float = 1.0
):
    """Factory for Ackermann steering dynamics"""
    # Implementation
    pass
```

---

## 4. UGV-Specific Cost Functions

### 4.1 Navigation Costs

```python
# In src/jax_mppi/costs/ugv.py

def create_waypoint_cost(
    Q_pos: jax.Array,      # (2, 2) - position weight
    Q_heading: float,       # Heading weight
    waypoint: jax.Array,    # (x, y, theta_desired)
    goal_tolerance: float = 0.5
):
    """
    Cost = ||pos - waypoint||_Q + Q_heading * (1 - cos(theta - theta_desired))
    """
    def cost(state, action):
        pos = state[:2]
        theta = state[2]

        pos_error = waypoint[:2] - pos
        pos_cost = pos_error @ Q_pos @ pos_error

        heading_error = waypoint[2] - theta
        heading_cost = Q_heading * (1 - jnp.cos(heading_error))

        # Smoothly decay cost near goal
        distance = jnp.linalg.norm(pos_error)
        decay = jnp.where(distance < goal_tolerance,
                          distance / goal_tolerance,
                          1.0)

        return decay * (pos_cost + heading_cost)

    return cost


def create_path_following_cost(
    Q_cross_track: float,   # Perpendicular error weight
    Q_heading: float,       # Heading alignment weight
    path_points: jax.Array  # (N, 2) waypoints
):
    """
    Cost for following a path (closest point projection)
    """
    def cost(state, action):
        pos = state[:2]
        theta = state[2]

        # Find closest path segment
        distances = jax.vmap(lambda p: jnp.linalg.norm(p - pos))(path_points)
        closest_idx = jnp.argmin(distances)

        # Cross-track error
        closest_point = path_points[closest_idx]
        cross_track_error = jnp.linalg.norm(pos - closest_point)

        # Heading alignment (if path provides orientation)
        # ... implementation

        return Q_cross_track * cross_track_error**2

    return cost
```

### 4.2 Terrain and Traversability Costs

```python
def create_terrain_cost(
    grid_map,              # GridMap with elevation/traversability
    Q_elevation: float,    # Penalty for elevation changes
    Q_roughness: float     # Penalty for rough terrain
):
    """
    Penalize difficult terrain and elevation changes
    """
    def cost(state, action):
        x, y = state[:2]

        # Get terrain properties at (x, y)
        elevation, traversability = grid_map.query_terrain(x, y)

        # Higher cost for steep slopes or rough terrain
        elevation_cost = Q_elevation * elevation**2
        roughness_cost = Q_roughness * (1.0 - traversability)

        return elevation_cost + roughness_cost

    return cost


def create_dynamic_obstacle_cost(
    obstacles: jax.Array,   # (N, 4) [x, y, vx, vy]
    Q_obs: jax.Array,       # (2, 2) inverse covariance
    prediction_horizon: int
):
    """
    Cost for predicted dynamic obstacle positions
    """
    def cost_t(state, action, t):
        pos = state[:2]

        # Predict obstacle positions at time t
        predicted_obs = obstacles[:, :2] + obstacles[:, 2:] * t * dt

        # Sum of Gaussian costs
        def single_obs_cost(obs_pos):
            diff = pos - obs_pos
            return jnp.exp(-0.5 * diff @ Q_obs @ diff)

        return jax.vmap(single_obs_cost)(predicted_obs).sum()

    return cost_t
```

### 4.3 Information-Aware Costs

```python
def create_information_gain_cost(
    fsmi_state,            # FSMI state with information map
    Q_info: float,         # Information gain weight (negative for reward)
    max_info_distance: float = 10.0
):
    """
    Reward exploring high-information regions (negative cost)
    """
    def cost(state, action):
        pos = state[:2]
        theta = state[2]

        # Query expected information gain at this pose
        info_gain = fsmi_state.query_information(pos, theta)

        # Distance decay (prefer nearby information)
        distance_weight = jnp.exp(-jnp.linalg.norm(pos) / max_info_distance)

        # Negative cost = reward
        return -Q_info * info_gain * distance_weight

    return cost
```

---

## 5. Sensor Models for Ground Vehicles

### 5.1 Horizontal LiDAR Model

```python
# In src/jax_mppi/i_mppi/sensors.py

@dataclass(frozen=True)
class LiDARConfig:
    """2D LiDAR configuration"""
    num_beams: int = 360              # Full circle or sector
    max_range: float = 30.0           # meters
    fov_angle: float = 2 * jnp.pi     # radians (360° or 270°)
    angular_resolution: float = None  # Auto-computed
    range_noise_std: float = 0.05     # meters
    height: float = 0.5               # Height above ground


def create_lidar_sensor_model(config: LiDARConfig):
    """
    Creates inverse sensor model for 2D LiDAR
    Similar to existing beam model but horizontal plane
    """
    angles = jnp.linspace(-config.fov_angle/2,
                          config.fov_angle/2,
                          config.num_beams)

    def inverse_sensor_model(
        pose: jax.Array,        # (x, y, theta)
        measurement: jax.Array, # (num_beams,) ranges
        grid_map: GridMap
    ) -> jax.Array:
        """
        Returns log-odds update for grid cells
        Output: (H, W) log-odds deltas
        """
        x, y, theta = pose

        # Beam endpoints in world frame
        beam_angles = theta + angles
        beam_x = x + measurement * jnp.cos(beam_angles)
        beam_y = y + measurement * jnp.sin(beam_angles)

        # Ray casting (reuse existing FSMI ray casting)
        log_odds_update = jax.vmap(
            lambda bx, by: _ray_trace_log_odds(
                grid_map, x, y, bx, by, measurement
            )
        )(beam_x, beam_y)

        return log_odds_update.sum(axis=0)  # Aggregate all beams

    return inverse_sensor_model
```

### 5.2 Forward-Facing Camera Model

```python
@dataclass(frozen=True)
class CameraConfig:
    """Monocular or stereo camera"""
    hfov: float = jnp.pi / 3          # Horizontal FOV (60°)
    vfov: float = jnp.pi / 4          # Vertical FOV (45°)
    max_range: float = 20.0           # Effective depth range
    resolution: tuple = (640, 480)
    height: float = 1.0               # Camera height
    tilt_angle: float = -0.1          # Downward tilt (radians)


def create_camera_frustum_model(config: CameraConfig):
    """
    FOV frustum for information-theoretic planning
    Simplified compared to full image processing
    """
    def compute_frustum_info(
        pose: jax.Array,        # (x, y, theta)
        grid_map: GridMap
    ) -> float:
        """
        Compute mutual information within camera frustum
        """
        x, y, theta = pose

        # Sample points within frustum
        ranges = jnp.linspace(0, config.max_range, 20)
        angles = jnp.linspace(-config.hfov/2, config.hfov/2, 30)

        # Grid of (range, angle) -> (x, y) in world frame
        r_grid, a_grid = jnp.meshgrid(ranges, angles)
        sample_x = x + r_grid * jnp.cos(theta + a_grid)
        sample_y = y + r_grid * jnp.sin(theta + a_grid)

        # Query entropy at each point
        entropies = jax.vmap(jax.vmap(
            lambda sx, sy: grid_map.query_entropy(sx, sy)
        ))(sample_x, sample_y)

        # Aggregate (weighted by distance decay)
        distance_weights = jnp.exp(-r_grid / config.max_range)
        total_info = (entropies * distance_weights).sum()

        return total_info

    return compute_frustum_info
```

---

## 6. Enhanced Grid Map for UGVs

### 6.1 Multi-Layer Grid Map

```python
# In src/jax_mppi/i_mppi/ugv_map.py

@register_pytree_node_class
@dataclass
class UGVGridMap:
    """
    Extended grid map for ground vehicles
    Includes occupancy, elevation, traversability, and information
    """
    # Existing fields
    occupancy: jax.Array        # (H, W) in [0, 1] - obstacle probability
    entropy: jax.Array          # (H, W) - information content
    origin: jax.Array           # (2,) world coordinates
    resolution: float           # meters/cell
    width: int
    height: int

    # New UGV-specific fields
    elevation: jax.Array        # (H, W) - height in meters
    traversability: jax.Array   # (H, W) in [0, 1] - 1=easy, 0=impassable
    roughness: jax.Array        # (H, W) - terrain roughness metric

    # Metadata
    elevation_scale: float = 1.0
    traversability_threshold: float = 0.3  # Below this = obstacle

    def query_terrain(self, x: float, y: float) -> tuple:
        """
        Query terrain properties at world coordinates (x, y)
        Returns: (elevation, traversability, is_occupied)
        """
        i, j = self.world_to_grid(x, y)

        # Bounds checking
        in_bounds = (0 <= i < self.height) & (0 <= j < self.width)

        elev = jnp.where(in_bounds, self.elevation[i, j], 0.0)
        trav = jnp.where(in_bounds, self.traversability[i, j], 0.0)
        occ = jnp.where(in_bounds, self.occupancy[i, j], 1.0)

        return elev, trav, occ

    def is_collision_free(self, x: float, y: float) -> bool:
        """Check if position is traversable"""
        _, trav, occ = self.query_terrain(x, y)
        return (trav > self.traversability_threshold) & (occ < 0.5)

    def update_from_lidar(
        self,
        pose: jax.Array,
        scan: jax.Array,
        sensor_model
    ):
        """
        Update map from LiDAR scan
        Returns: new UGVGridMap with updated occupancy and entropy
        """
        log_odds_update = sensor_model(pose, scan, self)

        # Convert log-odds to probability (existing pattern)
        new_occupancy = self._apply_log_odds_update(log_odds_update)

        # Update entropy (p * log p)
        new_entropy = self._compute_entropy(new_occupancy)

        return self.replace(occupancy=new_occupancy, entropy=new_entropy)
```

### 6.2 Terrain Loading from Heightmaps

```python
def create_ugv_map_from_heightmap(
    occupancy_image: np.ndarray,   # (H, W) grayscale
    heightmap_image: np.ndarray,   # (H, W) grayscale
    resolution: float = 0.1,
    elevation_scale: float = 5.0
) -> UGVGridMap:
    """
    Load map from images (PNG, etc.)
    occupancy_image: 0=free, 255=occupied
    heightmap_image: 0=low elevation, 255=high elevation
    """
    # Normalize to [0, 1]
    occupancy = jnp.array(occupancy_image / 255.0)
    elevation = jnp.array(heightmap_image / 255.0) * elevation_scale

    # Compute traversability from gradient
    grad_x = jnp.gradient(elevation, axis=1)
    grad_y = jnp.gradient(elevation, axis=0)
    slope = jnp.sqrt(grad_x**2 + grad_y**2)

    # Steeper slopes = lower traversability
    max_slope = 0.5  # ~26 degrees
    traversability = jnp.clip(1.0 - slope / max_slope, 0.0, 1.0)

    # Initial entropy (uniform prior)
    entropy = 4 * occupancy * (1 - occupancy)  # Max entropy at p=0.5

    return UGVGridMap(
        occupancy=occupancy,
        entropy=entropy,
        elevation=elevation,
        traversability=traversability,
        roughness=slope,
        origin=jnp.array([0.0, 0.0]),
        resolution=resolution,
        width=occupancy.shape[1],
        height=occupancy.shape[0]
    )
```

---

## 7. Biased MPPI for UGV

### 7.1 Reference Trajectory Tracking

```python
# In src/jax_mppi/i_mppi/ugv_planner.py

def biased_ugv_mppi_command(
    config: MPPIConfig,
    mppi_state: MPPIState,
    current_state: jax.Array,      # (nx,) UGV state
    reference_trajectory: jax.Array,  # (T, nx) from Layer 2
    reference_actions: jax.Array,   # (T, nu) from Layer 2
    dynamics,
    running_cost,
    terminal_cost,
    bias_ratio: float = 0.5,        # 50% biased toward reference
    key: jax.Array = None
):
    """
    Biased sampling MPPI for Layer 3 control

    Sampling strategy:
    - 50% of samples centered on reference trajectory
    - 50% of samples from exploration noise

    Cost includes:
    - Reference tracking cost
    - Information gain reward
    - Obstacle avoidance cost
    """
    T = config.horizon
    K = config.num_samples
    nu = config.nu

    K_biased = int(K * bias_ratio)
    K_explore = K - K_biased

    # Get reference for current horizon window
    ref_actions = reference_actions[:T]  # (T, nu)

    # Sample noise
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Biased samples: reference + small noise
    noise_biased = jax.random.multivariate_normal(
        subkey1,
        mean=jnp.zeros(nu),
        cov=mppi_state.noise_sigma * 0.1,  # 10% of exploration noise
        shape=(K_biased, T)
    )
    actions_biased = ref_actions[None, :, :] + noise_biased  # (K_biased, T, nu)

    # Exploration samples: current trajectory + full noise
    noise_explore = jax.random.multivariate_normal(
        subkey2,
        mean=jnp.zeros(nu),
        cov=mppi_state.noise_sigma,
        shape=(K_explore, T)
    )
    actions_explore = mppi_state.U[None, :, :] + noise_explore  # (K_explore, T, nu)

    # Concatenate
    actions = jnp.concatenate([actions_biased, actions_explore], axis=0)  # (K, T, nu)

    # Standard MPPI rollout and weighting
    costs = jax.vmap(
        lambda a: _rollout_cost(
            config, current_state, a, dynamics, running_cost, terminal_cost
        )
    )(actions)

    # Importance sampling weights
    weights = jax.nn.softmax(-costs / config.lambda_)

    # Update trajectory
    all_noise = jnp.concatenate([noise_biased, noise_explore], axis=0)
    delta_U = jnp.tensordot(weights, all_noise, axes=1)

    U_new = mppi_state.U + delta_U

    # Update state
    new_mppi_state = mppi_state.replace(U=U_new, key=key)

    # Return first action
    action = U_new[0]

    return action, new_mppi_state
```

### 7.2 Composite Cost Function

```python
def create_ugv_i_mppi_cost(
    Q_ref: float,              # Reference tracking weight
    Q_info: float,             # Information gain weight
    Q_obs: jax.Array,          # Obstacle avoidance
    reference_traj: jax.Array, # (T, nx)
    grid_map: UGVGridMap,
    sensor_config: LiDARConfig
):
    """
    Multi-objective cost for UGV I-MPPI
    """
    def running_cost(state, action, t):
        # 1. Reference tracking
        ref_state = reference_traj[t]
        tracking_error = state - ref_state
        tracking_cost = Q_ref * (tracking_error @ tracking_error)

        # 2. Information gain (negative cost = reward)
        pos = state[:2]
        theta = state[2]
        info_gain = grid_map.query_entropy_at(pos)
        info_cost = -Q_info * info_gain

        # 3. Obstacle avoidance
        _, trav, occ = grid_map.query_terrain(pos[0], pos[1])
        obstacle_cost = jnp.where(occ > 0.5, 1e6, 0.0)  # Hard constraint

        # Traversability penalty
        trav_cost = 10.0 * (1.0 - trav)

        return tracking_cost + info_cost + obstacle_cost + trav_cost

    return running_cost
```

---

## 8. Layer 2: FSMI Planner for UGV

### 8.1 Adapt Existing FSMI

The existing FSMI infrastructure in `src/jax_mppi/i_mppi/fsmi.py` can be largely reused, with modifications:

```python
# In src/jax_mppi/i_mppi/ugv_fsmi.py

def create_ugv_fsmi_trajectory_generator(
    dynamics,                  # UGV dynamics
    grid_map: UGVGridMap,
    sensor_config: LiDARConfig,
    num_trajectory_samples: int = 100,
    trajectory_horizon: int = 50,
    planning_dt: float = 0.2   # Longer timestep for Layer 2
):
    """
    Generate information-maximizing trajectories for UGV

    Key differences from quadrotor version:
    1. UGV dynamics constraints (turning radius, etc.)
    2. Horizontal sensor FOV instead of downward
    3. Terrain-aware sampling
    """
    def sample_trajectories(
        current_state: jax.Array,  # (nx,)
        goal_position: jax.Array,  # (2,) or (3,)
        key: jax.Array
    ) -> jax.Array:
        """
        Sample K candidate trajectories with exploration bias
        Returns: (K, T, nx)
        """
        # Strategy: Sample control sequences, rollout dynamics
        keys = jax.random.split(key, num_trajectory_samples)

        def sample_single_trajectory(k):
            # Sample control sequence (exploration + goal-directed)
            actions = _sample_ugv_actions(k, current_state, goal_position)

            # Rollout
            def step(state, action):
                next_state = dynamics(state, action)
                return next_state, next_state

            _, traj = jax.lax.scan(step, current_state, actions)
            return traj  # (T, nx)

        trajectories = jax.vmap(sample_single_trajectory)(keys)  # (K, T, nx)
        return trajectories

    def evaluate_fsmi(trajectories: jax.Array) -> jax.Array:
        """
        Evaluate FSMI for each trajectory
        Returns: (K,) information scores
        """
        def trajectory_info(traj):
            # Per-pose information (reuse existing FSMI logic)
            def pose_info(state):
                pos = state[:2]
                theta = state[2]

                # Entropy within sensor FOV
                entropy = _compute_fov_entropy(pos, theta, grid_map, sensor_config)
                return entropy

            # Sum over trajectory (with discount for redundancy)
            entropies = jax.vmap(pose_info)(traj)
            discount = jnp.exp(-jnp.arange(len(traj)) * 0.05)  # Decay
            total_info = (entropies * discount).sum()

            return total_info

        info_scores = jax.vmap(trajectory_info)(trajectories)  # (K,)
        return info_scores

    def select_best_trajectory(trajectories, info_scores, goal_position):
        """
        Select trajectory balancing information and goal progress
        """
        # Multi-objective: 70% information, 30% goal progress
        final_positions = trajectories[:, -1, :2]  # (K, 2)
        goal_distances = jax.vmap(
            lambda pos: jnp.linalg.norm(pos - goal_position)
        )(final_positions)

        # Normalize and combine
        info_norm = info_scores / info_scores.max()
        dist_norm = 1.0 - (goal_distances / goal_distances.max())

        combined_score = 0.7 * info_norm + 0.3 * dist_norm

        best_idx = jnp.argmax(combined_score)
        return trajectories[best_idx], best_idx

    return sample_trajectories, evaluate_fsmi, select_best_trajectory
```

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Files to Create:**
1. `src/jax_mppi/dynamics/ugv.py`
   - Unicycle kinematics
   - Differential drive dynamics
   - Ackermann steering
   - Factory functions with pytree registration

2. `src/jax_mppi/costs/ugv.py`
   - Waypoint tracking cost
   - Path following cost
   - Terrain cost
   - Dynamic obstacle cost

3. `tests/test_ugv_dynamics.py`
   - Unit tests for each dynamics model
   - JIT compilation tests
   - Pytree compatibility tests

**Deliverables:**
- Working UGV dynamics with 3 model variants
- Cost functions for navigation
- Passing test suite

---

### Phase 2: Sensing and Mapping (Week 3)

**Files to Create:**
1. `src/jax_mppi/i_mppi/sensors.py`
   - LiDAR sensor model
   - Camera frustum model
   - Inverse sensor model for occupancy updates

2. `src/jax_mppi/i_mppi/ugv_map.py`
   - `UGVGridMap` dataclass
   - Terrain querying functions
   - Heightmap loading utilities
   - Map update from LiDAR

3. `tests/test_ugv_sensors.py`
4. `tests/test_ugv_map.py`

**Deliverables:**
- Multi-layer grid map (occupancy, elevation, traversability)
- LiDAR sensor model for map updates
- Terrain loading from images

---

### Phase 3: Layer 3 Controller (Week 4)

**Files to Create:**
1. `src/jax_mppi/i_mppi/ugv_planner.py`
   - `biased_ugv_mppi_command()` function
   - Composite cost function for I-MPPI
   - Reference trajectory tracking

2. `examples/ugv/basic_navigation.py`
   - Simple waypoint navigation demo
   - Uses differential drive dynamics
   - No information gathering yet

3. `tests/test_ugv_mppi.py`

**Deliverables:**
- Working biased MPPI for UGV
- Basic navigation example
- Obstacle avoidance working

---

### Phase 4: Layer 2 Planner (Week 5)

**Files to Modify/Create:**
1. `src/jax_mppi/i_mppi/ugv_fsmi.py`
   - FSMI trajectory generator for UGV
   - Horizontal FOV information computation
   - Trajectory sampling with UGV constraints

2. `examples/ugv/i_mppi_exploration.py`
   - Full two-layer I-MPPI demo
   - Unknown environment exploration
   - Information gathering visualization

3. `tests/test_ugv_i_mppi.py`

**Deliverables:**
- Complete two-layer architecture
- Information-guided exploration
- End-to-end UGV demo

---

### Phase 5: Optimization and Benchmarking (Week 6)

**Files to Create:**
1. `examples/ugv/benchmarks.py`
   - Compare MPPI vs SMPPI vs KMPPI for UGV
   - Performance metrics (coverage, efficiency, safety)

2. `examples/ugv/autotuning.py`
   - Use evosax to tune MPPI hyperparameters
   - UGV-specific parameter optimization

3. `docs/ugv_tuning_guide.md`

**Deliverables:**
- Performance benchmarks
- Autotuned parameters for each dynamics model
- Comparison with quadrotor results

---

## 10. Parallelization Strategy

### 10.1 JAX Parallelization Patterns

Following existing codebase patterns:

```python
# Sample-level parallelism (K samples)
costs = jax.vmap(rollout_cost)(actions)  # (K, T, nu) -> (K,)

# Horizon-level parallelism (T steps)
_, states = jax.lax.scan(dynamics_step, init_state, actions)  # (T, nu) -> (T, nx)

# Batch trajectory parallelism (Layer 2)
info_scores = jax.vmap(evaluate_fsmi)(trajectories)  # (100, T, nx) -> (100,)

# Sensor beam parallelism
ray_costs = jax.vmap(ray_trace)(beam_endpoints)  # (360,) -> (360,)
```

### 10.2 GPU Optimization

**Key Techniques:**
1. **Pre-allocate arrays**: Avoid dynamic shapes
2. **JIT compile**: All hot loops via `@jax.jit`
3. **XLA optimization**: Use `static_argnames` for function arguments
4. **Minimize host-device transfers**: Keep data on GPU
5. **Batch operations**: Use `vmap` instead of Python loops

**Example:**
```python
# Good: Fully JIT-compiled
@partial(jax.jit, static_argnames=['dynamics', 'cost'])
def mppi_step(state, obs, dynamics, cost):
    # All operations in JAX
    pass

# Bad: Python loop with JIT
for i in range(K):
    costs[i] = jitted_rollout(actions[i])  # Slow!

# Good: Vectorized
costs = jax.vmap(jitted_rollout)(actions)  # Fast!
```

### 10.3 Multi-Layer Asynchronous Execution

```python
# Layer 2 runs at 10 Hz (every 10 Layer 3 steps)
layer2_update_interval = 10

for step in range(num_steps):
    # Layer 3: High-frequency control
    action, mppi_state = biased_mppi_command(...)
    state = dynamics(state, action)

    # Layer 2: Low-frequency replanning
    if step % layer2_update_interval == 0:
        reference_traj = fsmi_planner.generate_reference(state, goal)

    # Update map from sensor data
    if new_sensor_data_available:
        grid_map = grid_map.update_from_lidar(state, lidar_scan, sensor_model)
```

---

## 11. Testing and Validation

### 11.1 Unit Tests

```python
# tests/test_ugv_dynamics.py
def test_diffdrive_velocity_limits():
    dynamics, config = create_diffdrive_dynamics(max_v=2.0)
    state = jnp.array([0, 0, 0, 5.0, 0])  # Exceeds max velocity
    action = jnp.array([0, 0])

    next_state = dynamics(state, action)
    assert next_state[3] <= 2.0  # Velocity clipped

def test_pytree_compatibility():
    dynamics, config = create_diffdrive_dynamics()
    state = jnp.array([0, 0, 0, 0, 0])
    action = jnp.array([1, 0])

    # Should work with jax.jit
    jitted_dynamics = jax.jit(dynamics)
    next_state = jitted_dynamics(state, action)
    assert next_state.shape == (5,)
```

### 11.2 Integration Tests

```python
# tests/test_ugv_i_mppi.py
def test_full_pipeline():
    """End-to-end test of two-layer I-MPPI"""
    # Setup
    dynamics, _ = create_diffdrive_dynamics()
    grid_map = create_simple_test_map()

    # Layer 2
    trajectories = sample_trajectories(init_state, goal, key)
    info_scores = evaluate_fsmi(trajectories, grid_map)
    best_traj, _ = select_best_trajectory(trajectories, info_scores, goal)

    # Layer 3
    action, _ = biased_mppi_command(mppi_state, init_state, best_traj, ...)

    # Verify
    assert action.shape == (2,)  # Valid UGV action
    assert jnp.isfinite(action).all()
```

### 11.3 Benchmark Tests

```python
# examples/ugv/benchmarks.py
def benchmark_sample_throughput():
    """Measure samples/second for MPPI"""
    config = MPPIConfig(num_samples=1000, horizon=50)

    start = time.time()
    for _ in range(100):
        action, state = mppi_command(...)
    elapsed = time.time() - start

    throughput = (100 * config.num_samples) / elapsed
    print(f"Throughput: {throughput:.0f} samples/sec")

    # Target: >100k samples/sec on modern GPU
    assert throughput > 100000
```

---

## 12. Example Scenarios

### Scenario 1: Warehouse Exploration
- **Environment**: Indoor with aisles and obstacles
- **UGV**: Differential drive (warehouse robot)
- **Sensor**: 270° LiDAR
- **Task**: Explore unknown space, build occupancy map
- **Info Goal**: Maximize coverage

### Scenario 2: Search and Rescue
- **Environment**: Outdoor terrain with elevation
- **UGV**: Ackermann steering (car-like)
- **Sensor**: Forward camera + LiDAR
- **Task**: Search for targets while navigating terrain
- **Info Goal**: Check high-uncertainty regions

### Scenario 3: Perimeter Security
- **Environment**: Building perimeter with dynamic obstacles
- **UGV**: Differential drive (security robot)
- **Sensor**: 360° LiDAR
- **Task**: Patrol and detect anomalies
- **Info Goal**: Maintain situational awareness

---

## 13. Open Questions and Design Decisions

### Q1: Dynamics Model Choice
**Options:**
- A. Start with unicycle (simplest)
- B. Start with differential drive (realistic)
- C. Implement all three in parallel

**Recommendation**: **B** - Differential drive is the sweet spot for most UGV applications and captures inertial effects.

---

### Q2: Sensor Fusion
**Question**: Should we support multiple sensor types simultaneously?

**Options:**
- A. Single sensor (LiDAR only)
- B. Multi-sensor fusion (LiDAR + camera)

**Recommendation**: **A** for initial implementation, design API for **B** in future.

---

### Q3: Map Update Frequency
**Question**: When to update the occupancy map from sensor data?

**Options:**
- A. Every timestep (expensive)
- B. Every N timesteps (10-20 Hz)
- C. Asynchronously in background thread

**Recommendation**: **B** initially, explore **C** for real-time systems.

---

### Q4: Reference Trajectory Time Alignment
**Question**: How to handle timing mismatch between Layer 2 (10 Hz) and Layer 3 (50 Hz)?

**Options:**
- A. Interpolate reference trajectory
- B. Use nearest timestep
- C. Extrapolate using dynamics model

**Recommendation**: **A** - Linear interpolation for smooth tracking.

---

## 14. Success Metrics

### Performance Metrics
1. **Control Frequency**: Layer 3 should run at ≥50 Hz on CPU, ≥100 Hz on GPU
2. **Planning Frequency**: Layer 2 should run at ≥5 Hz
3. **Sample Throughput**: ≥100k samples/sec for MPPI
4. **Compilation Time**: Initial JIT <5 seconds

### Quality Metrics
1. **Information Gain**: Compare exploration efficiency vs. random walk
2. **Safety**: Zero collisions in known obstacles
3. **Goal Reaching**: <5% distance to goal
4. **Smoothness**: Low jerk/acceleration for differential drive

### Code Quality Metrics
1. **Test Coverage**: ≥80% for core modules
2. **Type Safety**: Full jaxtyping annotations
3. **Documentation**: Docstrings for all public functions
4. **Examples**: Working demos for each component

---

## 15. Future Extensions

### Short-term (1-3 months)
1. **Dynamic obstacles**: Moving pedestrians/vehicles
2. **Learned dynamics**: Use neural network models
3. **Real sensor integration**: ROS2 interface for real hardware
4. **3D elevation maps**: Full 3D terrain representation

### Medium-term (3-6 months)
1. **Multi-UGV coordination**: Fleet exploration
2. **Adaptive planning horizon**: Dynamic T based on environment
3. **Semantic mapping**: Object-level understanding
4. **Active SLAM**: Joint mapping and control

### Long-term (6-12 months)
1. **End-to-end learning**: Differentiable MPPI for policy gradient
2. **Sim-to-real transfer**: Domain randomization
3. **Hardware deployment**: Real-world testing on multiple platforms
4. **Benchmark suite**: Standardized evaluation environments

---

## 16. References and Related Work

### Key Papers
1. **MPPI**: Williams et al. "Information-Theoretic Model Predictive Control" (2017)
2. **I-MPPI**: Bottero et al. "Informative Path Planning for Active Mapping" (2020)
3. **UGV Control**: Thrun et al. "Probabilistic Robotics" (2005)
4. **JAX for Robotics**: Freeman et al. "Brax - Fast Physics Simulation" (2021)

### Existing Implementations
- **mppi_numba**: CPU-based Python MPPI
- **pytorch_mppi**: PyTorch GPU MPPI (less efficient than JAX)
- **STORM**: ROS-based MPPI for manipulation

### Codebase References
- Existing quadrotor I-MPPI in `src/jax_mppi/i_mppi/`
- FSMI implementation in `fsmi.py`
- Autotuning framework in `autotune*.py`

---

## Appendix A: File Structure

```
jax_mppi/
├── src/jax_mppi/
│   ├── dynamics/
│   │   ├── ugv.py                    # NEW: UGV dynamics models
│   │   └── quadrotor.py              # EXISTING
│   ├── costs/
│   │   ├── ugv.py                    # NEW: UGV-specific costs
│   │   └── basic.py                  # EXISTING
│   ├── i_mppi/
│   │   ├── ugv_map.py                # NEW: Multi-layer grid map
│   │   ├── ugv_fsmi.py               # NEW: FSMI for UGV
│   │   ├── ugv_planner.py            # NEW: Biased MPPI
│   │   ├── sensors.py                # NEW: Sensor models
│   │   ├── map.py                    # EXISTING (GridMap base)
│   │   └── fsmi.py                   # EXISTING (Reusable logic)
│   ├── mppi.py                       # EXISTING (Core algorithm)
│   └── types.py                      # MODIFY: Add UGV types
├── examples/
│   └── ugv/                          # NEW DIRECTORY
│       ├── basic_navigation.py       # NEW: Waypoint navigation
│       ├── i_mppi_exploration.py     # NEW: Full I-MPPI demo
│       ├── benchmarks.py             # NEW: Performance tests
│       └── autotuning.py             # NEW: Parameter optimization
├── tests/
│   ├── test_ugv_dynamics.py          # NEW
│   ├── test_ugv_costs.py             # NEW
│   ├── test_ugv_map.py               # NEW
│   ├── test_ugv_sensors.py           # NEW
│   └── test_ugv_i_mppi.py            # NEW
└── docs/
    ├── i_mppi_ugv_architecture_plan.md  # THIS FILE
    └── ugv_tuning_guide.md           # NEW (Phase 5)
```

---

## Appendix B: API Design Examples

### B.1 Basic Usage Pattern

```python
import jax
import jax.numpy as jnp
from jax_mppi.dynamics.ugv import create_diffdrive_dynamics
from jax_mppi.costs.ugv import create_waypoint_cost
from jax_mppi import MPPIConfig, MPPIState, mppi

# Setup
dynamics, dyn_config = create_diffdrive_dynamics(dt=0.05)
cost = create_waypoint_cost(
    Q_pos=jnp.eye(2) * 10.0,
    Q_heading=5.0,
    waypoint=jnp.array([10.0, 5.0, 0.0])
)

# Initialize MPPI
config = MPPIConfig(
    num_samples=1000,
    horizon=30,
    nx=5,
    nu=2,
    lambda_=1.0,
    **dyn_config
)

state = MPPIState.create(config)

# Control loop
current_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
for _ in range(100):
    action, state = mppi.command(
        config, state, current_state,
        dynamics=dynamics,
        running_cost=cost,
        terminal_cost=cost
    )
    current_state = dynamics(current_state, action)
    print(f"State: {current_state}, Action: {action}")
```

### B.2 I-MPPI Usage Pattern

```python
from jax_mppi.i_mppi.ugv_planner import biased_ugv_mppi_command
from jax_mppi.i_mppi.ugv_fsmi import create_ugv_fsmi_trajectory_generator
from jax_mppi.i_mppi.ugv_map import create_ugv_map_from_heightmap

# Load map
grid_map = create_ugv_map_from_heightmap(
    occupancy_image=occupancy_img,
    heightmap_image=heightmap_img,
    resolution=0.1
)

# Setup Layer 2 (FSMI planner)
fsmi_gen = create_ugv_fsmi_trajectory_generator(
    dynamics=dynamics,
    grid_map=grid_map,
    sensor_config=lidar_config
)

# Setup Layer 3 (Biased MPPI)
mppi_state = MPPIState.create(config)

# Two-layer control loop
current_state = initial_state
reference_traj = None
key = jax.random.PRNGKey(0)

for step in range(1000):
    # Layer 2: Replan every 10 steps
    if step % 10 == 0:
        key, subkey = jax.random.split(key)
        trajectories = fsmi_gen.sample_trajectories(current_state, goal, subkey)
        info_scores = fsmi_gen.evaluate_fsmi(trajectories)
        reference_traj, _ = fsmi_gen.select_best_trajectory(
            trajectories, info_scores, goal
        )

    # Layer 3: Track reference
    action, mppi_state = biased_ugv_mppi_command(
        config, mppi_state, current_state,
        reference_trajectory=reference_traj,
        dynamics=dynamics,
        running_cost=cost,
        bias_ratio=0.5
    )

    # Execute
    current_state = dynamics(current_state, action)

    # Update map (if sensor data available)
    if step % 5 == 0:
        scan = simulate_lidar(current_state, true_map)
        grid_map = grid_map.update_from_lidar(
            current_state, scan, sensor_model
        )
```

---

## Summary

This plan provides a comprehensive roadmap for implementing parallel I-MPPI for UGVs in JAX. The key design principles are:

1. **Leverage existing code**: Reuse core MPPI, FSMI, and map infrastructure
2. **Modular design**: Separate dynamics, costs, sensors, and planners
3. **JAX-native**: Pure functional, JIT-compilable, vectorized
4. **Incremental development**: 6-week phased approach
5. **Practical focus**: Target real UGV applications (warehouse, outdoor, security)

The implementation will extend the proven quadrotor I-MPPI architecture to ground vehicles while adding UGV-specific features like terrain awareness, horizontal sensing, and constrained dynamics.

**Next Steps:**
1. Review and approve this plan
2. Begin Phase 1: UGV dynamics implementation
3. Set up testing infrastructure
4. Iterate based on experimental results

---

**Document Version**: 1.0
**Last Updated**: 2026-02-12
**Author**: Claude (Anthropic)
