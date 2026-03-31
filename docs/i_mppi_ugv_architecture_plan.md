# Parallel I-MPPI Architecture for UGV: Design Plan

**Date**: 2026-02-12
**Status**: Draft
**Target**: JAX/Python Implementation

---

## 1. Executive Summary

This document outlines the design and implementation plan for a **parallel I-MPPI architecture** for Unmanned Ground Vehicles (UGVs). Unlike hierarchical approaches, this design runs **multiple MPPI controllers in parallel**, each optimizing different objectives (information gain, goal reaching, safety), and combines their outputs for robust, information-aware navigation.

### Key Objectives
1. **Parallel MPPI Controllers**: Multiple MPPI instances running simultaneously with different cost weightings
2. **UGV Dynamics Model**: Implement ground vehicle kinematics/dynamics models
3. **Terrain-Aware Planning**: Extend grid maps to include elevation and traversability
4. **Ground-Based Sensing**: Adapt FSMI for horizontal sensor models (LiDAR, cameras)
5. **Action Fusion**: Combine parallel controllers' outputs via voting, averaging, or switching
6. **JAX Parallelism**: Leverage vmap to run N controllers simultaneously on GPU

---

## 2. Architecture Overview

### 2.1 Parallel MPPI Portfolio

Instead of hierarchical layers, we run **N parallel MPPI controllers** simultaneously, each optimizing different objectives:

```
                    Current State (x, θ, v)
                            │
                ┌───────────┴───────────┐
                │                       │
        ┌───────▼───────┐      ┌───────▼───────┐      ┌───────▼───────┐
        │  MPPI #1      │      │  MPPI #2      │      │  MPPI #N      │
        │ (Explorer)    │      │ (Exploiter)   │ ...  │ (Cautious)    │
        ├───────────────┤      ├───────────────┤      ├───────────────┤
        │ Cost Weights: │      │ Cost Weights: │      │ Cost Weights: │
        │ - Info: 0.8   │      │ - Info: 0.1   │      │ - Info: 0.3   │
        │ - Goal: 0.1   │      │ - Goal: 0.8   │      │ - Goal: 0.3   │
        │ - Safety: 0.1 │      │ - Safety: 0.1 │      │ - Safety: 0.4 │
        └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
                │                      │                      │
                │ u₁(t)               │ u₂(t)               │ uₙ(t)
                │                      │                      │
                └──────────────────────┴──────────────────────┘
                                       │
                              ┌────────▼─────────┐
                              │  Action Fusion   │
                              │  (Vote/Avg/Mix)  │
                              └────────┬─────────┘
                                       │
                                   u*(t) → UGV
```

**Key Insight**: By running multiple MPPI controllers with different cost weights in parallel, we get:
- **Robustness**: Ensemble reduces variance
- **Multi-objective**: Simultaneously optimize exploration, exploitation, and safety
- **Adaptivity**: Switch between controllers based on context
- **GPU Efficiency**: All N controllers run via single `vmap` operation

### 2.2 Portfolio Design Strategies

#### Strategy 1: Exploration-Exploitation Portfolio

| Controller | Info Weight | Goal Weight | Safety Weight | Role |
|------------|-------------|-------------|---------------|------|
| Explorer   | 0.7         | 0.1         | 0.2           | Maximize information gain |
| Balanced   | 0.4         | 0.4         | 0.2           | Balance info + goal |
| Exploiter  | 0.1         | 0.7         | 0.2           | Reach goal quickly |
| Cautious   | 0.2         | 0.2         | 0.6           | Prioritize safety |

#### Strategy 2: Regional Decomposition
- Each controller optimizes for different spatial regions
- Controller 1: Left side exploration
- Controller 2: Right side exploration
- Controller 3: Forward progress
- Fusion: Select based on information density

#### Strategy 3: Temporal Decomposition
- Short-horizon controllers (fast reactions)
- Long-horizon controllers (strategic planning)
- Different time scales operating in parallel

### 2.3 Action Fusion Methods

**Method 1: Weighted Voting**
```python
def weighted_vote_fusion(actions, costs):
    """Select action from controller with lowest cost"""
    weights = jax.nn.softmax(-costs / temperature)
    return (actions * weights[:, None]).sum(axis=0)
```

**Method 2: Contextual Switching**
```python
def context_switching_fusion(actions, costs, context):
    """Switch controller based on context (distance to goal, info density)"""
    if context['near_goal']:
        return actions[exploiter_idx]  # Use exploiter near goal
    elif context['high_info_nearby']:
        return actions[explorer_idx]   # Use explorer in info-rich regions
    else:
        return actions[balanced_idx]   # Default to balanced
```

**Method 3: Trust-Weighted Average**
```python
def trust_weighted_fusion(actions, uncertainties):
    """Weight by inverse uncertainty (lower uncertainty = higher trust)"""
    trust = 1.0 / (uncertainties + 1e-6)
    weights = trust / trust.sum()
    return (actions * weights[:, None]).sum(axis=0)
```

### 2.4 Key Components

| Component | Description | JAX Pattern |
|-----------|-------------|-------------|
| Parallel MPPI | N controllers with different costs | `jax.vmap` over N configs |
| Per-Controller Sampling | K samples per controller | Nested `vmap`: outer=N, inner=K |
| UGV Dynamics | Shared dynamics model | `jax.jit` compiled, reused by all |
| Grid Map | Shared occupancy + info map | Pytree-registered, read-only during control |
| Action Fusion | Combine N actions → 1 output | Custom fusion function |
| Sensor Model | LiDAR/Camera FOV | `jax.vmap` over beams |

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

## 7. Parallel MPPI Controllers

### 7.1 Portfolio Configuration

```python
# In src/jax_mppi/i_mppi/parallel_mppi.py

@dataclass(frozen=True)
class MPPIPortfolioConfig:
    """Configuration for N parallel MPPI controllers"""
    num_controllers: int           # N - number of parallel MPPIs
    base_config: MPPIConfig        # Shared MPPI configuration
    cost_weights: jax.Array        # (N, num_objectives) weight matrix
    fusion_method: str = "weighted_vote"  # or "switching", "average"
    fusion_temperature: float = 1.0


@dataclass(frozen=True)
class CostWeights:
    """Multi-objective cost weights for a single controller"""
    information_gain: float = 0.0  # Weight for FSMI-based information
    goal_reaching: float = 1.0     # Weight for distance to goal
    obstacle_avoidance: float = 1.0  # Weight for safety
    terrain_cost: float = 0.5      # Weight for traversability
    control_effort: float = 0.1    # Weight for smooth controls


def create_exploration_portfolio() -> list[CostWeights]:
    """
    Create a standard exploration-exploitation portfolio
    Returns 4 controllers with different behavioral biases
    """
    return [
        # Explorer: Maximize information gain
        CostWeights(
            information_gain=0.7,
            goal_reaching=0.1,
            obstacle_avoidance=0.2,
            terrain_cost=0.0,
            control_effort=0.0
        ),
        # Balanced: Equal info + goal
        CostWeights(
            information_gain=0.4,
            goal_reaching=0.4,
            obstacle_avoidance=0.2,
            terrain_cost=0.0,
            control_effort=0.0
        ),
        # Exploiter: Reach goal quickly
        CostWeights(
            information_gain=0.1,
            goal_reaching=0.7,
            obstacle_avoidance=0.2,
            terrain_cost=0.0,
            control_effort=0.0
        ),
        # Cautious: Prioritize safety
        CostWeights(
            information_gain=0.2,
            goal_reaching=0.2,
            obstacle_avoidance=0.4,
            terrain_cost=0.2,
            control_effort=0.0
        ),
    ]
```

### 7.2 Parallel Execution via vmap

```python
def parallel_mppi_command(
    portfolio_config: MPPIPortfolioConfig,
    mppi_states: list[MPPIState],    # (N,) states for each controller
    current_state: jax.Array,        # (nx,) UGV state
    goal_state: jax.Array,           # (nx,) goal
    grid_map: UGVGridMap,
    dynamics,
    key: jax.Array
):
    """
    Run N MPPI controllers in parallel and fuse their actions

    Returns:
        - fused_action: (nu,) the combined action
        - mppi_states: (N,) updated states for each controller
        - controller_actions: (N, nu) individual actions (for analysis)
        - costs: (N,) total cost for each controller
    """
    N = portfolio_config.num_controllers

    # Create cost functions for each controller
    def create_cost_fn(weights: CostWeights):
        def running_cost(state, action, t):
            # Multi-objective cost combination
            info_cost = -weights.information_gain * compute_info_gain(
                state, grid_map
            )
            goal_cost = weights.goal_reaching * jnp.linalg.norm(
                state[:2] - goal_state[:2]
            )**2
            obs_cost = weights.obstacle_avoidance * compute_obstacle_cost(
                state, grid_map
            )
            terrain_cost = weights.terrain_cost * compute_terrain_cost(
                state, grid_map
            )
            control_cost = weights.control_effort * (action @ action)

            return info_cost + goal_cost + obs_cost + terrain_cost + control_cost

        return running_cost

    # Vectorize over N controllers
    def single_controller_step(mppi_state, cost_weights):
        """Run one MPPI controller"""
        cost_fn = create_cost_fn(cost_weights)

        action, new_state = mppi.command(
            portfolio_config.base_config,
            mppi_state,
            current_state,
            dynamics=dynamics,
            running_cost=cost_fn,
            terminal_cost=cost_fn
        )

        # Compute total cost for fusion
        total_cost = evaluate_trajectory_cost(
            current_state, action, cost_fn, dynamics
        )

        return action, new_state, total_cost

    # Run all N controllers in parallel (single vmap!)
    cost_weights_array = portfolio_config.cost_weights  # (N, num_objectives)

    actions, new_states, costs = jax.vmap(single_controller_step)(
        jnp.array(mppi_states),
        cost_weights_array
    )
    # actions: (N, nu)
    # new_states: (N, MPPIState)
    # costs: (N,)

    # Fuse actions based on selected method
    fused_action = fuse_actions(
        actions,
        costs,
        method=portfolio_config.fusion_method,
        temperature=portfolio_config.fusion_temperature
    )

    return fused_action, new_states, actions, costs
```

### 7.3 Action Fusion Methods

```python
def fuse_actions(
    actions: jax.Array,      # (N, nu) actions from N controllers
    costs: jax.Array,        # (N,) costs for each controller
    method: str = "weighted_vote",
    temperature: float = 1.0
) -> jax.Array:
    """
    Combine actions from multiple controllers into single action

    Args:
        actions: (N, nu) array of actions
        costs: (N,) array of costs (lower is better)
        method: Fusion strategy
        temperature: Softmax temperature for weighted methods

    Returns:
        fused_action: (nu,) single action
    """
    if method == "weighted_vote":
        # Softmax weighting based on cost
        weights = jax.nn.softmax(-costs / temperature)
        return (actions * weights[:, None]).sum(axis=0)

    elif method == "best_only":
        # Select action from best controller
        best_idx = jnp.argmin(costs)
        return actions[best_idx]

    elif method == "average":
        # Simple average (equal weight)
        return actions.mean(axis=0)

    elif method == "median":
        # Median (robust to outliers)
        return jnp.median(actions, axis=0)

    else:
        raise ValueError(f"Unknown fusion method: {method}")


def contextual_fusion(
    actions: jax.Array,
    costs: jax.Array,
    current_state: jax.Array,
    goal_state: jax.Array,
    grid_map: UGVGridMap,
    controller_types: list[str]  # ["explorer", "balanced", "exploiter", "cautious"]
) -> jax.Array:
    """
    Context-aware controller selection

    Selects controller based on current situation:
    - Near goal → use exploiter
    - High information density → use explorer
    - Near obstacles → use cautious
    - Default → use balanced
    """
    # Compute context features
    distance_to_goal = jnp.linalg.norm(current_state[:2] - goal_state[:2])
    local_info = grid_map.query_entropy_at(current_state[:2])
    local_occupancy = grid_map.query_occupancy_at(current_state[:2])

    # Decision logic
    near_goal_threshold = 2.0
    high_info_threshold = 0.7
    near_obstacle_threshold = 0.3

    # Priority-based selection
    if distance_to_goal < near_goal_threshold:
        # Near goal: use exploiter
        idx = controller_types.index("exploiter")
    elif local_occupancy > near_obstacle_threshold:
        # Near obstacles: use cautious
        idx = controller_types.index("cautious")
    elif local_info > high_info_threshold:
        # High information: use explorer
        idx = controller_types.index("explorer")
    else:
        # Default: use balanced
        idx = controller_types.index("balanced")

    return actions[idx]
```

---

## 8. Information Gain Computation (FSMI Integration)

### 8.1 Fast Shannon Mutual Information for UGV

The existing FSMI infrastructure in `src/jax_mppi/i_mppi/fsmi.py` can be reused and adapted for UGV horizontal sensing. Instead of a separate layer, FSMI is computed **per-pose within each controller's cost function**:

```python
# In src/jax_mppi/i_mppi/ugv_fsmi.py

def compute_info_gain(
    state: jax.Array,        # (nx,) UGV pose [x, y, θ, ...]
    grid_map: UGVGridMap,
    sensor_config: LiDARConfig
) -> float:
    """
    Compute expected information gain at a given pose

    This is called within MPPI cost functions (per-timestep evaluation)

    Returns:
        info_gain: Scalar information metric (higher = more informative)
    """
    x, y, theta = state[0], state[1], state[2]

    # Compute sensor FOV in world frame
    angles = jnp.linspace(
        theta - sensor_config.fov_angle / 2,
        theta + sensor_config.fov_angle / 2,
        sensor_config.num_beams
    )

    # Sample points within sensor range
    ranges = jnp.linspace(0, sensor_config.max_range, 20)  # Discretize range

    # Create grid of beam endpoints
    beam_x = x + jnp.outer(ranges, jnp.cos(angles)).flatten()
    beam_y = y + jnp.outer(ranges, jnp.sin(angles)).flatten()

    # Query entropy at each point
    entropies = jax.vmap(
        lambda bx, by: grid_map.query_entropy_at(jnp.array([bx, by]))
    )(beam_x, beam_y)

    # Weight by distance (closer points have higher weight)
    distances = jnp.linalg.norm(
        jnp.stack([beam_x - x, beam_y - y], axis=1),
        axis=1
    )
    distance_weights = jnp.exp(-distances / sensor_config.max_range)

    # Aggregate information (weighted sum)
    total_info = (entropies * distance_weights).sum()

    return total_info


def compute_full_trajectory_fsmi(
    trajectory: jax.Array,   # (T, nx) sequence of states
    grid_map: UGVGridMap,
    sensor_config: LiDARConfig,
    discount_factor: float = 0.95
) -> float:
    """
    Compute total information gain for a full trajectory

    Used for offline trajectory evaluation or analysis
    (not in the hot control loop)

    Args:
        trajectory: (T, nx) state sequence
        discount_factor: Temporal discount (reduce double-counting)

    Returns:
        total_fsmi: Cumulative information along trajectory
    """
    # Compute per-pose information
    per_pose_info = jax.vmap(
        lambda state: compute_info_gain(state, grid_map, sensor_config)
    )(trajectory)  # (T,)

    # Apply temporal discount
    discounts = discount_factor ** jnp.arange(len(trajectory))
    total_fsmi = (per_pose_info * discounts).sum()

    return total_fsmi


def create_information_field_cache(
    grid_map: UGVGridMap,
    sensor_config: LiDARConfig,
    heading_samples: int = 16
) -> jax.Array:
    """
    Pre-compute information field for all (x, y, θ) combinations

    This creates a lookup table (H, W, heading_samples) that can be
    queried during MPPI for fast information estimates

    Returns:
        info_field: (H, W, heading_samples) array of information values
    """
    H, W = grid_map.height, grid_map.width
    headings = jnp.linspace(0, 2 * jnp.pi, heading_samples, endpoint=False)

    def compute_cell_info(i, j, theta_idx):
        # Convert grid cell to world coordinates
        x, y = grid_map.grid_to_world(i, j)
        theta = headings[theta_idx]

        state = jnp.array([x, y, theta, 0.0, 0.0])  # Dummy velocities
        return compute_info_gain(state, grid_map, sensor_config)

    # Vectorize over all grid cells and headings
    info_field = jax.vmap(jax.vmap(jax.vmap(
        compute_cell_info,
        in_axes=(None, None, 0)  # vmap over headings
    ), in_axes=(None, 0, None)),  # vmap over width
        in_axes=(0, None, None)    # vmap over height
    )(
        jnp.arange(H), jnp.arange(W), jnp.arange(heading_samples)
    )

    return info_field  # (H, W, heading_samples)
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

### Phase 3: FSMI Integration (Week 3-4)

**Files to Create:**
1. `src/jax_mppi/i_mppi/ugv_fsmi.py`
   - `compute_info_gain()` function for per-pose information
   - `compute_full_trajectory_fsmi()` for trajectory evaluation
   - `create_information_field_cache()` for lookup table acceleration
   - Horizontal FOV information computation

2. `src/jax_mppi/costs/ugv.py` (extend)
   - Information-aware cost functions
   - Multi-objective cost with info gain term

3. `examples/ugv/basic_navigation.py`
   - Simple waypoint navigation demo
   - Uses differential drive dynamics
   - Single MPPI controller

4. `tests/test_ugv_fsmi.py`
5. `tests/test_ugv_mppi.py`

**Deliverables:**
- FSMI computation for UGV horizontal sensing
- Information-aware cost functions
- Basic navigation example working
- Obstacle avoidance working

---

### Phase 4: Parallel MPPI Portfolio (Week 4-5)

**Files to Create:**
1. `src/jax_mppi/i_mppi/parallel_mppi.py`
   - `MPPIPortfolioConfig` dataclass
   - `CostWeights` dataclass
   - `parallel_mppi_command()` function (main controller)
   - `fuse_actions()` with multiple fusion strategies
   - `contextual_fusion()` for context-aware selection
   - `create_exploration_portfolio()` factory

2. `examples/ugv/parallel_i_mppi.py`
   - Full parallel I-MPPI demo
   - 4 controllers (explorer, balanced, exploiter, cautious)
   - Unknown environment exploration
   - Visualization of controller outputs

3. `tests/test_parallel_mppi.py`

**Deliverables:**
- Complete parallel MPPI architecture
- N controllers running simultaneously via vmap
- Action fusion working (weighted vote, switching, etc.)
- Information-guided exploration demo

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

### 10.1 Multi-Level Parallelization

The parallel architecture exploits JAX's nested parallelization:

```python
# Level 1: Controller parallelism (N controllers)
actions, states, costs = jax.vmap(run_single_mppi)(
    mppi_states,      # (N, MPPIState)
    cost_weights      # (N, num_objectives)
)  # -> (N, nu), (N, MPPIState), (N,)

# Level 2: Sample-level parallelism (K samples per controller)
# Inside each controller:
rollout_costs = jax.vmap(rollout_single_trajectory)(
    perturbed_actions  # (K, T, nu)
)  # -> (K,)

# Level 3: Horizon-level sequential execution (T steps per sample)
# Inside each rollout:
_, states = jax.lax.scan(dynamics_step, init_state, actions)  # (T, nu) -> (T, nx)

# Level 4: Sensor beam parallelism (within cost computation)
entropies = jax.vmap(query_entropy)(beam_endpoints)  # (num_beams,) -> (num_beams,)
```

**Total Parallelism**: `N × K × num_beams` operations in parallel!

Example with N=4, K=1000, beams=360: **1.44 million parallel evaluations** per control step!

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

### 10.3 Synchronous Parallel Execution

```python
# All N controllers run in parallel at same frequency (50-100 Hz)
for step in range(num_steps):
    # Run all controllers in parallel (single vmap operation)
    fused_action, mppi_states, individual_actions, costs = parallel_mppi_command(
        portfolio_config,
        mppi_states,
        current_state,
        goal,
        grid_map,
        dynamics,
        key
    )

    # Execute fused action
    current_state = dynamics(current_state, fused_action)

    # Update map periodically (every 5-10 steps)
    if step % map_update_interval == 0 and new_sensor_data_available:
        grid_map = grid_map.update_from_lidar(
            current_state, lidar_scan, sensor_model
        )

    # Optional: Log individual controller behavior for analysis
    if step % 100 == 0:
        log_controller_diversity(individual_actions, costs)
```

**Key Advantages**:
- All controllers synchronized (no race conditions)
- Single GPU kernel launch per step (efficient)
- Simple timing model (no async complexity)
- Easy to debug and reason about

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
# tests/test_parallel_mppi.py
def test_parallel_mppi_pipeline():
    """End-to-end test of parallel I-MPPI"""
    # Setup
    dynamics, _ = create_diffdrive_dynamics()
    grid_map = create_simple_test_map()

    # Create portfolio config
    portfolio_config = MPPIPortfolioConfig(
        num_controllers=4,
        base_config=base_mppi_config,
        cost_weights=create_exploration_portfolio(),
        fusion_method="weighted_vote"
    )

    # Initialize N controller states
    mppi_states = [MPPIState.create(base_mppi_config) for _ in range(4)]

    # Run parallel controllers
    fused_action, new_states, actions, costs = parallel_mppi_command(
        portfolio_config,
        mppi_states,
        init_state,
        goal,
        grid_map,
        dynamics,
        key
    )

    # Verify outputs
    assert fused_action.shape == (2,)  # Valid UGV action
    assert jnp.isfinite(fused_action).all()
    assert len(actions) == 4  # All 4 controllers produced actions
    assert len(costs) == 4  # All 4 controllers evaluated

def test_controller_diversity():
    """Verify controllers produce different behaviors"""
    # ... setup same as above ...

    # Controllers should produce different actions
    assert not jnp.allclose(actions[0], actions[1])  # Explorer != Balanced
    assert not jnp.allclose(actions[1], actions[2])  # Balanced != Exploiter

def test_fusion_methods():
    """Test all fusion methods work"""
    for method in ["weighted_vote", "best_only", "average", "median"]:
        action = fuse_actions(test_actions, test_costs, method=method)
        assert action.shape == (2,)
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

### B.2 Parallel I-MPPI Usage Pattern

```python
from jax_mppi.i_mppi.parallel_mppi import (
    parallel_mppi_command,
    MPPIPortfolioConfig,
    create_exploration_portfolio,
    fuse_actions
)
from jax_mppi.i_mppi.ugv_map import create_ugv_map_from_heightmap
from jax_mppi.i_mppi.sensors import create_lidar_sensor_model
from jax_mppi.dynamics.ugv import create_diffdrive_dynamics

# Load map
grid_map = create_ugv_map_from_heightmap(
    occupancy_image=occupancy_img,
    heightmap_image=heightmap_img,
    resolution=0.1
)

# Setup dynamics
dynamics, dyn_config = create_diffdrive_dynamics(dt=0.05, max_v=2.0)

# Create base MPPI config
base_config = MPPIConfig(
    num_samples=1000,
    horizon=30,
    nx=5,
    nu=2,
    lambda_=1.0,
    **dyn_config
)

# Create portfolio of 4 controllers with different objectives
portfolio_config = MPPIPortfolioConfig(
    num_controllers=4,
    base_config=base_config,
    cost_weights=create_exploration_portfolio(),  # [explorer, balanced, exploiter, cautious]
    fusion_method="weighted_vote",
    fusion_temperature=1.0
)

# Initialize states for all N controllers
mppi_states = [MPPIState.create(base_config) for _ in range(4)]

# Parallel control loop
current_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, θ, v, ω]
goal = jnp.array([10.0, 10.0, 0.0, 0.0, 0.0])
key = jax.random.PRNGKey(0)

for step in range(1000):
    key, subkey = jax.random.split(key)

    # Run all 4 controllers in parallel (single vmap!)
    fused_action, mppi_states, individual_actions, costs = parallel_mppi_command(
        portfolio_config,
        mppi_states,
        current_state,
        goal,
        grid_map,
        dynamics,
        subkey
    )

    # Execute fused action
    current_state = dynamics(current_state, fused_action)

    # Update map from sensor data
    if step % 5 == 0:
        scan = simulate_lidar(current_state, true_map)
        grid_map = grid_map.update_from_lidar(
            current_state, scan, sensor_model
        )

    # Optional: Visualize controller diversity
    if step % 100 == 0:
        print(f"Step {step}:")
        print(f"  Explorer action: {individual_actions[0]}")
        print(f"  Balanced action: {individual_actions[1]}")
        print(f"  Exploiter action: {individual_actions[2]}")
        print(f"  Cautious action: {individual_actions[3]}")
        print(f"  Fused action: {fused_action}")
        print(f"  Costs: {costs}")
```

### B.3 Custom Portfolio Configuration

```python
# Define custom cost weights for specific application
custom_weights = [
    # Warehouse robot: prioritize safety and efficiency
    CostWeights(
        information_gain=0.2,
        goal_reaching=0.6,
        obstacle_avoidance=0.5,
        terrain_cost=0.0,
        control_effort=0.1
    ),
    # Search robot: maximize exploration
    CostWeights(
        information_gain=0.8,
        goal_reaching=0.1,
        obstacle_avoidance=0.3,
        terrain_cost=0.0,
        control_effort=0.0
    ),
    # Patrol robot: balance coverage and safety
    CostWeights(
        information_gain=0.5,
        goal_reaching=0.3,
        obstacle_avoidance=0.4,
        terrain_cost=0.1,
        control_effort=0.1
    ),
]

# Create custom portfolio
custom_portfolio = MPPIPortfolioConfig(
    num_controllers=3,
    base_config=base_config,
    cost_weights=jnp.array([w.to_array() for w in custom_weights]),
    fusion_method="contextual",  # Use context-aware switching
)

# Use contextual fusion for intelligent controller selection
action = contextual_fusion(
    individual_actions,
    costs,
    current_state,
    goal,
    grid_map,
    controller_types=["warehouse", "search", "patrol"]
)
```

---

## Summary

This plan provides a comprehensive roadmap for implementing **parallel I-MPPI** for UGVs in JAX. The key design principles are:

1. **True Parallelism**: Multiple MPPI controllers running simultaneously (not hierarchical layers)
2. **Portfolio Approach**: Diverse controllers with different objectives (exploration, exploitation, safety)
3. **Action Fusion**: Smart combination strategies (weighted voting, contextual switching, averaging)
4. **JAX-Native**: Pure functional, JIT-compilable, nested vmap for massive parallelism
5. **Modular Design**: Separate dynamics, costs, sensors, and fusion logic
6. **Incremental Development**: 5-week phased approach
7. **Practical Focus**: Target real UGV applications (warehouse, search & rescue, patrol)

### Key Innovations

**vs. Hierarchical I-MPPI:**
- ✅ All controllers run at same frequency (simpler timing)
- ✅ Single vmap operation (more GPU-efficient)
- ✅ No reference trajectory tracking overhead
- ✅ Ensemble robustness through diversity
- ✅ Adaptive behavior via contextual fusion

**Architecture Highlights:**
- **N = 4 controllers**: Explorer, Balanced, Exploiter, Cautious
- **K = 1000 samples** per controller
- **Total parallelism**: 4 × 1000 × 360 beams = **1.44M operations** per step
- **Target frequency**: 50-100 Hz on GPU

**UGV-Specific Features:**
- Differential drive, unicycle, and Ackermann dynamics
- Multi-layer grid maps (occupancy, elevation, traversability)
- Horizontal LiDAR and camera sensor models
- Terrain-aware cost functions

**Next Steps:**
1. Review and approve this plan
2. Begin Phase 1: UGV dynamics implementation
3. Phase 2: Sensing and mapping
4. Phase 3: FSMI integration
5. Phase 4: Parallel MPPI portfolio (main innovation!)
6. Phase 5: Optimization and benchmarking

---

**Document Version**: 2.0 (Revised for Parallel Architecture)
**Last Updated**: 2026-02-12
**Author**: Claude (Anthropic)
