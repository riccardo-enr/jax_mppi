from dataclasses import dataclass, field, replace
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .map import GridMap, world_to_grid


@dataclass
class FSMIConfig:
    # Legacy rectangular zone parameters (for backward compatibility)
    info_threshold: float = 20.0
    ref_speed: float = 2.0
    info_weight: float = 10.0
    motion_weight: float = 1.0

    # Grid-based FSMI parameters
    use_grid_fsmi: bool = True  # Toggle between grid FSMI and legacy geometric

    # Sensor parameters (Zhang et al. 2020)
    fov_rad: float = 1.57  # 90 degrees FOV
    num_beams: int = 16  # Rays per scan (keep low for MPPI speed)
    max_range: float = 5.0  # Meters
    ray_step: float = 0.1  # Ray integration resolution (10cm cells)
    sigma_range: float = 0.15  # Sensor noise std dev (Gaussian model)

    # Inverse Sensor Model (Log Odds)
    # p_occ=0.7 -> log(0.7/0.3) ≈ 0.85
    # p_emp=0.4 -> log(0.4/0.6) ≈ -0.4
    inv_sensor_model_occ: float = 0.85
    inv_sensor_model_emp: float = -0.4

    # FSMI optimization parameters
    gaussian_truncation_sigma: float = 3.0  # Truncate G_kj beyond 3σ
    trajectory_subsample_rate: int = 5  # Evaluate FSMI every N steps

    # Legacy geometric parameters (kept for backward compatibility)
    sensor_range: float = 6.0
    fov_half_angle_deg: float = 60.0
    info_scale: float = 20.0
    distance_sigma: float = 1.0
    info_depletion_rate: float = 20.0
    info_depletion_sigma: float = 1.0

    goal_pos: jax.Array = field(
        default_factory=lambda: jnp.array([9.0, 5.0, -2.0])
    )
    # FSMI params
    num_rays: int = 36
    ray_max_steps: int = 50
    sensor_range: float = 10.0
    # Weights
    gain_weight: float = 1.0
    dist_weight: float = 0.5
    min_gain_threshold: float = 2.0


def _entropy_proxy(p: jax.Array) -> jax.Array:
    """Returns proxy for entropy: 1.0 at p=0.5, 0.0 at p=0,1."""
    # 4 * p * (1 - p)
    return 4.0 * p * (1.0 - p)


@partial(jax.jit, static_argnames=["num_steps"])
def cast_ray_fsmi(
    origin: jax.Array,  # (2,)
    angle: float,
    map_grid: jax.Array,  # (H, W)
    map_origin: jax.Array,
    map_resolution: float,
    num_steps: int = 50,
) -> float:
    """
    Cast a single ray and compute information gain (entropy reduction).
    """
    step_size = map_resolution
    dx = jnp.cos(angle) * step_size
    dy = jnp.sin(angle) * step_size
    step_vec = jnp.array([dx, dy])

    def step_fn(carry, _):
        pos, visibility, current_gain = carry

        # Get grid index
        grid_pos = world_to_grid(pos, map_origin, map_resolution)
        ix = jnp.floor(grid_pos[0]).astype(jnp.int32)
        iy = jnp.floor(grid_pos[1]).astype(jnp.int32)

        # Check bounds
        h, w = map_grid.shape
        in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

        # Get cell probability
        # If out of bounds, treat as occupied (prob=1) to stop ray
        p = jax.lax.select(in_bounds, map_grid[iy, ix], 1.0)

        # Info Gain
        # Gain = visibility * Entropy(p)
        # Only unknown cells (p=0.5) contribute high entropy
        h_val = _entropy_proxy(p)
        gain_inc = visibility * h_val

        # Update visibility
        # Visibility reduces by probability of occlusion (p)
        # If p=1 (wall), vis becomes 0
        # If p=0 (free), vis stays same
        # If p=0.5 (unknown), vis reduces by 0.5
        new_vis = visibility * (1.0 - p)

        new_pos = pos + step_vec

        return (new_pos, new_vis, current_gain + gain_inc), None

    init_val = (origin, 1.0, 0.0)  # pos, visibility, gain
    (final_pos, final_vis, total_gain), _ = jax.lax.scan(
        step_fn, init_val, None, length=num_steps
    )

    return total_gain


@partial(jax.jit, static_argnames=["num_rays", "num_steps"])
def compute_fsmi_gain(
    origin: jax.Array,  # (2,)
    map_grid: jax.Array,
    map_origin: jax.Array,
    map_resolution: float,
    num_rays: int = 36,
    num_steps: int = 50,
) -> float:
    """
    Compute total FSMI gain for a viewpoint by casting rays in circle.
    """
    angles = jnp.linspace(0, 2 * jnp.pi, num_rays, endpoint=False)

    # vmap over angles
    def ray_fn(a):
        return cast_ray_fsmi(
            origin, a, map_grid, map_origin, map_resolution, num_steps
        )

    gains = jax.vmap(ray_fn)(angles)
    return jnp.sum(gains)


@partial(jax.jit, static_argnames=["width", "height", "resolution"])
def _update_grid_zones(
    grid: jax.Array,
    origin: jax.Array,
    resolution: float,
    info_zones: jax.Array,
    info_levels: jax.Array,
    width: int,
    height: int,
) -> jax.Array:
    """Updates grid probabilities based on info levels."""
    y_range = jnp.arange(height)
    x_range = jnp.arange(width)
    Y, X = jnp.meshgrid(y_range, x_range, indexing="ij")

    world_X = origin[0] + (X + 0.5) * resolution
    world_Y = origin[1] + (Y + 0.5) * resolution
    coords = jnp.stack([world_X.flatten(), world_Y.flatten()], axis=1)

    def get_new_prob(p_curr, pos):
        # We only update if p_curr != 1.0 (Wall)
        is_wall = p_curr >= 0.99

        zone_prob = 0.0
        for i in range(info_zones.shape[0]):
            z = info_zones[i]
            level = info_levels[i]
            center = z[:2]
            size = z[2:4]
            half_size = size / 2.0
            d = jnp.abs(pos - center) - half_size
            in_zone = jnp.all(d <= 0.0)
            p = 0.5 * jnp.clip(level / 100.0, 0.0, 1.0)
            zone_prob = jax.lax.select(in_zone, p, zone_prob)

        return jax.lax.select(is_wall, 1.0, zone_prob)

    new_vals = jax.vmap(lambda p_c, pos: get_new_prob(p_c, pos))(
        grid.flatten(), coords
    )
    return new_vals.reshape(height, width)


@register_pytree_node_class
@dataclass
class FSMITrajectoryGenerator:
    """
    Layer 2: FSMI-driven trajectory generator.

    Generates reference trajectories by selecting targets that maximize
    information gain while minimizing motion cost.

    config: FSMIConfig
    info_zones: jax.Array
    grid_map: GridMap

    def tree_flatten(self):
        return (self.info_zones, self.grid_map), (self.config,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        info_zones, grid_map = children
        config = aux[0]
        return cls(config=config, info_zones=info_zones, grid_map=grid_map)

    def update_map(self, info_levels: jax.Array):
        """Update the internal grid map based on current info levels."""
        new_grid = _update_grid_zones(
            self.grid_map.grid,
            self.grid_map.origin,
            self.grid_map.resolution,
            self.info_zones,
            info_levels,
            self.grid_map.width,
            self.grid_map.height,
        )
        new_map = replace(self.grid_map, grid=new_grid)
        return replace(self, grid_map=new_map)

    def select_target(self, current_pos: jax.Array):
        """Select best target using FSMI."""
        # Candidates: Info Zones centers + Goal
        candidates = self.info_zones[:, :2]  # (N, 2)
        goal = self.config.goal_pos[:2]

        # Evaluate candidates
        def score_candidate(cand_pos, is_goal):
            gain = compute_fsmi_gain(
                cand_pos,
                self.grid_map.grid,
                self.grid_map.origin,
                self.grid_map.resolution,
                self.config.num_rays,
                self.config.ray_max_steps,
            )
            dist = jnp.linalg.norm(cand_pos - current_pos[:2])

            score = (
                self.config.gain_weight * gain - self.config.dist_weight * dist
            )
            return score, gain

        # Score zones
        zone_scores, zone_gains = jax.vmap(lambda p: score_candidate(p, False))(
            candidates
        )

        # Score goal
        goal_score, goal_gain = score_candidate(goal, True)

        # Find best zone
        best_zone_idx = jnp.argmax(zone_scores)
        best_zone_gain = zone_gains[best_zone_idx]

        go_to_zone = best_zone_gain > self.config.min_gain_threshold

        target_pos_2d = jax.lax.select(
            go_to_zone, candidates[best_zone_idx], goal
        )

        # Construct 3D target (z fixed at -2.0)
        target_pos = jnp.array([target_pos_2d[0], target_pos_2d[1], -2.0])

        # Mode: 1 = Exploring, 0 = Homming
        mode = jax.lax.select(go_to_zone, 1, 0)

        return target_pos, mode
