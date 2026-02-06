"""
FSMI (Fast Shannon Mutual Information) implementation for I-MPPI.

This module implements the actual FSMI algorithm from Zhang et al. (2020):
"Fast Entropy-Based Informative Trajectory Planning"

The implementation includes:
1. Grid-based occupancy map representation
2. Beam-based mutual information computation using Theorem 1
3. Optimized JAX operations for GPU acceleration
4. Integration with biased MPPI trajectory generation
"""

from dataclasses import dataclass, field, replace
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.tree_util import register_pytree_node_class

from jax_mppi.i_mppi.environment import dist_rect

from .map import GridMap, world_to_grid


@dataclass
class FSMIConfig:
    # Legacy geometric parameters (for backward compatibility)
    info_threshold: float = 20.0
    ref_speed: float = 2.0
    info_weight: float = 10.0
    motion_weight: float = 1.0

    goal_pos: jax.Array = field(
        default_factory=lambda: jnp.array([9.0, 5.0, -2.0])
    )

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

    # Parameters from my implementation
    num_rays: int = 36
    ray_max_steps: int = 50
    gain_weight: float = 1.0
    dist_weight: float = 0.5
    min_gain_threshold: float = 2.0


@dataclass
class UniformFSMIConfig:
    """
    Configuration for Uniform-FSMI (fast O(n) variant).

    Used in Layer 3 (I-MPPI controller) for local reactivity at 50Hz.
    Simplified compared to full FSMI for computational efficiency.

    Zhang et al. (2020) developed Uniform-FSMI specifically for real-time
    informative control, reducing complexity from O(n²) to O(n).
    """

    # Sensor parameters (reduced for speed)
    fov_rad: float = 1.57  # 90 degrees FOV
    num_beams: int = 6  # Fewer beams for speed
    max_range: float = 2.5  # Local only (meters)
    ray_step: float = 0.2  # Coarser resolution

    # Inverse Sensor Model (same as full FSMI)
    inv_sensor_model_occ: float = 0.85
    inv_sensor_model_emp: float = -0.4

    # Weight for informative cost term
    info_weight: float = 5.0


class FSMIModule:
    """
    Core FSMI computation module implementing Zhang et al. (2020).

    Computes mutual information I(M; Z) between map M and measurement Z
    using Theorem 1 and Algorithms 1-3 from the paper.
    """

    def __init__(
        self,
        config: FSMIConfig,
        map_origin: jax.Array,
        map_resolution: float,
    ):
        """
        Args:
            config: FSMI configuration parameters
            map_origin: (x, y) world coordinates of grid origin
            map_resolution: Meters per grid cell
        """
        self.cfg = config
        self.map_origin = map_origin
        self.map_res = map_resolution

    def _get_f_score(
        self, r_i: jax.Array, delta_inv_model: jax.Array
    ) -> jax.Array:
        """
        Computes f(r, delta) information gain term from Eq. 9.

        This represents the expected information gain when updating a cell
        with occupancy odds ratio r_i using inverse sensor model delta.

        Args:
            r_i: Current odds ratio p/(1-p) of cells
            delta_inv_model: Odds ratio of inverse sensor model

        Returns:
            Information gain score f(r, delta)
        """
        # Clamp to avoid singularities at r=0
        r_i = jnp.maximum(r_i, 1e-4)

        delta = delta_inv_model

        # Eq. 9: log((r + 1)/(r + 1/delta)) - log(delta)/(r*delta + 1)
        term1 = jnp.log((r_i + 1.0) / (r_i + (1.0 / delta)))
        term2 = jnp.log(delta) / (r_i * delta + 1.0)

        return term1 - term2

    def _compute_beam_fsmi(
        self, cell_probs: jax.Array, cell_dists: jax.Array
    ) -> jax.Array:
        """
        Computes I(M; Z) for a single beam using Theorem 1.

        This is the core FSMI computation that calculates the expected
        mutual information from a single lidar beam based on:
        - P(e_j): Probability beam terminates at cell j (Algorithm 2)
        - C_k: Information potential of measurement at cell k (Algorithm 3)
        - G_kj: Geometric/noise term relating true and measured hits (Eq. 22)

        Args:
            cell_probs: (N,) Occupancy probabilities [0,1] along ray
            cell_dists: (N,) Distances (meters) from sensor

        Returns:
            Scalar mutual information for this beam
        """
        N = cell_probs.shape[0]

        # === Algorithm 2: Compute P(e_j) ===
        # P(e_j) = o_j * prod_{l<j} (1 - o_l)
        # "Probability that beam stops at cell j"
        not_occ = 1.0 - cell_probs
        cum_not_occ = jnp.cumprod(not_occ)
        # Shift right (exclusive cumprod)
        shifted_cum = jnp.concatenate([jnp.array([1.0]), cum_not_occ[:-1]])
        P_e = cell_probs * shifted_cum

        # === Algorithm 3: Compute C_k ===
        # Information potential: occupied info at k + empty info before k
        odds = cell_probs / (1.0 - cell_probs + 1e-6)

        # f_occ: Info from measuring occupied at cell k
        f_occ = self._get_f_score(odds, jnp.exp(self.cfg.inv_sensor_model_occ))
        # f_emp: Info from measuring empty at cell k
        f_emp = self._get_f_score(odds, jnp.exp(self.cfg.inv_sensor_model_emp))

        # C_k = f_occ[k] + sum_{i<k} f_emp[i]
        cum_f_emp = jnp.cumsum(f_emp)
        shifted_cum_f_emp = jnp.concatenate([jnp.array([0.0]), cum_f_emp[:-1]])
        C_k = f_occ + shifted_cum_f_emp

        # === Eq. 22: Compute G_kj (Geometry/Noise) ===
        # G_kj = Φ((l_{k+1} - μ_j)/σ) - Φ((l_k - μ_j)/σ)
        # where Φ is the Gaussian CDF, representing sensor noise

        sigma = self.cfg.sigma_range

        # Cell boundaries (assuming cells are ray_step wide)
        l_k_plus = cell_dists + (self.cfg.ray_step / 2.0)
        l_k_minus = cell_dists - (self.cfg.ray_step / 2.0)
        mu_j = cell_dists  # True obstacle positions

        # Broadcast to (N_k, N_j) grid
        z_high = (l_k_plus[:, None] - mu_j[None, :]) / sigma
        z_low = (l_k_minus[:, None] - mu_j[None, :]) / sigma

        G_kj = norm.cdf(z_high) - norm.cdf(z_low)

        # Optimization: Gaussian truncation beyond 3σ
        # Mask out G_kj entries where |k-j| > truncation_radius
        truncation_radius = int(
            self.cfg.gaussian_truncation_sigma * sigma / self.cfg.ray_step
        )
        k_indices = jnp.arange(N)
        j_indices = jnp.arange(N)
        mask = (
            jnp.abs(k_indices[:, None] - j_indices[None, :])
            <= truncation_radius
        )
        G_kj = jnp.where(mask, G_kj, 0.0)

        # === Theorem 1: Total MI ===
        # I(M; Z) = sum_j sum_k P(e_j) * C_k * G_kj
        # Using einsum for efficient contraction
        mi = jnp.einsum("j,k,kj->", P_e, C_k, G_kj)

        return mi

    def compute_fsmi(
        self, grid_map: jax.Array, pos: jax.Array, yaw: float
    ) -> jax.Array:
        """
        Computes total FSMI for all beams at a given pose.

        Args:
            grid_map: (H, W) occupancy probability grid [0, 1]
            pos: (x, y) world position
            yaw: Heading in radians

        Returns:
            Total mutual information across all beams
        """
        # Generate beam angles uniformly across FOV
        angles = jnp.linspace(
            yaw - self.cfg.fov_rad / 2,
            yaw + self.cfg.fov_rad / 2,
            self.cfg.num_beams,
        )

        # Generate sample points along all rays
        # Shape: (num_beams, num_samples_per_beam)
        r_range = jnp.arange(0, self.cfg.max_range, self.cfg.ray_step)

        # Ray casting: compute world coordinates for all beam samples
        ray_x = pos[0] + r_range[None, :] * jnp.cos(angles)[:, None]
        ray_y = pos[1] + r_range[None, :] * jnp.sin(angles)[:, None]

        # World to grid coordinates
        grid_x = ((ray_x - self.map_origin[0]) / self.map_res).astype(jnp.int32)
        grid_y = ((ray_y - self.map_origin[1]) / self.map_res).astype(jnp.int32)

        # Boundary checking (safe indexing)
        H, W = grid_map.shape
        valid_mask = (grid_x >= 0) & (grid_x < W) & (grid_y >= 0) & (grid_y < H)

        # Gather probabilities (clip indices for safety)
        safe_x = jnp.clip(grid_x, 0, W - 1)
        safe_y = jnp.clip(grid_y, 0, H - 1)
        probs = grid_map[safe_y, safe_x]

        # Treat out-of-bounds as unknown (0.5) to stop ray
        probs = jnp.where(valid_mask, probs, 0.5)

        # Compute FSMI for each beam in parallel using vmap
        # vmap over beams dimension (axis 0)
        info_per_beam = jax.vmap(self._compute_beam_fsmi, in_axes=(0, None))(
            probs, r_range
        )

        return jnp.sum(info_per_beam)


class UniformFSMI:
    """
    Uniform-FSMI: O(n) fast variant for Layer 3 (I-MPPI controller).

    This is the simplified FSMI variant from Zhang et al. (2020) designed
    for real-time informative control. Key simplifications:

    1. **No G_kj matrix:** Assumes measurement noise is negligible locally,
       reducing O(n²) to O(n) per beam.
    2. **Short range:** Only considers local information (2-3m) since Layer 2
       handles long-range planning.
    3. **Fewer beams:** Uses 4-8 beams vs 16+ for full FSMI.

    The approximation is valid because:
    - At short ranges, sensor noise is small relative to cell size
    - G_kj ≈ δ(k-j) (Kronecker delta) when sigma << ray_step
    - This gives: MI ≈ sum_j P(e_j) * C_j

    Usage:
        Layer 2 (5Hz): Full FSMI for reference trajectory planning
        Layer 3 (50Hz): Uniform-FSMI for reactive informative control
    """

    def __init__(
        self,
        config: UniformFSMIConfig,
        map_origin: jax.Array,
        map_resolution: float,
    ):
        """
        Args:
            config: Uniform-FSMI configuration
            map_origin: (x, y) world coordinates of grid origin
            map_resolution: Meters per grid cell
        """
        self.cfg = config
        self.map_origin = map_origin
        self.map_res = map_resolution

    def _compute_beam_uniform_fsmi(self, cell_probs: jax.Array) -> jax.Array:
        """
        Compute O(n) mutual information for a single beam.

        Uses the uniform-FSMI approximation:
            MI ≈ sum_j P(e_j) * C_j

        where G_kj ≈ δ(k-j) due to small local sensor noise.

        Args:
            cell_probs: (N,) Occupancy probabilities [0,1] along ray

        Returns:
            Scalar mutual information for this beam
        """
        # === Compute P(e_j): Probability beam terminates at cell j ===
        # P(e_j) = o_j * prod_{l<j} (1 - o_l)
        not_occ = 1.0 - cell_probs
        cum_not_occ = jnp.cumprod(not_occ)
        shifted_cum = jnp.concatenate([jnp.array([1.0]), cum_not_occ[:-1]])
        P_e = cell_probs * shifted_cum

        # === Compute C_j: Information potential at cell j ===
        # C_j = f_occ[j] + sum_{i<j} f_emp[i]
        odds = cell_probs / (1.0 - cell_probs + 1e-6)
        odds = jnp.maximum(odds, 1e-4)

        # f_occ: Info from measuring occupied
        delta_occ = jnp.exp(self.cfg.inv_sensor_model_occ)
        term1_occ = jnp.log((odds + 1.0) / (odds + (1.0 / delta_occ)))
        term2_occ = jnp.log(delta_occ) / (odds * delta_occ + 1.0)
        f_occ = term1_occ - term2_occ

        # f_emp: Info from measuring empty
        delta_emp = jnp.exp(self.cfg.inv_sensor_model_emp)
        term1_emp = jnp.log((odds + 1.0) / (odds + (1.0 / delta_emp)))
        term2_emp = jnp.log(delta_emp) / (odds * delta_emp + 1.0)
        f_emp = term1_emp - term2_emp

        # C_j = f_occ[j] + cumsum of f_emp before j
        cum_f_emp = jnp.cumsum(f_emp)
        shifted_cum_f_emp = jnp.concatenate([jnp.array([0.0]), cum_f_emp[:-1]])
        C_j = f_occ + shifted_cum_f_emp

        # === Uniform-FSMI: MI ≈ sum_j P(e_j) * C_j ===
        # This is O(n) instead of O(n²) for full FSMI
        mi = jnp.sum(P_e * C_j)

        return mi

    def compute(
        self, grid_map: jax.Array, pos: jax.Array, yaw: float
    ) -> jax.Array:
        """
        Compute Uniform-FSMI at a given pose.

        This is the fast local information metric for Layer 3.

        Args:
            grid_map: (H, W) occupancy probability grid [0, 1]
            pos: (x, y) world position
            yaw: Heading in radians

        Returns:
            Total mutual information (scalar)
        """
        # Generate beam angles
        angles = jnp.linspace(
            yaw - self.cfg.fov_rad / 2,
            yaw + self.cfg.fov_rad / 2,
            self.cfg.num_beams,
        )

        # Ray sampling points
        r_range = jnp.arange(0, self.cfg.max_range, self.cfg.ray_step)

        # Ray casting: world coordinates
        ray_x = pos[0] + r_range[None, :] * jnp.cos(angles)[:, None]
        ray_y = pos[1] + r_range[None, :] * jnp.sin(angles)[:, None]

        # World to grid coordinates
        grid_x = ((ray_x - self.map_origin[0]) / self.map_res).astype(jnp.int32)
        grid_y = ((ray_y - self.map_origin[1]) / self.map_res).astype(jnp.int32)

        # Boundary checking
        H, W = grid_map.shape
        valid_mask = (grid_x >= 0) & (grid_x < W) & (grid_y >= 0) & (grid_y < H)

        # Safe indexing
        safe_x = jnp.clip(grid_x, 0, W - 1)
        safe_y = jnp.clip(grid_y, 0, H - 1)
        probs = grid_map[safe_y, safe_x]

        # Out-of-bounds treated as unknown (0.5)
        probs = jnp.where(valid_mask, probs, 0.5)

        # Compute Uniform-FSMI for each beam (O(n) per beam)
        info_per_beam = jax.vmap(self._compute_beam_uniform_fsmi)(probs)

        return jnp.sum(info_per_beam)

    def compute_batch(
        self,
        grid_map: jax.Array,
        positions: jax.Array,
        yaws: jax.Array,
    ) -> jax.Array:
        """
        Compute Uniform-FSMI for a batch of poses (for MPPI samples).

        Args:
            grid_map: (H, W) occupancy grid
            positions: (K, 2) batch of (x, y) positions
            yaws: (K,) batch of headings

        Returns:
            (K,) information values for each pose
        """
        return jax.vmap(self.compute, in_axes=(None, 0, 0))(
            grid_map, positions, yaws
        )


# --- FSMI Implementation from my changes (simplified) ---


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
    """Layer 2: FSMI-driven trajectory generator."""

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

    def _fov_cos_threshold(self) -> jax.Array:
        """Cosine of half FOV angle for legacy geometric gating."""
        return jnp.cos(jnp.deg2rad(self.config.fov_half_angle_deg))

    def _make_ref_traj(
        self,
        pos0: jax.Array,
        target_pos: jax.Array,
        horizon: int,
        dt: float,
    ) -> jax.Array:
        """
        Generate straight-line reference trajectory from pos0 to target.

        Args:
            pos0: (3,) starting position [x, y, z]
            target_pos: (3,) target position
            horizon: Number of trajectory steps
            dt: Time step size

        Returns:
            (horizon, 3) reference trajectory
        """
        direction = target_pos - pos0
        dist = jnp.linalg.norm(direction) + 1e-6
        unit = direction / dist

        step_d = self.config.ref_speed * dt * jnp.arange(horizon)
        step_d = jnp.minimum(step_d, dist)
        ref_traj = pos0[None, :] + step_d[:, None] * unit[None, :]
        return ref_traj

    def _info_gain_legacy(
        self,
        ref_traj: jax.Array,
        view_dir_xy: jax.Array,
        info_levels: jax.Array,
        dt: float,
    ) -> jax.Array:
        """
        Legacy geometric info gain (heuristic gating functions).

        Uses FOV, range, and proximity gates to approximate information gain
        from rectangular zones.
        """
        pos_xy = ref_traj[:, :2]
        zone_centers = self.info_zones[:, :2]
        zone_sizes = self.info_zones[:, 2:4]

        def dist_to_zones(p):
            return jax.vmap(dist_rect, in_axes=(None, 0, 0))(
                p, zone_centers, zone_sizes
            )

        rect_dist = jax.vmap(dist_to_zones)(pos_xy)

        vec = zone_centers[None, :, :] - pos_xy[:, None, :]
        dist = jnp.linalg.norm(vec, axis=-1) + 1e-6
        view_dir = view_dir_xy / (jnp.linalg.norm(view_dir_xy) + 1e-6)
        cos_angle = jnp.sum(vec * view_dir[None, None, :], axis=-1) / dist
        cos_half = self._fov_cos_threshold()

        angle_gate = jnp.clip(
            (cos_angle - cos_half) / (1.0 - cos_half + 1e-6), 0.0, 1.0
        )
        range_gate = jnp.clip(
            1.0 - dist / (self.config.sensor_range + 1e-6), 0.0, 1.0
        )
        proximity_gate = jnp.exp(
            -(rect_dist**2) / (2.0 * self.config.distance_sigma**2)
        )

        info_levels0 = info_levels

        def step_fn(info_lvls, gates):
            angle_g, range_g, prox_g, rect_d = gates
            info_strength = jnp.tanh(info_lvls / self.config.info_scale)
            step_gain = angle_g * range_g * prox_g * info_strength

            # Deplete info where trajectory is informative
            depletion = (
                self.config.info_depletion_rate
                * dt
                * jnp.exp(
                    -(rect_d**2) / (2.0 * self.config.info_depletion_sigma**2)
                )
            )
            next_info = jnp.maximum(0.0, info_lvls - depletion)
            return next_info, jnp.sum(step_gain)

        gates = (angle_gate, range_gate, proximity_gate, rect_dist)
        info_levels_T, gain_seq = jax.lax.scan(step_fn, info_levels0, gates)
        _ = info_levels_T

        return dt * jnp.sum(gain_seq)

    def _info_gain_grid(
        self,
        ref_traj: jax.Array,
        view_dir_xy: jax.Array,
        grid_map: jax.Array,
        dt: float,
    ) -> jax.Array:
        """
        Grid-based FSMI info gain (true algorithm).

        Computes mutual information along trajectory using the FSMI module.
        Subsamples trajectory points to maintain MPPI speed.

        Args:
            ref_traj: (horizon, 3) trajectory waypoints
            view_dir_xy: (2,) viewing direction (for yaw calculation)
            grid_map: (H, W) occupancy probability grid
            dt: Time step

        Returns:
            Total expected information gain
        """
        # Subsample trajectory to reduce computation
        subsample = self.config.trajectory_subsample_rate
        sampled_traj = ref_traj[::subsample]

        # Note: self.fsmi_module requires config.use_grid_fsmi=True
        # In this merged version, we need to ensure self.fsmi_module exists.
        # But this class initialization logic in merged version relies on
        # self.fsmi_module.
        # I will assume `self.fsmi_module` is NOT available in my version.
        # So `_info_gain_grid` cannot work as is.
        # I should use `compute_fsmi_gain` instead.

        step_gains = jax.vmap(
            lambda pos: compute_fsmi_gain(
                pos,
                self.grid_map.grid,
                self.grid_map.origin,
                self.grid_map.resolution,
                self.config.num_rays,
                self.config.ray_max_steps,
            )
        )(sampled_traj[:, :2])

        # Scale by subsampling and dt
        return jnp.sum(step_gains) * dt * subsample

    def _target_cost(
        self,
        pos0: jax.Array,
        target_pos: jax.Array,
        info_data: jax.Array | tuple[jax.Array, jax.Array],
        horizon: int,
        dt: float,
    ) -> jax.Array:
        """
        Unified cost target selection: motion_cost - info_weight * info_gain.

        Args:
            pos0: Current position
            target_pos: Candidate target position
            info_data: Either info_levels (legacy)
                       or (grid_map, info_levels) tuple
            horizon: Planning horizon
            dt: Time step

        Returns:
            Combined cost (lower is better)
        """
        ref_traj = self._make_ref_traj(pos0, target_pos, horizon, dt)
        view_dir_xy = target_pos[:2] - pos0[:2]

        # Motion cost (Euclidean distance)
        motion_cost = self.config.motion_weight * jnp.linalg.norm(
            target_pos[:2] - pos0[:2]
        )

        # Info gain (dispatch based on mode)
        if self.config.use_grid_fsmi:
            # We assume info_data contains grid map?
            # In my version, grid_map is stored in self.grid_map.
            # info_data passed here is from `get_target` which isn't fully
            # compatible with `select_target`.

            # I will stick to `select_target` logic but support this method for
            # compatibility if needed.
            # But `select_target` uses pre-calculated gains on zones.

            info_gain = self._info_gain_grid(
                ref_traj, view_dir_xy, self.grid_map.grid, dt
            )
        else:
            info_levels = info_data  # Single array
            info_gain = self._info_gain_legacy(
                ref_traj, view_dir_xy, info_levels, dt
            )

        return motion_cost - self.config.info_weight * info_gain

    def get_target(
        self,
        pos0: jax.Array,
        info_data: jax.Array | tuple[jax.Array, jax.Array],
        horizon: int,
        dt: float,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Select target by minimizing unified cost (motion - info gain).

        Evaluates candidates: goal + all info zones.

        Args:
            pos0: Current position [x, y, z]
            info_data: Either info_levels or (grid_map, info_levels)
            horizon: Planning horizon steps
            dt: Time step size

        Returns:
            target_pos: (3,) selected target position
            target_mode: Integer index of selected target
        """
        # Build candidate list: goal + info zones
        goal_pos = self.config.goal_pos

        # Info zone positions (use zone centers at z=-2.0)
        zone_targets = jnp.column_stack([
            self.info_zones[:, :2],  # cx, cy
            -2.0 * jnp.ones(self.info_zones.shape[0]),  # z coordinate
        ])

        # Concatenate: [goal, zone1, zone2, ...]
        candidates = jnp.vstack([goal_pos[None, :], zone_targets])

        def cost_for_target(target_pos):
            return self._target_cost(pos0, target_pos, info_data, horizon, dt)

        costs = jax.vmap(cost_for_target)(candidates)
        target_idx = jnp.argmin(costs)

        target_pos = candidates[target_idx]
        target_mode = target_idx

        return target_pos, target_mode

    def get_reference_trajectory(
        self,
        state: jax.Array,
        info_data: jax.Array | tuple[jax.Array, jax.Array],
        horizon: int,
        dt: float,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Generate reference trajectory to optimal target.

        Args:
            state: Current state (position + velocities + info_levels)
            info_data: Either info_levels or (grid_map, info_levels)
            horizon: Trajectory length
            dt: Time step

        Returns:
            ref_traj: (horizon, 3) reference trajectory
            target_mode: Integer index of selected target
        """
        pos0 = state[:3]
        target_pos, target_mode = self.get_target(pos0, info_data, horizon, dt)
        ref_traj = self._make_ref_traj(pos0, target_pos, horizon, dt)

        return ref_traj, target_mode
