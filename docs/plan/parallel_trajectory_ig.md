# Plan: Parallel Trajectory Information Gain & Parallel I-MPPI

**Branch:** `feat/parallel-traj-ig`
**Reference:** `docs/src/i_mppi.qmd` — §Trajectory-Based FSMI, §Parallel I-MPPI Architecture

## Overview

Two related improvements to the I-MPPI architecture:

**Part A — Trajectory-Level FSMI** (improve Layer 2 trajectory evaluation):
Replace the entropy-proxy `compute_fsmi_gain()` in `_info_gain_grid()` with true FSMI (Theorem 1), and add overlap-aware methods for dense trajectories.

**Part B — Parallel I-MPPI** (new architecture replacing sequential L2→L3):
Eliminate the reference trajectory. A single MPPI runs at 50 Hz and consults a precomputed **information potential field** $\mathcal{I}(x,y)$ updated concurrently at 5–10 Hz.

---

## Part A: Trajectory-Level FSMI

### Current State

`_info_gain_grid()` in `fsmi.py` evaluates trajectory-level information by:

1. Subsampling the reference trajectory (every `trajectory_subsample_rate=5` steps)
2. Vmapping `compute_fsmi_gain()` over sampled waypoints — 360° rays, entropy proxy `4*p*(1-p)`
3. Summing scalar gains

Limitations:
- Uses entropy proxy, not true FSMI (Theorem 1)
- No beam overlap handling between consecutive viewpoints
- 360° rays ignore directional FOV

### Step A1: `FSMIModule.compute_fsmi_batch()`

**File:** `src/jax_mppi/i_mppi/fsmi.py`

- [x] Add `compute_fsmi_batch(grid_map, positions, yaws)` to `FSMIModule`
- Signature: `(jax.Array [H,W], jax.Array [N,2], jax.Array [N]) -> jax.Array [N]`
- Implementation: `jax.vmap(self.compute_fsmi, in_axes=(None, 0, 0))(grid_map, positions, yaws)`
- Validate: output matches N individual `compute_fsmi` calls

### Step A2: Direct Summation (Method 1)

**File:** `src/jax_mppi/i_mppi/fsmi.py` — update `_info_gain_grid()`

- [x] Replace `compute_fsmi_gain` calls with `FSMIModule.compute_fsmi_batch()`
- [x] Compute per-pose yaw from consecutive trajectory points: `atan2(dy, dx)`
- [x] Sum per-pose MI values (scaled by `dt * subsample_rate`)
- This is the simplest upgrade: true FSMI instead of entropy proxy, same summation

### Step A3: Discount Factor (Method 3)

**File:** `src/jax_mppi/i_mppi/fsmi.py` — new function

- [x] Add `_fsmi_trajectory_discounted(ref_traj, grid_map, fsmi_module, dt, decay=0.7)`
- For each sampled pose, compute a dense cell mask `(N_poses, H, W)` — which cells are within FOV+range
- Compute `previous_views[i, h, w]` = number of earlier poses that see cell `(h,w)`: `cumsum(masks, axis=0) - masks`
- Per-pose discount: `weight[i] = exp(-decay * mean_over_cells(previous_views[i] * mask[i]))`
- Final MI: `sum(mi[i] * weight[i])`
- Fully parallel via vmap for mask computation

### Step A4: Conservative Parallel Filtering (Method 2)

**File:** `src/jax_mppi/i_mppi/fsmi.py` — new function

- [x] Add `_fsmi_trajectory_filtered(ref_traj, grid_map, fsmi_module, dt)`
- Compute per-pose cell masks `(N_poses, H, W)` (same as A3)
- First-hit mask: `first_hit[i] = (argmin_over_poses(any_mask, axis=0) == i)` — cell belongs to the first pose that sees it
- `independent_fraction[i] = sum(first_hit[i] & mask[i]) / sum(mask[i])`
- Scale: `mi_filtered[i] = mi[i] * independent_fraction[i]`

### Step A5: Config & Dispatch

**File:** `src/jax_mppi/i_mppi/fsmi.py`

- [x] Add to `FSMIConfig`:
  - `trajectory_ig_method: str = "direct"` (options: `"direct"`, `"discount"`, `"filtered"`)
  - `trajectory_ig_decay: float = 0.7`
- [x] Wire `_info_gain_grid()` to dispatch based on `config.trajectory_ig_method`

### Cell Mask Design Decision

Dense boolean mask `(N_poses, H, W)`:
- For 10 subsampled poses on a 200×200 grid → 400K bools ≈ 0.4 MB. Acceptable.
- First-hit via `jnp.argmin` along pose axis is fully parallel.
- Sparse alternatives are harder in JAX and unnecessary at this grid size.

---

## Part B: Parallel I-MPPI Architecture

### Current State

Sequential architecture:
```
Layer 2 (5 Hz) → reference trajectory τ_ref
Layer 3 (50 Hz) → biased_mppi_command(U_ref=τ_ref) + Uniform-FSMI
```

Problems:
- Layer 3 cannot act on new information until next L2 replan (200 ms latency)
- Biased sampling anchored to a single reference limits path discovery
- Two separate optimization problems that may conflict

### Target Architecture

```
FSMI Field Generator (5–10 Hz) → I(x,y) potential field (cached)
Single MPPI (50 Hz) → cost = obstacles + bounds + field_lookup + local_uniform_fsmi
```

No reference trajectory. No biased sampling. MPPI discovers informative paths by following the spatial gradient of the information potential field.

### Step B1: Bilinear Interpolation Utility

**File:** `src/jax_mppi/i_mppi/map.py`

- [ ] Add `interp2d(field, field_origin, field_res, query_points) -> jax.Array`
  - `field`: `(Nx, Ny)` float array
  - `field_origin`: `(2,)` world coords of `field[0,0]`
  - `field_res`: scalar, meters per cell
  - `query_points`: `(M, 2)` world coordinates
  - Returns: `(M,)` interpolated values
  - Implementation:
    ```python
    def interp2d(field, origin, res, points):
        # Convert to continuous grid indices
        gx = (points[:, 0] - origin[0]) / res
        gy = (points[:, 1] - origin[1]) / res
        # Floor indices
        ix = jnp.floor(gx).astype(jnp.int32)
        iy = jnp.floor(gy).astype(jnp.int32)
        # Fractional parts
        fx = gx - ix
        fy = gy - iy
        # Clamp
        Nx, Ny = field.shape
        ix0 = jnp.clip(ix, 0, Nx - 2)
        iy0 = jnp.clip(iy, 0, Ny - 2)
        # Bilinear
        v00 = field[ix0, iy0]
        v10 = field[ix0 + 1, iy0]
        v01 = field[ix0, iy0 + 1]
        v11 = field[ix0 + 1, iy0 + 1]
        return (v00 * (1-fx)*(1-fy) + v10 * fx*(1-fy)
              + v01 * (1-fx)*fy + v11 * fx*fy)
    ```
  - Out-of-bounds queries return 0.0 (no information outside field)
  - Must be JIT-compatible and vmap-friendly

### Step B2: Information Potential Field Computation

**File:** `src/jax_mppi/i_mppi/fsmi.py` — new class or function

- [ ] Add `compute_info_field(fsmi_module, grid_map, uav_pos, config) -> (field, field_origin)`

  **Parameters** (add to new `InfoFieldConfig` or extend `FSMIConfig`):
  - `field_res: float = 0.5` — meters per field cell
  - `field_extent: float = 5.0` — half-width of local workspace
  - `n_yaw: int = 8` — number of candidate yaw angles

  **Algorithm:**
  1. Build coarse grid of `(x, y)` positions centered on UAV:
     ```python
     xs = jnp.arange(-field_extent, field_extent, field_res) + uav_pos[0]
     ys = jnp.arange(-field_extent, field_extent, field_res) + uav_pos[1]
     positions = jnp.stack(jnp.meshgrid(xs, ys, indexing="ij"), axis=-1)  # (Nx, Ny, 2)
     ```
  2. Define yaw angles: `psis = jnp.linspace(0, 2*pi, n_yaw, endpoint=False)`
  3. Evaluate FSMI at every `(position, yaw)` pair via double vmap:
     ```python
     flat_pos = positions.reshape(-1, 2)  # (Nx*Ny, 2)
     fsmi_fn = lambda pos, psi: fsmi_module.compute_fsmi(grid_map, pos, psi)
     gains = vmap(vmap(fsmi_fn, in_axes=(None, 0)), in_axes=(0, None))(flat_pos, psis)
     # gains: (Nx*Ny, n_yaw)
     ```
  4. Max over yaw: `field = gains.max(axis=-1).reshape(Nx, Ny)`
  5. Return `(field, jnp.array([xs[0], ys[0]]))`

  **Computational budget:**
  - 10×10 m at 0.5 m → 20×20 = 400 positions × 8 yaws = 3,200 FSMI evals
  - Each FSMI eval: 16 beams × ~50 ray steps → ~25K flops
  - Total: ~80M flops — well within GPU budget at 5–10 Hz

  **Static shape consideration:**
  - `field_extent` and `field_res` determine array shapes → use `static_argnames` or precompute grid positions outside JIT and pass as argument

### Step B3: Field-Based MPPI Cost Function

**File:** `src/jax_mppi/i_mppi/environment.py` — new cost function

- [ ] Add `parallel_imppi_running_cost(state, action, t, *, grid_map, grid_origin, grid_resolution, info_field, field_origin, field_res, uniform_fsmi_fn, lambda_info, lambda_local)`

  **Cost terms:**
  ```
  J = J_collision + J_grid + J_bounds + J_height + J_control
    + J_field(x,y)           ← NEW: medium-range strategic guidance
    + J_local(x,y,yaw)       ← existing: Uniform-FSMI for immediate viewpoint
  ```

  - `J_field = -lambda_info * interp2d(info_field, field_origin, field_res, pos_xy)`
  - `J_local = -lambda_local * uniform_fsmi_fn(grid_map, pos_xy, yaw)`
  - Drop `J_target` (no reference trajectory to track)
  - Keep all safety costs (collision, bounds, height, grid obstacle)

  **Key difference from `informative_running_cost`:**
  - No `target` parameter
  - `info_field` + `field_origin` + `field_res` replace reference trajectory tracking
  - `lambda_info` and `lambda_local` are separate weights (tunable)

### Step B4: Parallel I-MPPI Simulation Loop

**File:** `docs/examples/sim_utils.py` — new function `build_parallel_sim_fn()`

- [ ] Implement dual-rate main loop:

  ```python
  def build_parallel_sim_fn(config, fsmi_module, uniform_fsmi, grid_map_obj, ...):
      # Precompute field grid positions (static shapes)
      field_xs = jnp.arange(-field_extent, field_extent, field_res)
      field_ys = jnp.arange(-field_extent, field_extent, field_res)
      Nx, Ny = len(field_xs), len(field_ys)

      def step_fn(carry, t):
          state, ctrl_state, info_field, field_origin, grid, done_step = carry

          # --- Low-rate: recompute field every N steps ---
          recompute = (t % field_update_interval == 0)
          uav_pos = state[:3]
          new_field, new_origin = jax.lax.cond(
              recompute,
              lambda: compute_info_field(fsmi_module, grid, uav_pos, ...),
              lambda: (info_field, field_origin),
          )

          # --- High-rate: standard MPPI (no bias) with field cost ---
          def cost_fn(x, u, t_step):
              return parallel_imppi_running_cost(
                  x, u, t_step,
                  grid_map=grid, info_field=new_field,
                  field_origin=new_origin, ...
              )

          action, new_ctrl = mppi_command(config, ctrl_state, state, dynamics, cost_fn)

          # --- Apply control, update state & grid ---
          new_state = dynamics(state, action)
          new_grid = _update_grid_from_info(initial_grid, zone_masks, new_state[13:])

          return (new_state, new_ctrl, new_field, new_origin, new_grid, done_step), (new_state,)

      return jax.lax.scan(step_fn, init_carry, jnp.arange(sim_steps))
  ```

  **Note on `jax.lax.cond` for field update:**
  Both branches must have the same output shape. The false branch returns the cached field unchanged. Since field dimensions are fixed (determined by `field_extent`/`field_res`), this is safe.

  **Alternative: always recompute but mask.**
  If `lax.cond` causes tracing issues with the FSMI vmap, always compute the field but use `jnp.where(recompute, new_field, old_field)`. This wastes compute but avoids shape ambiguity.

### Step B5: Configuration

**File:** `src/jax_mppi/i_mppi/fsmi.py`

- [ ] Add `InfoFieldConfig` dataclass:
  ```python
  @dataclass
  class InfoFieldConfig:
      field_res: float = 0.5           # meters per field cell
      field_extent: float = 5.0        # half-width of local workspace [m]
      n_yaw: int = 8                   # candidate yaw angles
      field_update_interval: int = 10  # MPPI steps between field updates
      lambda_info: float = 10.0        # field lookup weight
      lambda_local: float = 5.0        # Uniform-FSMI weight
  ```

### Step B6: Target Selection Fallback

The Parallel I-MPPI architecture removes explicit target selection. However, the field naturally provides directional guidance:

- Unknown zones have high FSMI → high field values → MPPI steers toward them
- Depleted zones have low FSMI → low field values → MPPI ignores them
- When all zones are explored, field is ~0 everywhere → need goal fallback

- [ ] Add goal attraction as a fallback cost term when max field value < threshold:
  ```python
  goal_active = jnp.max(info_field) < min_field_threshold
  J_goal = jnp.where(goal_active, w_goal * jnp.linalg.norm(pos - goal_pos), 0.0)
  ```

  Or: always include a small goal attraction that is dominated by info cost when zones remain:
  ```python
  J_goal = w_goal_base * dist_to_goal  # small constant pull toward goal
  ```

---

## Part C: Tests & Validation

### Step C1: Unit Tests — Trajectory FSMI

- [ ] `test_compute_fsmi_batch`: batch output matches N individual calls
- [ ] `test_direct_summation`: known grid with one unknown zone → MI > 0
- [ ] `test_discount_leq_direct`: `MI_discount <= MI_direct` for overlapping trajectory
- [ ] `test_filtered_leq_direct`: `MI_filtered <= MI_direct`
- [ ] `test_filtered_eq_direct_nonoverlapping`: non-overlapping poses → `MI_filtered == MI_direct`
- [ ] `test_all_methods_jit`: all three methods compile and run under `jax.jit`

### Step C2: Unit Tests — Info Field & Interpolation

- [ ] `test_interp2d_corners`: interpolation at grid points equals exact values
- [ ] `test_interp2d_midpoint`: value at midpoint is average of corners
- [ ] `test_interp2d_oob`: out-of-bounds returns 0.0
- [ ] `test_compute_info_field_shape`: output shape matches expected `(Nx, Ny)`
- [ ] `test_info_field_unknown_zone`: field peaks near unknown zone centers
- [ ] `test_info_field_explored`: field values drop after zone depletion

### Step C3: Integration Test — Parallel I-MPPI Loop

- [ ] Run `build_parallel_sim_fn()` for ~200 steps on the standard 3-zone environment
- [ ] Verify: UAV moves toward zones (not random walk)
- [ ] Verify: info levels deplete during simulation
- [ ] Verify: field updates occur every `field_update_interval` steps
- [ ] Compare with sequential I-MPPI on same scenario (qualitative)

### Step C4: Benchmarks

- [ ] Wall-clock: 3 trajectory FSMI methods on 50-waypoint trajectory, 16 beams
- [ ] Wall-clock: `compute_info_field` for 20×20 grid, 8 yaws (target: < 50 ms on GPU)
- [ ] Wall-clock: full parallel I-MPPI step including field lookup (target: < 20 ms at 50 Hz)

---

## Implementation Order

| Priority | Step | Description | Depends On |
|----------|------|-------------|------------|
| 1 | A1 | `compute_fsmi_batch` | — |
| 2 | A2 | Direct summation (replace entropy proxy) | A1 |
| 3 | B1 | `interp2d` utility | — |
| 4 | B2 | `compute_info_field` | A1, B1 |
| 5 | B3 | Field-based cost function | B1, B2 |
| 6 | B4 | Parallel I-MPPI sim loop | B2, B3 |
| 7 | B5 | Configuration | B2, B3 |
| 8 | B6 | Goal fallback | B3 |
| 9 | A3 | Discount factor method | A1 |
| 10 | A4 | Conservative filtering | A1 |
| 11 | A5 | Config & dispatch for Part A | A2, A3, A4 |
| 12 | C1–C4 | Tests & benchmarks | all above |

**Rationale:** B1–B4 (Parallel I-MPPI) is the higher-value deliverable. A3/A4 (overlap methods) are refinements that can come later since direct summation is often sufficient when trajectory subsampling is coarse.

## Design Decisions

### Field vs Reference: When to Use Each

| Scenario | Recommendation |
|----------|---------------|
| Sparse unknown zones, open space | Parallel I-MPPI (field) — MPPI can freely explore |
| Dense obstacles, narrow corridors | Sequential I-MPPI (reference) — biased sampling helps |
| Real-time on GPU | Parallel — fewer sequential dependencies |
| CPU-only | Sequential — field computation is expensive without GPU |

Both architectures should remain available. The parallel variant is an **alternative** sim loop, not a replacement.

### Field Staleness

The field is stale by up to `field_update_interval / mppi_rate` seconds (e.g., 10/50 = 0.2 s). This is acceptable because:
- FSMI values change slowly (grid updates are incremental)
- The Uniform-FSMI local term provides immediate viewpoint feedback
- MPPI's stochastic sampling can adapt between field updates

### Yaw-Dependent vs Yaw-Independent Field

Start with yaw-independent (`max over yaw`). This is simpler and works well for omnidirectional or wide-FOV sensors. Yaw-dependent variant (`(Nx, Ny, N_psi)` field with 3D interpolation) can be added later if needed for narrow-FOV cameras.

### Field Extent & Resolution Trade-offs

| Parameter | Value | Grid Size | FSMI Evals | Notes |
|-----------|-------|-----------|------------|-------|
| 5m, 0.5m | default | 20×20 | 3,200 | Good for local planning |
| 10m, 1.0m | coarse | 20×20 | 3,200 | Same cost, wider view |
| 10m, 0.5m | fine | 40×40 | 12,800 | 4× cost, better resolution |
| 5m, 0.25m | dense | 40×40 | 12,800 | High-res local field |

The field resolution need not match the occupancy grid resolution. A coarser field is fine because bilinear interpolation smooths the values.
