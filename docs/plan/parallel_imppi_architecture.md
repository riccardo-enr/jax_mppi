# Parallel I-MPPI Architecture Overview

## High-Level Architecture

The system implements a **two-layer hierarchical informative planning** system where both layers run **within a single JIT-compiled simulation function** at 50 Hz, with the strategic layer cached and updated at 5 Hz.

| Layer   | Component                    | Frequency | Complexity | Purpose                          |
|---------|------------------------------|-----------|------------|----------------------------------|
| Layer 2 | FSMI + Info Field + Gradient | 5 Hz      | O(n²)     | Strategic planning via field gradient |
| Layer 3 | Biased MPPI + Uniform-FSMI   | 50 Hz     | O(n)      | Reactive trajectory tracking     |

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Parallel I-MPPI Loop (50 Hz)                     │
│                                                                     │
│  Every 10 steps (5 Hz) ── Layer 2:                                  │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ compute_info_field()                                       │     │
│  │   FSMIModule (full FSMI, O(n²))                            │     │
│  │   20×20 grid × 8 yaw angles = 3200 FSMI evaluations       │     │
│  │   → info_field: (20, 20) potential field                   │     │
│  │                                                            │     │
│  │ field_gradient_trajectory()                                │     │
│  │   Climb field gradient at 2.0 m/s                          │     │
│  │   → ref_traj: (40, 3) reference path                      │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Every step (50 Hz) ── Layer 3:                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ fov_grid_update()  →  updated occupancy grid               │     │
│  │                                                            │     │
│  │ mppi.command() with cost =                                 │     │
│  │   target_tracking(ref_traj)                                │     │
│  │   - λ_local × Uniform-FSMI(pos, yaw)     (local info)     │     │
│  │   - λ_info  × field_lookup(info_field)    (strategic)      │     │
│  │   + collision_avoidance(walls + grid)                      │     │
│  │   + goal_attraction                                        │     │
│  │   → action: [thrust, ωx, ωy, ωz]                          │     │
│  │                                                            │     │
│  │ step_quadrotor() → next 13D state (RK4)                   │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Files

| File | Key Contents |
|------|-------------|
| `examples/i_mppi/sim_utils.py` | `build_parallel_sim_fn()`, `fov_grid_update()`, `field_gradient_trajectory()`, `interp2d()` |
| `src/jax_mppi/i_mppi/fsmi.py` | `FSMIModule`, `UniformFSMI`, `InfoFieldConfig`, `compute_info_field()` |
| `src/jax_mppi/i_mppi/environment.py` | `step_quadrotor()`, `informative_running_cost()`, `quat_to_yaw()` |
| `src/jax_mppi/i_mppi/planner.py` | `biased_mppi_command()`, `biased_smppi_command()`, `biased_kmppi_command()` |
| `src/jax_mppi/i_mppi/map.py` | `GridMap`, `rasterize_environment()`, coordinate conversions |

---

## Component Details

### Layer 2: Strategic Planning (5 Hz)

#### FSMIModule — Full FSMI (Zhang et al. 2020, Theorem 1)

Computes expected mutual information I(M; Z) for a sensor at a given pose.

**Algorithm per pose:**
1. Cast `num_beams` rays across FOV (16 rays, 90° FOV)
2. For each beam, compute:
   - **P(e_j)**: Probability beam terminates at cell j = `occ[j] × ∏(1 - occ[l]) for l < j`
   - **C_k**: Information potential (gain if cell measured as occupied/empty)
   - **G_kj**: Gaussian sensor model (CDF-based noise term, N×N matrix → O(n²))
3. Total MI = Σ P(e_j) · C_k · G_kj over all cells j, k

#### compute_info_field()

1. Build 10m×10m local grid centered on UAV (0.5m spacing → 20×20)
2. Evaluate FSMI at each position × 8 yaw candidates (vmapped)
3. Max over yaw → yaw-independent potential field

#### field_gradient_trajectory()

1. Compute ∂field/∂x, ∂field/∂y via `numpy.gradient`
2. Walk 40 steps at 2.0 m/s following normalized gradient
3. Output: (40, 3) reference trajectory for MPPI to track

---

### Layer 3: Reactive Control (50 Hz)

#### UniformFSMI — Fast O(n) Variant

Same as FSMIModule but **omits G_kj matrix** (valid at short range where sensor noise << cell size, so G_kj ≈ δ(k-j)):

```
MI ≈ Σ_j P(e_j) · C_j    (O(n) instead of O(n²))
```

Config: 6 beams, 2.5m range, 90° FOV.

#### Biased MPPI (planner.py)

Mixture sampling:
- **50% nominal**: Standard Gaussian noise (exploration)
- **50% biased**: Noise shifted toward reference trajectory (exploitation)

```
noise_biased = noise_base + (U_ref - U_nominal)
```

#### MPPI Cost Function

```
J = wall_cost
  + grid_obstacle_cost
  + target_weight × ||pos - ref_traj[t]||²     (track Layer 2 reference)
  - λ_local × UniformFSMI(pos, yaw)            (local info gain)
  - λ_info × interp2d(info_field, pos)          (strategic field lookup)
  + goal_weight × ||pos - goal||                (constant goal attraction)
  + height_cost                                 (maintain z = -2m)
  + 0.01 × ||u||²                              (action regularization)
```

---

### FOV-Based Grid Updates (50 Hz)

Every step, `fov_grid_update()`:
1. Cast 24 rays across 90° FOV (up to 4.0m)
2. For each visible cell:
   - Non-obstacle + line-of-sight clear → set to 0.01 (known-free)
   - First obstacle on ray → set to 0.99 (known-occupied)
3. Preserves wall positions; clears unknown cells in FOV

---

### Termination

Simulation ends when **both** conditions are met:
1. **Goal reached**: `||pos - [9, 5, -2]|| < 1.5m`
2. **Map explored**: `fraction of unknown cells < 5%`

Unknown detection via entropy proxy: `4·p·(1-p)` → 1.0 at p=0.5, ~0 at known cells.

---

## Simulation State (jax.lax.scan carry)

| Field | Shape | Description |
|-------|-------|-------------|
| `current_state` | (13,) | Quadrotor state [pos, vel, quat, omega] |
| `ctrl_state` | — | MPPI internal state (mean control sequence) |
| `info_field` | (Nx, Ny) | Cached FSMI potential field |
| `field_origin` | (2,) | World coordinates of field[0,0] |
| `grid` | (H, W) | Occupancy grid (updated every step) |
| `ref_traj` | (40, 3) | Reference trajectory from gradient |
| `done_step` | scalar | Termination flag |

---

## Design Rationale

**Why two layers?**
- Full FSMI is O(n²) per pose — too expensive for 50 Hz cost evaluation across K×H samples
- Uniform-FSMI at O(n) provides immediate reactive feedback
- Info field + gradient trajectory provide medium-range strategic guidance

**Why biased MPPI?**
- 50% exploitation samples efficiently follow Layer 2 guidance
- 50% exploration samples prevent local optima traps
- Smooth trajectories from biasing (vs chaotic pure-noise sampling)

**Why FOV grid updates at 50 Hz?**
- Immediate collision avoidance feedback
- Self-regulating: cleared cells (p→0) don't contribute to info cost via entropy weighting `4·p·(1-p)`
- Cost function always sees current occupancy state

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_SAMPLES` | 1000 | MPPI rollout samples |
| `HORIZON` | 40 | MPPI planning horizon (steps) |
| `LAMBDA` | 0.1 | MPPI temperature |
| `FSMI_BEAMS` | 12 | Full FSMI ray count |
| `FSMI_RANGE` | 10.0m | Full FSMI max range |
| `FIELD_RES` | 0.5m | Info field cell size |
| `FIELD_EXTENT` | 5.0m | Info field half-width |
| `FIELD_N_YAW` | 8 | Candidate yaw angles |
| `FIELD_UPDATE_INTERVAL` | 10 | Steps between field recomputation |
| `LAMBDA_INFO` | 20.0 | Field lookup cost weight |
| `LAMBDA_LOCAL` | 10.0 | Uniform-FSMI cost weight |
| `REF_SPEED` | 2.0 m/s | Gradient trajectory speed |
| `REF_HORIZON` | 40 | Gradient trajectory steps |
| `TARGET_WEIGHT` | 1.0 | Reference tracking weight |
| `GOAL_WEIGHT` | 0.2 | Goal attraction weight |
