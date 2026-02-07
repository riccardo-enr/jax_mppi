# I-MPPI Architecture Alignment Plan

## Problem Statement

The current I-MPPI implementation is missing the **informative term in Layer 3 (fast loop)**. The architecture description specifies a two-layer system:

1. **Layer 2 (FSMI Analyzer):** ~5-10 Hz - Full FSMI for reference trajectory generation
2. **Layer 3 (I-MPPI Controller):** ~50 Hz - Fast control with **Uniform-FSMI** for local reactivity

Currently, Layer 3 only does trajectory tracking (`running_cost`), which reduces I-MPPI to standard MPPI with a biased reference. This loses:

- Reactive viewpoint maintenance during disturbances
- Handling of occlusions detected between Layer 2 updates
- True informative control behavior

## Required Changes

### 1. Implement Uniform-FSMI Module

Create a simplified O(n) FSMI variant for the fast loop:

```python
class UniformFSMI:
    """
    Uniform-FSMI from Zhang et al. (2020) for O(n) computation.

    Simplifications vs full FSMI:
    - Assumes uniform sensor noise
    - Uses shorter ray range (local, 2-3m)
    - Fewer beams
    - No G_kj matrix computation (biggest speedup)
    """
```

Key differences from full FSMI:

- Sensor noise is uniform (constant sigma)
- Ray range limited to 2-3m (local)
- Reduced beam count (4-8 vs 16)
- Direct sum without G_kj matrix: `MI ≈ sum_j P(e_j) * C_j`

### 2. Create Informative Cost Function

Add to `src/jax_mppi/i_mppi/environment.py`:

```python
def informative_running_cost(
    state: jax.Array,
    action: jax.Array,
    t: int,
    target: jax.Array,
    grid_map: jax.Array,
    uniform_fsmi: UniformFSMI,
    info_weight: float = 5.0,
) -> jax.Array:
    """
    Cost function with informative term:
    J = tracking_cost - info_weight * uniform_fsmi(state)
    """
    tracking = running_cost(state, action, t, target)
    info_gain = uniform_fsmi.compute(grid_map, state[:2], yaw)
    return tracking - info_weight * info_gain
```

### 3. Update I-MPPI Simulation

Modify `examples/i_mppi/i_mppi_simulation.py`:

```python
# Layer 3: Biased I-MPPI with informative cost
cost_fn = partial(
    informative_running_cost,
    target=ref_traj,
    grid_map=grid_map,
    uniform_fsmi=uniform_fsmi,
    info_weight=local_info_weight,
)
```

### 4. Update Examples Directory Structure

Reorganize `examples/i_mppi/` to clearly show the architecture:

```text
examples/i_mppi/
├── __init__.py
├── i_mppi_simulation.py      # Main: Layer 2 (FSMI) + Layer 3 (I-MPPI with Uniform-FSMI)
├── i_mppi_simulation_legacy.py  # Legacy: geometric zones
├── fsmi_grid_demo.py         # Demo: Full FSMI visualization
├── uniform_fsmi_demo.py      # NEW: Demonstrate Uniform-FSMI speedup
└── architecture_comparison.py # NEW: Compare with/without Layer 3 info term
```

## Implementation Steps

### Step 1: Implement Uniform-FSMI

- [ ] Add `UniformFSMI` class to `fsmi.py`
- [ ] Implement O(n) MI computation
- [ ] Add configuration parameters (short_range, few_beams)
- [ ] Benchmark vs full FSMI

### Step 2: Update Cost Functions

- [ ] Add `informative_running_cost()` to environment.py
- [ ] Make grid_map accessible in cost function (via closure or state)
- [ ] Handle yaw extraction from state

### Step 3: Update I-MPPI Simulation

- [ ] Integrate Uniform-FSMI into Layer 3 cost
- [ ] Add config parameters for local info weight
- [ ] Maintain backward compatibility flag

### Step 4: Add Demonstration Examples

- [ ] Create `uniform_fsmi_demo.py`
- [ ] Create `architecture_comparison.py` showing the difference

### Step 5: Documentation

- [ ] Update README in examples/i_mppi/
- [ ] Document the two-layer architecture

## Configuration Parameters

### Layer 2 (Full FSMI) - Slow Path

```python
FSMIConfig(
    use_grid_fsmi=True,
    num_beams=16,
    max_range=5.0,
    ray_step=0.15,
    trajectory_subsample_rate=8,
    info_weight=15.0,
)
```

### Layer 3 (Uniform-FSMI) - Fast Path

```python
UniformFSMIConfig(
    num_beams=6,          # Reduced
    max_range=2.5,        # Local only
    ray_step=0.2,         # Coarser
    info_weight=5.0,      # Lower weight (reactive)
)
```

## Expected Performance

| Component | Rate | Computation |
| :--- | :--- | :--- |
| Layer 2 (Full FSMI) | 5 Hz | ~40-50ms |
| Layer 3 (Uniform-FSMI per sample) | 50 Hz | ~0.5ms |
| Total control cycle | 50 Hz | <20ms |

## Benefits of Proper Architecture

1. **Reactive Viewpoint Maintenance:** During disturbances, Layer 3 locally optimizes viewing angle
2. **Occlusion Handling:** Can detect and respond to occlusions between Layer 2 updates
3. **True I-MPPI:** Not just biased trajectory tracking

## Status

- [x] Step 1: Uniform-FSMI implementation (UniformFSMI class in fsmi.py)
- [x] Step 2: Cost function update (informative_running_cost in environment.py)
- [x] Step 3: Simulation integration (i_mppi_simulation.py updated)
- [ ] Step 4: Demo examples (optional: architecture_comparison.py)
- [x] Step 5: Documentation (README.md updated)
