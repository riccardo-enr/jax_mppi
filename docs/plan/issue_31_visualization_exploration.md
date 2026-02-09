# Issue #31: Refactor Visualization and Increase Exploration Incentive

## Problem Summary

The current `examples/i_mppi/i_mppi_simulation.py` has two main issues:

1. **Visualization Overcrowding**: 4 plots (2D, 3D, occupancy grid) with overlapping elements
2. **Insufficient Exploration**: Robot visits first info zone then goes straight to goal

## Analysis

### Current Configuration (i_mppi_simulation.py:200-232)

**Layer 2 (FSMI Analyzer, ~5 Hz):**

- `info_weight=25.0` - information weight
- `motion_weight=0.5` - motion cost weight
- Ratio: 50:1 (info:motion)

**Layer 3 (Uniform-FSMI, ~50 Hz):**

- `info_weight=5.0` - local information weight

### Root Cause Analysis

The issue is NOT the Layer 2/3 weights alone. Looking at `environment.py:146`:

```python
info_cost = -10.0 * info_gain
target_cost = 1.0 * dist_target
```

The `running_cost` function (used for baseline) has `-10.0` info multiplier.
But `informative_running_cost` (Layer 3) has `info_weight=5.0` which is relatively low.

**Key insight**: After exploring Area 1, the info levels deplete (`info_levels` go to 0), so the controller no longer sees value in nearby zones. The reference trajectory from Layer 2 should be driving exploration, but the `target_cost` weight of `1.0` is too weak compared to goal attraction.

---

## Implementation Plan

### Step 1: Visualization Refactor

- [ ] Remove the 3D view (row 1, col 2) - not adding value
- [ ] Remove the full occupancy grid heatmap (row 2) - cluttered
- [ ] Keep single 2D trajectory plot with:
  - Walls (gray rectangles)
  - Info zones (yellow rectangles with labels)
  - Start/goal markers
  - Controller trajectories (MPPI, KMPPI only - no SMPPI)
- [ ] Add optional info zone visit indicators (checkmarks when explored)

**Files to modify:**

- `examples/i_mppi/i_mppi_simulation.py` (lines 541-776)

### Step 2: Increase Exploration Incentive

- [ ] **Layer 2 (FSMIConfig):**
  - Increase `info_weight` from 25.0 to 50.0 or higher
  - Decrease `motion_weight` from 0.5 to 0.1
  - Ratio target: ~500:1 (info:motion)

- [ ] **Layer 3 (UniformFSMIConfig):**
  - Increase `info_weight` from 5.0 to 15.0-20.0

- [ ] **Environment costs (environment.py):**
  - Reduce `target_cost` weight from 1.0 to 0.3-0.5
  - This allows reference trajectory (with info incentive) to override pure goal-seeking

**Files to modify:**

- `examples/i_mppi/i_mppi_simulation.py` (lines 200-232)
- `src/jax_mppi/i_mppi/environment.py` (line 154, optionally line 231)

### Step 3: Controller Configuration

- [ ] Remove SMPPI from controller list (causes issues per issue notes)
- [ ] Run only MPPI and KMPPI
- [ ] Verify both visit 2+ info zones before goal

**Files to modify:**

- `examples/i_mppi/i_mppi_simulation.py` (line 509)

### Step 4: Validation

- [ ] Run simulation and verify:
  - MPPI trajectory visits multiple info zones
  - KMPPI trajectory visits multiple info zones
  - Visualization is clean and readable
- [ ] Update output image path and verify saved correctly

---

## Testing Notes

**DO NOT RUN SMPPI** - causes visualization/computation issues.

Run command:

```bash
uv run python examples/i_mppi/i_mppi_simulation.py
```

### Success Criteria

1. Visualization shows 1 clean 2D plot
2. Both MPPI and KMPPI visit 2+ info zones (check final `info_levels` in output)
3. Trajectories clearly show exploration behavior before goal convergence

---

## Weight Tuning Guide

If exploration is still insufficient:

| Parameter | Current | Try | Effect |
| :--- | :--- | :--- | :--- |
| Layer 2 info_weight | 25.0 | 100.0 | Stronger exploration in reference |
| Layer 2 motion_weight | 0.5 | 0.05 | Allow longer detours |
| Layer 3 info_weight | 5.0 | 25.0 | Stronger local info seeking |
| target_cost (env) | 1.0 | 0.2 | Weaker goal pull |

If robot gets stuck or oscillates:

- Reduce info_weight
- Increase motion_weight
- Check for collision cost inflation near walls

---

## Documentation

After implementation, add brief explanation in code comments about:

- Why SMPPI is excluded
- Weight choices for exploration vs. goal-seeking trade-off
- Visualization design rationale
