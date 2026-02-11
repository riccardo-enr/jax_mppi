# Fix CUDA I-MPPI Simulation Issues

## Context

The C++/CUDA I-MPPI feature parity implementation (Phase 1-8) is complete and builds successfully. FSMI unit tests pass, but the full `informative_sim` diverges because the controller never receives updated cost parameters (info field, reference trajectory) during the simulation loop. There are also build quality issues (Eigen/CUDA warnings) and the pixi-installed `.so` is stale.

## Issues

### Issue 1: Controller cost is stale (CRITICAL)

**Root cause**: `MPPIController` stores `cost_` by value (line 170 of `mppi.cuh`). The `rollout_kernel` receives `cost` by value at each kernel launch, copying the controller's internal `cost_` member. In `informative_sim.cu`, the sim loop updates a *local* `cost` variable (lines 200-202) but never propagates these changes to `controller.cost_`.

The existing `update_cost_params(goal, lambda_goal)` in `i_mppi.cuh:63` only updates two fields. The info field pointer (`info_field.d_field`), field origin/dimensions, and reference trajectory pointer are never updated.

**Fix**: Add `set_cost(const Cost& c)` method to `MPPIController` that replaces `cost_` entirely. Then call it from the sim loop after updating cost fields.

**Files**:
- `third_party/cuda-mppi/include/mppi/controllers/mppi.cuh` — add `set_cost()` to base class
- `third_party/cuda-mppi/src/informative_sim.cu` — call `controller.set_cost(cost)` after updating cost fields

### Issue 2: Eigen/CUDA constexpr warnings

**Root cause**: Eigen's CUDA complex number headers trigger `calling a constexpr __host__ function from a __host__ __device__ function` warnings. These are cosmetic but noisy.

**Fix**: Add `--expt-relaxed-constexpr` to CUDA compile flags in CMakeLists.txt.

**Files**:
- `third_party/cuda-mppi/CMakeLists.txt` — add `target_compile_options` with `--expt-relaxed-constexpr`

### Issue 3: Stale pixi-installed `cuda_mppi.so`

**Root cause**: The `.so` in `.pixi/envs/dev/lib/python3.12/site-packages/` is from a previous build and lacks the new types (OccupancyGrid2D, InfoField, QuadrotorDynamics, etc.). Python imports pick up the stale one.

**Fix**: Rebuild via `pixi run` or `pip install` from the build directory so the installed `.so` matches the source.

**Files**:
- Run `pixi run` build task or `pip install -e third_party/cuda-mppi/` to update

## Implementation

### Step 1: Add `set_cost()` to MPPIController

In `third_party/cuda-mppi/include/mppi/controllers/mppi.cuh`, add to the public section:

```cpp
void set_cost(const Cost& cost) { cost_ = cost; }
```

This is safe because the cost struct is trivially copyable (all POD + device pointers). The next `compute()` call will pass the updated `cost_` to the rollout kernel.

Also add a getter for symmetry:
```cpp
const Cost& cost() const { return cost_; }
Cost& cost() { return cost_; }
```

### Step 2: Fix informative_sim.cu sim loop

Replace lines 197-207 in `informative_sim.cu`:

```cpp
// Update cost fields
cost.ref_trajectory = d_ref_traj;
cost.ref_horizon    = ifc.ref_horizon;
cost.info_field     = info_field;

// Propagate to controller
controller.set_cost(cost);
```

### Step 3: Suppress Eigen/CUDA warnings

In `third_party/cuda-mppi/CMakeLists.txt`, after the CUDA standard setting:

```cmake
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
```

### Step 4: Rebuild pixi package

```bash
pixi run build-cuda  # or equivalent task
```

If no pixi build task exists for cuda-mppi, install manually:
```bash
cd third_party/cuda-mppi && pip install .
```

### Additional Issues Found During Implementation

#### Issue 4: `get_action()` returns unscaled controls

The rollout kernel applies `u = (u_nom + noise) * u_scale`, but `get_action()` returns raw `u_nom` without scaling. The sim loop must scale: `action = get_action() * u_scale`.

#### Issue 5: Hardcoded `learning_rate = 0.1f` in MPPI update

`weighted_update_kernel` was called with hardcoded 0.1f learning rate in both `mppi.cuh` and `i_mppi.cuh`. Standard MPPI uses 1.0. Made it configurable via `MPPIConfig::learning_rate` (default 1.0).

#### Issue 6: Zero-initialized `u_nom` for quadrotor

`u_nom` starts at 0, but hover requires thrust = mass * gravity ≈ 19.62N. With `u_scale=10`, initialized `u_nom[t][0] = 1.962` (hover/u_scale) so the controller starts at equilibrium.

## Verification (completed)

1. Build: all targets compile clean, no warnings
2. `fsmi_unit_test`: 6/6 tests pass
3. `informative_sim`: quadrotor maintains stable flight (z ≈ -2.0), depletes zone 0 from 99→5
4. Python bindings build successfully
