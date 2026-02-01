# JAX MPPI Implementation Plan

Port `pytorch_mppi` to JAX, producing a functional, JIT-compilable MPPI library.

## Status (Jan 31, 2026)

**Overall Progress:** Phase 6 complete (Autotuning system fully implemented with CMA-ES, Ray Tune, and CMA-ME support).

### Implementation Status by Phase

- **Phase 1: Core MPPI** ✅ **COMPLETE**
  - 353 lines implemented in `src/jax_mppi/mppi.py`
  - All core features from pytorch_mppi ported
  - 115 lines of unit tests in `tests/test_mppi.py`
  
- **Phase 2: Pendulum Integration** ✅ **COMPLETE**
  - 270 lines in `examples/pendulum.py` (full-featured example with CLI)
  - 282 lines in `tests/test_pendulum.py` (8 comprehensive integration tests)
  - All tests passing, swing-up and stabilization verified
  
- **Phase 3: Smooth MPPI (SMPPI)** ✅ **COMPLETE**
  - 634 lines implemented in `src/jax_mppi/smppi.py`
  - All SMPPI features: action_sequence, smoothness cost, dual bounds, integration
  - 580 lines in `tests/test_smppi.py` (18 comprehensive tests)
  - All tests passing
  
- **Phase 4: Kernel MPPI (KMPPI)** ✅ **COMPLETE**
  - 660 lines implemented in `src/jax_mppi/kmppi.py`
  - RBFKernel, kernel interpolation, control point optimization
  - 595 lines in `tests/test_kmppi.py` (23 comprehensive tests)
  - All tests passing (53/53 total tests pass)
  
- **Phase 5: Smooth Comparison Example** ✅ **COMPLETE**
  - 442 lines in `examples/smooth_comparison.py`
  - Compares MPPI, SMPPI, and KMPPI on 2D navigation with obstacle avoidance
  - Includes visualization with 4 subplots: trajectories, costs, controls, smoothness
  - Supporting modules: `src/jax_mppi/costs/` and `src/jax_mppi/dynamics/`
  
- **Phase 6: Autotuning** ✅ **COMPLETE**
  - 656 lines in `src/jax_mppi/autotune.py` - Core CMA-ES autotuning
  - 375 lines in `src/jax_mppi/autotune_global.py` - Ray Tune global search
  - 218 lines in `src/jax_mppi/autotune_qd.py` - CMA-ME quality diversity
  - 305 lines in `tests/test_autotune.py` (21 unit tests)
  - 247 lines in `tests/test_autotune_integration.py` (4 integration tests)
  - 321 lines in `examples/autotune_pendulum.py` - Full demonstration
  - 90 lines in `examples/autotune_basic.py` - Minimal example
  - All 25 tests passing
  
### Package Size Comparison

| Package                | Core Code  | Tests      | Examples   | Total          |
| ---------------------- | ---------- | ---------- | ---------- | -------------- |
| **pytorch_mppi**       | 1214 lines | ~500 lines | ~800 lines | ~2500 lines    |
| **jax_mppi** (current) | 2919 lines | 2124 lines | 681 lines  | **5724 lines** |
| **Completion %**       | 240%       | 425%       | 85%        | **229%**       |

Core code now includes: mppi.py (353), smppi.py (634), kmppi.py (660), autotune.py (656), autotune_global.py (375), autotune_qd.py (218), plus supporting modules.

### Feature Parity Matrix

| Feature | pytorch_mppi | jax_mppi | Status |
| :--- | :--- | :--- | :--- |
| **Core MPPI Algorithm** | ✓ | ✓ | ✅ Complete |
| Basic sampling & weighting | ✓ | ✓ | ✅ |
| Control bounds (u_min/u_max) | ✓ | ✓ | ✅ |
| Control scaling (u_scale) | ✓ | ✓ | ✅ |
| Partial updates (u_per_command) | ✓ | ✓ | ✅ |
| Step-dependent dynamics | ✓ | ✓ | ✅ |
| Stochastic dynamics (rollout_samples) | ✓ | ✓ | ✅ |
| Sample null action | ✓ | ✓ | ✅ |
| Noise absolute cost | ✓ | ✓ | ✅ |
| Terminal cost function | ✓ | ✓ | ✅ |
| Shift nominal trajectory | ✓ | ✓ | ✅ |
| Get rollouts (visualization) | ✓ | ✓ | ✅ |
| Reset controller | ✓ | ✓ | ✅ |
| **Smooth MPPI (SMPPI)** | ✓ | ✓ | ✅ Complete |
| Action sequence tracking | ✓ | ✓ | ✅ |
| Smoothness penalty | ✓ | ✓ | ✅ |
| Separate action/control bounds | ✓ | ✓ | ✅ |
| Delta_t integration | ✓ | ✓ | ✅ |
| Shift with continuity | ✓ | ✓ | ✅ |
| **Kernel MPPI (KMPPI)** | ✓ | ✓ | ✅ Complete |
| Kernel interpolation | ✓ | ✓ | ✅ |
| RBF kernel | ✓ | ✓ | ✅ |
| Support point optimization | ✓ | ✓ | ✅ |
| Time grid management (Tk/Hs) | ✓ | ✓ | ✅ |
| Solve-based interpolation | ✓ | ✓ | ✅ |
| **Autotuning** | ✓ | ✓ | ✅ Complete |
| CMA-ES local tuning | ✓ | ✓ | ✅ |
| Ray Tune global search | ✓ | ✓ | ✅ |
| CMA-ME quality diversity | ✓ | ✓ | ✅ |
| Parameter types (lambda, sigma, mu, horizon) | ✓ | ✓ | ✅ |
| All MPPI variants support | ✓ | ✓ | ✅ |
| **Examples** | | | |
| Pendulum swing-up | ✓ | ✓ | ✅ Complete |
| Smooth MPPI comparison | ✓ | ✓ | ✅ Complete |
| Autotuning example | ✓ | ✓ | ✅ Complete |
| Pendulum with learned dynamics | ✓ | ✗ | 🔴 Not planned |

### Current File Structure

```text
jax_mppi/
├── pyproject.toml              ✅ Exists
├── README.md                   ✅ Exists
├── LICENSE                     ✅ Exists  
├── src/jax_mppi/
│   ├── __init__.py            ✅ Exists (updated for autotune)
│   ├── types.py               ✅ Exists (9 lines)
│   ├── mppi.py                ✅ Exists (353 lines) - COMPLETE
│   ├── smppi.py               ✅ Exists (634 lines) - COMPLETE
│   ├── kmppi.py               ✅ Exists (660 lines) - COMPLETE
│   ├── autotune.py            ✅ Exists (656 lines) - COMPLETE
│   ├── autotune_global.py     ✅ Exists (375 lines) - COMPLETE
│   ├── autotune_qd.py         ✅ Exists (218 lines) - COMPLETE
│   ├── costs/                 ✅ Exists (supporting modules)
│   └── dynamics/              ✅ Exists (supporting modules)
├── tests/
│   ├── test_mppi.py           ✅ Exists (115 lines) - COMPLETE
│   ├── test_pendulum.py       ✅ Exists (282 lines) - COMPLETE
│   ├── test_smppi.py          ✅ Exists (580 lines) - COMPLETE
│   ├── test_autotune.py       ✅ Exists (305 lines, 21 tests) - COMPLETE
│   └── test_autotune_integration.py ✅ Exists (247 lines, 4 tests) - COMPLETE
│   └── test_kmppi.py          ✅ Exists (595 lines) - COMPLETE
├── examples/
│   ├── pendulum.py            ✅ Exists (270 lines) - COMPLETE
│   ├── smooth_comparison.py   ✅ Exists (442 lines) - COMPLETE
│   ├── autotune_pendulum.py   ✅ Exists (321 lines) - COMPLETE
│   └── autotune_basic.py      ✅ Exists (90 lines) - COMPLETE
└── docs/
    └── plan/
        └── porting_pytorch_jax.md ✅ This file
```

### Recommended Next Steps

**Priority Order:**

1. **Phase 3: SMPPI Implementation** (High Priority)
   - Core functionality that adds smoothness to control
   - Estimated ~250-300 lines for smppi.py
   - Estimated ~150-200 lines for tests
   - Reference: `../pytorch_mppi/src/pytorch_mppi/mppi.py` (SMPPI class)

2. **Phase 4: KMPPI Implementation** (High Priority)
   - Novel contribution with kernel interpolation
   - Estimated ~300-350 lines for kmppi.py
   - Estimated ~150-200 lines for tests
   - Reference: `../pytorch_mppi/src/pytorch_mppi/mppi.py` (KMPPI class)

3. **Phase 5: Smooth Comparison Example** (Medium Priority)
   - Demonstrates value of SMPPI and KMPPI
   - Estimated ~200-250 lines
   - Reference: `../pytorch_mppi/tests/smooth_mppi.py`

4. **Additional Examples** (Low Priority)
   - Pendulum with learned dynamics
   - More complex environments

5. **Phase 6: Autotuning** (Optional/Stretch)
   - Advanced feature for hyperparameter optimization
   - Estimated ~300-400 lines
   - Reference: `../pytorch_mppi/src/pytorch_mppi/autotune.py`

## Design Decisions

### API Style: Functional with dataclass state containers

Use `@jax.tree_util.register_dataclass` (or `flax.struct.dataclass`) to hold MPPI state (nominal trajectory `U`, PRNG key, config). All core functions are pure: `command(state, mppi_state) -> (action, mppi_state)`.

**Rationale:** Idiomatic JAX — pure functions compose with `jit`, `vmap`, `grad`. No mutable `self`. Avoids heavyweight dependencies like Equinox for what is fundamentally a numerical algorithm.

### Key JAX mappings from PyTorch

| PyTorch                                    | JAX                              |
| ------------------------------------------ | -------------------------------- |
| `torch.distributions.MultivariateNormal`   | `jax.random.multivariate_normal` |
| `tensor.to(device)`                        | `jax.device_put` / automatic     |
| Python for-loop over horizon               | `jax.lax.scan`                   |
| `@handle_batch_input` decorator            | `jax.vmap`                       |
| `torch.roll`                               | `jnp.roll`                       |
| `torch.linalg.solve`                       | `jnp.linalg.solve`               |
| In-place mutation (`self.U = ...`)         | Return new state (pytree)        |

---

## Notes from `../pytorch_mppi` review (Jan 2026)

Actionable parity items to carry over:

- **SMPPI semantics:** maintains `action_sequence` separately from lifted control `U`; integrates with `delta_t`; smoothness cost from `diff(action_sequence)`.
- **SMPPI bounds:** support `action_min`/`action_max` distinct from `u_min`/`u_max` (control-derivative bounds).
- **KMPPI internals:** keep `theta` as control points; build `Tk`/`Hs` time grids; kernel interpolation via `solve(Ktktk, K)`; batch interpolation with `vmap`.
- **Sampling options:** `rollout_samples` (M), `sample_null_action`, `noise_abs_cost` (abs(noise) in action cost).
- **Rollouts:** `get_rollouts` handles `state` batch and dynamics that may augment state (take first `nx`).

---

## Package Structure

```text
jax_mppi/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/jax_mppi/
│   ├── __init__.py          # Public API exports
│   ├── mppi.py              # Core MPPI (MPPIConfig, MPPIState, command, reset, etc.)
│   ├── smppi.py             # Smooth MPPI variant
│   ├── kmppi.py             # Kernel MPPI variant + TimeKernel / RBFKernel
│   ├── types.py             # Type aliases, protocols for Dynamics/Cost callables
│   └── autotune.py          # Autotuning (CMA-ES wrapper, parameter search)
├── tests/
│   ├── test_mppi.py         # Unit tests for core MPPI
│   ├── test_smppi.py        # Unit tests for SMPPI
│   ├── test_kmppi.py        # Unit tests for KMPPI
│   └── test_pendulum.py     # Integration test with pendulum env
├── examples/
│   ├── pendulum.py          # Gym pendulum with true dynamics
│   ├── pendulum_approximate.py  # Learned dynamics
│   └── smooth_comparison.py # MPPI vs SMPPI vs KMPPI
└── docs/
    └── plan/
```

---

## Phased Implementation

### Phase 1: Project scaffolding + Core MPPI

**Files:** `pyproject.toml`, `src/jax_mppi/types.py`, `src/jax_mppi/mppi.py`, `src/jax_mppi/__init__.py`

1. **`pyproject.toml`** — project metadata, deps: `jax[cuda13]`, `jaxlib`, optional `gymnasium` for examples.

2. **`types.py`** — Type definitions:

   ```python
   # Dynamics: (state, action) -> next_state  or  (state, action, t) -> next_state
   DynamicsFn = Callable[..., jax.Array]
   # Cost: (state, action) -> scalar_cost  or  (state, action, t) -> scalar_cost
   RunningCostFn = Callable[..., jax.Array]
   # Terminal: (states, actions) -> scalar_cost
   TerminalCostFn = Callable[[jax.Array, jax.Array], jax.Array]
   ```

3. **`mppi.py`** — Core implementation:

   **Data structures (registered as JAX pytrees):**

   ```python
   @dataclass
   class MPPIConfig:
       # Static config (not traced through JAX)
       num_samples: int       # K
       horizon: int           # T
       nx: int
       nu: int
       lambda_: float
       u_scale: float
       u_per_command: int
       step_dependent_dynamics: bool
       rollout_samples: int   # M
       rollout_var_cost: float
       rollout_var_discount: float
       sample_null_action: bool
       noise_abs_cost: bool

   @dataclass
   class MPPIState:
       # Dynamic state (carried through JAX transforms)
       U: jax.Array           # (T, nu) nominal trajectory
       u_init: jax.Array      # (nu,) default action for shift
       noise_mu: jax.Array    # (nu,)
       noise_sigma: jax.Array # (nu, nu)
       noise_sigma_inv: jax.Array
       u_min: jax.Array | None
       u_max: jax.Array | None
       key: jax.Array         # PRNG key
   ```

   **Functions:**

   ```python
   def create(
       nx, nu, noise_sigma, num_samples=100, horizon=15, lambda_=1.0,
       noise_mu=None, u_min=None, u_max=None, u_init=None, U_init=None,
       u_scale=1, u_per_command=1, step_dependent_dynamics=False,
       rollout_samples=1, rollout_var_cost=0., rollout_var_discount=0.95,
       sample_null_action=False, noise_abs_cost=False, key=None,
   ) -> tuple[MPPIConfig, MPPIState]:
       """Factory: create config + initial state."""

   def command(
       config: MPPIConfig,
       mppi_state: MPPIState,
       current_obs: jax.Array,
       dynamics: DynamicsFn,
       running_cost: RunningCostFn,
       terminal_cost: TerminalCostFn | None = None,
       shift: bool = True,
   ) -> tuple[jax.Array, MPPIState]:
       """Compute optimal action and return updated state."""

   def reset(config: MPPIConfig, mppi_state: MPPIState, key: jax.Array) -> MPPIState:
       """Reset nominal trajectory."""

   def get_rollouts(
       config: MPPIConfig, mppi_state: MPPIState,
       current_obs: jax.Array, dynamics: DynamicsFn,
       num_rollouts: int = 1,
   ) -> jax.Array:
       """Forward-simulate trajectories for visualization."""
   ```

   **Internal functions (all JIT-compatible):**
   - `_shift_nominal(mppi_state) -> MPPIState` — `jnp.roll` + set last to `u_init`
   - `_sample_noise(key, K, T, noise_mu, noise_sigma) -> (noise, new_key)` — sample from multivariate normal
   - `_compute_rollout_costs(config, current_obs, perturbed_actions, dynamics, running_cost, terminal_cost)` — uses `jax.lax.scan` over horizon, `jax.vmap` over K samples
   - `_compute_weights(costs, lambda_)` — softmax importance weighting
   - `_bound_action(action, u_min, u_max)` — `jnp.clip`

   **Key JAX patterns:**
   - Rollout loop: `jax.lax.scan` with carry = `(state,)`, xs = `actions[t]`
   - Batch over K samples: `jax.vmap(_single_rollout, in_axes=(0, None, ...))`
   - Batch over M rollout samples (stochastic dynamics): nested vmap or scan
   - All internal functions decorated with `@jax.jit` or called inside a top-level jitted `command`

4. **Unit test:** `tests/test_mppi.py`
   - Test `create()` produces valid config/state
   - Test `command()` returns correct shape
   - Test cost reduction over iterations on simple 1D problem
   - Test bounds are respected

### Phase 2: Pendulum example (integration test)

**Files:** `examples/pendulum.py`, `tests/test_pendulum.py`

1. Implement pendulum dynamics as a pure JAX function (no gym dependency for core test)
2. Run MPPI loop, verify convergence (swing-up or stabilization)
3. Optional: gym rendering wrapper for visualization

### Phase 3: Smooth MPPI (SMPPI)

**Files:** `src/jax_mppi/smppi.py`, `tests/test_smppi.py`

1. **Data structures:**

   ```python
   @dataclass
   class SMPPIState(MPPIState):
       action_sequence: jax.Array  # (T, nu) actual actions
       w_action_seq_cost: float
       delta_t: float
       action_min: jax.Array | None
       action_max: jax.Array | None
   ```

2. **Functions:** Same API as `mppi.py` but with:
   - `_shift_nominal` shifts both `U` (velocity) and `action_sequence`
   - `_compute_perturbed_actions` integrates velocity to get actions
   - `_compute_total_cost` adds smoothness penalty: `||diff(actions)||^2`
   - `reset()` zeros both `U` and `action_sequence`
   - `change_horizon()` keeps both `U` and `action_sequence` in sync (truncate/extend)

3. **Test:** Verify smoother trajectories than base MPPI on 2D navigation

### Phase 4: Kernel MPPI (KMPPI)

**Files:** `src/jax_mppi/kmppi.py`, `tests/test_kmppi.py`

1. **Kernel abstractions:**

   ```python
   def rbf_kernel(t, tk, sigma=1.0):
       d = jnp.sum((t[:, None] - tk) ** 2, axis=-1)
       return jnp.exp(-d / (2 * sigma ** 2 + 1e-8))

   def kernel_interpolate(t, tk, coeffs, kernel_fn):
       K_t_tk = kernel_fn(t, tk)
       K_tk_tk = kernel_fn(tk, tk)
       weights = jnp.linalg.solve(K_tk_tk, K_t_tk.T).T
       return weights @ coeffs
   ```

2. **Data structures:**

   ```python
   @dataclass
   class KMPPIState(MPPIState):
       theta: jax.Array         # (num_support_pts, nu)
       num_support_pts: int
   ```

3. **Functions:** Override `_compute_perturbed_actions` to sample sparse + interpolate. Update `theta` instead of `U`.
   - Build `Tk` and `Hs` time grids on init and on horizon changes
   - Use `kernel_interpolate()` with `solve(Ktktk, K)` (avoid explicit inverse)
   - Batch interpolate with `jax.vmap` for K samples

4. **Test:** Verify fewer parameters produce smooth trajectories

### Phase 5: Smooth comparison example

**Files:** `examples/smooth_comparison.py`

- Side-by-side MPPI vs SMPPI vs KMPPI on 2D navigation
- Plot trajectories and control signals

### Phase 6: Autotuning (stretch goal)

**Files:** `src/jax_mppi/autotune.py`

- Wrap CMA-ES (`cmaes` or `evosax` for JAX-native) for sigma/lambda/horizon tuning
- Simpler than pytorch_mppi's framework — skip Ray Tune and QD initially
- Functional API: `tune_step(eval_fn, params, optimizer_state) -> (params, optimizer_state)`

---

## Verification Strategy

1. **Unit tests (per phase):** `pytest tests/` — shape checks, cost reduction, bounds
2. **Pendulum benchmark:** Compare convergence (total reward) against pytorch_mppi on same scenario
3. **JIT correctness:** Ensure `jax.jit(command)` produces identical results to non-jitted version
4. **Performance:** Benchmark `command()` latency vs pytorch_mppi (JAX should win after warmup due to XLA compilation)
5. **Smooth variants:** Visual comparison of trajectory smoothness

### Test setup options (src layout)

IMPORTANT: You should always use the virtual environment. To run the tests and all of the other python files.

- Option A: add a `tests/conftest.py` to insert `src` into `sys.path`.
- Option B: run tests after `uv pip install -e .` (editable install).

## Dependencies

**Core:** `jax[cuda13]`, `jaxlib`, `numpy`
**Testing:** `pytest`, `gymnasium[classic_control]`
**Autotuning (optional):** `cmaes` or `evosax`
**Examples (optional):** `matplotlib`, `gymnasium`

---

## Actionable Task Checklist

### Core MPPI (Phase 1)

- [x] Mirror `pytorch_mppi` signature flags: `rollout_samples`, `sample_null_action`, `noise_abs_cost`.
- [x] Implement `get_rollouts` handling: accept single or batched `state`; allow dynamics that augment state (take `:nx`).
- [x] Add `shift_nominal_trajectory` via `jnp.roll` + `u_init` fill.
- [x] Implement action cost with optional `abs(noise)` branch.
- [x] Add `u_per_command` slicing and `u_scale` application in `command`.

### SMPPI (Phase 3)

- [x] Carry `action_sequence` in state and integrate `U` with `delta_t`.
- [x] Implement distinct action bounds (`action_min`/`action_max`) vs control bounds (`u_min`/`u_max`).
- [x] Add smoothness cost from `diff(action_sequence)` and weight `w_action_seq_cost`.
- [x] Ensure `reset()` updates both `U` and `action_sequence`.
- [x] Implement proper shift with action continuity (hold last value).
- [x] Implement dual bounding system (_bound_control and _bound_action).
- [x] Recompute effective noise after bounding for accurate cost.

### KMPPI (Phase 4)

- [x] Implement `theta` control points + interpolation kernel (RBF by default).
- [x] Build `Tk`/`Hs` grids and re-build on horizon changes.
- [x] Use `solve(Ktktk, K)` for interpolation weights (no explicit inverse).
- [x] Shift `theta` via interpolation when shifting nominal trajectory.
- [x] Implement RBFKernel with configurable sigma.
- [x] Noise sampling in control point space.
- [x] Batched interpolation with vmap.

### Autotune + Examples (Phase 6)

- [ ] Mirror autotune interface from `pytorch_mppi/autotune*.py` at a minimal level (evaluation fn + optimizer loop).
- [ ] Port `tests/auto_tune_parameters.py` logic into a JAX-friendly example.
