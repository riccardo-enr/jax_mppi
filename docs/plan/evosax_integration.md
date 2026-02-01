# Evosax Integration Plan

**Goal:** Add evosax as a JAX-native optimization backend for the autotuning framework in jax-mppi.

**Status:** In Progress

---

## Overview

Evosax is a JAX-native library for evolutionary strategies that provides highly efficient, JIT-compiled optimization algorithms. Integrating it into the autotuning framework will:

1. **Provide JAX-native optimization** - Full JIT compilation of the entire tuning loop
2. **Enable GPU acceleration** - Evolutionary strategies running entirely on GPU
3. **Add diverse algorithms** - CMA-ES, OpenES, SNES, Sep-CMA-ES, and more
4. **Improve performance** - Eliminate Python overhead in the optimization loop
5. **Simplify dependencies** - Pure JAX implementation, no external C++ dependencies

### Why Evosax?

- **Performance:** Fully JIT-compiled ES algorithms vs. Python-based `cma` library
- **JAX ecosystem fit:** Natural integration with JAX-based MPPI code
- **GPU support:** Can run entire autotuning on GPU without host-device transfers
- **Algorithm variety:** 15+ evolutionary strategies in one package
- **Maintained:** Active development and well-documented

---

## Current State Analysis

### Existing Autotuning Architecture

The current autotuning system has three modules:

1. **`autotune.py`** (656 lines) - Core framework + CMA-ES via `cma` library
   - Abstract `Optimizer` base class
   - `CMAESOpt` using Python `cma` package
   - Parameter abstractions: `LambdaParameter`, `NoiseSigmaParameter`, `MuParameter`, `HorizonParameter`
   - `Autotune` orchestrator class

2. **`autotune_global.py`** (375 lines) - Ray Tune for global search
   - `RayOptimizer` for distributed hyperparameter search
   - Global parameter variants with search spaces
   - Integration with HyperOpt and BayesOpt

3. **`autotune_qd.py`** (218 lines) - CMA-ME quality diversity
   - `CMAMEOpt` using `ribs` library
   - Archive-based diversity preservation

### Architecture Pattern

All optimizers follow the `Optimizer` ABC:

```python
class Optimizer(abc.ABC):
    @abc.abstractmethod
    def setup_optimization(
        self, initial_params: np.ndarray,
        evaluate_fn: Callable[[np.ndarray], EvaluationResult]
    ) -> None:
        ...

    @abc.abstractmethod
    def optimize_step(self) -> EvaluationResult:
        ...

    def optimize_all(self, iterations: int) -> EvaluationResult:
        ...
```

---

## Implementation Plan

### Step 1: Add evosax dependency

**File:** `pyproject.toml`

**Changes:**
- Add `evosax>=0.1.0` to `[project.optional-dependencies.autotuning]` section
- Add to `[project.optional-dependencies.dev]` section as well

**Reasoning:** Keep evosax optional like other autotuning dependencies, allowing users to install only what they need.

---

### Step 2: Create evosax optimizer module ✅ DONE

**New File:** `src/jax_mppi/autotune_evosax.py` (~387 lines)

**Contents (Implemented):**

#### 2.1 EvoSaxOptimizer base class

Implements `Optimizer` ABC with evosax backends:

```python
class EvoSaxOptimizer(Optimizer):
    """JAX-native evolutionary strategies using evosax.

    Fully JIT-compiled optimization loop with GPU support.

    Attributes:
        strategy: evosax strategy name (e.g., "CMA_ES", "OpenES", "SNES")
        population: Population size
        num_generations: Number of generations per optimize_step
        maximize: Whether to maximize objective (default: False for cost minimization)
    """

    def __init__(
        self,
        strategy: str = "CMA_ES",
        population: int = 10,
        num_generations: int = 1,
        sigma_init: float = 0.1,
        maximize: bool = False,
        es_params: dict | None = None,
    ):
        ...
```

**Key features:**
- Strategy selection from evosax's 15+ algorithms
- JIT-compiled ask-evaluate-tell loop
- Support for both single-step and batch optimization
- Configurable ES hyperparameters via `es_params`

#### 2.2 JAX-native evaluation wrapper

```python
def _create_jax_evaluate_fn(
    evaluate_fn: Callable[[np.ndarray], EvaluationResult],
    maximize: bool = False,
) -> Callable[[jax.Array], float]:
    """Wrap Python evaluation function for JAX compatibility.

    Handles conversion between JAX arrays and numpy arrays,
    extracts scalar cost from EvaluationResult.
    """
    ...
```

**Challenge:** The user-provided `evaluate_fn` may not be JAX-pure (e.g., it might use numpy, modify state, etc.). Need to handle this gracefully.

**Solutions:**
- **Option A:** Use `jax.pure_callback` to call non-pure evaluation functions
- **Option B:** Require evaluation function to be JAX-pure for evosax optimizer
- **Option C:** Provide both pure and impure modes

**Recommendation:** Start with Option C - detect if evaluation is JAX-pure, use direct JIT if yes, use `pure_callback` if no.

#### 2.3 Batched evaluation support

Evosax can leverage `vmap` for parallel fitness evaluation:

```python
def _batch_evaluate(
    solutions: jax.Array,  # (population, param_dim)
    evaluate_fn: Callable,
) -> jax.Array:  # (population,)
    """Evaluate population in parallel using vmap."""
    return jax.vmap(evaluate_fn)(solutions)
```

**Benefits:**
- GPU parallelization of fitness evaluations
- Significant speedup when dynamics/cost are JIT-compiled

**Challenges:**
- Requires evaluation to be JAX-pure and vmappable
- May need sequential fallback for non-pure evaluations

#### 2.4 Algorithm-specific optimizers

Provide convenience classes for common strategies:

```python
class CMAESOpt(EvoSaxOptimizer):
    """CMA-ES optimizer (evosax backend)."""
    def __init__(self, population: int = 10, sigma: float = 0.1, **kwargs):
        super().__init__(strategy="CMA_ES", population=population,
                         sigma_init=sigma, **kwargs)

class OpenESOpt(EvoSaxOptimizer):
    """OpenAI's Evolution Strategies."""
    def __init__(self, population: int = 100, sigma: float = 0.1, **kwargs):
        super().__init__(strategy="OpenES", population=population,
                         sigma_init=sigma, **kwargs)

class SepCMAESOpt(EvoSaxOptimizer):
    """Separable CMA-ES (faster for high dimensions)."""
    def __init__(self, population: int = 10, sigma: float = 0.1, **kwargs):
        super().__init__(strategy="Sep_CMA_ES", population=population,
                         sigma_init=sigma, **kwargs)
```

---

### Step 3: Update main autotune module ✅ DONE

**File:** `src/jax_mppi/autotune.py`

**Changes:**
- Update module docstring to mention evosax as an option
- Update examples to show evosax usage

**Example addition:**
```python
>>> # With evosax (JAX-native, GPU-accelerated)
>>> from jax_mppi import autotune_evosax
>>> tuner = jmppi.autotune.Autotune(
...     params_to_tune=[...],
...     evaluate_fn=evaluate,
...     optimizer=autotune_evosax.CMAESOpt(population=10),
... )
```

---

### Step 4: Update package exports ✅ DONE

**File:** `src/jax_mppi/__init__.py`

**Changes:**
```python
# Add conditional import
try:
    from . import autotune_evosax
except ImportError:
    autotune_evosax = None  # evosax not installed
```

**Reasoning:** Keep it optional - don't break imports if evosax isn't installed.

---

### Step 5: Create tests ✅ DONE

**New File:** `tests/test_autotune_evosax.py` (~408 lines)

**Test coverage:**

1. **Basic functionality tests**
   - Test EvoSaxOptimizer initialization with various strategies
   - Test setup_optimization() creates valid ES state
   - Test optimize_step() returns EvaluationResult
   - Test optimize_all() finds better solutions

2. **Strategy-specific tests**
   - Test CMAESOpt on simple quadratic function
   - Test OpenESOpt convergence
   - Test SepCMAESOpt on high-dimensional problem

3. **Integration tests**
   - Test with actual MPPI parameter tuning (simple 1D system)
   - Test with LambdaParameter and NoiseSigmaParameter
   - Verify results comparable to CMAESOpt from `cma` library

4. **JAX-specific tests**
   - Test JIT compilation of optimization loop
   - Test with JAX-pure evaluation function
   - Test with non-pure evaluation function (using pure_callback)
   - Test batched evaluation with vmap

5. **Edge cases**
   - Test with single parameter
   - Test with multi-dimensional parameters
   - Test parameter bounds enforcement
   - Test with invalid strategy name (should raise clear error)

**Test structure example:**
```python
def test_evosax_cmaes_simple():
    """Test CMA-ES on simple quadratic objective."""
    # Minimize ||x - target||^2
    target = np.array([1.0, 2.0, 3.0])

    def evaluate_fn(x: np.ndarray) -> EvaluationResult:
        cost = float(np.sum((x - target) ** 2))
        return EvaluationResult(
            mean_cost=cost,
            rollouts=jnp.zeros((1, 1, 3)),
            params={},
            iteration=0,
        )

    optimizer = EvoSaxOptimizer(strategy="CMA_ES", population=10)
    optimizer.setup_optimization(
        initial_params=np.zeros(3),
        evaluate_fn=evaluate_fn,
    )

    best = optimizer.optimize_all(iterations=50)

    # Should converge close to target
    assert best.mean_cost < 0.1
```

---

### Step 6: Create example ✅ DONE

**New File:** `examples/autotune_evosax_comparison.py` (~307 lines)

**Purpose:** Compare evosax vs. cma library performance

**Contents:**
1. Setup simple pendulum MPPI tuning task
2. Run with `CMAESOpt` from `cma` library (baseline)
3. Run with `CMAESOpt` from evosax
4. Run with other evosax strategies (OpenES, Sep-CMA-ES)
5. Compare:
   - Convergence speed (iterations to threshold)
   - Wall-clock time
   - Final performance
6. Generate comparison plots:
   - Convergence curves for each optimizer
   - Time comparison bar chart
   - Parameter trajectory plots

**Expected outcome:** Evosax should be faster in wall-clock time due to JIT compilation, especially with GPU.

---

### Step 7: Documentation ✅ DONE

#### 7.1 Update README.md ✅ DONE

Add evosax to the autotuning section:

```markdown
### Autotuning

JAX-MPPI supports automatic hyperparameter tuning with multiple backends:

- **CMA-ES** (via `cma` library) - Classic evolution strategy
- **CMA-ES** (via `evosax`) - JAX-native, GPU-accelerated  ⚡ **NEW**
- **Ray Tune** - Distributed hyperparameter search
- **CMA-ME** (via `ribs`) - Quality diversity optimization

Install with: `pip install jax-mppi[autotuning] evosax`
```

#### 7.2 Add evosax examples to docstrings

Update autotune.py module docstring with evosax example.

#### 7.3 Create migration guide

Document for users switching from `cma` to `evosax`:

```markdown
## Migrating from cma to evosax

**Before:**
```python
from jax_mppi.autotune import CMAESOpt
opt = CMAESOpt(population=10, sigma=0.1)
```

**After:**
```python
from jax_mppi.autotune_evosax import CMAESOpt
opt = CMAESOpt(population=10, sigma=0.1)
```

**Benefits:** 5-10x faster with GPU acceleration
```

---

## Implementation Details

### Handling Non-Pure Evaluations

The main challenge is that user evaluation functions may not be JAX-pure.

**Strategy:**

```python
def _wrap_evaluation(evaluate_fn, maximize):
    """Wrap evaluation function for JAX compatibility."""

    # Try to detect if function is JAX-pure
    # by checking if it uses only JAX operations

    def jax_eval(x: jax.Array) -> float:
        # Convert to numpy for non-pure functions
        x_np = np.array(x)
        result = evaluate_fn(x_np)
        cost = result.mean_cost
        return -cost if maximize else cost

    # For non-pure functions, use pure_callback
    if not _is_jax_pure(evaluate_fn):
        @jax.jit
        def wrapped(x):
            return jax.pure_callback(
                jax_eval,
                jax.ShapeDtypeStruct((), jnp.float32),
                x
            )
        return wrapped
    else:
        return jax.jit(jax_eval)
```

### Batched vs Sequential Evaluation

Provide both modes:

```python
class EvoSaxOptimizer(Optimizer):
    def __init__(self, ..., batched_evaluation: bool = False):
        self.batched_evaluation = batched_evaluation

    def optimize_step(self):
        solutions = self.es_state.ask()

        if self.batched_evaluation:
            # Parallel evaluation with vmap
            fitness = jax.vmap(self.evaluate_fn)(solutions)
        else:
            # Sequential evaluation (safer for non-pure functions)
            fitness = jnp.array([
                self.evaluate_fn(x) for x in solutions
            ])

        self.es_state = self.es_state.tell(fitness)
        ...
```

### Parameter Constraints

Evosax doesn't natively handle box constraints. Options:

1. **Rejection sampling:** Reject invalid samples (wasteful)
2. **Clipping:** Clip to bounds after sampling (biases distribution)
3. **Repair:** Project invalid samples to feasible region
4. **Penalty:** Add penalty for constraint violation

**Recommendation:** Use clipping (Option 2) for simplicity, add note in docs that proper constrained optimization should use constrained ES variants if needed.

```python
def _apply_bounds(x, lower, upper):
    """Apply box constraints via clipping."""
    if lower is not None or upper is not None:
        x = jnp.clip(x, lower, upper)
    return x
```

---

## Available Evosax Strategies

Strategies to support (from evosax):

### Gradient-free ES
1. **CMA_ES** - Classic Covariance Matrix Adaptation
2. **Sep_CMA_ES** - Separable CMA-ES (faster for high-dim)
3. **IPOP_CMA_ES** - Increasing population CMA-ES
4. **BIPOP_CMA_ES** - Bi-population CMA-ES
5. **OpenES** - OpenAI's Natural Evolution Strategies
6. **SNES** - Separable Natural Evolution Strategies
7. **xNES** - Exponential Natural Evolution Strategies
8. **SimpleGA** - Simple Genetic Algorithm
9. **PersistentES** - Persistent Evolution Strategies
10. **LES** - Learned Evolution Strategies (meta-learned)

### Recommendation for defaults:
- **Default:** CMA_ES (well-tested, robust)
- **High-dimensional:** Sep_CMA_ES (scales better)
- **Large population budget:** OpenES (naturally parallelizable)

---

## Migration from cma library

### Advantages of evosax

| Feature | `cma` library | `evosax` |
|---------|---------------|----------|
| Language | Python + C | Pure JAX |
| JIT compilation | ❌ | ✅ |
| GPU acceleration | ❌ | ✅ |
| Batched evaluation | ❌ | ✅ (via vmap) |
| Integration with JAX code | ⚠️ (numpy conversion) | ✅ (native) |
| Algorithm variety | CMA-ES variants only | 15+ strategies |
| Performance (CPU) | Good | Similar |
| Performance (GPU) | N/A | Excellent (5-10x) |

### When to use each

**Use `cma` library when:**
- You need the original CMA-ES implementation
- Your evaluation function has complex Python dependencies
- You're not using GPU

**Use `evosax` when:**
- You want GPU acceleration
- Your MPPI code is already JIT-compiled
- You want to experiment with different ES algorithms
- You need batched parallel evaluation

---

## Testing Strategy

### Unit tests
- Test each strategy initialization
- Test optimize_step produces valid results
- Test parameter bounds enforcement
- Test with different parameter dimensions

### Integration tests
- Compare convergence to `cma` library on same problems
- Test with actual MPPI parameter tuning
- Verify GPU execution (if GPU available)

### Performance benchmarks
- Compare wall-clock time vs `cma` library
- Measure JIT compilation overhead
- Profile GPU vs CPU performance

### Regression tests
- Ensure results are deterministic with fixed seed
- Verify backward compatibility with existing Optimizer API

---

## Potential Issues & Solutions

### Issue 1: JIT compilation overhead

**Problem:** First call to evosax optimizer incurs JIT compilation cost.

**Solution:**
- Document warmup requirement
- Provide `warmup()` method that JIT-compiles with dummy data
- Consider pre-compilation for common parameter dimensions

### Issue 2: Non-pure evaluation functions

**Problem:** Most user evaluation functions are not JAX-pure (use numpy, I/O, etc.).

**Solution:**
- Use `jax.pure_callback` to wrap impure functions
- Provide clear error messages when incompatible operations are detected
- Document limitations and workarounds

### Issue 3: Memory usage with large populations

**Problem:** GPU memory may be limited for large populations.

**Solution:**
- Add memory usage estimates in docs
- Provide chunk-based evaluation for very large populations
- Default to reasonable population sizes

### Issue 4: Determinism

**Problem:** JAX PRNG behaves differently than numpy.random.

**Solution:**
- Document PRNG handling
- Ensure reproducibility with fixed JAX random keys
- Provide utility to seed evosax optimizer

---

## Timeline Estimate

| Step | Description | Estimated Lines | Time |
|------|-------------|----------------|------|
| 1 | Add dependency to pyproject.toml | ~5 | 5 min |
| 2 | Implement autotune_evosax.py | ~350 | 4-6 hours |
| 3 | Update autotune.py docs | ~20 | 30 min |
| 4 | Update __init__.py | ~5 | 5 min |
| 5 | Create test_autotune_evosax.py | ~250 | 3-4 hours |
| 6 | Create example comparison script | ~200 | 2-3 hours |
| 7 | Update documentation | ~100 | 1-2 hours |
| **Total** | | **~930 lines** | **11-16 hours** |

---

## Success Criteria

### Functional Requirements
- [ ] EvoSaxOptimizer implements Optimizer ABC correctly
- [ ] All evosax strategies can be instantiated
- [ ] Optimization converges on test problems
- [ ] Integration with existing Autotune class works
- [ ] Parameter bounds are respected

### Performance Requirements
- [ ] Evosax is faster than `cma` on GPU (>2x speedup)
- [ ] Evosax is competitive with `cma` on CPU (within 20%)
- [ ] JIT compilation overhead is acceptable (<5s for typical problems)

### Quality Requirements
- [ ] All tests pass (>95% coverage)
- [ ] Documentation is clear and complete
- [ ] Examples run without errors
- [ ] Code follows existing style conventions

### Integration Requirements
- [ ] Works with all parameter types (Lambda, NoiseSigma, Mu, Horizon)
- [ ] Compatible with existing Autotune orchestrator
- [ ] No breaking changes to existing API
- [ ] Optional dependency (doesn't break install if evosax not available)

---

## Future Extensions

### Short-term (post-MVP)
1. Add more evosax strategies (GLD, LM-MA-ES, etc.)
2. Implement proper constrained optimization variants
3. Add support for multi-objective optimization
4. Create Jupyter notebook tutorial

### Medium-term
1. Integrate with autotune_qd.py for quality diversity
2. Add learned evolution strategies (LES) with meta-learning
3. Implement adaptive strategy selection
4. Add visualization of ES state (e.g., covariance ellipsoids)

### Long-term
1. Develop JAX-native quality diversity framework
2. Add support for multi-fidelity optimization
3. Implement distributed evosax with multi-GPU support
4. Create benchmarking suite comparing all optimizers

---

## References

- [evosax documentation](https://github.com/RobertTLange/evosax)
- [JAX documentation](https://jax.readthedocs.io/)
- Current autotuning implementation: `src/jax_mppi/autotune.py`
- Original pytorch_mppi autotune: `../pytorch_mppi/src/pytorch_mppi/autotune.py`

---

## Open Questions

1. **Should we deprecate the `cma` library backend?**
   - Probably not - keep both for compatibility
   - Users can choose based on their needs

2. **Should batched evaluation be default?**
   - No - requires JAX-pure evaluation functions
   - Make it opt-in with clear documentation

3. **Which evosax strategies should have convenience classes?**
   - Start with: CMA_ES, Sep_CMA_ES, OpenES
   - Add more based on user feedback

4. **Should we add evosax to core dependencies or keep it optional?**
   - Keep optional - maintains lightweight core
   - Document installation clearly

---

## Notes

- Evosax is actively maintained by Robert Lange
- Current version: 0.1.x (check latest before implementing)
- JAX-native means entire optimization loop can run on GPU
- Consider adding evosax to CI/CD pipeline for testing
