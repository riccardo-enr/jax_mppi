# Testing Guide

This guide explains the testing stack for `jax_mppi` and provides instructions on how to run and write tests.

## Running Tests

The project uses `pytest` for running tests. You can run all tests using `uv`:

```bash
uv run pytest
```

To run a specific test file:

```bash
uv run pytest tests/test_mppi.py
```

To run a specific test case:

```bash
uv run pytest tests/test_mppi.py::TestMPPICommand::test_command_returns_correct_shapes
```

## Test Suite Structure

The tests are located in the `tests/` directory and mirror the source code structure where appropriate. The test suite is divided into several files, each covering a specific flavor or aspect of the library.

### Core MPPI Flavors

*   **`tests/test_mppi.py`**: Tests for the base MPPI implementation (`jax_mppi.mppi`).
    *   **Goal**: Ensure the correctness of the core algorithm, state management, and configuration options.
    *   **Scope**:
        *   **Initialization**: Verifies that `create()` returns correct shapes and types for `config` and `state`.
        *   **Command Generation**: Tests the `command()` function to ensure it generates valid actions within bounds and correctly updates the state.
        *   **Configuration Options**: Validates various settings like `u_per_command` (multi-step control), `step_dependent_dynamics` (time-varying systems), `sample_null_action` (ensuring baseline inclusion), and `u_scale` (control authority scaling).
        *   **Integration**: Includes basic convergence tests to verify that the cost decreases over iterations (e.g., `TestMPPIIntegration`).

*   **`tests/test_smppi.py`**: Tests for Smooth MPPI (`jax_mppi.smppi`).
    *   **Goal**: Verify that the "smooth" variant correctly operates in the lifted velocity control space and produces continuous action sequences.
    *   **Scope**:
        *   **Lifted Space**: Checks that the internal state (`U`) represents control velocity/acceleration, while `action_sequence` represents the integrated actions.
        *   **Smoothness**: Verifies that the smoothness cost penalty (`w_action_seq_cost`) effectively reduces action variance.
        *   **Bounds**: Tests that bounds are respected for both the control velocity (`u_min`/`u_max`) and the final action (`action_min`/`action_max`).
        *   **Continuity**: checks that the `shift` operation maintains continuity in the action space, preventing jumps during receding horizon updates.

*   **`tests/test_kmppi.py`**: Tests for Kernel MPPI (`jax_mppi.kmppi`).
    *   **Goal**: Ensure that kernel-based interpolation works correctly and that optimization occurs effectively in the reduced control point space.
    *   **Scope**:
        *   **Kernels**: Tests the properties of time-domain kernels (e.g., `RBFKernel`), such as shape and distance decay.
        *   **Interpolation**: Verifies that control points (`theta`) are correctly mapped to full trajectories (`U`) via `_kernel_interpolate`, preserving values at control points.
        *   **Optimization**: Checks that the MPPI update rule is applied to the control points (`theta`) rather than the full trajectory.
        *   **Smoothness**: Confirms that the resulting trajectories are smooth due to the kernel properties (e.g., by checking second derivatives).

### Integration & Examples

*   **`tests/test_pendulum.py`**: End-to-end integration tests using a Pendulum environment.
    *   **Goal**: Validate that the algorithms can solve a concrete, non-linear control task.
    *   **Scope**:
        *   **Stabilization**: Tests if MPPI can stabilize the pendulum at the upright position.
        *   **Swing-up**: Tests the more difficult task of swinging up from a hanging position.
        *   **Physics**: Sanity checks the pendulum dynamics and cost functions.

### Autotuning

*   **`tests/test_autotune.py`**: Unit tests for the autotuning framework (`jax_mppi.autotune`).
    *   **Goal**: Verify the components of the hyperparameter optimization system.
*   **`tests/test_autotune_integration.py`**: Integration tests for autotuning.
    *   **Goal**: Ensure that the autotuner can successfully improve performance on a benchmark task (finding better parameters than the default).

## Writing New Tests

When adding new features or fixing bugs, please add corresponding tests.

1.  **Locate the appropriate test file**: If you are modifying `mppi.py`, add tests to `tests/test_mppi.py`.
2.  **Use Class-Based Structure**: Group related tests into classes (e.g., `TestMPPIBasics`, `TestMPPICommand`).
3.  **Property-Based Testing**: Where possible, test properties (e.g., "output shape depends on input shape in this way") rather than just hardcoded values.
4.  **Integration Tests**: For significant algorithmic changes, ensure that `tests/test_pendulum.py` still passes or add a similar simple control task to verify efficacy.
5.  **JAX Compatibility**: Ensure tests check that functions can be JIT-compiled if they are intended to be used within `jax.jit`.

### Example Test Case

```python
def test_new_feature(self):
    nx, nu = 2, 1
    config, state = mppi.create(nx=nx, nu=nu, noise_sigma=jnp.eye(nu))

    # ... perform action ...
    action, new_state = mppi.command(config, state, ...)

    # ... assert expected behavior ...
    assert action.shape == (nu,)
```
