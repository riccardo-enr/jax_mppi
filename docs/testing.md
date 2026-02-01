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
    *   Covers creation of config and state.
    *   Verifies shapes and types of inputs/outputs.
    *   Tests command generation, including bounding and noise sampling.
    *   Tests various configuration options like `u_per_command`, `step_dependent_dynamics`, `sample_null_action`, `u_scale`, etc.
    *   Includes integration tests verifying cost reduction.

*   **`tests/test_smppi.py`**: Tests for Smooth MPPI (`jax_mppi.smppi`).
    *   Tests the lifted control space (velocity control) and action integration.
    *   Verifies smoothness properties and cost penalties on action variations.
    *   Tests bounds on both control velocity (`U`) and final actions.
    *   Checks the `shift` operation which must maintain continuity in action space.

*   **`tests/test_kmppi.py`**: Tests for Kernel MPPI (`jax_mppi.kmppi`).
    *   Tests kernel functions (e.g., `RBFKernel`) and interpolation mechanics.
    *   Verifies that control points (`theta`) are correctly mapped to full trajectories (`U`).
    *   Tests that optimization happens in the reduced control point space.
    *   Checks smoothness properties resulting from kernel interpolation.

### Integration & Examples

*   **`tests/test_pendulum.py`**: End-to-end integration tests using a Pendulum environment.
    *   Verifies that MPPI can stabilize and swing up a pendulum.
    *   Checks that physics and cost functions behave as expected.
    *   Serves as a sanity check that the algorithm actually solves control tasks.

### Autotuning

*   **`tests/test_autotune.py`**: Unit tests for the autotuning framework (`jax_mppi.autotune`).
*   **`tests/test_autotune_integration.py`**: Integration tests for autotuning, verifying that it can optimize hyperparameters for a given task.

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
