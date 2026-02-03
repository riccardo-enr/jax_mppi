# CUDA/C++ MPPI Implementation Plan

## Objective
Implement CUDA/C++ versions of MPPI, SMPPI, and KMPPI controllers within the `src/cuda_mppi` directory, using `../MPPI-Generic` as a reference for high-performance CUDA implementation patterns.

## Goals
1.  Create a C++/CUDA project structure within `src/cuda_mppi`.
2.  Implement the standard MPPI algorithm (mirroring `src/jax_mppi/mppi.py`).
3.  Implement the Smooth MPPI (SMPPI) algorithm (mirroring `src/jax_mppi/smppi.py`).
4.  Implement the Kernel MPPI (KMPPI) algorithm (mirroring `src/jax_mppi/kmppi.py`).
5.  Ensure the implementations are self-contained or have clear interfaces (even if not fully hooked up to Python yet).

## Directory Structure
We will create `src/cuda_mppi` with the following structure:

```
src/cuda_mppi/
├── CMakeLists.txt              # Build configuration
├── include/
│   └── mppi/
│       ├── controllers/
│       │   ├── mppi.cuh        # Standard MPPI header
│       │   ├── smppi.cuh       # Smooth MPPI header
│       │   └── kmppi.cuh       # Kernel MPPI header
│       ├── core/
│       │   ├── mppi_common.cuh # Common structures and utilities
│       │   └── kernels.cuh     # Shared CUDA kernels (rollout, cost, etc.)
│       ├── dynamics/
│       │   └── dynamics.cuh    # Dynamics interface and base classes
│       ├── costs/
│       │   └── costs.cuh       # Cost function interface
│       └── utils/
│           └── cuda_utils.cuh  # CUDA helper functions
└── src/
    ├── controllers/
    │   ├── mppi.cu             # Standard MPPI implementation
    │   ├── smppi.cu            # Smooth MPPI implementation
    │   └── kmppi.cu            # Kernel MPPI implementation
    ├── core/
    │   └── kernels.cu          # Kernel implementations
    └── utils/
        └── cuda_utils.cu       # Utility implementations
```

## Implementation Details

### 1. Core Components (`include/mppi/core/`)
-   **`mppi_common.cuh`**: Define data structures for state, configuration, and control sequences.
-   **`kernels.cuh`**:
    -   `rollout_kernel`: Generic kernel to propagate dynamics and compute costs for $K$ samples over $T$ timesteps.
    -   `reduce_cost_kernel`: Kernel to compute weighted averages of trajectories.

### 2. Dynamics & Costs Interfaces (`include/mppi/dynamics/`, `include/mppi/costs/`)
-   Define template interfaces or base classes for `Dynamics` and `RunningCost` so that specific system models (like Quadrotor) can be plugged in.
-   *Note*: Since we are focusing on the *controllers*, we will provide a simple example dynamics (e.g., Double Integrator or simple Quadrotor) to verify compilation, but the main focus is the controller logic.

### 3. Controllers

#### Standard MPPI (`mppi.cu/cuh`)
-   **Logic**:
    -   Sample noise $\epsilon \sim \mathcal{N}(0, \Sigma)$.
    -   Compute $u_{per} = u_{nom} + \epsilon$.
    -   Rollout dynamics using $u_{per}$.
    -   Compute costs $J(\tau)$.
    -   Compute weights $w \propto \exp(-J/\lambda)$.
    -   Update $u_{nom} \leftarrow u_{nom} + \sum w \epsilon$.
-   **CUDA**: Use block-per-sample or thread-per-sample approach depending on horizon/state size. `MPPI-Generic` often uses block-y striding.

#### Smooth MPPI (`smppi.cu/cuh`)
-   **Logic**:
    -   Sample noise in velocity space $\delta v$.
    -   Integrate to get actions $u$.
    -   Add smoothness cost $\sum (\Delta u)^2$.
    -   Update velocity sequence $v_{nom}$.
-   **CUDA**: Needs a kernel that handles the integration step (velocity -> action) before the rollout.

#### Kernel MPPI (`kmppi.cu/cuh`)
-   **Logic**:
    -   Control trajectory parameterized by control points $\theta$ and kernel $K(\cdot, \cdot)$.
    -   Sample noise on $\theta$.
    -   Interpolate $\theta \to u(t)$.
    -   Rollout.
    -   Update $\theta$.
-   **CUDA**: Needs a kernel multiplication/interpolation step before rollout.

## Execution Plan

1.  [x] **Setup**: Create directory structure and `CMakeLists.txt`.
2.  [x] **Common**: Implement `mppi_common.cuh` and basic `cuda_utils.cuh`.
3.  [x] **Dynamics/Cost**: Define minimal interfaces.
4.  [x] **MPPI**: Implement `mppi.cuh` and `mppi.cu`.
5.  [x] **SMPPI**: Implement `smppi.cuh` and `smppi.cu`.
6.  [x] **KMPPI**: Implement `kmppi.cuh` and `kmppi.cu`.
7.  [x] **Verification**: Create a dummy `main.cu` to instantiate these controllers and verify they compile.

## Python Integration (Phase 2)

### Objective
Expose the C++ MPPI controllers to Python to allow direct usage from the `jax_mppi` package, potentially replacing the JAX implementation for performance-critical sections.

### Strategy
1.  **Binding Library**: Use `nanobind` (efficient, small footprint) to create Python bindings for the C++ classes.
2.  **Data Transfer**:
    -   **Basic**: Accept NumPy arrays (CPU) and copy to GPU in C++.
    -   **Advanced (Zero-Copy)**: Accept `DLPack` capsules (from `jax.Array` or `torch.Tensor`) to pass GPU pointers directly to the C++ controllers, avoiding CPU-GPU transfers.

### Implementation Steps
1.  [x] **Project Config**: 
    -   Update `pyproject.toml` to support C++ extensions (e.g., using `scikit-build-core`).
    -   Add dependencies: `nanobind`, `scikit-build-core`.
2.  [x] **Bindings Code**:
    -   Create `src/cuda_mppi/bindings/bindings.cpp`.
    -   Expose `MPPIConfig` struct as a Python class.
    -   [x] Expose `MPPIController`, `SMPPIController`, `KMPPIController` classes.
    -   [x] Bind methods like `compute(state)` and `get_action()`.
    -   [x] Implement type casters for `Eigen::VectorXf` <-> `numpy.ndarray` (using `nanobind/eigen/dense.h`).
3.  [x] **CMake Update**:
    -   Add `nanobind_add_module` target.
    -   Link against `cuda_mppi` and CUDA libraries.
4.  [x] **Integration**:
    -   Create a Python wrapper module (e.g., `jax_mppi.cuda`) that imports the extension.
    -   Add tests in `tests/` to verify correctness against the JAX implementation.

## Phase 3: Runtime Dynamics Compilation (NVRTC) ✅

### Objective
Allow users to define dynamics and cost functions in Python (initially as C++ code strings, or eventually transpiled from JAX) and compile the specialized MPPI controller at runtime. This avoids the need to recompile the shared library for every new system.

### Strategy
1.  **NVRTC (NVIDIA Runtime Compilation)**: Use NVRTC to compile CUDA C++ code strings into PTX at runtime.
2.  **CUDA Driver API**: Use the Driver API (`cuModuleLoadData`, `cuLaunchKernel`) to load the compiled PTX and launch the `rollout_kernel`.
3.  **Warm Start**: The compilation happens once during the "warm start" phase (controller initialization), enabling high-performance rollouts thereafter.

### Implementation Steps
1.  [x] **Build Config**: Link against `nvrtc` and `cuda` (Driver API).
2.  [x] **JIT Compiler Class (`src/cuda_mppi/include/mppi/jit/jit_compiler.hpp`)**:
    -   Inputs: Strings for `dynamics_struct_code` and `cost_struct_code`.
    -   Action: Constructs the full `.cu` source code (headers + user structs + template instantiation).
    -   Output: Compiles to PTX using `nvrtcProgramCompile`.
    -   Updated wrapper generation to work with Driver API.
3.  [x] **JIT Controller (`JITMPPIController`)**:
    -   A generic controller class that holds `CUfunction` handles instead of hardcoded kernels.
    -   `compute()` method launches the generated kernel via `cuLaunchKernel`.
    -   Implemented in `include/mppi/controllers/jit_mppi.hpp` and `src/jit/jit_mppi_controller.cpp`.
4.  [x] **Python Interface**:
    -   Expose `JITMPPIController` to Python via nanobind.
    -   Example usage:
        ```python
        dynamics_code = """
        struct UserDynamics {
            __device__ void step(...) { ... }
        };
        """
        cost_code = """
        struct UserCost {
            __device__ float compute(...) { ... }
            __device__ float terminal_cost(...) { ... }
        };
        """
        controller = cuda_mppi.JITMPPIController(config, dynamics_code, cost_code, include_paths)
        ```
5.  [x] **Verification & Examples**:
    -   Implemented `examples/cuda_pendulum_jit.py` - complete pendulum swing-up example with matplotlib plotting.
    -   Created `include/mppi/jit/examples.hpp` with example templates for common systems:
        - Pendulum dynamics and cost
        - Double integrator dynamics and cost
        - Cart-pole dynamics and cost
    -   Created `examples/JIT_EXAMPLES_README.md` with comprehensive documentation.
    -   Note: Interactive pygame visualization can be added in future enhancement.

### Files Created/Modified
- **New Files**:
  - `include/mppi/controllers/jit_mppi.hpp` - JIT controller header
  - `src/jit/jit_mppi_controller.cpp` - JIT controller implementation
  - `include/mppi/jit/examples.hpp` - Example code templates
  - `examples/cuda_pendulum_jit.py` - Pendulum swing-up example
  - `examples/JIT_EXAMPLES_README.md` - JIT examples documentation

- **Modified Files**:
  - `src/jit/jit_compiler.cpp` - Updated wrapper generation for Driver API compatibility
  - `CMakeLists.txt` - Added JIT sources to build
  - `bindings/bindings.cu` - Added Python bindings for JITMPPIController

### Usage Example
```python
from jax_mppi import cuda_mppi
import numpy as np
import os

# Set include path
os.environ['CUDA_MPPI_INCLUDE_DIR'] = '/path/to/src/cuda_mppi/include'

# Configure MPPI
config = cuda_mppi.MPPIConfig(
    num_samples=1000,
    horizon=50,
    nx=2, nu=1,
    lambda_=1.0,
    dt=0.02,
    u_scale=5.0,
    w_action_seq_cost=0.0,
    num_support_pts=10
)

# Define custom dynamics and cost
dynamics_code = """..."""  # See examples.hpp for templates
cost_code = """..."""

# Create JIT controller (compilation happens here, ~1-5 seconds)
controller = cuda_mppi.JITMPPIController(
    config, dynamics_code, cost_code,
    [os.environ['CUDA_MPPI_INCLUDE_DIR']]
)

# Use controller
state = np.array([1.0, 0.0], dtype=np.float32)
controller.compute(state)
action = controller.get_action()
controller.shift()
```

## References
-   `src/jax_mppi/*.py` (Main reference for the MPPI Implementation)
-   `../MPPI-Generic` (Reference for CUDA patterns)
-   NVRTC Documentation
