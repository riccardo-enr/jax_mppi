# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-02-03

### Changed

- **CUDA Backend Architecture**: Extracted `cuda_mppi` implementation into a separate git repository ([riccardo-enr/cuda_mppi](https://github.com/riccardo-enr/cuda_mppi))
- **Build System**: Updated CMake references to use submodule at `third_party/cuda_mppi` instead of `src/cuda_mppi`
- **Development Workflow**: Repository now requires `git clone --recursive` to include CUDA components

### Fixed

- Updated include paths in JIT examples to reference submodule location
- Updated justfile commands to use correct CUDA include directory

## [0.2.0] - 2026-02-03

### Added

- **CUDA Backend**: High-performance C++/CUDA implementation of MPPI (`cuda_mppi`) for significant speedups.
- **JIT Compilation**: Runtime compilation of user-defined dynamics and costs using NVRTC, allowing pure Python strings to be compiled into optimized CUDA kernels.
- **Python Bindings**: Seamless integration via `nanobind`, exposing the CUDA controller to the Python ecosystem.
- **Examples**:
    - `cuda_pendulum_jit.py`: A complete example demonstrating the JIT-compiled controller with a pendulum environment.
- **Documentation**: Comprehensive documentation for the CUDA implementation plans and usage.

### Changed

- Updated build system to `scikit-build-core` for mixed Python/C++ builds.

## [0.1.8] - 2026-02-01

### Changed

- Version bump only

## [0.1.7] - 2026-02-01

### Changed

- Version bump only

## [0.1.6] - 2026-02-01

### Changed

- Updated PyPI publish workflow configuration

## [0.1.5] - 2026-02-01

### Added

- JAX-native evosax autotuning backend with CMA-ES, Sep-CMA-ES, and OpenES optimizers
- Evosax vs CMA-ES comparison example script
- Comprehensive evosax optimizer test suite
- MIT License file
- Dedicated testing documentation

### Changed

- README autotuning section with evosax usage guidance and optimizer matrix
- Pyproject dependency groups for autotuning and autotuning-extra, plus new `chex` dependency
- CI test workflow to install autotuning dependencies
- Autotuning modules refactored to support evosax backend integration

### Fixed

- Evosax API usage updates for current evosax versions
- MPPI `create` API usage in the evosax comparison example
- Example output to report `noise_sigma` correctly and avoid NumPy scalar conversion deprecation warnings

## [0.1.1] - 2026-02-01

### Changed

- Updated minimum Python version requirement to 3.12
- Updated Ruff target version to Python 3.12
- Removed support for Python 3.9, 3.10, 3.11 (targeting 3.12+)
- Prepared package for initial PyPI release with complete metadata
- Updated README with PyPI installation instructions

### Added

- GitHub Actions workflow for automated PyPI publishing via trusted publishing
- Comprehensive PyPI metadata (classifiers, keywords, project URLs)
- CHANGELOG following Keep a Changelog format

## [0.1.0] - 2026-02-01

### Added

- Core MPPI controller implementation with functional, JIT-compilable design
- Smooth MPPI (SMPPI) variant with action sequence maintenance and smoothness costs
- Kernel MPPI (KMPPI) variant using kernel interpolation for control points
- Multiple sampling strategies:
  - Standard Gaussian noise sampling
  - Colored noise sampling
  - Time-correlated noise sampling
- Cost function composition system for flexible objective design
- Basic cost functions (quadratic state/action costs, terminal costs)
- Linear dynamics models for testing and prototyping
- Autotuning capabilities:
  - CMA-ES optimizer for hyperparameter optimization
  - Ray Tune integration for global search
  - Quality Diversity (QD) optimization via ribs library
- GPU acceleration via JAX backend
- Type safety with jaxtyping annotations
- Pure functional design with explicit state management using dataclasses
- Full JIT compilation support for high-performance control loops
- Support for `jax.vmap` batch processing and `jax.lax.scan` horizon loops
- Examples:
  - Pendulum environment example
  - Basic autotuning example
  - Autotuning pendulum example
  - Smooth MPPI comparison example
- Comprehensive test suite
- PyPI package configuration with setuptools
- MIT License
- Documentation structure with scientific theory for MPPI variants

[unreleased]: https://github.com/riccardo-enr/jax_mppi/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/riccardo-enr/jax_mppi/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/riccardo-enr/jax_mppi/compare/v0.1.8...v0.2.0
[0.1.8]: https://github.com/riccardo-enr/jax_mppi/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/riccardo-enr/jax_mppi/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/riccardo-enr/jax_mppi/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/riccardo-enr/jax_mppi/compare/v0.1.1...v0.1.5
[0.1.1]: https://github.com/riccardo-enr/jax_mppi/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/riccardo-enr/jax_mppi/releases/tag/v0.1.0
