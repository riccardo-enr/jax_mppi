# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/riccardo-enr/jax_mppi/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/riccardo-enr/jax_mppi/releases/tag/v0.1.0
