# Plan: Add cuda_mppi as git submodule

## Status: Completed

## Overview
Extract the CUDA MPPI implementation from `src/cuda_mppi/` into a separate git repository and integrate it as a submodule at `third_party/cuda_mppi`.

## Decisions Made
- New repository: `riccardo-enr/cuda_mppi`
- Submodule path: `third_party/cuda_mppi`
- Feature branch: `feat/cuda-mppi-submodule`
- Track branch: `main` (in submodule)

## Implementation Steps

### 1. Create New cuda_mppi Repository
- [x] Use `gh repo create` to create `riccardo-enr/cuda_mppi`
- [x] Set appropriate description and visibility
- [x] Initialize with README.md

### 2. Initialize cuda_mppi Repository
- [x] Clone the new repository locally
- [x] Copy contents from `src/cuda_mppi/` to repository root
- [x] Create proper repository structure:
  - CMakeLists.txt (standalone build)
  - README.md (usage and build instructions)
  - LICENSE (MIT, matching main project)
  - .gitignore
- [x] Commit and push initial code

### 3. Create Feature Branch in jax_mppi
- [x] Create branch: `feat/cuda-mppi-submodule`
- [x] Checkout the new branch

### 4. Add Submodule to jax_mppi
- [x] Create `third_party/` directory
- [x] Run: `git submodule add https://github.com/riccardo-enr/cuda_mppi.git third_party/cuda_mppi`
- [x] Remove old `src/cuda_mppi/` directory
- [x] Commit submodule addition

### 5. Update Build System
- [x] Update root `CMakeLists.txt`:
  - Change `add_subdirectory(src/cuda_mppi)` to `add_subdirectory(third_party/cuda_mppi)`
- [x] Verify `pyproject.toml` (should work via scikit-build-core automatically)
- [x] Check for any hardcoded paths in:
  - Python source files
  - CMake files
  - Documentation

### 6. Test Build
- [x] Clean build: `rm -rf build/ _skbuild/`
- [x] Test clone from scratch with submodules
- [x] Build and install package: `uv pip install -e .`
- [x] Run test: `python examples/test_cuda_mppi.py`
- [x] Verify JIT examples work

### 7. Update Documentation
- [x] Add submodule instructions to README.md:
  ```bash
  # Clone with submodules
  git clone --recursive https://github.com/riccardo-enr/jax_mppi.git

  # Or if already cloned
  git submodule update --init --recursive
  ```
- [x] Document build requirements in both repositories
- [x] Update any relevant documentation links

### 8. Create Pull Request
- [x] Commit all changes with conventional commits
- [x] Push feature branch
- [x] Create PR referencing issue #23
- [x] Move this plan to `completed/` directory

## Rollback Plan
If issues arise, we can:
1. Remove submodule: `git submodule deinit -f third_party/cuda_mppi`
2. Delete from git: `git rm -f third_party/cuda_mppi`
3. Remove from .gitmodules
4. Restore original src/cuda_mppi from git history
