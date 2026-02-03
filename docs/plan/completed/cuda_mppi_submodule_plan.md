# Plan: Add cuda_mppi as git submodule

## Status: In Progress

## Overview
Extract the CUDA MPPI implementation from `src/cuda_mppi/` into a separate git repository and integrate it as a submodule at `third_party/cuda_mppi`.

## Decisions Made
- New repository: `riccardo-enr/cuda_mppi`
- Submodule path: `third_party/cuda_mppi`
- Feature branch: `feat/cuda-mppi-submodule`
- Track branch: `main` (in submodule)

## Implementation Steps

### 1. Create New cuda_mppi Repository
- [ ] Use `gh repo create` to create `riccardo-enr/cuda_mppi`
- [ ] Set appropriate description and visibility
- [ ] Initialize with README.md

### 2. Initialize cuda_mppi Repository
- [ ] Clone the new repository locally
- [ ] Copy contents from `src/cuda_mppi/` to repository root
- [ ] Create proper repository structure:
  - CMakeLists.txt (standalone build)
  - README.md (usage and build instructions)
  - LICENSE (MIT, matching main project)
  - .gitignore
- [ ] Commit and push initial code

### 3. Create Feature Branch in jax_mppi
- [ ] Create branch: `feat/cuda-mppi-submodule`
- [ ] Checkout the new branch

### 4. Add Submodule to jax_mppi
- [ ] Create `third_party/` directory
- [ ] Run: `git submodule add https://github.com/riccardo-enr/cuda_mppi.git third_party/cuda_mppi`
- [ ] Remove old `src/cuda_mppi/` directory
- [ ] Commit submodule addition

### 5. Update Build System
- [ ] Update root `CMakeLists.txt`:
  - Change `add_subdirectory(src/cuda_mppi)` to `add_subdirectory(third_party/cuda_mppi)`
- [ ] Verify `pyproject.toml` (should work via scikit-build-core automatically)
- [ ] Check for any hardcoded paths in:
  - Python source files
  - CMake files
  - Documentation

### 6. Test Build
- [ ] Clean build: `rm -rf build/ _skbuild/`
- [ ] Test clone from scratch with submodules
- [ ] Build and install package: `uv pip install -e .`
- [ ] Run test: `python examples/test_cuda_mppi.py`
- [ ] Verify JIT examples work

### 7. Update Documentation
- [ ] Add submodule instructions to README.md:
  ```bash
  # Clone with submodules
  git clone --recursive https://github.com/riccardo-enr/jax_mppi.git

  # Or if already cloned
  git submodule update --init --recursive
  ```
- [ ] Document build requirements in both repositories
- [ ] Update any relevant documentation links

### 8. Create Pull Request
- [ ] Commit all changes with conventional commits
- [ ] Push feature branch
- [ ] Create PR referencing issue #23
- [ ] Move this plan to `completed/` directory

## Rollback Plan
If issues arise, we can:
1. Remove submodule: `git submodule deinit -f third_party/cuda_mppi`
2. Delete from git: `git rm -f third_party/cuda_mppi`
3. Remove from .gitmodules
4. Restore original src/cuda_mppi from git history
