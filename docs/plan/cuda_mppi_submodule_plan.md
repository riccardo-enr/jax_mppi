# Plan: Add cuda_mppi as git submodule

1. Identify the `cuda_mppi` repository URL and desired pinned commit/branch/tag.
2. Ensure the target path for the submodule is correct (e.g., `third_party/cuda_mppi` or `external/cuda_mppi`) and does not conflict with existing files.
3. Add the submodule at the chosen path using git, and record the chosen commit (or branch) in `.gitmodules`.
4. Update any build or package references to use the submodule path (e.g., CMake, Python packaging, docs).
5. Verify a clean clone works with submodules and document the required clone/update commands in `README.md`.
