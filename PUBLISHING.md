# Publishing jax-mppi to PyPI

**Quick Start**: For the recommended automated workflow, see [docs/RELEASE.md](docs/RELEASE.md).

This document covers manual publishing and initial setup.

## Prerequisites

1. **Create PyPI accounts**:
   - TestPyPI: <https://test.pypi.org/account/register/>
   - PyPI: <https://pypi.org/account/register/>

2. **Generate API tokens** (recommended over passwords):
   - TestPyPI: <https://test.pypi.org/manage/account/token/>
   - PyPI: <https://pypi.org/manage/account/token/>

   Save tokens securely - you'll use them for authentication.

3. **Configure credentials** (optional but recommended):
   Create/edit `~/.pypirc`:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-YOUR-PRODUCTION-TOKEN-HERE

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-YOUR-TEST-TOKEN-HERE
   ```

## Publishing Workflow

### 1. Update Version

Before publishing, update the version in both files:
- `pyproject.toml` - line 3: `version = "0.3.0"`
- `pixi.toml` - line 3: `version = "0.3.0"`

Follow [semantic versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH` (e.g., `0.3.0` → `0.3.1` for bug fixes)

### 2. Update Changelog

Document changes in `CHANGELOG.md` following your existing format.

### 3. Commit Version Changes

```bash
git add pyproject.toml pixi.toml CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
git tag vX.Y.Z
git push && git push --tags
```

### 4. Build the Package

```bash
# Using pixi (recommended)
pixi run -e dev build-sdist

# This will:
# - Clean old builds
# - Create source distribution (.tar.gz) only
# - Output to dist/
```

**Why source-only?** This package contains CUDA/C++ extensions with platform-specific builds. Publishing only the source distribution (`.tar.gz`) allows users to build the package for their specific system and CUDA version. Binary wheels (`.whl`) would require manylinux compatibility and wouldn't work across different CUDA installations.

### 5. Test on TestPyPI (Recommended First Time)

```bash
# Upload to TestPyPI
pixi run -e dev publish-test

# Test installation in a fresh environment
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ jax-mppi
```

Note: TestPyPI uses a separate package index, so dependencies will be pulled from the main PyPI.

### 6. Publish to PyPI

Once verified on TestPyPI:

```bash
# Upload to production PyPI
pixi run -e dev publish
```

**⚠️ Warning**: Once published to PyPI, you cannot delete or modify that version. You must publish a new version for any changes.

### 7. Verify Installation

```bash
# In a fresh environment
pip install jax-mppi

# Test import
python -c "import jax_mppi; print(jax_mppi.__version__)"
```

## Manual Upload (Alternative)

If you prefer manual control:

```bash
# Build
pixi run -e dev build

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Quick Reference Commands

```bash
# Clean build artifacts
pixi run -e dev clean-build

# Build source distribution only (recommended for this package)
pixi run -e dev build-sdist

# Build both source and wheel (wheel won't upload to PyPI)
pixi run -e dev build

# Check package validity
pixi run -e dev check-package

# Publish to TestPyPI (source distribution only)
pixi run -e dev publish-test

# Publish to PyPI (production, source distribution only)
pixi run -e dev publish
```

## Troubleshooting

### "File already exists" error
You're trying to upload a version that already exists. Bump the version number and rebuild.

### Authentication failures
- Ensure API tokens are correctly set in `~/.pypirc`
- Or provide credentials interactively when prompted
- Username should be `__token__` when using API tokens

### Missing dependencies during install
- Check `dependencies` in `pyproject.toml` are correct
- Ensure version constraints are appropriate (not too strict)

### Import errors after install
- Verify package structure: source code must be in `src/jax_mppi/`
- Check `__init__.py` files exist in all package directories

## GitHub Release (Optional)

After publishing to PyPI, create a GitHub release:

```bash
gh release create vX.Y.Z dist/* \
  --title "vX.Y.Z" \
  --notes "See CHANGELOG.md for details"
```

Or create manually at: <https://github.com/riccardo-enr/jax_mppi/releases>

## Automation (Future)

Consider setting up GitHub Actions to automate publishing when you push a version tag. Example workflow location: `.github/workflows/publish.yml`
