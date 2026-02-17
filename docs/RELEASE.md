# Release Workflow

This document describes the automated and manual release process for jax-mppi.

## Quick Release (Recommended)

Use the automated release script:

```bash
# 1. Prepare the release (updates versions and changelog)
pixi run -e dev prepare-release 0.3.1

# 2. Review and edit CHANGELOG.md with your changes

# 3. Commit, tag, and push
git add pyproject.toml pixi.toml CHANGELOG.md
git commit -m "chore: bump version to 0.3.1"
git tag v0.3.1
git push && git push --tags
```

**That's it!** GitHub Actions will automatically:
- Build the source distribution
- Publish to PyPI
- Create a GitHub release with the tarball attached

## Release Script Usage

### Dry run (preview changes)
```bash
pixi run -e dev dry-release 0.3.1
```

### Prepare release
```bash
pixi run -e dev prepare-release 0.3.1
```

The script will:
1. ✅ Validate version format
2. ✅ Check git status
3. ✅ Update `pyproject.toml`
4. ✅ Update `pixi.toml`
5. ✅ Add new section to `CHANGELOG.md`
6. ✅ Display next steps

## Manual Release Process

If you prefer manual control or the automated workflow fails:

### 1. Update version manually

Edit `pyproject.toml`:
```toml
version = "0.3.1"
```

Edit `pixi.toml`:
```toml
version = "0.3.1"
```

### 2. Update CHANGELOG.md

Add a new section:
```markdown
## [0.3.1] - 2024-02-12

### Added
- New feature X

### Changed
- Updated behavior Y

### Fixed
- Bug fix Z
```

### 3. Commit and tag

```bash
git add pyproject.toml pixi.toml CHANGELOG.md
git commit -m "chore: bump version to 0.3.1"
git tag v0.3.1
git push && git push --tags
```

### 4. Manual publish (if GitHub Actions fails)

```bash
pixi run -e dev publish
```

### 5. Create GitHub release manually

```bash
gh release create v0.3.1 dist/*.tar.gz \
  --title "v0.3.1" \
  --notes "See CHANGELOG.md for details"
```

## GitHub Actions Workflow

The workflow (`.github/workflows/publish-pypi.yml`) triggers on:
- **Tag push**: Any tag matching `v*.*.*` (e.g., `v0.3.1`)
- **Manual dispatch**: Can be triggered manually from GitHub Actions UI

### What it does:

1. **Build**: Creates source distribution
2. **Verify**: Checks package validity and version matches tag
3. **Publish**: Uploads to PyPI using `PYPI_API_TOKEN` secret
4. **Release**: Creates GitHub release with tarball and notes

### Setup Requirements

Add your PyPI token as a GitHub secret:

1. Go to: <https://github.com/riccardo-enr/jax_mppi/settings/secrets/actions>
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: Your PyPI API token (starts with `pypi-`)
5. Click "Add secret"

You can use the same token that's in your `~/.pypirc` file.

## Versioning Guidelines

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes, incompatible API changes
- **MINOR** (0.3.0): New features, backwards-compatible
- **PATCH** (0.3.1): Bug fixes, backwards-compatible

Examples:
- `0.3.0` → `0.3.1`: Bug fix release
- `0.3.1` → `0.4.0`: New feature release
- `0.4.0` → `1.0.0`: First stable release or breaking changes

## Troubleshooting

### GitHub Actions fails with "version mismatch"

The tag version doesn't match `pyproject.toml`. Either:
- Delete the tag: `git tag -d v0.3.1 && git push --delete origin v0.3.1`
- Fix `pyproject.toml` and create a new patch version

### "File already exists" on PyPI

You can't re-upload the same version. Bump to a new version:
- `0.3.1` → `0.3.2` for another attempt

### GitHub Actions missing PYPI_API_TOKEN

Add the secret as described in "Setup Requirements" above.

### Manual publish needed

If GitHub Actions fails, you can always publish manually:
```bash
pixi run -e dev publish
```

## Pre-release Versions

For alpha/beta/rc releases:

```bash
# Update to pre-release version
pixi run -e dev prepare-release 0.4.0a1  # alpha
pixi run -e dev prepare-release 0.4.0b1  # beta
pixi run -e dev prepare-release 0.4.0rc1 # release candidate

# Tag and push
git add pyproject.toml pixi.toml CHANGELOG.md
git commit -m "chore: bump version to 0.4.0a1"
git tag v0.4.0a1
git push && git push --tags
```

Users install with:
```bash
pip install jax-mppi==0.4.0a1  # Specific pre-release
pip install --pre jax-mppi      # Latest pre-release
```

## Release Checklist

Before releasing:

- [ ] All tests pass: `pixi run -e dev test`
- [ ] Code is formatted: `pixi run -e dev lint`
- [ ] Type checking passes: `pixi run -e dev type-check`
- [ ] CHANGELOG.md is updated with all changes
- [ ] Version follows semantic versioning
- [ ] Git working directory is clean
- [ ] On main branch (or appropriate release branch)

After releasing:

- [ ] Verify package on PyPI: <https://pypi.org/project/jax-mppi/>
- [ ] Test installation: `pip install jax-mppi==X.Y.Z`
- [ ] GitHub release created with correct notes
- [ ] Announce release (optional): Twitter, Reddit, etc.
