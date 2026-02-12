# Release Checklist

## Quick Release (3 steps)

```bash
# 1. Prepare release
pixi run -e dev prepare-release X.Y.Z

# 2. Edit CHANGELOG.md with your changes

# 3. Commit, tag, and push
git add pyproject.toml pixi.toml CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
git tag vX.Y.Z
git push && git push --tags
```

GitHub Actions will automatically publish to PyPI! 🚀

---

## Pre-Release Checks

- [ ] All tests pass: `pixi run -e dev test`
- [ ] Code formatted: `pixi run -e dev lint`
- [ ] Types checked: `pixi run -e dev type-check`
- [ ] On main branch with clean working directory

## Post-Release

- [ ] Verify on PyPI: https://pypi.org/project/jax-mppi/
- [ ] Test install: `pip install jax-mppi==X.Y.Z`
- [ ] Check GitHub release created

---

See [docs/RELEASE.md](../docs/RELEASE.md) for full documentation.
