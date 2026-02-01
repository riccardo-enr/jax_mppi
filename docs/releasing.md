# Releasing Guide

This guide describes how to release a new version of `jax_mppi` to PyPI.

## Prerequisites

The release process is automated using GitHub Actions, but it requires the repository to be configured as a Trusted Publisher on PyPI.

### PyPI Trusted Publisher Setup

1. Log in to your [PyPI account](https://pypi.org/).
2. Go to **Publishing** in your project settings (or create a new project if this is the first release).
3. Add a new **Trusted Publisher**.
4. Select **GitHub**.
5. Enter the following details:
    * **Owner**: `riccardo-enr`
    * **Repository name**: `jax_mppi`
    * **Workflow name**: `publish.yml`
    * **Environment name**: (Leave empty)
6. Click **Add**.

This allows the GitHub Action to authenticate with PyPI using OIDC tokens without needing a long-lived API token or password.

## Release Process

To release a new version:

1. **Update Version**:
    Update the version number in `pyproject.toml`:

    ```toml
    [project]
    version = "0.1.6"  # Example version
    ```

2. **Commit and Push**:
    Commit the version change and push to `main`.

    ```bash
    git add pyproject.toml
    git commit -m "Bump version to 0.1.6"
    git push origin main
    ```

3. **Create Tag**:
    Create a git tag for the release. The tag must start with `v`.

    ```bash
    git tag v0.1.6
    git push origin v0.1.6
    ```

4. **Wait for Action**:
    The `Publish to PyPI` GitHub Action will automatically run when the tag is pushed. It will build the package and upload it to PyPI.

5. **Verify**:
    Check the [PyPI page](https://pypi.org/project/jax-mppi/) to confirm the new version is available.
