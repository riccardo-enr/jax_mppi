#!/usr/bin/env python3
"""
Release preparation script for jax-mppi.

Usage:
    python scripts/prepare_release.py <new_version> [--dry-run]

Example:
    python scripts/prepare_release.py 0.3.1
    python scripts/prepare_release.py 0.4.0 --dry-run
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(
    cmd: list[str], check: bool = True
) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def validate_version(version: str) -> bool:
    """Validate semantic version format."""
    pattern = r"^\d+\.\d+\.\d+$"
    return bool(re.match(pattern, version))


def get_current_version(file_path: Path, pattern: str) -> str:
    """Extract current version from a file."""
    content = file_path.read_text()
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Could not find version in {file_path}")
    return match.group(1)


def update_file_version(
    file_path: Path, old_version: str, new_version: str, dry_run: bool = False
):
    """Update version in a file."""
    content = file_path.read_text()
    updated = content.replace(
        f'version = "{old_version}"', f'version = "{new_version}"'
    )

    if content == updated:
        print(f"⚠️  No version found to update in {file_path}")
        return False

    if dry_run:
        print(f"Would update {file_path}: {old_version} -> {new_version}")
    else:
        file_path.write_text(updated)
        print(f"✅ Updated {file_path}: {old_version} -> {new_version}")

    return True


def update_changelog(new_version: str, dry_run: bool = False) -> bool:
    """Add a new version section to CHANGELOG.md."""
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        print("⚠️  CHANGELOG.md not found, skipping")
        return False

    content = changelog_path.read_text()
    today = datetime.now().strftime("%Y-%m-%d")

    # Find where to insert the new version
    new_section = f"""## [{new_version}] - {today}

### Added
-

### Changed
-

### Fixed
-

"""

    # Insert after the first heading (assuming "# Changelog" or similar)
    lines = content.split("\n")
    insert_index = 0
    for i, line in enumerate(lines):
        if line.startswith("## "):  # Find first version section
            insert_index = i
            break

    if insert_index == 0:
        # No version sections found, insert after first line
        insert_index = 1

    lines.insert(insert_index, new_section)
    updated_content = "\n".join(lines)

    if dry_run:
        print(
            f"Would add new section to CHANGELOG.md for version {new_version}"
        )
    else:
        changelog_path.write_text(updated_content)
        print(f"✅ Added new section to CHANGELOG.md for version {new_version}")
        print(
            "⚠️  Please edit CHANGELOG.md to add your changes before committing!"
        )

    return True


def check_git_status() -> bool:
    """Check if git working directory is clean."""
    result = run_command(["git", "status", "--porcelain"], check=False)
    if result.returncode != 0:
        print("❌ Not in a git repository")
        return False

    if result.stdout.strip():
        print("⚠️  Git working directory is not clean:")
        print(result.stdout)
        response = input("Continue anyway? (y/N): ")
        return response.lower() == "y"

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a new release for jax-mppi"
    )
    parser.add_argument("version", help="New version number (e.g., 0.3.1)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--skip-git-check", action="store_true", help="Skip git status check"
    )
    args = parser.parse_args()

    new_version = args.version
    dry_run = args.dry_run

    # Validate version format
    if not validate_version(new_version):
        print(f"❌ Invalid version format: {new_version}")
        print("   Expected format: MAJOR.MINOR.PATCH (e.g., 0.3.1)")
        sys.exit(1)

    # Check git status
    if not args.skip_git_check and not check_git_status():
        sys.exit(1)

    print(
        f"\n{'DRY RUN: ' if dry_run else ''}Preparing release {new_version}\n"
    )

    # Get current versions
    pyproject_path = Path("pyproject.toml")
    pixi_path = Path("pixi.toml")

    try:
        current_pyproject = get_current_version(
            pyproject_path, r'version = "([^"]+)"'
        )
        current_pixi = get_current_version(pixi_path, r'version = "([^"]+)"')

        if current_pyproject != current_pixi:
            print(
                f"⚠️  Version mismatch: pyproject.toml={current_pyproject}, pixi.toml={current_pixi}"
            )
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                sys.exit(1)

        current_version = current_pyproject
        print(f"Current version: {current_version}")
        print(f"New version: {new_version}\n")

    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Update version files
    updated_files = []
    if update_file_version(
        pyproject_path, current_version, new_version, dry_run
    ):
        updated_files.append("pyproject.toml")
    if update_file_version(pixi_path, current_version, new_version, dry_run):
        updated_files.append("pixi.toml")

    # Update changelog
    if update_changelog(new_version, dry_run):
        updated_files.append("CHANGELOG.md")

    if not updated_files:
        print("\n❌ No files were updated")
        sys.exit(1)

    print(
        f"\n{'Would update' if dry_run else 'Updated'} files: {', '.join(updated_files)}"
    )

    if dry_run:
        print("\n✅ Dry run complete. Run without --dry-run to apply changes.")
        return

    # Provide next steps
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("\n1. Edit CHANGELOG.md to document your changes")
    print("\n2. Review the changes:")
    print("   git diff")
    print("\n3. Commit and tag the release:")
    print(f"   git add {' '.join(updated_files)}")
    print(f'   git commit -m "chore: bump version to {new_version}"')
    print(f"   git tag v{new_version}")
    print("   git push && git push --tags")
    print("\n4. Publish to PyPI:")
    print("   pixi run -e dev publish")
    print("\n5. Create GitHub release:")
    print(f"   gh release create v{new_version} dist/*.tar.gz \\")
    print(f'     --title "v{new_version}" \\')
    print('     --notes "See CHANGELOG.md for details"')
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
