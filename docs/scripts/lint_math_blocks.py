#!/usr/bin/env python3
"""
Linter for math block formatting in markdown files.

Enforces the following rules:
1. Display math blocks use \\[ ... \\] delimiters (not $$ ... $$)
2. Unclosed \\[ blocks are reported
"""

import sys
from pathlib import Path


def check_math_formatting(file_path: Path) -> list[str]:
    """Check math block formatting in a markdown file."""
    errors = []
    lines = file_path.read_text().split("\n")
    in_code_block = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track fenced code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        # Check for $$ delimiters (should use \[ \] instead)
        if "$$" in line:
            errors.append(
                f"{file_path}:{i}: Use \\[ and \\] instead of $$ for display math"
            )

        # Check for unclosed \[ blocks
        if r"\[" in line and r"\]" not in line:
            found_close = False
            for j in range(i, min(i + 10, len(lines))):
                if r"\]" in lines[j]:
                    found_close = True
                    break
            if not found_close:
                errors.append(f"{file_path}:{i}: Unclosed math block \\[")

    return errors


def main():
    """Run linter on all markdown files."""
    repo_root = Path(__file__).parent.parent
    md_files = sorted(repo_root.glob("docs/**/*.md"))

    all_errors = []
    for md_file in md_files:
        all_errors.extend(check_math_formatting(md_file))

    if all_errors:
        for error in all_errors:
            print(error)
        print(f"\n✗ Found {len(all_errors)} math formatting error(s)")
        sys.exit(1)
    else:
        print("✓ All math blocks formatted correctly!")
        sys.exit(0)


if __name__ == "__main__":
    main()
