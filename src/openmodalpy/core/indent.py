#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Common ruff format options (uncomment to use):
RUFF_OPTIONS = [
    # File handling
    # '--force-exclude',            # Enforce exclude patterns even for explicitly named files
    "--respect-gitignore",  # Don't format files ignored by .gitignore
    # Line length options (choose one)
    # '--line-length=79',          # PEP 8 standard
    # '--line-length=88',          # Black default
    # '--line-length=100',         # More modern standard
    "--line-length=320",  # Wider screens
    # Note: The following options must be set in pyproject.toml or ruff.toml:
    # [tool.ruff.format]
    # quote-style = "single"       # Use single quotes
    # quote-style = "double"       # Use double quotes (default)
    # quote-style = "preserve"     # Keep existing quote style
    # indent-style = "space"       # Use spaces (default)
    # indent-style = "tab"         # Use tabs
    # docstring-code-format = true # Format code snippets in docstrings
    # Advanced options
    # '--preview',                 # Use the preview style for newer formatting features
    # '--stdin-filename',          # The name of the file when reading from stdin
]


def find_python_files(path, non_recursive=False):
    """Find all Python files in the given path. If non_recursive is True, only search the top-level directory."""
    path = Path(path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return []

    if path.is_file():
        if path.is_symlink():
            print(f"Skipping symbolic link: {path}")
            return []
        return [path] if path.suffix == ".py" else []

    python_files = []
    if non_recursive:
        for file in path.iterdir():
            if file.is_file() and file.suffix == ".py" and not file.is_symlink():
                python_files.append(file)
            elif file.is_symlink():
                print(f"Skipping symbolic link: {file}")
    else:
        for root, _, files in os.walk(path):
            root_path = Path(root)
            if ".git" in root_path.parts:
                continue
            for file in files:
                file_path = root_path / file
                if file.endswith(".py") and not file_path.is_symlink():
                    python_files.append(file_path)
                elif file_path.is_symlink():
                    print(f"Skipping symbolic link: {file_path}")

    return sorted(python_files)


def format_code(path, check_only=False, sort_imports=True, non_recursive=False):
    """Format Python files using ruff."""
    path = Path(path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return False

    # Check ruff installation
    result = subprocess.run(["ruff", "--version"], capture_output=True)
    if result.returncode != 0:
        print("Ruff is not installed. Please install it using:")
        print("pip install ruff")
        return False

    # Find all Python files
    python_files = find_python_files(path, non_recursive=non_recursive)
    if not python_files:
        print("No Python files found.")
        return True

    if check_only:
        print("\nCheck-only mode, no changes will be made.")
        return True

    print("\n Formatting files...")
    success = True
    for i, file in enumerate(python_files, start=1):
        print(f"{i:2d}. {file}")

        # Sort imports if requested
        if sort_imports:
            cmd = ["ruff", "check", "--select", "I"]
            cmd.append("--fix" if not check_only else "--diff")
            cmd.append(str(file))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stderr:
                print(result.stderr)
            if result.returncode != 0:
                success = False
                continue

        # Format code
        cmd = ["ruff", "format"]
        if check_only:
            cmd.extend(["--check", "--diff"])
        cmd.extend(RUFF_OPTIONS)
        cmd.append(str(file))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(description="Format Python files using ruff with clear output")
    parser.add_argument("path", nargs="?", default=".", help="Path to file or directory to format (default: current directory, non-recursive)")
    parser.add_argument("--check", action="store_true", help="Check if files are formatted without making changes")
    parser.add_argument("--no-sort-imports", action="store_true", help="Skip sorting imports")
    args = parser.parse_args()

    # If the path argument is default ('.') and not explicitly given, only format *.py in current dir
    non_recursive = False
    if "path" not in sys.argv or (len(sys.argv) > 1 and sys.argv[1] == "."):  # covers both 'indent.py' and 'indent.py .' cases
        # If user did not specify a path, or specified '.', do non-recursive
        non_recursive = True

    if not format_code(args.path, args.check, not args.no_sort_imports, non_recursive=non_recursive):
        sys.exit(1)


if __name__ == "__main__":
    main()
