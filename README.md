# care-survival

This package is a work in progress.

## Usage with uv

To run the main entry point

`uv run bin/main.py`

To build the Python package

`uv build`

To publish to TestPyPI

`uv publish --index testpypi --token <token>`

To publish to PyPI

`uv publish --token <token>`

To add a new package as a dependency

`uv add <package>; uv lock`, then reload the nix development environment
