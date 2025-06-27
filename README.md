# care-survival

Upgrading survival models with CARE.

## Building the Python package

This project uses uv to manage Python dependencies.
To build the Python package, run `uv build`.
To update the lockfile, use `uv lock`.

## Running the simulation scripts

This project uses just to run scripts.
To execute all of the simulations, run `just`.
The recipes can be found in the justfile.

## Publishing to PyPI

To publish to TestPyPI with an API token, run

```
uv publish --index testpypi --token <token>
```

To publish to PyPI with an API token, run

```
uv publish --token <token>
```
