# care-survival

Upgrading survival models with CARE.

## Building the Python package

This project uses
[uv](https://github.com/astral-sh/uv)
to manage Python dependencies.
To build the Python package, run `uv build`.
To update the lockfile, use `uv lock`.

## Running the simulation scripts

This project uses
[just](https://github.com/casey/just)
and [parallel](https://www.gnu.org/software/parallel/)
to run scripts.
To execute all of the simulations and generate the plots,
run `just`.
The recipes can be found in the
[justfile](https://github.com/WGUNDERWOOD/care-survival/blob/main/justfile).

The scripts to run the SCORE2 data analysis
are also available in the bin directory,
but require access to UK Biobank data.

## Example usage

## Publishing to PyPI

To publish to TestPyPI with an API token, run

```
uv publish --index testpypi --token <token>
```

To publish to PyPI with an API token, run

```
uv publish --token <token>
```
