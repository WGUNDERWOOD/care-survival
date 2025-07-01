# care-survival

Upgrading survival models with CARE.

This repository contains a Python implementation
of the methodology proposed by
[Underwood, Reeve, Feng, Lambert, Mukherjee and Samworth, 2025](https://arxiv.org/abs/2506.23870).

## Building the Python package

This project uses
[uv](https://github.com/astral-sh/uv)
to manage Python dependencies.
To build the Python package, run `uv build`.
To update the lockfile, use `uv lock`.

## Running the simulation scripts

For running scripts,
[just](https://github.com/casey/just)
and [parallel](https://www.gnu.org/software/parallel/)
should be installed.
To execute all of the simulations and generate the plots,
run `just`.
The recipes can be found in the
[justfile](https://github.com/WGUNDERWOOD/care-survival/blob/main/justfile).

The scripts to run the SCORE2 data analysis
are also available in the bin directory,
but require access to UK Biobank data.

## Example usage

```python
import numpy as np
import care_survival

# generate some sample data
n = 60
d = 2
X = np.random.random((n, d))
T = X.sum(axis=1) + 1 + np.random.random(n)
I = (np.random.random(n) > 0.8) * 1
f = (-X.sum(axis=1) / 2 + np.random.random(n)).reshape(-1, 1)

# split data into training and validation sets
n_train = int(np.ceil(n / 2))
X_train = X[0:n_train]
T_train = T[0:n_train]
I_train = I[0:n_train]
f_train = f[0:n_train]
X_valid = X[n_train:n]
T_valid = T[n_train:n]
I_valid = I[n_train:n]
f_valid = f[n_train:n]

# define a kernel and use the feature map optimisation method
a = 1
p = 2
kernel = care_survival.PolynomialKernel(a, p)
method = "feature_map"

# set up the kernel tuning parameters
n_gammas = 20
gamma_min = 1e-6
gamma_max = 1e1

# set up the simplex tuning parameters
simplex_resolution = 0.05

# compute concordance score on all data
with_concordance = ["train", "valid"]

# fit CARE
care = care_survival.care(
    X_train,
    T_train,
    I_train,
    f_train,
    X_valid,
    T_valid,
    I_valid,
    f_valid,
    kernel,
    method,
    n_gammas,
    gamma_min,
    gamma_max,
    simplex_resolution,
    with_concordance,
)

# view diagnostics
best = care.best["aggregated"]["ln"]["valid"]
print("best theta value:", best.theta)
print("best gamma value:", best.gamma)
print("concordance index:", best.score["concordance"]["valid"])
```

## Publishing to PyPI

First build the package with

```
uv build
```

To publish to TestPyPI with an API token, run

```
uv publish --index testpypi --token <token>
```

To publish to PyPI with an API token, run

```
uv publish --token <token>
```
