# Installation

This development branch targets Python 3.12–3.14 and COMPAS 3. For a published
release, install the version appropriate for your COMPAS environment from PyPI.

```sh
pip install compas_dr
```

To test this branch, clone the repository and use its lockfile. The locked
environment resolves COMPAS to the exact revision used for compatibility
verification.

```sh
git clone https://github.com/BlockResearchGroup/compas_dr
cd compas_dr
uv sync --locked --group docs
uv run --no-sync pytest --doctest-modules src/compas_dr tests
uv run --no-sync mkdocs build --strict
```

The legacy `hpc` extra still installs Numba. The package's solvers use the
same code paths with or without this extra.

```sh
uv sync --locked --extra hpc
```
