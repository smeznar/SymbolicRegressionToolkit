# Contributing

Thank you for your interest in contributing to SymbolicRegressionToolkit!

## Setting up a development environment

1. Fork the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/SymbolicRegressionToolkit.git
   cd SymbolicRegressionToolkit
   ```

2. Install the package in editable mode with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

   To also work on the ML-based approaches (EDHiE), install the optional dependencies too:

   ```bash
   pip install -e ".[dev,approaches]"
   ```

3. Install the pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Making changes

Create a branch for your work:

```bash
git checkout -b my-feature
```

## Running tests

Run the full test suite including doctests:

```bash
pytest --doctest-modules --cov=SRToolkit --cov-report=term-missing SRToolkit/ tests/
```

Tests marked `benchmark` require the Feynman/Nguyen datasets to be downloaded locally and are skipped by default. To run them:

```bash
pytest -m benchmark
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting (line length 120, Python 3.10 target). The pre-commit hooks run automatically on `git commit`, or you can run them manually:

```bash
ruff check .   # lint
ruff format .  # format
```

## Building the documentation

```bash
pip install -e ".[dev]"
mkdocs build
```

The built site is written to `site/`. To preview it locally with live reload:

```bash
mkdocs serve
```

Docstrings follow the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) and are rendered automatically by mkdocstrings. Cross-references use the `[Name][full.dotted.path]` syntax.

## Submitting a pull request

1. Push your branch to your fork:

   ```bash
   git push origin my-feature
   ```

2. Open a pull request against the `master` branch of `smeznar/SymbolicRegressionToolkit`.
3. Describe what the PR does and, if it fixes a bug, link the relevant issue.
4. Make sure all CI checks pass before requesting a review.