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

## Writing docstrings

Docstrings are the library's **reference documentation and its specification**: tests are designed from them, so they describe what a function is *supposed* to do. Aim for the precision of NumPy/SciPy and the consistency of scikit-learn, in the concise Google syntax.

**Where content belongs:**

| Content | Location |
|---|---|
| What a function/class does, its parameters, returns and errors | Function/class docstring |
| Concepts, background, how the pieces fit together | Module docstring |
| End-to-end walkthroughs | `examples/` |
| Edge cases and exhaustive behaviour checks | `tests/` |

**Rules:**

1. **Summary**: one line, imperative mood ("Return…", "Fit…", "Compute…"), ending with a period. It says *what*, not *how*.
2. **Description** (optional): only what a caller needs: semantics, side effects (caching, mutation of state, files written), randomness and seeding, guarantees such as symmetry or determinism, notable cost. Do not narrate the implementation. If it grows past about 8 lines, the material is probably explanation and belongs in the module docstring.
3. **Args**: types live in the annotations, not the prose. Give the meaning, valid values or range, array **shape** (e.g. `(n_samples, n_features)`), and what `None` or the default means.
4. **Returns**: what is returned, its shape, and every **special value** (`inf`, `nan`, empty list) and when it occurs.
5. **Raises**: every exception a caller can observe, with its trigger condition. Never list exceptions that cannot occur.
6. **Examples**: 1–3 short doctests showing typical use. They are documentation, not a test suite: edge cases go in `tests/`.
7. **Notes / References** (optional): formulas, and papers for methods (e.g. BED, ProGED, EDHiE).
8. **Section order**: summary, description, `Args`, `Returns` / `Yields`, `Raises`, `Examples`, `Notes`, `References`.
9. **Classes**: describe what the object represents. `__init__` arguments are documented in the class docstring (`merge_init_into_class` is enabled). Public attributes go in an `Attributes` section.
10. **Private helpers** (`_name`): a one-line summary is enough unless the helper implements a non-trivial algorithm.

**Example:**

```python
def rmse(predict: PredictFn, X: np.ndarray, y: np.ndarray) -> float:
    """Return the root-mean-square error of a model's predictions.

    Args:
        predict: Callable mapping ``X`` to predictions of shape ``(n_samples,)``.
        X: Input data of shape ``(n_samples, n_features)``.
        y: Targets of shape ``(n_samples,)``.

    Returns:
        ``sqrt(mean((predict(X) - y) ** 2))``. ``nan`` if any prediction is ``nan``.

    Examples:
        >>> rmse(lambda X: X[:, 0], np.array([[1.0], [2.0]]), np.array([1.0, 4.0]))
        1.4142135623730951
    """
```

**Checking a module strictly** (the project config still ignores some docstring rules codebase-wide while modules are being reviewed):

```bash
ruff check SRToolkit/path/to/module.py --isolated --preview --select D,DOC --config 'lint.pydocstyle.convention="google"'
pytest --doctest-modules SRToolkit/path/to/module.py
mkdocs build --strict
```

## Writing tests

Tests check what the code is **supposed** to do (the docstring), not what it currently does.

- Design tests from the signature and docstring first, then use coverage to find untested branches. `python scripts/spec_view.py <module>` prints a module's spec without function bodies.
- Use the strongest oracle available: a hand-computed or analytic value, then a trusted reference implementation (SciPy, NumPy, SymPy), then a property (Hypothesis), then a metamorphic relation. Snapshot ("same as last time") tests are a last resort.
- Every assertion must be able to fail for a plausible bug. `isinstance(result, float)` on its own is not a test.
- Never set an expected value by copying the code's current output. A failing test is a finding: decide whether the code, the docstring or the test is wrong. Record confirmed bugs as `@pytest.mark.xfail(strict=True, reason="BUG: ...")`.
- Test through the public API. Use `pytest.raises(SpecificError, match="...")`, seed all randomness, and mock only external boundaries (network, filesystem locations, time).
- Keep tests lean, because every test has to be worth reading. Every test has a stated reason:
  - **clause**: a behaviour the docstring states;
  - **boundary**: an edge of the valid inputs the docstring defines (n = 1, empty, smallest allowed value);
  - **implied**: a contract users rely on without it being written (inputs not mutated, determinism under a seed, no global side effects), a mathematical consequence of the documented definition, or consistency between functions;
  - **regression**: a specific bug that was fixed.

  A test without one of these reasons does not belong. Implied behaviour that users should rely on explicitly belongs in the docstring. Use the strongest one or two oracles per behaviour, parametrized tables for edge cases and invalid inputs, and one `xfail` per distinct bug. Tests that kill no mutant the others miss, and are not the only test for their reason, should be merged or deleted.
- Coverage finds untested code; mutation testing shows whether tested code is actually checked. Always run mutmut with a worker cap and a memory limit, because each worker imports the full scientific stack:

  ```bash
  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 mutmut run --max-children 4 "SRToolkit.utils.measures*"
  ```

## Submitting a pull request

1. Push your branch to your fork:

   ```bash
   git push origin my-feature
   ```

2. Open a pull request against the `master` branch of `smeznar/SymbolicRegressionToolkit`.
3. Describe what the PR does and, if it fixes a bug, link the relevant issue.
4. Make sure all CI checks pass before requesting a review.
