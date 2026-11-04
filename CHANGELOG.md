### SymbolicRegressionToolkit (unreleased)
- Ranking functions are now pluggable: `RankingFunction` with the built-in `RMSERanking` and `BEDRanking`. `SR_evaluator` and `SR_dataset` accept a built-in name (`"rmse"`, `"bed"`) or the import path of a custom ranking.
- Datasets can carry a held-out test set (`test_X` / `test_y`, `dataset.test`) and extra data for your own measures (`aux_data`), stored as `DataSplit`s. Both are saved, cached and shared with the dataset, and can be regenerated from their own data source.
- New `HoldoutMetric` augmenter scores results on held-out data. It reuses the constants fitted during the search by default; `refit_constants=True` refits them on the held-out data instead.
- BED:
  - takes its evaluation points from the dataset's samplers when the dataset has them, otherwise from `X`;
  - can rank expressions against the observed targets `y` when no ground-truth expression is given;
  - is fixed per dataset by default (`ranking_seed`, the dataset's seed), so every run on a dataset is scored by the same objective; `bed_resample=True` resamples it per run.
- Added an `rmse` regression metric.
- Python 3.14 support: wheels are built for 3.10–3.14, and CI tests all five versions.
- Removed the `editdistance` dependency (no wheels for Python ≥ 3.13); `edit_distance` now uses a pure-Python Levenshtein implementation.
- Installing from source without a C compiler now works as documented: the Cython extension is skipped and the pure-Python evaluator is used. A `RuntimeWarning` is issued the first time the `"stack"` backend falls back, and `SRToolkit.utils.expression_compiler.CYTHON_AVAILABLE` tells which evaluator is in use. Set `SRTOOLKIT_REQUIRE_CYTHON=1` to make a failed build an error.
- `SR_results` can be iterated over directly (`for result in results:`); previously such a loop ended with an `AssertionError` (#5).
- Results record the evaluation trace: `EvalResult.trace` lists, for every counted `evaluate_expr` call, the index into `all_models` of the requested expression (cache hits included). New `EvalResult.best_so_far()`, `calls_to_threshold(threshold=None)` and `best_error_within(budgets)` compute the incumbent history, the number of calls needed to reach a threshold, and the best error within given budgets.
- `import SRToolkit` no longer imports PyTorch: `EDHiE` and `EDHiEConfig` are loaded the first time they are used. This cuts the import's memory use from about 770 MB to 150 MB and makes it faster.
- `PredictFn`, the type of the `predict` argument of the regression metrics, is exported from `SRToolkit.utils`.
- `create_behavior_matrix` and `bed` take a `backend` argument (default `"stack"`), and BED ranking now respects the `backend` setting, which previously only applied to RMSE ranking and `get_metric`.
- The docstrings of `SRToolkit.utils.measures` now fully specify each function: valid inputs, special values (`nan`, `inf`), exceptions, and how `bed` handles non-finite values.
- Saved results and configs (`ModelResult`, `EvalResult`, `ApproachConfig`, `HoldoutMetric`) now round-trip exactly: numpy arrays keep their dtype and shape (including empty axes such as `(0, 3)`) and object arrays are saved element by element, tuples stay tuples, and dicts with non-string keys (int, float, tuple, numpy scalars) keep their keys. Files written by earlier versions still load. The supported values and the file format are documented on the new `SRToolkit.utils.serialization` reference page.
- Expression constraints (`SRToolkit.utils.expression_constraints`): `ExpressionConstraint` checks a candidate before scoring (`pre_check`, on the expression) and/or after (`post_check`, on the fitted model), and returns a `CheckResult` (passed, message, amount). Built-ins: `MaxConstants`, `MaxTokens`, `MaxTreeDepth`, `MaxTreeSize`, `RequiredVariables`, `MaxSymbolCount`, `NoNesting` and `OutputBounds`. `OutputBounds` takes one bound for all points or one per point. A check that raises counts as failed: its `CheckResult` has `error` set, a required constraint then gives status `"error"`, and an advisory one is recorded in `advisory_violations`.
- `SR_evaluator` takes `required_constraints` (a failed check gives status `"rejected"`) and `advisory_constraints` (failed checks are recorded in `ModelResult.advisory_violations` without changing the error). `check_constraints(expr)` runs the pre-checks without evaluating, billing or recording anything, and returns the failed required and advisory checks as two lists. `max_constants` / `max_expr_length` add a `MaxConstants` / `MaxTokens` in front of the list unless it already has one.

**Breaking changes:**
- Removed the `RMSE`, `BED` and `R2` augmenters; use `HoldoutMetric`.
- `EstimationSettings` is renamed to `RankingSettings`.
- Removed the `bed_X` and `domain_bounds` settings; pass the points as `X` with `num_points_sampled=-1`, or give samplers.
- `ground_truth` must be an expression (token list or `Node`); a precomputed behavior matrix goes in the `bed_gt` setting.
- Removed `SR_evaluator.to_dict` / `from_dict`; serialise the dataset instead.
- Datasets are saved with `format_version: 3`; loading `format_version: 1` configs is no longer supported (version 2 still loads).
- `EvalResult` stores `all_models` (in first-seen order), `trace`, `success_threshold` and `top_k`; `min_error`, `best_expr`, `success`, `num_evaluated` and `evaluation_calls` are now read-only properties derived from them, so the constructor no longer takes them. `top_models` is now a method, `top_models(k=None)`, returning the `k` best models (default `top_k`); it returns the same objects as `all_models`, breaks ties in error by expression length (shorter first) and then by first-seen order, and is no longer written to JSON. Results are saved with `format_version: 2`; results saved by earlier versions cannot be loaded and raise a `ValueError` saying so.
- `SR_evaluator.models` is a list in first-seen order instead of a dict keyed by the expression string, and `total_evaluations` is a read-only property (`len(trace)`).
- Calls to `evaluate_expr` after evaluation has stopped are no longer counted in `total_evaluations` / `evaluation_calls`.
- The module-level sampling helpers in `SRToolkit.dataset.sampling` (`uniform_sampling`, `log_uniform_sampling`, …) are now private; use the `UniformSampling`, `LogUniformSampling` and `IntegerUniformSampling` classes.
- Regression metrics (`mse`, `rmse`, `mae`, `r2`, `nrmse`) raise a `ValueError` unless `predict(X)` and `y` are 1-D of length `len(X)`. Previously a `(n, 1)` target was silently broadcast against `(n,)` predictions into an `(n, n)` array, giving a meaningless value.
- `r2` and `nrmse` with constant targets return `1.0` / `0.0` for an exact prediction and `-inf` / `inf` otherwise, instead of `nan`. `predict` is now always called, so its exceptions propagate.
- `REGRESSION_METRICS` is a read-only mapping.
- `tokens_to_tree`, `compile_expr` and `compile_expr_rmse` raise a `TypeError` for an expression that is not a list of token strings or a `Node` (previously a bare `Exception`, or numpy's "expected str instance" message). Numbers in token lists must be strings (`"1.5"`); passing `1.5` now says so.
- Invalid arguments to `edit_distance`, `tree_edit_distance`, `create_behavior_matrix` and `bed` raise `TypeError` or `ValueError` naming the offending parameter, instead of a bare `Exception`, `UnboundLocalError` or `AttributeError` (a string or tuple passed to `edit_distance` used to be compared character by character). `bed` and `create_behavior_matrix` now validate `num_consts_sampled`, `consts_bounds`, `num_points_sampled`, matrix shapes and the row counts of matrices against `X`.
- Values that cannot be saved as JSON (complex numbers, `datetime64`, sets, masked arrays, arbitrary objects) in augmentations, ranking data or approach configs now raise a `TypeError` from `to_dict` naming the value's location (for results the full path, e.g. `all_models[3].ranking_data['z']`), instead of failing later in `json.dump`. Saved arrays now include `dtype` and `shape`, and tuples and non-string-keyed dicts are written in a new form, so files saved by this version cannot be read by earlier versions.
- `ModelResult.from_dict` raises a `ValueError` for a status that is not one of `STATUSES`, instead of failing later in `status_counts` with a `KeyError`.
- The grammar base class `Constraint` is renamed `GrammarConstraint`.
- The `status_detail` of a rejected expression names the constraint, e.g. `"MaxConstants: 9 constants, limit 8"` instead of `"9 constants exceed max_constants=8"`.
- `max_constants` / `max_expr_length` must be ints (numpy ints accepted); other values raise a `ValueError` instead of being silently ignored. Giving one together with a required constraint of the same class and a different limit raises a `ValueError`.

**Fixes:**
- A dataset with custom `constant_bounds` scored every expression with a constant as NaN after a JSON round trip (`from_dict`, `from_archive`, `ExperimentGrid`).
- A `Node` ground truth is now serialised with the dataset's own symbol library.
- `SR_dataset.refresh()` regenerates the test and aux data too, and `data_cache.remove` deletes them.
- Saving results no longer fails on dict keys that are numpy scalars (e.g. `np.int64`) or on object arrays holding arrays.
- `SR_dataset` now describes BED datasets correctly.
- Compiled expressions given fewer constants than they use raise a `ValueError` on every backend. The default `"stack"` backend used to read past the end of the array and return garbage values. A `SampleSource` whose ground truth has a free constant (`C`) now fails with a clear error, as `SR_dataset.resample` already did, instead of generating a dataset with garbage `y`; the custom-dataset guide example that used one is fixed (#4).
- `EvalResult` saves `metadata` like its other fields, so tuples, numpy arrays and numpy scalars in it round-trip exactly; previously a numpy value made `json.dump` fail and tuples came back as lists. A numpy `wall_time` no longer breaks saving either.
- `add_augmentation` (on `ModelResult` and `EvalResult`) stores a shallow copy of `data` and no longer adds `"_type"` to the caller's dict.
- `EvalResult.calls_to_threshold` ignores models with a non-finite error, like the other trace metrics.
- Augmenters with `scope="all"` annotated the top models twice (e.g. `ExpressionToLatex` and `ExpressionToLatex_1`).
- Fixed examples in the benchmarking, grammars, quickstart and experiments guides and the README (a wrong benchmark, a derivation that never finished, inconsistent file names) (#5).
- `bed` gave wrong values when called with two behavior matrices that have different numbers of columns, e.g. a ground-truth matrix sampled with a different `num_consts_sampled`. `BEDRanking` always compares matrices of equal width, so rankings were not affected.
- `bed` returned 0 for points where one expression is `nan` and the other infinite, or where they are `+inf` and `-inf`. Such points now give `inf`; points where both expressions are the same infinity give 0, as documented.
- Evaluating an expression on an `X` with too few columns (or a 1-D `X`) read past the end of the array on the default `"stack"` backend and returned garbage; it now raises a `ValueError`. This affected `compile_expr`, `create_behavior_matrix` and `bed`.
- `tree_edit_distance` priced relabelling by the character edit distance between symbols when the optional `editdistance` package was installed, so results depended on the environment. It now always uses unit costs.
- Regression metrics no longer emit NumPy warnings on empty or non-finite input; the returned `nan` / `inf` is the signal.
- `MaxOccurrences` let a grammar rule containing the symbol several times (e.g. `E -> x * x`) exceed the limit.
- An `SR_evaluator` whose required constraints allow more than 8 free constants no longer has its parameter estimator reject such expressions at its default limit of 8.

### SymbolicRegressionToolkit-1.6.0 (2026-06-23)
- New backends for expression evaluations (about 1.5x speedup for parameter estimation)
- SymbolLibrary can now be as a global default or used with the ContextManager (with SymbolLibrary()...:)
- Added grammars (CFG, PCFG) and the ability to add constraints to them.
- Added a TimeLimitCallback
- Added wallclock time to experiment
- Added some default constraints that work as built-ins and reference implementations
- Added guides to the documentation
- Added some tests
- Reworked how sharing code/benchmark works:
  - Code is shared through bundles, which can be installed and are directly usable after
  - Datasets/benchmarks are versioned and can be shared via .zip files, of through configs implementing DataSources
  - Configs should be shared directly and should be run through pack, so they are directly usable
  - ExperimentGrid can now be exported together with all the bundles and data it needs. Custom code is shipped only via re-packed installed bundles or explicit `.srtk` files passed to `export(additional_bundles=...)` (no more guessing a bundle from a loose source file); uncovered classes warn, or raise with `export(strict=True)`

### SymbolicRegressionToolkit-1.5.0 (2026-04-14)
- Reworked how SR_results and ResultAugmenters work
- Improved documentation
- Fixed some bugs
- Made the interface more consistent and easier to use
- Two instances of SR_benchmarks are now in separate files and allow resampling of data
- All (or at least all relevant) classes can now be saved and loaded/transformed into and from a dictionary
- Added some tests (a lot more to go)
- Added new examples

**New Features:**

- Added a base class for approaches and implemented ProGED and EDHiE
- Added a way to run generate and run experiments easily
- Added the callback system to allow custom events during experiments


### SymbolicRegressionToolkit-1.4.0 (2025-10-28)

- Updated documentation
- Rewrote the SR_dataset and SR_benchmark classes
- New readme and logo

**New Features:**

- Added ResultAugmenter to augment result of evaluation with additional measures, simplification of best expressions, etc.
- Added distance measures
- Added BED as a possible ranking measure for evaluation


### SymbolicRegressionToolkit-1.3.2 (2025-07-07)

- Updated documentation

**New Feature:**

- Added a way to generate expressions
- Expression simplification now more or less works

**Bug Fixes:**

- Feynman dataset and Nguyen datasets are now almost done
- Small fixes to different functionalities


### SymbolicRegressionToolkit-1.2.6 (2025-04-26)

- Updated documentation
- Added the change log

**New Features:**

- Expressions can now be transformed into latex code
- Added Dataset and benchmark objects that create evaluators for Symbolic Regression models
- Added modified versions of feynman and nguyen benchmarks

**Bug Fixes:**

- Fixed expressions with constants only evaluating to one value instead of an array


### SymbolicRegressionToolkit-1.1.0 (2024-12-10)

- Project restructure

**New Features:**

- Added documentation
- Expanded upon examples

### SymbolicRegressionToolkit-1.0.0 (2024-12-06)

- Initial release
- Expression compilation
- Parameter estimation
- Model evaluation
- Examples for parameter estimation and performance evaluation
