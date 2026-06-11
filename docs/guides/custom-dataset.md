---
title: Creating a Custom Dataset
---

# Creating a Custom Dataset

## Defining a SymbolLibrary

The [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] defines which tokens an approach may use. Start from the curated default set and restrict it to what's relevant for your problem:

```python
from SRToolkit.utils import SymbolLibrary

# Restrict to a specific token set for a 2-variable problem
sl = SymbolLibrary.from_symbol_list(
    ["+", "-", "*", "/", "sin", "cos", "exp", "sqrt", "^2", "^3", "C"],
    num_variables=2,
)
```

Full list of supported default tokens is documented in
[SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

## Creating a standalone SR_dataset

[SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] wraps input data and evaluation settings for one problem:

```python
import numpy as np
from SRToolkit.dataset import SR_dataset
from SRToolkit.utils import SymbolLibrary

rng = np.random.default_rng(0)
X = rng.uniform(0.5, 5.0, size=(1000, 2))
y = X[:, 0] ** 2 + np.sin(X[:, 1])

sl = SymbolLibrary.from_symbol_list(
    ["+", "*", "sin", "^2", "C"], num_variables=2
)

dataset = SR_dataset(
    X=X,
    y=y,
    symbol_library=sl,
    dataset_name="my_equation",
    original_equation="x0^2 + sin(x1)",
    ground_truth=["X_0", "^2", "+", "sin", "(", "X_1", ")"],
    max_evaluations=50000,
    success_threshold=1e-6,
    constant_bounds=(-10.0, 10.0),
    max_expr_length=20,
)
```

Key parameters:

| Parameter | Purpose |
|---|---|
| `ground_truth` | Token list of the known solution; used to compute BED and check success |
| `success_threshold` | Error below which an expression counts as solved |
| `constant_bounds` | Search range for free constants during parameter fitting |
| `max_evaluations` | Budget passed to the SR approach |
| `max_expr_length` | Maximum token list length the evaluator will accept |

## Creating a dataset from an expression and samplers

A very common case is having only the target expression and a sampling spec for its
inputs — no data arrays yet. [`SR_dataset.from_samplers`][SRToolkit.dataset.sr_dataset.SR_dataset.from_samplers]
covers it in one call: it draws `X` from the samplers (one per variable), evaluates the
ground truth to produce `y`, and records a
[`SampleSource`][SRToolkit.dataset.data_source.SampleSource] so the data can be
regenerated or resampled later.

```python
from SRToolkit.dataset import SR_dataset
from SRToolkit.dataset.sampling import UniformSampling

dataset = SR_dataset.from_samplers(
    ground_truth=["X_0", "^2", "+", "sin", "(", "X_1", ")"],
    samplers=[UniformSampling(0.5, 5.0), UniformSampling(0.5, 5.0)],
    n_samples=1000,
    seed=42,
)
print(dataset.X.shape)   # (1000, 2)
```

`symbol_library` defaults to one variable per sampler and `original_equation` is filled in
from a token-list ground truth, so the minimal call is just the expression and the
samplers. Use `ranking_function="bed"` for a dataset that only needs inputs (no `y`). The
same estimation settings as the constructor (`max_evaluations`, `success_threshold`,
`constant_bounds`, …) are accepted as keyword arguments.

## Building a benchmark with SR_benchmark

Group multiple datasets into a reusable benchmark by subclassing [SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark]. Provide a `data_source` for each dataset so data is generated (or downloaded) automatically on first use and cached locally for subsequent runs.

```python
from SRToolkit.dataset import SR_benchmark
from SRToolkit.dataset.data_source import SampleSource
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.utils import SymbolLibrary

_SYMBOL_LIST = ["+", "-", "*", "/", "sin", "cos", "^2", "^3", "C"]


class MyBenchmark(SR_benchmark):
    __data_version__ = "1.0.0"

    def __init__(self):
        super().__init__("my_benchmark", version="1.0.0")
        self._populate()

    def _populate(self):
        sl_1v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, num_variables=1)
        sl_2v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, num_variables=2)

        self.add_dataset(sl_1v, None, dataset_name="eq1", ranking_function="rmse", max_evaluations=50000,
                         ground_truth=["X_0", "^2", "+", "C"], original_equation="x^2 + c", success_threshold=1e-6,
                         seed=42, samplers=[UniformSampling(0.5, 5.0, uses_negative=False)],
                         data_source=SampleSource(n_samples=10000, seed=42))
        self.add_dataset(sl_2v, None, dataset_name="eq2", ranking_function="rmse", max_evaluations=50000,
                         ground_truth=["sin", "(", "X_0", ")", "+", "X_1", "^2"], original_equation="sin(x0) + x1^2",
                         success_threshold=1e-6, seed=42,
                         samplers=[UniformSampling(0.5, 5.0), UniformSampling(0.5, 5.0)],
                         data_source=SampleSource(n_samples=10000, seed=42))
```

### Shortcut: `add_from_samplers`

When every dataset is "expression + samplers", [`add_from_samplers`][SRToolkit.dataset.sr_benchmark.SR_benchmark.add_from_samplers]
is a shorthand for the `add_dataset(None, …, data_source=SampleSource(...))` form above: it
wires up the [`SampleSource`][SRToolkit.dataset.data_source.SampleSource] for you and
defaults the symbol library to one variable per sampler. Data is still generated lazily on
the first `create_dataset()`.

```python
self.add_from_samplers(
    ground_truth=["X_0", "^2", "+", "C"],
    samplers=[UniformSampling(0.5, 5.0, uses_negative=False)],
    dataset_name="eq1",
    n_samples=10000,
    seed=42,
    success_threshold=1e-6,
)
```

### The `data_source` field

`data_source` is a [DataSource][SRToolkit.dataset.data_source.DataSource] that tells the
cache layer where the raw arrays come from. It describes the data's *origin* only — the
problem's input distribution lives in `samplers`, which stays available for
[resample][SRToolkit.dataset.sr_dataset.SR_dataset.resample] regardless of the source.

| Source | Example | Behaviour |
|---|---|---|
| [SampleSource][SRToolkit.dataset.data_source.SampleSource] | `SampleSource(n_samples=N, seed=S)` | Generate from the dataset's `samplers` + `ground_truth` on first use; reproducible |
| [UrlSource][SRToolkit.dataset.data_source.UrlSource] | `UrlSource("https://...")` | Download a zip archive on first use; all datasets in the archive extracted at once |
| `None` | `None` | Data was supplied directly (e.g. as arrays) and already lives in the cache; raises `FileNotFoundError` if absent |

Need a source of your own (a simulation, a database query, a custom loader)? Subclass
[DataSource][SRToolkit.dataset.data_source.DataSource] and implement `to_dict` /
`from_dict` / `materialize`; it round-trips without any registration.

Data is generated or downloaded once and stored under
`<user_data_dir>/SRToolkit/data/<benchmark>/<version>/` (dots and hyphens in the
version become underscores, e.g. `1.0.0` → `1_0_0`). All subsequent calls
to `create_dataset()` load from this cache instantly.

### Resampling

To get a fresh draw with a different size or seed, pass `n_samples` (and optionally `seed`) to `create_dataset()`. The returned dataset contains newly-sampled data; the cache is not modified.

```python
bm = MyBenchmark()

# Standard load (from cache)
ds = bm.create_dataset("eq1")

# Fresh 500-sample draw
ds_small = bm.create_dataset("eq1", n_samples=500, seed=7)
print(ds_small.X.shape)   # (500, 1)
```

### Serialising a custom benchmark

Set `benchmark` and `version` on the benchmark (already done via `super().__init__`) and call `to_dict()` to get a portable JSON config with no filesystem side effects:

```python
bm = MyBenchmark()
config = bm.to_dict()   # pure dict — no .npz written

import json
with open("my_benchmark.json", "w") as f:
    json.dump(config, f, indent=2)
```

Reconstruct from the config file (data materialises on first `create_dataset`):

```python
from SRToolkit.dataset import SR_benchmark

bm2 = SR_benchmark.from_dict("my_benchmark.json")
ds = bm2.create_dataset("eq1")
```

For self-contained sharing (data embedded in a single file), use `to_archive()` instead — see the [Sharing guide](sharing.md).

## Loading a dataset someone shared

To consume a dataset or benchmark built elsewhere, pick the loader that matches how it was
shared:

| Shared as | Load with |
|---|---|
| A self-contained `.zip` (config + data) | `SR_dataset.from_archive(path)` / `SR_benchmark.from_archive(path)` |
| A hosted `.zip` archive | `SR_dataset.from_url(url)` / `SR_benchmark.from_url(url)` |
| A JSON config / recipe | `SR_dataset.from_dict(path_or_dict)` / `SR_benchmark.from_dict(...)` |

The trade-offs between these channels — shipping data versus shipping a recipe that
regenerates by sampling — are covered in the [Sharing guide](sharing.md).
