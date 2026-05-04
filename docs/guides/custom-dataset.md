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

## Building a benchmark with SR_benchmark

Group multiple datasets into a reusable benchmark by subclassing [SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark]:

```python
import os
import numpy as np
from platformdirs import user_data_dir
from SRToolkit.dataset import SR_benchmark
from SRToolkit.utils import SymbolLibrary

_SYMBOL_LIST = ["+", "-", "*", "/", "sin", "cos", "^2", "^3", "C"]

class MyBenchmark(SR_benchmark):
    def __init__(self, dataset_directory=os.path.join(user_data_dir("SRToolkit"), "my_benchmark")):
        super().__init__("MyBenchmark", dataset_directory)
        self._populate()

    def _populate(self):
        os.makedirs(self.base_dir, exist_ok=True)
        sl_1v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, num_variables=1)
        sl_2v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, num_variables=2)

        self.add_dataset(
            "",                         # "" → load from base_dir/dataset_name.npz
            sl_1v,
            dataset_name="eq1",
            ground_truth=["X_0", "^2", "+", "C"],
            original_equation="x^2 + c",
            ranking_function="rmse",
            max_evaluations=50000,
            success_threshold=1e-6,
            constant_bounds=(-10.0, 10.0),
            max_expr_length=20,
            seed=None,
        )
        self.add_dataset(
            "",
            sl_2v,
            dataset_name="eq2",
            ground_truth=["sin", "(", "X_0", ")", "+", "X_1", "^2"],
            original_equation="sin(x0) + x1^2",
            ranking_function="rmse",
            max_evaluations=50000,
            success_threshold=1e-6,
            constant_bounds=(-10.0, 10.0),
            max_expr_length=20,
            seed=None,
        )

    def resample(self, dataset_name: str, n: int, seed=None):
        from SRToolkit.utils import SymbolLibrary
        from SRToolkit.utils.expression_compiler import compile_expr
        info = self.datasets[dataset_name]
        rng = np.random.default_rng(seed)
        sl = SymbolLibrary.from_dict(info["symbol_library"])
        n_vars = sl.num_variables
        X = rng.uniform(0.5, 5.0, size=(n, n_vars))
        f = compile_expr(info["ground_truth"], sl)
        y = f(X, np.array([]))
        return X, y
```

### The `add_dataset` data argument

The first positional argument to `add_dataset` controls where data comes from:

| Value | Behaviour |
|---|---|
| `""` | Load `{base_dir}/{dataset_name}.npz` from disk |
| `np.ndarray` (X only) | Compute `y` from `ground_truth`, then save `.npz` |
| `(X, y)` tuple | Use directly and save `.npz` |

Pass a numpy array on first run to generate and cache the data, then switch to `""` for subsequent loads — or generate the `.npz` files externally and always use `""`.

### Generating `.npz` files

```python
import numpy as np

rng = np.random.default_rng(42)
X = rng.uniform(0.5, 5.0, size=(10000, 2))
y = np.sin(X[:, 0]) + X[:, 1] ** 2
np.savez("eq2.npz", X=X, y=y)
```
