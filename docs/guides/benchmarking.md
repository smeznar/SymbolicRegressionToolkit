---
title: Benchmarking SR Approaches
---

# Benchmarking SR Approaches

## Loading a benchmark

SRToolkit ships with three benchmark collections. Data is downloaded or generated automatically on first use.

```python
from SRToolkit.dataset import Feynman, Nguyen, SRSD_Feynman

bm = Feynman()       # 100 physics equations, downloads ~10 MB on first use
bm = Nguyen()        # 10 polynomial / trig expressions
bm = SRSD_Feynman()  # 120 physics equations with per-variable sampling
```

## Listing and creating datasets

```python
# Print a summary table of all datasets
bm.list_datasets()

# Filter by number of variables
names = bm.list_datasets(num_variables=2, verbose=False)

# Create an SR_dataset ready for evaluation
dataset = bm.create_dataset(names[0])

print(dataset.X.shape)          # (n_samples, n_variables)
print(dataset.y.shape)          # (n_samples,)
print(dataset.original_equation)
```

## Evaluating an approach

Pass any [SR_approach][SRToolkit.approaches.sr_approach.SR_approach] to `dataset.evaluate_approach()`:

```python
from SRToolkit.approaches import ProGED

model = ProGED()

# Run 5 independent experiments and keep the top 20 expressions each
results = dataset.evaluate_approach(model, num_experiments=5, top_k=20, initial_seed=0)
```

`results` is an [SR_results][SRToolkit.evaluation.sr_evaluator.SR_results] object. Inspect individual runs:

```python
for r in results:
    print(r.dataset_name, r.approach_name, r.best_expr, r.best_error, r.success)
```

## Resampling data

Each benchmark exposes `resample()` to draw a fresh sample using the same distribution as the original data:

```python
X, y = bm.resample("I.16.6", n=500, seed=42)
```

For SRSD_Feynman, this uses the per-variable sampling objects (log-uniform, linear, integer) defined for each equation.

## Using callbacks

Callbacks attach to the evaluator and fire events during search. Pass them to `evaluate_approach()`:

```python
from SRToolkit.evaluation.callbacks import (
    EarlyStoppingCallback,
    LoggingCallback,
    ProgressBarCallback,
)

results = dataset.evaluate_approach(
    model,
    num_experiments=5,
    callbacks=[
        ProgressBarCallback(),
        EarlyStoppingCallback(threshold=1e-6),
        LoggingCallback(log_file="logs/{dataset_name}_{seed}.log"),
    ],
)
```

| Callback | Behaviour |
|---|---|
| `ProgressBarCallback` | tqdm progress bar updated after each evaluation |
| `EarlyStoppingCallback(threshold)` | stops the search once error drops below `threshold` |
| `LoggingCallback(log_file)` | appends each new best expression to a file or stdout |

### Writing a custom callback

Subclass [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] and override only the events you need. Return `False` from any handler to stop the search early.

```python
from SRToolkit.evaluation.callbacks import SRCallbacks, BestExpressionFound

class PrintEveryImprovement(SRCallbacks):
    def on_best_expression(self, event: BestExpressionFound):
        print(f"  [{event.evaluation_number}] {event.expression}  error={event.error:.4g}")
```
