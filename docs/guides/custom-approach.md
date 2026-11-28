---
title: Implementing a Custom Approach
---

# Implementing a Custom Approach

All SR approaches subclass [Approach][SRToolkit.approaches.approach.Approach] and implement `search()`, or the step interface `start()`, `ask()` and `tell()` (see the [module docs][SRToolkit.approaches.approach]). Optionally, override `adapt()` when your approach benefits from a pre-training or warm-up phase.

## Minimal example

```python
import numpy as np
from SRToolkit.approaches.approach import Approach, ApproachConfig
from SRToolkit.evaluation import Evaluator
from SRToolkit.utils import SymbolLibrary, generate_n_expressions
from dataclasses import dataclass

@dataclass
class RandomSearchConfig(ApproachConfig):
    name: str = "RandomSearch"
    batch_size: int = 50

class RandomSearch(Approach):
    def __init__(self, batch_size: int = 50):
        super().__init__(RandomSearchConfig(batch_size=batch_size))
        self.batch_size = batch_size

    def search(self, evaluator: Evaluator, seed: int | None = None):
        rng = np.random.default_rng(seed)
        sl = evaluator.symbol_library

        while not evaluator.should_stop:
            exprs = generate_n_expressions(sl, self.batch_size, verbose=False)
            for expr in exprs:
                evaluator.evaluate_expr(expr)
                if evaluator.should_stop:
                    return
```

### Key rules for `search()`

- Call `evaluator.evaluate_expr(expr)` for every candidate expression. The evaluator handles parameter fitting, caching, and result bookkeeping.
- Check `evaluator.should_stop` regularly and finish search when it is `True`.
- To end the search early (e.g. the search space is exhausted), call `evaluator.stop("reason")`; the reason is saved in the results as `stop_reason`.
- Do **not** access target values directly — use the evaluator as the sole interface to `y`.

## The `adapt()` lifecycle

`adapt()` is called before `search()` and receives `X` (no `y`), the dataset's background knowledge and its required constraints (set up, so their checks can run). Use what you need for warm-up that does not require knowing the target:

```python
def adapt(self, X, background_knowledge, required_constraints):
    # Train a generative model, build a PCFG, pre-compute statistics, etc.
    self._grammar = build_grammar(background_knowledge.symbol_library)
```

Control when adaptation happens via the `adaptation_scope` property:

| Value | When `adapt()` is called |
|---|---|
| `"never"` (default) | Never — skip `adapt()` entirely |
| `"once"` | Once per run; with a state path the state is saved and reloaded for later runs |
| `"experiment"` | Before every `search()` call |

For `"once"` scope, also implement `save_adapted_state()` and `load_adapted_state()`:

```python
import torch

@property
def adaptation_scope(self):
    return "once"

def save_adapted_state(self, path: str):
    torch.save(self._model.state_dict(), path)

def load_adapted_state(self, path: str):
    self._model.load_state_dict(torch.load(path))
```

!!! note "Heavy dependencies"
    If your approach needs heavy libraries (e.g. PyTorch, like the built-in
    [EDHiE][SRToolkit.approaches.EDHiE.EDHiE]), do not make them hard dependencies of the toolkit.
    Declare them in an optional extra and guard the import so users without the extra get a clear
    message instead of an opaque `ImportError`:

    ```python
    try:
        import torch
    except ImportError:
        raise ImportError(
            "MyApproach requires PyTorch. Install it manually or with "
            "pip install 'symbolic-regression-toolkit[approaches]'"
        )
    ```

## Configuration with ApproachConfig

Subclass [ApproachConfig][SRToolkit.approaches.approach.ApproachConfig] to make your approach serialisable. This enables saving/loading grids and running jobs from the command line:

```python
@dataclass
class RandomSearchConfig(ApproachConfig):
    name: str = "RandomSearch"
    batch_size: int = 50
```

The config is serialised automatically when the grid is saved. To support CLI execution, your Approach subclass should also implement [from_config][SRToolkit.approaches.approach.Approach.from_config]:

```python
@classmethod
def from_config(cls, config: dict) -> "RandomSearch":
    return cls(batch_size=config.get("batch_size", 50))
```

## Using the approach

Once implemented, the approach integrates directly with the rest of the toolkit:

```python
from SRToolkit.dataset import Nguyen

bm = Nguyen()
dataset = bm.create_dataset("NG-1")
model = RandomSearch(batch_size=100)
results = dataset.evaluate_approach(model, num_experiments=3)
```
