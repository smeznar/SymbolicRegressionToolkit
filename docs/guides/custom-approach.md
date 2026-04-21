---
title: Implementing a Custom Approach
---

# Implementing a Custom Approach

All SR approaches subclass [SR_approach][SRToolkit.approaches.sr_approach.SR_approach] and implement two abstract methods: `prepare()` and `search()`. Optionally, override `adapt()` when your approach benefits from a pre-training or warm-up phase.

## Minimal example

```python
import numpy as np
from SRToolkit.approaches.sr_approach import SR_approach, ApproachConfig
from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import SymbolLibrary, generate_n_expressions
from dataclasses import dataclass
from typing import Optional

@dataclass
class RandomSearchConfig(ApproachConfig):
    name: str = "RandomSearch"
    batch_size: int = 50

class RandomSearch(SR_approach):
    def __init__(self, batch_size: int = 50):
        super().__init__(RandomSearchConfig(batch_size=batch_size))
        self.batch_size = batch_size

    def prepare(self):
        pass  # stateless — nothing to reset

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        sl = sr_evaluator.symbol_library

        while not sr_evaluator.should_stop():
            exprs = generate_n_expressions(self.batch_size, sl, seed=int(rng.integers(1e9)))
            for expr in exprs:
                sr_evaluator.evaluate_expr(expr)
                if sr_evaluator.should_stop():
                    return
```

### Key rules for `search()`

- Call `sr_evaluator.evaluate_expr(expr)` for every candidate expression. The evaluator handles parameter fitting, caching, and result bookkeeping.
- Check `sr_evaluator.should_stop()` (or `sr_evaluator.total_evaluations >= sr_evaluator.max_evaluations`) regularly and return when it is `True`.
- Do **not** access target values directly — use the evaluator as the sole interface to `y`.

## The `adapt()` lifecycle

`adapt()` is called before `search()` and receives only `X` (no `y`). Use it for data-driven warm-up that does not require knowing the target:

```python
def adapt(self, X: np.ndarray, symbol_library: SymbolLibrary):
    # Train a generative model, build a PCFG, pre-compute statistics, etc.
    self._grammar = build_grammar(symbol_library)
```

Control when adaptation happens via the `adaptation_scope` property:

| Value | When `adapt()` is called |
|---|---|
| `"never"` (default) | Never — skip `adapt()` entirely |
| `"once"` | Once per symbol library; state is saved and reloaded for subsequent runs |
| `"experiment"` | Before every `search()` call |

For `"once"` scope, also implement `save_adapted_state()` and `load_adapted_state()`:

```python
import torch

@property
def adaptation_scope(self):
    return "once"

def save_adapted_state(self, path: str):
    torch.save(self._model.state_dict(), path + ".pt")

def load_adapted_state(self, path: str):
    self._model.load_state_dict(torch.load(path + ".pt"))
```

## Configuration with ApproachConfig

Subclass [ApproachConfig][SRToolkit.approaches.sr_approach.ApproachConfig] to make your approach serialisable. This enables saving/loading grids and running jobs from the command line:

```python
@dataclass
class RandomSearchConfig(ApproachConfig):
    name: str = "RandomSearch"
    batch_size: int = 50
```

The config is serialised automatically when the grid is saved. To support CLI execution, also implement `from_config()`:

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
dataset = bm.create_dataset("Nguyen-1")
model = RandomSearch(batch_size=100)
results = dataset.evaluate_approach(model, num_experiments=3)
```
