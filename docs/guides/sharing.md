---
title: Sharing Custom Implementations
---

# Sharing Custom Implementations

SRToolkit's serialization model is built around a single contract: every custom class is identified by its **fully-qualified Python module path** (e.g. `meznar_pcfg_grammar.MyGrammarConstraint`). When you call `to_dict`, this path is embedded in the JSON. When another machine calls `from_dict`, it uses `importlib` to import that module and reconstruct the object. Sharing works as long as the recipient can import the class at the same path.

This guide covers the best practices for structuring and naming shareable files, then walks through everything that can be shared.

## Best practices

### Name files to avoid collisions

Use the convention `{author}_{descriptor}.py` — for example:

```
meznar_pcfg_grammar.py
smith_gp_approach.py
jones_feynman_sampler.py
```

The descriptor should say what the file contains, not just "custom". This ensures that when multiple contributors' files land in the same working directory they do not overwrite each other.

### Use the working directory as the sharing unit

Python adds `.` to `sys.path` automatically, so any `.py` file in the working directory is importable by name. The simplest sharing model is:

```
experiment/
├── meznar_pcfg_grammar.py   # custom constraint
├── smith_gp_approach.py     # custom approach
├── grammar.json             # Grammar.to_dict() output
├── dataset.json             # SR_dataset.to_dict() output
├── approach_config.json     # ApproachConfig.to_dict() output
└── requirements.txt         # SRToolkit version pin
```

Zip the directory and share it. The recipient unpacks and runs from inside it.

### Never define shared classes in a script's `__main__` scope

When a script is run directly, Python sets `__module__ = "__main__"` on every class defined in it. The path `"__main__.MyApproach"` is meaningless on any other machine. Always define classes in a named module file and import from there:

```python
# good: class lives in smith_gp_approach.py, importable by that name
from smith_gp_approach import GeneticProgramming

# bad: class defined in the script being run — serializes as __main__.GeneticProgramming
class GeneticProgramming(SR_approach): ...
```

### Avoid defining classes in notebooks

Classes defined in Jupyter cells also get `__module__ = "__main__"`. Define the class in a `.py` file and import it into the notebook instead:

```python
# In your notebook:
from meznar_pcfg_grammar import PCFGConstraint  # importable, serializable
```

### Verify importability before serializing

After writing your module file, confirm the class is importable under the exact path that will be embedded in the JSON:

```bash
python -c "from meznar_pcfg_grammar import PCFGConstraint; print('ok')"
```

If this fails, `from_dict` will fail on any other machine too.

### Pin the SRToolkit version

The serialization format and internal class paths are tied to library internals. Include the version in a `requirements.txt`:

```
SRToolkit==0.x.y
```

---

## What can be shared

### Symbol library

[SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] is fully self-contained and has no importlib dependency — all information is stored as plain data.

```python
import json
from SRToolkit.utils import SymbolLibrary

sl = SymbolLibrary.from_symbol_list(["+", "-", "*", "sin", "^2"], num_variables=2)

with open("symbol_library.json", "w") as f:
    json.dump(sl.to_dict(), f)

with open("symbol_library.json") as f:
    sl2 = SymbolLibrary.from_dict(json.load(f))
```

### Grammar

[Grammar][SRToolkit.utils.grammar.Grammar] serializes its rules and constraints. Built-in constraints are fully supported. Custom constraints require a `.py` file on the path (see [Custom constraints](#custom-constraints) below).

```python
import json
from SRToolkit.utils.grammar import Grammar

with open("grammar.json", "w") as f:
    json.dump(g.to_dict(), f)

with open("grammar.json") as f:
    g2 = Grammar.from_dict(json.load(f))
```

### Dataset

[SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] serializes its symbol library, sampling configuration, metadata, and a reference to the data files. The data itself is saved separately as `.npz` files, so `to_dict` takes a `base_path` that controls where those files are written.

```python
import json
from pathlib import Path

base = Path("my_dataset")
base.mkdir(exist_ok=True)

with open(base / "dataset.json", "w") as f:
    json.dump(dataset.to_dict(base_path=str(base)), f)
```

Distribute the entire `my_dataset/` directory. The recipient loads it with:

```python
from SRToolkit.dataset import SR_dataset

with open("my_dataset/dataset.json") as f:
    ds = SR_dataset.from_dict(json.load(f))
```

Custom samplers embedded in the dataset require a `.py` file on the path (see [Custom samplers](#custom-samplers) below).

### Approach configuration

[ApproachConfig][SRToolkit.approaches.sr_approach.ApproachConfig] stores all constructor parameters plus the fully-qualified class path of the approach. The recipient needs your approach `.py` file on their path.

```python
import json

with open("approach_config.json", "w") as f:
    json.dump(my_approach.config.to_dict(), f)
```

Approach configs are normally consumed by [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] rather than loaded directly — see [Sharing a complete experiment](#sharing-a-complete-experiment) below.

### Callbacks

[SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] subclasses serialize their constructor parameters alongside `callback_class`. Built-in callbacks ([ProgressBarCallback][SRToolkit.evaluation.callbacks.ProgressBarCallback], [EarlyStoppingCallback][SRToolkit.evaluation.callbacks.EarlyStoppingCallback], [LoggingCallback][SRToolkit.evaluation.callbacks.LoggingCallback]) require no extra files.

```python
import json
from SRToolkit.evaluation.callbacks import EarlyStoppingCallback

cb = EarlyStoppingCallback(threshold=1e-6)

with open("callback.json", "w") as f:
    json.dump(cb.to_dict(), f)
```

Callbacks are normally passed to [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] and reconstructed automatically — see [Sharing a complete experiment](#sharing-a-complete-experiment) below.

### Custom constraints

Provide the constraint class in a module file following the naming convention, then share the grammar JSON alongside it:

```python
# meznar_constraints.py
from SRToolkit.utils.grammar import Constraint

class PhysicsConstraint(Constraint):
    def __init__(self, forbidden_terminals):
        self.forbidden = frozenset(forbidden_terminals)

    def allows(self, slot, rule, global_):
        return self.forbidden.isdisjoint(rule.rhs)

    def to_dict(self):
        return {**super().to_dict(), "forbidden_terminals": sorted(self.forbidden)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["forbidden_terminals"])
```

The recipient places `meznar_constraints.py` in their working directory; `Grammar.from_dict` resolves the class automatically.

### Custom samplers

Same pattern as constraints — provide the sampler class in a module file:

```python
# meznar_samplers.py
from SRToolkit.dataset.sampling import Sampler
import numpy as np

class GaussianSampler(Sampler):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, n, rng=None):
        rng = np.random.default_rng(rng)
        return rng.normal(self.mean, self.std, n)

    def to_dict(self):
        return {
            "sampler_class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])
```

`sampling_from_dict` from `SRToolkit.dataset.sampling` dispatches via the `sampler_class` key, so the recipient only needs the module file on their path.

### Custom callbacks

Same pattern — module file alongside the JSON:

```python
# meznar_callbacks.py
from SRToolkit.evaluation.callbacks import SRCallbacks

class BestExprCallback(SRCallbacks):
    def __init__(self, output_file):
        self.output_file = output_file

    def on_evaluation(self, result, evaluator):
        with open(self.output_file, "w") as f:
            f.write(str(result.best_expr))

    def to_dict(self):
        return {**super().to_dict(), "output_file": self.output_file}

    @classmethod
    def from_dict(cls, d):
        return cls(d["output_file"])
```

---

## Sharing a complete experiment

The natural unit for sharing a full experiment is an [ExperimentGrid][SRToolkit.experiments.ExperimentGrid]. Calling `save()` (or `save_commands()`, which calls it automatically) writes a self-contained directory:

```
results/
├── grid.json                          # grid specification
├── _datasets/
│   └── velocity/
│       ├── velocity.json              # SR_dataset.to_dict() output
│       └── velocity.npz              # data
├── _approaches/
│   └── meznar_gp_config.json          # ApproachConfig.to_dict() output
└── _callbacks.json                    # list of callback dicts (if any)
```

Share the entire `results/` directory alongside any custom `.py` files it depends on. The recipient loads with:

```python
from SRToolkit.experiments import ExperimentGrid

grid = ExperimentGrid.load("results/grid.json")
```

Datasets and approaches are reconstructed lazily when jobs run, so the load is fast. To execute a single job directly:

```python
jobs = grid.create_jobs()
jobs[0].run()
```

For HPC or parallel runs, generate a commands file and dispatch from there:

```bash
python -m SRToolkit.experiments commands \
    --grid results/grid.json \
    --out results/commands.txt

# then dispatch each line, e.g.:
bash results/commands.txt
```

Individual jobs can also be run via the CLI without loading the full grid:

```bash
python -m SRToolkit.experiments run_job \
    --dataset results/_datasets/velocity/velocity.json \
    --approach results/_approaches/meznar_gp_config.json \
    --info    results/velocity/meznar_gp/exp_0/info.json \
    --callbacks results/_callbacks.json
```

!!! note
    Only include custom `.py` files that are actually needed. A standard experiment using only built-in approaches, constraints, and samplers needs no `.py` files alongside the `results/` directory — the JSON files are fully self-contained.
