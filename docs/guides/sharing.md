---
title: Sharing Custom Implementations
---

# Sharing Custom Implementations

Sharing in SRToolkit splits along three axes that travel differently. Keep them
separate and each one becomes easy to forward; mix them — embed raw arrays in a config,
ship a whole directory, define a class in a notebook — and they stop being portable.

| Axis | What it is | How to share |
|---|---|---|
| **[Code](#code-srtk-bundles)** | Custom approaches, constraints, samplers, callbacks, data sources | A `.srtk` **bundle** — install once, importable from anywhere |
| **[Configs](#configs)** | Plain JSON describing a component — a dataset, benchmark, sampler, grammar, callback, … | Share the `.json` directly — code and data stay separate |
| **[Data](#data-datasets-and-benchmarks)** | `.npz` arrays in a versioned cache | Reached through the config's `data_source` (a URL or a sampler recipe), or shipped inside a self-contained archive |

The three are deliberately decoupled: code is versioned independently of the settings that
use it, and a config carries only a *pointer* to its data, never the arrays themselves. If you
just want to hand someone a dataset, benchmark or experiment grid **with its data and code**,
jump to [sharing a whole object](#sharing-a-whole-object): one `.zip`, one call on each end.

---

## Code (`.srtk` bundles)

A bundle is a single `.srtk` zip archive holding your Python source files and a manifest.
Use it to share custom approaches, constraints, samplers, callbacks, or data sources — any
class that serialises through a `*_class` path. Configs are **not** part of the code: how a
model works (the bundle) and the settings you ran it with (the config) are shared separately
so each can be versioned on its own.

### Creating a bundle

```python
from SRToolkit.bundle import pack

pack(
    files=["my_approach/approach.py", "my_approach/ops.py"],
    out_path="meznar_gp.srtk",
    name="meznar-gp",
    version="0.1.0",
    author="meznar",
    python_deps=["torch>=2.0"],        # optional; checked at install time
    srtk_min_version="1.5.0",          # optional
    configs=["meznar_settings.json"],  # optional; see "Using a shared config"
)
```

Each file is stored under `src/` by its basename, so basenames must be unique.

A bundle should hold **all** your additions — pack every custom approach, constraint,
sampler, and so on into one `.srtk` rather than one bundle per class.

!!! note "Imports between bundled files aren't rewritten"
    Packing copies your `.py` files verbatim and installs them as sibling modules of one
    package (`srtk_bundles.<name>_<version>`); your `import` statements are left untouched.
    A single subclass kept in its own self-contained file always works. When one
    implementation is **split across several files**, those files must import each other with
    **relative** imports (`from .ops import foo`) — an absolute `import ops` resolves to a
    top-level module that won't exist on the recipient's machine. Imports of `SRToolkit`,
    declared `python_deps`, and the standard library are unaffected.

If `configs=[...]` is given, each JSON config is copied alongside the code with a
`.srtk.json` suffix and two metadata keys injected — `_bundle` and `_version`. That
annotated copy is what you ship with the bundle; the recipient passes it straight to any
`from_dict` and the class-path rewrite happens automatically (see below).

### Installing a bundle

```python
from SRToolkit.bundle import install

install("meznar_gp.srtk")
```

After install, the bundle lives under
`<user_data_dir>/SRToolkit/srtk_bundles/<safe_name>_<version_slug>/` and its parent is added
to `sys.path`, making it importable as `srtk_bundles.<safe_name>_<version_slug>`. The
`sys.path` entry is re-added automatically whenever an annotated config is loaded via
`from_dict`.

### Using a shared config

An annotated `.srtk.json` config can be passed directly to any `from_dict` — no extra step.
The dispatchers that handle configs containing custom classes
([`Sampler.from_dict`][SRToolkit.dataset.sampling.Sampler.from_dict],
[`DataSource.from_dict`][SRToolkit.dataset.data_source.DataSource.from_dict],
[`GrammarConstraint.from_dict`][SRToolkit.utils.grammar.constraints.GrammarConstraint.from_dict],
[`Grammar.from_dict`][SRToolkit.utils.grammar.Grammar.from_dict], and the approach loader
used by [`ExperimentGrid`][SRToolkit.experiments.ExperimentGrid]) detect the `_bundle` key
and rewrite every `*_class` dotted path to the installed bundle's import prefix. Paths
starting with `SRToolkit.` are left unchanged.

```python
import json
from SRToolkit.utils.grammar import Grammar

raw = json.load(open("meznar_grammar.srtk.json"))
grammar = Grammar.from_dict(raw)   # _bundle key triggers the rewrite automatically
```

If the config wasn't annotated at pack time, or you want to bind against a specific version,
call [`bind_config`][SRToolkit.bundle.bind_config] explicitly:

```python
from SRToolkit.bundle import bind_config

config = bind_config(raw, "meznar-gp")               # latest installed version
config = bind_config(raw, "meznar-gp", version="0.1.0")
```

### Using bundle code directly

```python
from SRToolkit.bundle import enable_bundle_imports
enable_bundle_imports()   # only needed in a fresh Python session

from srtk_bundles.meznar_gp_0_1_0.approach import MyApproach
approach = MyApproach(...)
```

### Listing and removing bundles

```python
from SRToolkit.bundle import list_installed, uninstall

for entry in list_installed():
    print(entry["name"], entry["version"], entry["author"])

uninstall("meznar-gp", version="0.1.0")   # version omitted → latest by semver
```

### Writing shareable custom classes

Every shareable class follows the same recipe: define it in a named `.py` file, implement
`to_dict` and `from_dict`, and make sure `to_dict` embeds a fully-qualified `*_class` path so
the matching dispatcher can reconstruct it via `importlib` — no central registry.

```python
# meznar_constraints.py
from SRToolkit.utils.grammar import GrammarConstraint

class PhysicsConstraint(GrammarConstraint):
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

Custom [samplers][SRToolkit.dataset.sampling.Sampler],
[callbacks][SRToolkit.evaluation.callbacks.Callback], and
[data sources][SRToolkit.dataset.data_source.DataSource] follow the same pattern, dispatched
via `sampler_class`, `callback_class`, and `source_class` respectively.

!!! warning "Never define shared classes in `__main__`"
    That includes scripts run directly with `python my_script.py` and Jupyter notebook cells.
    Python sets `__module__ = "__main__"` on classes defined there, so the serialised path
    `"__main__.MyApproach"` is meaningless on any other machine. Define the class in a named
    `.py` file and import it.

---

## Configs

A config is a pure, JSON-safe dict produced by `to_dict()` and reconstructed by `from_dict()`.
This pattern runs throughout the package: samplers, grammars, constraints, callbacks, data
sources, symbol libraries, and whole datasets, benchmarks, and experiment grids all serialise
this way — and custom subclasses do too (see
[Writing shareable custom classes](#writing-shareable-custom-classes)). Sharing a config
shares the *settings*, decoupled from the code that implements them and the data they point at.

The rest of this section focuses on **dataset and benchmark** configs — the most common thing
to share — but the mechanics apply to any config. For datasets specifically, the config is the
**recipe** model of sharing: the JSON carries samplers and a `data_source` pointer, *not* the
data arrays. The recipient calls `from_dict` and the data is regenerated (by sampling) or
downloaded on first use. To make the data travel *with* the config instead, use
[share.save](#sharing-a-whole-object).

### Serialising a benchmark

Call `to_dict()` on any `Benchmark` to get a plain dict — no files are written.

```python
from SRToolkit.dataset import Nguyen

bm = Nguyen()
config = bm.to_dict()          # plain dict — safe to json.dump
```

Save it to disk to share as a file:

```python
import json
with open("nguyen_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

### Reconstructing a benchmark from config

Pass either the dict or the file path to `Benchmark.from_dict()`. Each dataset stays lazy:
its data is materialised from the embedded `data_source` only when `create_dataset()` is
called.

```python
from SRToolkit.dataset import Benchmark

bm = Benchmark.from_dict(config)            # from a dict in memory
bm = Benchmark.from_dict("nguyen_config.json")   # or directly from a JSON file

dataset = bm.create_dataset("NG-1")            # materialises this dataset's data on demand
```

### Standalone dataset configs

`Dataset.to_dict()` works the same way, but the instance must have its `benchmark` and
`version` fields set — both are required so the cache layer can locate the data. The easiest
way to get a config-only dataset is
[`from_samplers`][SRToolkit.dataset.dataset.Dataset.from_samplers], which records a
[`SampleSource`][SRToolkit.dataset.data_source.SampleSource] for you:

```python
from SRToolkit.dataset import Dataset
from SRToolkit.dataset.sampling import UniformSampling

ds = Dataset.from_samplers(
    ground_truth=["X_0", "^2", "+", "C"],
    samplers=[UniformSampling(0.5, 5.0)],
    n_samples=10000,
    seed=42,
    dataset_name="my_eq",
    benchmark="my_project",    # namespace — required before to_dict
    version="1.0.0",
)

config = ds.to_dict()          # pure JSON — no arrays embedded
```

!!! note "`to_dict` may persist arrays"
    When a dataset's `data_source` is `None`, its in-memory arrays are the only copy of the
    data, so `to_dict()` writes them into the cache version directory to keep the config
    reloadable. For `SampleSource` / `UrlSource` datasets there are no arrays to write and the
    call has no filesystem side effects.

### The recipe model: regenerate by sampling

Build a sampling-backed dataset with
[`from_samplers`][SRToolkit.dataset.dataset.Dataset.from_samplers] (single dataset,
above) or
[`add_from_samplers`][SRToolkit.dataset.benchmark.Benchmark.add_from_samplers] (within a
benchmark), then ship the JSON. The recipient's `from_dict` regenerates the arrays by drawing
from the stored samplers:

```python
sender_config = ds.to_dict()                       # → my_eq.json, no .npz
recipient = Dataset.from_dict(sender_config)    # data regenerated by sampling here
```

For a single dataset, `from_dict` materialises immediately; within a benchmark,
`Benchmark.from_dict` keeps each entry lazy and the first `create_dataset()` triggers the
sampling.

!!! warning "Reproducibility depends on the seed"
    Regeneration reproduces the **exact** data only when the `SampleSource` carries a fixed
    `seed`. With `seed=None` the recipient gets statistically equivalent but *different*
    points. If everyone must evaluate on identical numbers, set a seed, or share the dataset with
    [share.save](#sharing-a-whole-object) so the data travels with the config.

!!! danger "Reconstruction imports the classes named in the config"
    `from_dict` instantiates the sampler, source, and constraint classes named by the config's
    `*_class` paths. Only load configs you trust.

---

## Data (datasets and benchmarks)

A dataset's arrays never live in its config — they live in the local **cache** at
`<user_data_dir>/SRToolkit/data/<benchmark>/<version>/`, and a config only points at them.
What does the pointing is the `data_source`.

### Data sources

The `data_source` is a [DataSource][SRToolkit.dataset.data_source.DataSource] object that
controls where the raw arrays come from. It captures the data's *origin* only; the problem's
input distribution lives in `samplers` and stays available for resampling no matter which
source is used.

| Source | Example | When to use |
|---|---|---|
| [SampleSource][SRToolkit.dataset.data_source.SampleSource] | `SampleSource(n_samples=10000, seed=42)` | Generate from the stored samplers + ground truth; reproducible only with a fixed `seed` |
| [UrlSource][SRToolkit.dataset.data_source.UrlSource] | `UrlSource("https://...")` | Public data served as a `.zip` archive, downloaded once into the cache |
| [FallbackSource][SRToolkit.dataset.data_source.FallbackSource] | `FallbackSource([UrlSource("https://..."), SampleSource(seed=42)])` | Try each source in order — download canonical data, regenerate from samplers if unavailable |
| `None` | `None` | Data was supplied directly (as arrays) and already lives in the cache; fail fast if absent |

The built-in benchmarks (`Feynman`, `Nguyen`, `SRSD_Feynman`) give each dataset a
[FallbackSource][SRToolkit.dataset.data_source.FallbackSource] of
`[UrlSource(archive), SampleSource(...)]`: they download their canonical data once from a
hosted archive — so every machine benchmarks on identical inputs — and regenerate locally
from the stored samplers only if the download is unavailable. Because the whole chain lives
in the config, that preference travels with a shared recipe, archive or grid (constructing
the benchmark with `force_generate=True` instead pins the data_source to sampling only).

Need something exotic? Subclass [DataSource][SRToolkit.dataset.data_source.DataSource] and
implement `to_dict` / `from_dict` / `materialize` — custom sources round-trip without
registration (see [Writing shareable custom classes](#writing-shareable-custom-classes)) and
travel to other machines via bundles.

!!! info "Drift detection"
    The cache stores a hash of each entry's `data_source` + `samplers`. If that config later
    changes — a different `url`, a new `n_samples`/`seed`, or a switch between source kinds —
    the next load warns that the cached bytes are stale and points you to `refresh()`.

### Hosting data for a `UrlSource`

A [UrlSource][SRToolkit.dataset.data_source.UrlSource] downloads a `.zip` into the cache the
first time a dataset needs its data. The zip holds `<key>.npz` files at its root or under
`data/` (the layout of the built-in benchmarks' hosted archives), or it is an archive written
by [share.save](#sharing-a-whole-object), from which only the data of the dataset's own
benchmark and version is taken.

### Cache management

The [`data_cache`][SRToolkit.dataset.data_cache] module is the housekeeping interface for the
materialised `.npz` files (the version directory uses underscores: `1.0.0` → `1_0_0`).

```python
from SRToolkit.dataset import data_cache

# List every cached dataset
for entry in data_cache.list():
    print(entry["benchmark"], entry["version"], entry["key"], entry["size_bytes"])

# Get the expected path for a specific entry (may not exist yet)
p = data_cache.dataset_path("feynman", "1.0.0", "I.16.6")

# Remove a single dataset, a whole version, or an entire benchmark
data_cache.remove("feynman", "1.0.0", "I.16.6")   # one dataset (+ its _gt.npy / .meta.json)
data_cache.remove("feynman", "1.0.0")             # one version
data_cache.remove("feynman")                      # the whole benchmark

# Garbage-collect: keep only the latest version per benchmark (or wipe everything)
data_cache.gc()                  # drop all but the latest version of each benchmark
data_cache.gc(keep_latest=False) # wipe the entire cache

# Force-refresh a single dataset from a URL source
from SRToolkit.dataset.data_source import UrlSource
data_cache.refresh("feynman", "1.0.0", "I.16.6", source=UrlSource("https://..."))
```

You can also refresh through the `Dataset` instance (required for a `SampleSource`, which
needs the dataset's samplers, and the only option when `data_source` is set):

```python
ds = bm.create_dataset("I.16.6")
ds.refresh()   # re-materialises from data_source; reloads self.X / self.y
```

---

## Sharing a whole object

[share.save][SRToolkit.share.save] writes a dataset, benchmark or experiment grid into one
`.zip` with everything the recipient needs: the config, the data, and the custom code it uses.
[share.load][SRToolkit.share.load] reads it back, from a path or a URL, and returns the object.

```python
from SRToolkit import share

share.save(bm, "nguyen.zip")
bm = share.load("nguyen.zip")                          # or share.load("https://example.org/nguyen.zip")
dataset = bm.create_dataset("NG-1")                    # the data comes from the archive
```

**Data.** By default (`data="all"`) every dataset's data is shipped, so the recipient needs no
network and no samplers. `data="auto"` leaves data the recipient can reproduce (a `UrlSource`, a
`SampleSource` with a seed, or a `FallbackSource` with at least one such source) to be
downloaded or regenerated, and ships the rest; `data="none"` ships the config and code only.
Loaded configs keep their `data_source`, so `refresh()` still works.

**Code.** Any custom class the object uses (an approach, sampler, data source, constraint,
callback, ...) must come from a [bundle](#code-srtk-bundles) to load elsewhere. `save` re-packs
the installed bundles that define such classes; for code that isn't installed, build a bundle
with [pack][SRToolkit.bundle.pack] and pass it:

```python
from SRToolkit.bundle import pack

pack(files=["approach.py", "ops.py"], out_path="my_ap.srtk", name="my-ap", version="1.0.0")
share.save(grid, "my_run.zip", bundles=["my_ap.srtk"], results=True)   # results=True: grids only
```

A custom class covered by no bundle gives a warning (an error with `strict=True`). Classes
defined in `__main__` (a script run directly, or a notebook cell) can't be shipped; move them
into a named `.py` module first (see
[Writing shareable custom classes](#writing-shareable-custom-classes)).

**Loading code.** `share.load` never runs shipped code on its own: it warns about bundles that
aren't installed yet, and `install_bundles=True` installs them (with
[install][SRToolkit.bundle.install]'s confirmation prompt). Classes bind to the installed bundle
automatically.

```python
grid = share.load("my_run.zip", install_bundles=True, results_dir="my_results")
```

For a grid, `results_dir` defaults to the archive name minus `.zip`; shipped results are copied
there without overwriting existing files. The archive layout is described in the
[share][SRToolkit.share] reference.

---

## Sharing a complete experiment

An [`ExperimentGrid`][SRToolkit.experiments.ExperimentGrid] is a *composite* config: it
references all three axes at once — approach and callback configs, dataset configs (each with
its `data_source`), and the run metadata. It serialises three ways:

- **The recipe:** [`to_dict()`][SRToolkit.experiments.ExperimentGrid.to_dict] returns one
  JSON-safe dict with every dataset, approach and callback config inlined, and no data arrays
  or machine-local state (`results_dir`, `adapted_states` paths).
  [`from_dict()`][SRToolkit.experiments.ExperimentGrid.from_dict] rebuilds the grid; the
  recipient must be able to reach the data and have any custom bundles installed.

  ```python
  config = grid.to_dict()                       # plain JSON — email it, commit it
  grid = ExperimentGrid.from_dict(config, results_dir="my_results")
  ```

- **Local persistence:** [`checkpoint()`][SRToolkit.experiments.ExperimentGrid.checkpoint]
  writes the recipe plus your local `adapted_states` paths to `results_dir/grid.json`, and
  `from_dict("…/grid.json")` resumes it, with `results_dir` set to wherever the file now lives.
  This is for your own run, not for handing to someone else.

- **Everything:** [share.save](#sharing-a-whole-object) ships the grid with its data, custom
  code and, with `results=True`, its finished runs.

See the [Experiments guide](experiments.md#saving-loading-and-sharing-a-grid) for the full
runner workflow.
