---
title: Running Experiments
---

# Running Experiments

[ExperimentGrid][SRToolkit.experiments.ExperimentGrid] manages the cross-product of datasets × approaches × seeds. Jobs can run locally or be dispatched to an HPC cluster, and results are automatically resumed if a run is interrupted.

## Setting up an ExperimentGrid

```python
from SRToolkit.dataset import Feynman
from SRToolkit.approaches import EDHiE, ProGED
from SRToolkit.experiments import ExperimentGrid

bm = Feynman()
datasets = [bm.create_dataset(n) for n in bm.list_datasets(num_variables=2, verbose=False)[:4]]

# EDHiE adapts once per symbol space; map each dataset to a shared weights file
adapted_states = {
    "EDHiE": {ds.dataset_name: "states/edhie_2v.pt" for ds in datasets}
}

grid = ExperimentGrid(
    datasets=datasets,
    approaches=[ProGED(), EDHiE()],
    num_experiments=5,
    results_dir="results/feynman_run",
    initial_seed=0,
    adapted_states=adapted_states,
)
```

Results land at `results_dir/{dataset}/{approach}/exp_{seed}.json`. Re-running never overwrites completed jobs.

## Adapting models

For approaches with `adaptation_scope="once"` (like EDHiE), the state must be adapted once per (approach, dataset) pair before the experiments run. For a local run, call `adapt_if_missing()` — it adapts every pair whose state file is absent and saves it to the path given in `adapted_states`:

```python
grid.adapt_if_missing()
```

When several datasets share one `state_path` (as in the EDHiE example above, where adaptation is per symbol space rather than per dataset), that path is adapted **once** — on the first dataset listed for it — and the remaining datasets reuse the result. An informational warning lists the shared paths so an accidental collision is easy to spot; if adaptation should differ per dataset, give each dataset a distinct `state_path`.

For a cluster run you don't need this: `save_commands` emits one parallel-safe adapt command per pair into a separate prepare file (see below).

## Running jobs locally

```python
for job in grid.create_jobs():   # skips already-completed jobs by default
    job.run()
```

## Generating commands for parallel execution

For larger experiments, write a commands file and run it with GNU Parallel or a SLURM array:

```python
grid.save_commands("results/feynman_run/commands.sh")
```

`save_commands` writes the grid to `results_dir/grid.json` and **materialises every
dataset's data into the cache once, in this process** — so data is fetched a single time
on the machine writing the commands (typically the only node with network access on a
cluster), and the parallel workers only read it. It then emits one self-contained CLI call
per job, each referencing that single file plus the identifiers needed to rebuild the job:

```bash
python -m SRToolkit.experiments run_job \
    --grid results/feynman_run/grid.json \
    --dataset I.12.1 --approach ProGED --seed 0
```

If any `adaptation_scope="once"` approach still needs state, a sibling `prepare_commands.sh`
is written too, with one independent adapt command per (approach, dataset) pair. **Run the
prepare file to completion first, then the experiments file** — a platform-agnostic two
phases, no scheduler-specific dependencies needed:

```bash
cat results/feynman_run/prepare_commands.sh | parallel -j 8   # only if it exists
cat results/feynman_run/commands.sh        | parallel -j 8
```

Or submit each phase to SLURM (the second array depending on the first):

```bash
sbatch --array=1-$(wc -l < commands.sh) run_array.sh
```

## Tracking progress

```python
grid.progress()
```

Prints a dataset × approach table showing `done/total` completed experiments:

```
Dataset        ProGED    EDHiE
-----------   -------   ------
I.12.1           5/5      3/5
I.12.2           2/5      0/5
```

On a cluster you can print the same table from the shell without opening Python:

```bash
python -m SRToolkit.experiments progress --grid results/feynman_run/grid.json
```

## Loading results

```python
results = grid.load_results("I.12.1", "ProGED")

for r in results:
    print(r.best_expr, r.min_error, r.success)
```

## Saving, loading, and sharing a grid

A grid serialises three different ways, each for a different audience:

| Goal | Call | Produces |
|---|---|---|
| Resume your own run in place | [save][SRToolkit.experiments.ExperimentGrid.save] / [load][SRToolkit.experiments.ExperimentGrid.load] | one `results_dir/grid.json` (recipe + local `adapted_states`) |
| Hand someone the recipe | [to_dict][SRToolkit.experiments.ExperimentGrid.to_dict] / [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict] | one JSON dict, no local paths |
| Hand someone everything | [export][SRToolkit.experiments.ExperimentGrid.export] / [from_export][SRToolkit.experiments.ExperimentGrid.from_export] | one `.zip`: recipe + bundled code + unreachable data + `MANIFEST.md` |

`save()` writes a single file and `load()` reads it back, re-anchoring `results_dir` to
wherever the file currently lives — so a run directory can be moved or mounted elsewhere and
still resumes correctly:

```python
grid.save()                                   # → results/feynman_run/grid.json
grid = ExperimentGrid.load("results/feynman_run/grid.json")
```

To share the *setup* without your machine-local paths or results, pass the
[to_dict][SRToolkit.experiments.ExperimentGrid.to_dict] recipe (a plain dict). The recipient
needs the same datasets reachable (downloaded, regenerated from a seed, or already cached) and
any custom `.srtk` bundles installed.

To share *everything* in one file, use
[export][SRToolkit.experiments.ExperimentGrid.export], which writes a single `.zip`:

```python
grid.export("share/feynman_run.zip")          # recipe + data/ + bundles/ + MANIFEST.md
```

It archives only the data the recipient can't reach on their own (`null`-source and
seedless-`sample` datasets), best-effort packs custom classes into `bundles/`, and writes a
`MANIFEST.md` with the exact install/load steps. The recipient loads it in one call —
`from_export` imports the bundled data into the cache and extracts any `bundles/` next to the
archive (defaulting `results_dir` to the archive name minus `.zip`):

```python
grid = ExperimentGrid.from_export("share/feynman_run.zip", results_dir="my_results")
```

If the export carried custom code, install those bundles from the extracted
`<results_dir>/bundles/` before running jobs (class binding is deferred to run time, so the
order of `from_export` and `install` doesn't matter — only that install happens first).

See [Sharing Custom Implementations](sharing.md#sharing-a-complete-experiment) for how this
fits the code / config / data sharing model.
