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

For approaches with `adaptation_scope="once"` (like EDHiE), call `adapt_if_missing()` before creating jobs. It adapts each (approach, dataset) pair whose state file is absent and saves it to the path given in `adapted_states`:

```python
grid.adapt_if_missing()
```

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

Each line is a self-contained CLI call:

```bash
python -m SRToolkit.experiments run_job \
    --dataset results/feynman_run/_datasets/I.12.1/I.12.1.json \
    --approach results/feynman_run/_approaches/ProGED_config.json \
    --info    results/feynman_run/I.12.1/ProGED/exp_0/info_0.json
```

Run with GNU Parallel:

```bash
cat results/feynman_run/commands.sh | parallel -j 8
```

Or submit to SLURM:

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

## Loading results

```python
results = grid.load_results("I.12.1", "ProGED")

for r in results:
    print(r.best_expr, r.best_error, r.success)
```
