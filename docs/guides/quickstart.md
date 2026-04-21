---
title: Quickstart
---

# Quickstart

## Installation

Install the latest stable release:

```bash
pip install symbolic-regression-toolkit
```

Or install directly from the repository (recommended for the latest features):

```bash
pip install git+https://github.com/smeznar/SymbolicRegressionToolkit
```

For approaches that require PyTorch and pymoo (EDHiE):

```bash
pip install 'symbolic-regression-toolkit[approaches]'
```

## Your first benchmark run

The snippet below loads two datasets from the Feynman benchmark, runs both ProGED and EDHiE on them, and prints progress.

```python
import numpy as np
from SRToolkit.dataset import Feynman
from SRToolkit.approaches import EDHiE, ProGED
from SRToolkit.experiments import ExperimentGrid

# Load two 2-variable datasets from the Feynman benchmark
bm = Feynman()
ds_names = bm.list_datasets(num_variables=2, verbose=False)
datasets = [bm.create_dataset(ds_names[0]), bm.create_dataset(ds_names[1])]

# EDHiE trains a VAE once per symbol space; reuse the same weights for both datasets
# since they share the same number of variables
adapted_states = {
    "EDHiE": {ds_names[0]: "state_2v.pt", ds_names[1]: "state_2v.pt"}
}

grid = ExperimentGrid(
    datasets=datasets,
    approaches=[ProGED(), EDHiE()],
    num_experiments=3,
    results_dir="results/quickstart",
    adapted_states=adapted_states,
)

# Adapt any models whose state files are missing
grid.adapt_if_missing()

# Run all experiments sequentially
for job in grid.create_jobs():
    job.run()

# Print a completion table
grid.progress()
```

For larger runs, replace the `for` loop with `grid.save_commands("commands.sh")` and execute the file in parallel — see the [Running Experiments](experiments.md) guide.
