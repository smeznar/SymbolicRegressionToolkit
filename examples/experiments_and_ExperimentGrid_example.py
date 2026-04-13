from SRToolkit.approaches import EDHiE, ProGED
from SRToolkit.dataset import Feynman
from SRToolkit.experiments import ExperimentGrid

# Load the Feynman benchmark and pick two 2-variable datasets to run on.
bm = Feynman()
ds_names = bm.list_datasets(num_variables=2, verbose=False)
dataset1 = bm.create_dataset(ds_names[0])
dataset1.max_evaluations = 5000  # We lower max_evaluations to make the example faster
dataset2 = bm.create_dataset(ds_names[1])
dataset2.max_evaluations = 5000  # We lower max_evaluations to make the example faster

# Define the SR approaches to benchmark.
# EDHiE requires a pre-trained/adapted model state; ProGED needs no time consuming adaptation.
ae = EDHiE()
ap = ProGED()

# Map each (approach, dataset) pair to a file where the adapted model state will
# be saved. Both datasets reuse the same file here because they share the same
# number of variables, so one adapted state covers both.
adapted_states = {ae.name: {ds_names[0]: "adapted_state_2_vars.pt", ds_names[1]: "adapted_state_2_vars.pt"}}

# Build the experiment grid: every combination of approach × dataset will be run
# num_experiments times (with different random seeds). Results are written under
# results_dir, specifically to "results_dir/{dataset}/{approach}/exp_{seed}.json".
eg = ExperimentGrid(
    approaches=[ap, ae],
    datasets=[dataset1, dataset2],
    num_experiments=2,
    results_dir="../results/",
    adapted_states=adapted_states,
)

# Write a shell script of CLI commands that can be executed in parallel, e.g.:
#   cat commands.sh | parallel -j 4
eg.save_commands("commands.sh")

# Run adaptation for any approach/dataset pair whose adapted state file is missing.
# This is a no-op if all state files already exist.
eg.adapt_if_missing()

# Collect all pending jobs (skip any whose result file already exists on disk).
jobs = eg.create_jobs(skip_completed=True)

# Print a dataset × approach progress table showing how many experiments have
# completed so far.
eg.progress()

# Run each job sequentially in this process. To parallelize, use the generated
# commands.sh instead.
for job in jobs:
    job.run()

# Print the final progress table once all jobs are done.
eg.progress()
