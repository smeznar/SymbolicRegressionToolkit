import numpy as np

from SRToolkit.approaches import ApproachConfig, ProGED, SR_approach
from SRToolkit.dataset import Feynman
from SRToolkit.experiments import ExperimentGrid
from SRToolkit.utils import generate_n_expressions


# ------------------------------------- Defining a custom approach -------------------------------------------
# A new SR approach is created by subclassing SR_approach and implementing the search method. The search method
# receives an SR_evaluator and submits candidate expressions to it via evaluate_expr until the evaluator
# signals that the budget is exhausted or the success threshold has been reached (evaluator.should_stop).
# Here we implement the simplest possible approach: random search, which samples expressions from a
# probabilistic grammar automatically derived from the dataset's symbol library.
class RandomSearch(SR_approach):
    def __init__(self):
        # Hyperparameters are encapsulated in an ApproachConfig (or a subclass of it). RandomSearch has no
        # hyperparameters, so the base config with just a name is enough. The name is used to identify the
        # approach in result files and progress tables.
        super().__init__(ApproachConfig(name="RandomSearch"))

    def prepare(self):
        # Called once before search to set up any internal state (e.g. loading models). RandomSearch is
        # stateless, so there is nothing to do here.
        pass

    def search(self, sr_evaluator, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Generate candidate expressions in batches and submit them to the evaluator. The evaluator fits
        # free constants, computes the error, caches duplicates, and flips should_stop when the evaluation
        # budget is exhausted or an expression reaches the success threshold.
        while not sr_evaluator.should_stop:
            expressions = generate_n_expressions(sr_evaluator.symbol_library, 100, verbose=False)
            for expression in expressions:
                sr_evaluator.evaluate_expr(expression)

    @classmethod
    def from_config(cls, config):
        # Reconstructs the approach from a serialized config. ExperimentGrid uses this when running jobs
        # (and for CLI/HPC execution via run_job), so every approach used in a grid must implement it.
        # RandomSearch has no hyperparameters, so we can ignore the config's contents.
        return cls()


if __name__ == "__main__":
    # ------------------------------------ Running it on a benchmark ---------------------------------------
    # The ExperimentGrid creates one job per (dataset, approach, seed) combination. Passing Feynman() runs
    # the approach on all 100 datasets of the Feynman benchmark with 10 independent seeds each. Dataset
    # files are downloaded and cached automatically on first use.
    # Note: the full grid (100 datasets x 10 seeds) takes a long time to run sequentially. To try things
    # out, pass a few datasets instead, e.g. datasets=[Feynman().create_dataset("I.6.2a")].
    # Passing multiple approaches benchmarks them under identical conditions: same data, same
    # evaluation budget, same parameter estimation, same seeds. For approaches that require an
    # adaptation step before search (e.g. EDHiE), see experiments_and_ExperimentGrid_example.py.
    dataset = Feynman().create_dataset("I.6.2a")
    dataset.max_evaluations = 10000

    grid = ExperimentGrid(
        datasets=dataset,
        approaches=[RandomSearch(), ProGED()],
        num_experiments=3,
        results_dir="results/",
    )

    # Option 1: run all jobs sequentially in this process. Jobs that already have results are skipped,
    # so the run can be interrupted and resumed.
    all_jobs = grid.create_jobs()
    for i, job in enumerate(all_jobs):
        print(f"Running experiment {i}/{len(all_jobs)} - {job.approach_name}: {job.dataset_name}")
        job.run()

    # Option 2: instead of running jobs in-process, generate one CLI command per job for parallel
    # execution, e.g. with GNU Parallel (cat commands.sh | parallel -j 4) or a SLURM array.
    # grid.save_commands("commands.sh")

    # Show a summary table of completed experiments and inspect the results of one (dataset, approach)
    # pair. load_results merges the per-seed result files into a single SR_results object.
    grid.progress()
    results = grid.load_results("I.6.2a", "RandomSearch")
    results.print_results()
    results = grid.load_results("I.6.2a", "ProGED")
    results.print_results()
