from SRToolkit.approaches import ProGED
from SRToolkit.dataset import Feynman
from SRToolkit.evaluation import EarlyStoppingCallback, HoldoutMetric, LoggingCallback, ProgressBarCallback

if __name__ == "__main__":
    # Create the feynman benchmark
    benchmark = Feynman()
    # Create an instance of the I.16.6 dataset
    dataset = benchmark.create_dataset("I.16.6")
    # Create an instance of ProGED
    model = ProGED()

    # Define callbacks
    es = EarlyStoppingCallback(1e-7)
    lc = LoggingCallback(log_file="log.log")
    pb = ProgressBarCallback()

    # Evaluate the approach on the dataset
    results = dataset.evaluate_approach(model, num_experiments=1, initial_seed=1, callbacks=[es, lc, pb])

    # Draw a fresh I.16.6 dataset to use as a test set (does not touch the cached canonical data)
    test_dataset = benchmark.create_dataset("I.16.6", n_samples=1000, seed=2)
    # Define an augmenter that computes the test RMSE by reusing each model's train-fitted
    # parameters (no refitting) on the held-out data — only on the top expressions.
    rmse_augmenter = HoldoutMetric(
        (test_dataset.X, test_dataset.y), metric="rmse", symbol_library=dataset.symbol_library
    )
    # Augment top expressions with test RMSE
    results.augment(rmse_augmenter)

    # Print the results with detailed information (Top expressions and their augmentations)
    results.print_results(detailed=True)
