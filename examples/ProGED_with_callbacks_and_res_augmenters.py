from SRToolkit.approaches import ProGED
from SRToolkit.dataset import Feynman
from SRToolkit.evaluation import RMSE, EarlyStoppingCallback, LoggingCallback, ProgressBarCallback, SR_evaluator

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
    X, y = test_dataset.X, test_dataset.y
    # Define an augmenter that computes the RMSE on the test set (only on the top 20 expressions)
    rmse_augmenter = RMSE(SR_evaluator(X, y))
    # Augment top expressions with test RMSE
    results.augment(rmse_augmenter)

    # Print the results with detailed information (Top expressions and their augmentations)
    results.print_results(detailed=True)
