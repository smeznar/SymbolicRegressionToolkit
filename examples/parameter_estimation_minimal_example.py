import numpy as np

from SRToolkit.evaluation import ParameterEstimator


if __name__ == "__main__":
    # A simple example of parameter estimation
    X = np.array(
        [
            [1, 2],
            [8, 4],
            [5, 4],
            [7, 9],
        ]
    )
    y = np.array([6, 6, 9, 24.5])

    # Create an instance of the ParameterEstimator class with the input data and target values
    pe = ParameterEstimator(X, y)

    # Estimate the parameters of the expression
    print(pe.estimate_parameters(["C", "*", "B", "-", "A"]))
