import numpy as np

from PESR.parameter_estimator import ParameterEstimator


if __name__ == "__main__":
    # A simple example of parameter estimation
    X = np.array([[1, 2],
                  [8, 4],
                  [5, 4],
                  [7, 9],])
    y = np.array([6, 6, 9, 24.5])
    pe = ParameterEstimator(X, y)
    print(pe.estimate_parameters(["C", "*", "B", "-", "A"]))