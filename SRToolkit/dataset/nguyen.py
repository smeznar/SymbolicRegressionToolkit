"""
Nguyen symbolic regression benchmark.
"""

import os
from typing import Optional, Tuple

import numpy as np
from platformdirs import user_data_dir

from SRToolkit.utils.expression_compiler import expr_to_executable_function
from SRToolkit.utils.symbol_library import SymbolLibrary

from .sr_benchmark import SR_benchmark, download_benchmark_data

_BOUNDS = {
    "NG-1": [(-20, 20)],
    "NG-2": [(-20, 20)],
    "NG-3": [(-20, 20)],
    "NG-4": [(-20, 20)],
    "NG-5": [(-20, 20)],
    "NG-6": [(-20, 20)],
    "NG-7": [(1, 100)],
    "NG-8": [(0, 100)],
    "NG-9": [(-20, 20), (-20, 20)],
    "NG-10": [(-20, 20), (-20, 20)],
}


class Nguyen(SR_benchmark):
    """
    The Nguyen symbolic regression benchmark.

    Contains 10 expressions without constant parameters (first 4 are polynomials, first 8 use
    one variable, last 2 use two variables). The benchmark ships with pre-generated data.

    For more information about the Nguyen benchmark, see:
    <https://doi.org/10.1007/s10710-010-9121-2>

    Examples:
        >>> benchmark = Nguyen()
        >>> len(benchmark.list_datasets(verbose=False))
        10

    Args:
        dataset_directory: Directory where dataset files are stored or will be downloaded to.
            Defaults to the platform-appropriate user data directory (e.g. ``~/.local/share/SRToolkit/nguyen`` on Linux).
    """

    def __init__(self, dataset_directory: str = os.path.join(user_data_dir("SRToolkit"), "nguyen")):
        super().__init__("Nguyen", dataset_directory)
        self._populate()

    def _populate(self):
        # fmt: off
        seed = None
        url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/nguyen.zip"
        download_benchmark_data(url, self.base_dir)
        # we create a SymbolLibrary with 1 and with 2 variables
        # Each library contains +, -, *, /, sin, cos, exp, log, sqrt, ^2, ^3
        sl_1v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "^2", "^3"], 1)
        sl_2v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "^2", "^3"], 2)

        self.metadata = {
            "description": "Symbolic regression benchmark with 10 expressions that don't contain constant "
            "parameters. First 4 are polynomials of different degrees. First eight expressions "
            "contain 1 variable, last two expressions contain two variables. This benchmark "
            "doesn't contain the original data, only expressions",
            "citation": """@article{Uy2011,
    author={Uy, Nguyen Quang and Hoai, Nguyen Xuan and O'Neill, Michael and McKay, R. I. and Galv{\'a}n-L{\'o}pez, Edgar},
    title={Semantically-based crossover in genetic programming: application to real-valued symbolic regression},
    journal={Genetic Programming and Evolvable Machines},
    year={2011},
    month={Jun},
    day={01},
    volume={12},
    number={2},
    pages={91-119},
}""",
        }

        # Add datasets to the benchmark
        self.add_dataset(
            "",
            sl_1v,
            dataset_name="NG-1",
            ranking_function="rmse",
            ground_truth=["X_0", "+", "X_0", "^2", "+", "X_0", "^3"],
            original_equation="y = x+x^2+x^3",
            max_evaluations=100000,
            max_expr_length=50,
            success_threshold=1e-7,
            dataset_metadata=self.metadata,
            seed=seed,
        )

        self.add_dataset(
            "",
            sl_1v,
            dataset_name="NG-2",
            ranking_function="rmse",
            ground_truth=["X_0", "+", "X_0", "^2", "+", "X_0", "^3", "+", "X_0", "*", "X_0", "^3"],
            original_equation="y = x+x^2+x^3+x^4",
            max_evaluations=100000,
            max_expr_length=50,
            success_threshold=1e-7,
            dataset_metadata=self.metadata,
            seed=seed,
        )

        self.add_dataset(
            "",
            sl_1v,
            dataset_name="NG-3",
            ranking_function="rmse",
            ground_truth=[
                "X_0",
                "+",
                "X_0",
                "^2",
                "+",
                "X_0",
                "^3",
                "+",
                "X_0",
                "*",
                "X_0",
                "^3",
                "+",
                "X_0",
                "^2",
                "*",
                "X_0",
                "^3",
            ],
            original_equation="y = x+x^2+x^3+x^4+x^5",
            max_evaluations=100000,
            max_expr_length=50,
            success_threshold=1e-7,
            dataset_metadata=self.metadata,
            seed=seed,
        )

        self.add_dataset(
            "",
            sl_1v,
            dataset_name="NG-4",
            ranking_function="rmse",
            ground_truth=[
                "X_0",
                "+",
                "X_0",
                "^2",
                "+",
                "X_0",
                "^3",
                "+",
                "X_0",
                "*",
                "X_0",
                "^3",
                "+",
                "X_0",
                "^2",
                "*",
                "X_0",
                "^3",
                "+",
                "X_0",
                "^3",
                "*",
                "X_0",
                "^3",
            ],
            original_equation="y = x+x^2+x^3+x^4+x^5+x^6",
            max_evaluations=100000,
            max_expr_length=50,
            success_threshold=1e-7,
            dataset_metadata=self.metadata,
            seed=seed,
        )

        self.add_dataset(
            "",
            sl_1v,
            dataset_name="NG-5",
            ranking_function="rmse",
            ground_truth=["sin", "(", "X_0", "^2", ")", "*", "cos", "(", "X_0", ")", "-", "1"],
            original_equation="y = sin(x^2)*cos(x)-1",
            max_evaluations=100000,
            max_expr_length=50,
            success_threshold=1e-7,
            dataset_metadata=self.metadata,
            seed=seed,
        )

        self.add_dataset(
            "",
            sl_1v,
            dataset_name="NG-6",
            ranking_function="rmse",
            ground_truth=["sin", "(", "X_0", ")", "+", "sin", "(", "X_0", "+", "X_0", "^2", ")"],
            original_equation="y = sin(x)+sin(x+x^2)",
            max_evaluations=100000,
            max_expr_length=50,
            success_threshold=1e-7,
            dataset_metadata=self.metadata,
            seed=seed,
        )

        self.add_dataset(
            "",
            sl_1v,
            dataset_name="NG-7",
            ranking_function="rmse",
            ground_truth=["log", "(", "1", "+", "X_0", ")", "+", "log", "(", "1", "+", "X_0", "^2", ")"],
            original_equation="y = log(1+x)+log(1+x^2)",
            max_evaluations=100000,
            max_expr_length=50,
            success_threshold=1e-7,
            dataset_metadata=self.metadata,
            seed=seed,
        )

        self.add_dataset(
            "",
            sl_1v,
            dataset_name="NG-8",
            ranking_function="rmse",
            ground_truth=["sqrt", "(", "X_0", ")"],
            original_equation="y = sqrt(x)",
            max_evaluations=100000,
            max_expr_length=50,
            success_threshold=1e-7,
            dataset_metadata=self.metadata,
            seed=seed,
        )

        self.add_dataset(
            "",
            sl_2v,
            dataset_name="NG-9",
            ranking_function="rmse",
            ground_truth=["sin", "(", "X_0", ")", "+", "sin", "(", "X_1", "^2", ")"],
            original_equation="y = sin(x)+sin(y^2)",
            max_evaluations=100000,
            max_expr_length=50,
            success_threshold=1e-7,
            dataset_metadata=self.metadata,
            seed=seed,
        )

        self.add_dataset(
            "",
            sl_2v,
            dataset_name="NG-10",
            ranking_function="rmse",
            ground_truth=["2", "*", "sin", "(", "X_0", ")", "*", "cos", "(", "X_1", ")"],
            original_equation="y = 2*sin(x)*cos(y)",
            max_evaluations=100000,
            max_expr_length=50,
            success_threshold=1e-7,
            dataset_metadata=self.metadata,
            seed=seed,
        )

    # fmt: on

    def resample(self, dataset_name: str, n: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fresh data for a dataset by sampling new inputs and evaluating the ground truth.

        Variable bounds are taken from ``_BOUNDS``.

        Examples:
            >>> benchmark = Nguyen('data/nguyen/')
            >>> X, y = benchmark.resample('NG-1', n=200, seed=42)
            >>> X.shape
            (200, 1)

        Args:
            dataset_name: Name of the dataset to resample.
            n: Number of new samples to generate.
            seed: Random seed for reproducibility.

        Returns:
            A tuple ``(X, y)`` of numpy arrays with shapes ``(n, n_vars)`` and ``(n,)``.

        Raises:
            ValueError: If the dataset has no ground truth expression.
        """
        info = self.datasets[dataset_name]
        if info.get("ground_truth") is None:
            raise ValueError(f"Dataset '{dataset_name}' has no ground truth expression — cannot compute y.")
        bounds = _BOUNDS[dataset_name]
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(seed)
        X_new = rng.uniform(lb, ub, size=(n, len(bounds)))
        sl = SymbolLibrary.from_dict(info["symbol_library"])
        f = expr_to_executable_function(info["ground_truth"], sl)
        y_new = f(X_new, np.array([]))
        return X_new, y_new
