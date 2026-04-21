"""
Nguyen symbolic regression benchmark.
"""

import os
import warnings
from typing import Optional

from platformdirs import user_data_dir

from SRToolkit.utils.symbol_library import SymbolLibrary

from .sampling import UniformSampling
from .sr_benchmark import SR_benchmark, download_benchmark_data


class Nguyen(SR_benchmark):
    """
    The Nguyen symbolic regression benchmark.

    Contains 10 expressions without constant parameters (first 4 are polynomials, first 8 use
    one variable, last 2 use two variables). The benchmark ships with pre-generated data. If the
    download fails, data is generated from the stored per-variable samplers using ``n_samples``
    points and the given ``seed``.

    References:
        [Uy et al. (2011)][cite-nguyen]

    Examples:
        >>> benchmark = Nguyen()
        >>> len(benchmark.list_datasets(verbose=False))
        10

    Args:
        dataset_directory: Directory where dataset files are stored or will be downloaded to.
            Defaults to the platform-appropriate user data directory (e.g. ``~/.local/share/SRToolkit/nguyen`` on Linux).
        n_samples: Number of samples to generate per dataset when falling back to sampler-based
            data generation (i.e. when the download fails or ``force_generate=True``). Defaults to ``1000``.
        seed: Random seed used for sampler-based data generation. Defaults to ``42``.
        force_generate: If ``True``, skip downloading/loading pre-generated data and always
            generate fresh data from samplers. Defaults to ``False``.
    """

    def __init__(
        self,
        dataset_directory: str = os.path.join(user_data_dir("SRToolkit"), "nguyen"),
        n_samples: int = 10000,
        seed: Optional[int] = 42,
        force_generate: bool = False,
    ):
        super().__init__("Nguyen", dataset_directory)
        self._n_samples = n_samples
        self._seed = seed
        self._force_generate = force_generate
        self._populate()

    def _populate(self):
        # fmt: off
        seed = self._seed
        url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/nguyen.zip"
        if not self._force_generate:
            try:
                download_benchmark_data(url, self.base_dir)
            except Exception as e:
                warnings.warn(
                    f"[Nguyen] Could not download benchmark data ({e}). "
                    "Data will be generated from samplers on first access."
                )
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
            samplers=[UniformSampling(0, 20)],
            n_samples=self._n_samples,
            force_generate=self._force_generate,
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
            samplers=[UniformSampling(0, 20)],
            n_samples=self._n_samples,
            force_generate=self._force_generate,
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
            samplers=[UniformSampling(0, 20)],
            n_samples=self._n_samples,
            force_generate=self._force_generate,
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
            samplers=[UniformSampling(0, 20)],
            n_samples=self._n_samples,
            force_generate=self._force_generate,
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
            samplers=[UniformSampling(0, 20)],
            n_samples=self._n_samples,
            force_generate=self._force_generate,
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
            samplers=[UniformSampling(0, 20)],
            n_samples=self._n_samples,
            force_generate=self._force_generate,
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
            samplers=[UniformSampling(1, 100, uses_negative=False)],
            n_samples=self._n_samples,
            force_generate=self._force_generate,
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
            samplers=[UniformSampling(0, 100, uses_negative=False)],
            n_samples=self._n_samples,
            force_generate=self._force_generate,
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
            samplers=[UniformSampling(0, 20), UniformSampling(0, 20)],
            n_samples=self._n_samples,
            force_generate=self._force_generate,
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
            samplers=[UniformSampling(0, 20), UniformSampling(0, 20)],
            n_samples=self._n_samples,
            force_generate=self._force_generate,
        )

    # fmt: on
