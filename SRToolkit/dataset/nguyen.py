"""
Nguyen symbolic regression benchmark.
"""

import warnings
from typing import Optional

from SRToolkit.utils.symbol_library import SymbolLibrary

from .data_source import SampleSource, UrlSource
from .sampling import UniformSampling
from .sr_benchmark import SR_benchmark


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
        n_samples: Number of samples to generate per dataset when ``force_generate=True``
            (sampler-based data generation). Defaults to ``10000``.
        seed: Random seed used for sampler-based data generation. Defaults to ``42``.
        force_generate: If ``True``, generate fresh data from the stored samplers instead of
            downloading the pre-generated data. Defaults to ``False``.
    """

    __data_version__ = "1.0.0"

    def __init__(
        self,
        n_samples: int = 10000,
        seed: Optional[int] = 42,
        force_generate: bool = False,
    ):
        super().__init__("Nguyen", version="1.0.0")
        self._n_samples = n_samples
        self._seed = seed
        self._force_generate = force_generate
        self._populate()

    def _populate(self):
        # fmt: off
        seed = self._seed
        url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/nguyen.zip"
        # The canonical data is downloaded once from the archive (see _ensure_data) so every
        # machine benchmarks on identical inputs. Each dataset's own data_source is a
        # SampleSource: a transparent, per-dataset fallback that regenerates the data from
        # that dataset's samplers if the download is unavailable (or force_generate is set).
        self._archive_source = UrlSource(url)
        data_source = SampleSource(n_samples=self._n_samples, seed=seed)
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
        self.add_dataset(sl_1v, None, dataset_name="NG-1", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], original_equation="y = x+x^2+x^3",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(0, 20)], data_source=data_source)

        self.add_dataset(sl_1v, None, dataset_name="NG-2", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["X_0", "+", "X_0", "^2", "+", "X_0", "^3", "+", "X_0", "*", "X_0", "^3"],
                         original_equation="y = x+x^2+x^3+x^4", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(0, 20)], data_source=data_source)

        self.add_dataset(sl_1v, None, dataset_name="NG-3", ranking_function="rmse", max_evaluations=100000,
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
                         ], original_equation="y = x+x^2+x^3+x^4+x^5", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(0, 20)], data_source=data_source)

        self.add_dataset(sl_1v, None, dataset_name="NG-4", ranking_function="rmse", max_evaluations=100000,
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
                         ], original_equation="y = x+x^2+x^3+x^4+x^5+x^6", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(0, 20)], data_source=data_source)

        self.add_dataset(sl_1v, None, dataset_name="NG-5", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sin", "(", "X_0", "^2", ")", "*", "cos", "(", "X_0", ")", "-", "1"],
                         original_equation="y = sin(x^2)*cos(x)-1", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(0, 20)], data_source=data_source)

        self.add_dataset(sl_1v, None, dataset_name="NG-6", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sin", "(", "X_0", ")", "+", "sin", "(", "X_0", "+", "X_0", "^2", ")"],
                         original_equation="y = sin(x)+sin(x+x^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(0, 20)], data_source=data_source)

        self.add_dataset(sl_1v, None, dataset_name="NG-7", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["log", "(", "1", "+", "X_0", ")", "+", "log", "(", "1", "+", "X_0", "^2", ")"],
                         original_equation="y = log(1+x)+log(1+x^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(1, 100, uses_negative=False)],
                         data_source=data_source)

        self.add_dataset(sl_1v, None, dataset_name="NG-8", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sqrt", "(", "X_0", ")"], original_equation="y = sqrt(x)",
                         success_threshold=1e-7, seed=seed, dataset_metadata=self.metadata,
                         samplers=[UniformSampling(0, 100, uses_negative=False)], data_source=data_source)

        self.add_dataset(sl_2v, None, dataset_name="NG-9", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["sin", "(", "X_0", ")", "+", "sin", "(", "X_1", "^2", ")"],
                         original_equation="y = sin(x)+sin(y^2)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(0, 20), UniformSampling(0, 20)],
                         data_source=data_source)

        self.add_dataset(sl_2v, None, dataset_name="NG-10", ranking_function="rmse", max_evaluations=100000,
                         ground_truth=["2", "*", "sin", "(", "X_0", ")", "*", "cos", "(", "X_1", ")"],
                         original_equation="y = 2*sin(x)*cos(y)", success_threshold=1e-7, seed=seed,
                         dataset_metadata=self.metadata, samplers=[UniformSampling(0, 20), UniformSampling(0, 20)],
                         data_source=data_source)

    # fmt: on

    def _ensure_data(self, dataset_name: str) -> None:
        """Download the canonical Nguyen archive into the cache once, unless ``force_generate``.

        On a cache miss the whole archive is fetched and every ``NG-*.npz`` extracted, so the
        next ``create_dataset`` reads the same data on any machine. If the download fails we
        warn and let each dataset's ``SampleSource`` regenerate the data locally.
        """
        if self._force_generate:
            return
        from SRToolkit.dataset import data_cache

        cache_path = data_cache.dataset_path(self.benchmark_name, self.version, dataset_name)
        if cache_path.exists():
            return
        try:
            self._archive_source.materialize(cache_path, self.datasets[dataset_name])
        except Exception as e:
            warnings.warn(
                f"[Nguyen] Could not download the canonical data ({e}); falling back to local "
                f"sampling. Generated data may differ across machines.",
                stacklevel=2,
            )
