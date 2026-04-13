"""
Benchmark collection for symbolic regression datasets.
"""

import copy
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from platformdirs import user_data_dir
from typing_extensions import Unpack

from SRToolkit.utils.expression_compiler import expr_to_executable_function
from SRToolkit.utils.expression_tree import Node
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import EstimationSettings

from .sr_dataset import SR_dataset


class SR_benchmark:
    def __init__(
        self,
        benchmark_name: str,
        base_dir: str,
        datasets: Optional[List[Union[SR_dataset, Tuple[str, SR_dataset]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        A named, persistent collection of symbolic regression datasets.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of SR_benchmark with additional functionality
            >>> len(benchmark.list_datasets(verbose=False))
            100

        Args:
            benchmark_name: Name of this benchmark.
            base_dir: Directory where dataset files are stored or will be written.
            datasets: Initial datasets to add. Each element can be an
                [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] instance (auto-named as
                ``"<benchmark_name>_<index>"``) or a ``(name, SR_dataset)`` tuple.
            metadata: Optional dictionary of benchmark-level metadata (e.g. citation, description).

        Raises:
            ValueError: If any element of ``datasets`` is not an
                [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] or a valid ``(name, SR_dataset)`` tuple.
        """
        self.benchmark_name = benchmark_name
        self.base_dir = base_dir
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.metadata = {} if metadata is None else metadata
        if datasets is not None:
            for i, dataset in enumerate(datasets):
                if isinstance(dataset, SR_dataset):
                    self.add_dataset_instance(benchmark_name + "_" + str(i + 1), dataset)
                elif isinstance(dataset, tuple) and isinstance(dataset[0], str) and isinstance(dataset[1], SR_dataset):
                    self.add_dataset_instance(dataset[0], dataset[1])
                else:
                    raise ValueError(
                        "[SR_benchmark] Dataset inside the datasets argument must be either a tuple "
                        "(name, SR_dataset) or a SR_dataset instance."
                    )

    def add_dataset_instance(self, dataset_name: str, dataset: SR_dataset):
        """
        Adds an instance of the SR_dataset class to the benchmark.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of SR_benchmark with additional functionality
            >>> dataset = benchmark.create_dataset('I.16.6')
            >>> isinstance(dataset, SR_dataset)
            True
            >>> bm = SR_benchmark("BM", "data/bm")
            >>> bm.add_dataset_instance("I.16.6", dataset)

        Args:
             dataset_name: The name of the dataset.
             dataset: An instance of the SR_dataset class.

        Raises:
            Exception: If the dataset name already exists in the benchmark.
        """
        if dataset_name in self.datasets:
            raise ValueError(f"Dataset {dataset_name} already exists in the benchmark.")
        else:
            self.datasets[dataset_name] = {}
        self.datasets[dataset_name]["sr_dataset"] = dataset
        self.datasets[dataset_name]["num_variables"] = dataset.X.shape[1]

    def add_dataset(
        self,
        dataset: Union[str, np.ndarray, Tuple[np.ndarray, np.ndarray]],
        symbol_library: SymbolLibrary,
        dataset_name: Optional[str] = None,
        ranking_function: str = "rmse",
        max_evaluations: int = -1,
        ground_truth: Optional[Union[List[str], Node, np.ndarray]] = None,
        original_equation: Optional[str] = None,
        success_threshold: Optional[float] = None,
        seed: Optional[int] = None,
        dataset_metadata: Optional[dict] = None,
        **kwargs: Unpack[EstimationSettings],
    ):
        """
        Adds a dataset to the benchmark.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> fey_benchmark = Feynman() # Feynman is a specific instance of SR_benchmark with additional functionality
            >>> benchmark = SR_benchmark("BM", "data/bm")
            >>> benchmark.add_dataset(fey_benchmark.base_dir+"/I.14.3.npz", SymbolLibrary.default_symbols(3),
            ...       dataset_name="I.14.3", ranking_function="rmse", ground_truth = ["X_0", "*", "X_1", "*", "X_2"],
            ...       original_equation="U = m*g*z", max_evaluations=100000, max_expr_length=50,
            ...       success_threshold=1e-7, dataset_metadata={}, constant_bounds=(-5.0, 5.0),
            ...       seed = 42)
            >>> len(benchmark.list_datasets(verbose=False))
            1

        Args:
            dataset: Data used in the dataset. Can be:
                - A string representing the path to a NumPy archive (.npz) containing the dataset. It should either
                the absolute path to the data, path relative to the base_dir 'base_dir'/'dataset', or empty, in that
                case the dataset will be loaded from 'base_dir'/'dataset_name'.npz. The .npz file must contain the
                features (saved in 'X') and if 'rmse' is used as the ranking function, the target (saved in 'y').
                - A 2d numpy array containing the features (X). If 'rmse' is used as the ranking function, ground truth
                should also be provided to calculate the target (y). Once added, the data will be saved at
                'base_dir'/'dataset_name'.npz.
                - A tuple containing the features (X) and the target (y). If 'bed' is used as the ranking function,
                the target will be ignored. Once added, the data will be saved at 'base_dir'/'dataset_name'.npz.
            symbol_library: The symbol library to use.
            dataset_name: The name of the dataset. If None, a name will be generated automatically as
                'benchmark_name'_'index+1'.
            ranking_function: The ranking function used during evaluation. Can be: 'rmse', 'bed'.
            max_evaluations: The maximum number of expressions to evaluate. Less than 0 means no limit.
            ground_truth: Ground truth expression as a token list in infix notation, a
                [Node][SRToolkit.utils.expression_tree.Node] tree, or a numpy behavior array. Required when
                ``ranking_function="bed"``.
            original_equation: Human-readable string of the original equation.
            success_threshold: Error threshold below which an expression is considered successful. If ``None``,
                no threshold is applied.
            seed: Random seed for reproducibility. ``None`` means no seed is set.
            dataset_metadata: Optional dictionary of dataset-level metadata (merged with benchmark metadata).
            **kwargs: Optional estimation settings passed to
                [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator].
                Supported keys: ``method``, ``tol``, ``gtol``, ``max_iter``, ``constant_bounds``,
                ``initialization``, ``max_constants``, ``max_expr_length``, ``num_points_sampled``,
                ``bed_X``, ``num_consts_sampled``, ``domain_bounds``.

        Raises:
            ValueError: When BED ranking function is used but ground truth is not provided. When dataset is given as
                a string (directory) that doesn't exist, is not a valid .npz file, or is a .npz file that doesn't
                contain one array for the BED ranking function (X) or two array for the RMSE ranking function (X, y).
                When the argument dataset is an array, ranking function RMSE and there is no ground truth or the
                expression given as the ground truth cannot be evaluated...
        """
        if dataset_name is None:
            dataset_name = f"{self.benchmark_name}_{len(self.datasets) + 1}"

        self.datasets[dataset_name] = {}
        self.datasets[dataset_name]["format_version"] = 1
        self.datasets[dataset_name]["dataset_name"] = dataset_name
        self.datasets[dataset_name]["symbol_library"] = symbol_library.to_dict()
        self.datasets[dataset_name]["ranking_function"] = ranking_function
        self.datasets[dataset_name]["max_evaluations"] = max_evaluations

        self.datasets[dataset_name]["success_threshold"] = success_threshold

        self.datasets[dataset_name]["seed"] = seed
        merged_metadata = copy.deepcopy(self.metadata)
        if dataset_metadata:
            merged_metadata.update(dataset_metadata)
        self.datasets[dataset_name]["dataset_metadata"] = merged_metadata

        if "bed_X" in kwargs and kwargs["bed_X"] is not None:
            kwargs["bed_X"] = kwargs["bed_X"].tolist()

        self.datasets[dataset_name]["kwargs"] = kwargs
        self.datasets[dataset_name]["original_equation"] = original_equation
        self.datasets[dataset_name]["ground_truth"] = ground_truth

        if ground_truth is None:
            if ranking_function == "bed":
                raise ValueError("[SR_benchmark.add_dataset] For 'bed' ranking, the ground truth must be provided. ")
            else:
                warnings.warn(
                    "[SR_benchmark.add_dataset] 'ground_truth' argument not provided. We recommend providing it "
                    "for more transparent evaluation."
                )
        else:
            if original_equation is None:
                if isinstance(ground_truth, str):
                    self.datasets[dataset_name]["original_equation"] = "y = " + ground_truth
                elif isinstance(ground_truth, list):
                    self.datasets[dataset_name]["original_equation"] = "y = " + "".join(ground_truth)

        if isinstance(dataset, str):
            dataset_path = None
            if os.path.exists(dataset):
                dataset_path = dataset
            elif dataset != "" and os.path.exists(f"{self.base_dir}/{dataset}"):
                dataset_path = f"{self.base_dir}/{dataset}"
            elif os.path.exists(f"{self.base_dir}/{dataset_name}.npz"):
                dataset_path = f"{self.base_dir}/{dataset_name}.npz"

            if dataset_path is None:
                error_msg = (
                    f"[SR_benchmark.add_dataset] Could not find the dataset file. "
                    f"Expected locations:\n"
                    f"- Absolute path: '{dataset}'\n"
                    f"- Relative to base_dir: '{self.base_dir}/{dataset}'\n"
                    f"- NPZ with the name of the dataset in base_dir: '{self.base_dir}/{dataset_name}.npz'"
                )
                raise FileNotFoundError(error_msg)

            self.datasets[dataset_name]["dataset_path"] = dataset_path

            try:
                data = np.load(self.datasets[dataset_name]["dataset_path"], allow_pickle=False)
            except IOError as e:
                error_msg = (
                    f"[SR_benchmark.add_dataset] Could not load dataset from path '{self.datasets[dataset_name]}' "
                    f"using np.load. The file may be corrupt or not a valid NumPy archive (.npz). "
                    f"Original error: {e}"
                )
                raise IOError(error_msg) from e

            if ranking_function == "rmse":
                if not (isinstance(data, np.lib.npyio.NpzFile) and "X" in data and "y" in data):
                    error_msg = (
                        f"[SR_benchmark.add_dataset] For 'rmse' ranking, the dataset file "
                        f"('{self.datasets[dataset_name]['dataset_path']}') must be a .npz NumPy archive containing "
                        f"both 'X' (features) and 'y' (targets). It should be created via `np.savez(path, X=X, y=y)`."
                    )
                    raise ValueError(error_msg)

            elif ranking_function == "bed":
                if not (isinstance(data, np.lib.npyio.NpzFile) and "X" in data):
                    error_msg = (
                        f"[SR_benchmark.add_dataset] For 'bed' ranking, the dataset file "
                        f"('{self.datasets[dataset_name]['dataset_path']}') must be a .npz NumPy archive "
                        f"containing 'X' (features). It should be created via `np.savez(path, X=X)`."
                    )
                    raise ValueError(error_msg)

            num_variables = data["X"].shape[1]

        elif isinstance(dataset, np.ndarray):
            if ranking_function == "rmse" and ground_truth is not None:
                if isinstance(ground_truth, np.ndarray):
                    raise ValueError(
                        "[SR_benchmark.add_dataset] For 'rmse' ranking, the ground truth must be a string or a SRToolkit.utils.Node object. "
                    )
                try:
                    expr = expr_to_executable_function(ground_truth, symbol_library)
                    y = expr(dataset, None)
                except Exception as e:
                    raise Exception(
                        f"[SR_benchmark.add_dataset] Could not evaluate the ground truth. Original error: {e}"
                    )
                if not os.path.isdir(self.base_dir):
                    os.makedirs(self.base_dir)
                np.savez(f"{self.base_dir}/{dataset_name}.npz", X=dataset, y=y, allow_pickle=False)
            elif ranking_function == "rmse" and ground_truth is None:
                raise ValueError(
                    "[SR_benchmark.add_dataset] For 'rmse' ranking, if the dataset argument is a numpy "
                    "array, the ground truth must be provided in order for the target values to be "
                    "calculated."
                )
            elif ranking_function == "bed":
                if not os.path.isdir(self.base_dir):
                    os.makedirs(self.base_dir)
                np.savez(f"{self.base_dir}/{dataset_name}.npz", X=dataset, allow_pickle=False)

            self.datasets[dataset_name]["dataset_path"] = f"{self.base_dir}/{dataset_name}.npz"
            num_variables = dataset.shape[1]

        elif isinstance(dataset, tuple):
            if not isinstance(dataset[0], np.ndarray) or not isinstance(dataset[1], np.ndarray):
                raise ValueError(
                    "[SR_benchmark.add_dataset] When dataset argument is provided as a tuple, both "
                    "values must be a numpy array. The first array represents the features ('X'), "
                    "the second array represents the targets ('y')."
                )
            if ranking_function == "bed":
                warnings.warn(
                    "[SR_benchmark.add_dataset] 'bed' ranking only utilizes the array with features. "
                    "Array with targets will be ignored."
                )
            if not os.path.isdir(self.base_dir):
                os.makedirs(self.base_dir)
            np.savez(f"{self.base_dir}/{dataset_name}.npz", X=dataset[0], y=dataset[1], allow_pickle=False)
            self.datasets[dataset_name]["dataset_path"] = f"{self.base_dir}/{dataset_name}.npz"
            num_variables = dataset[0].shape[1]

        else:
            raise ValueError(
                "[SR_benchmark.add_dataset] The dataset argument must be a string, a numpy array, "
                "or a tuple containing two numpy arrays."
            )

        self.datasets[dataset_name]["num_variables"] = num_variables

    def create_dataset(self, dataset_name: str) -> SR_dataset:
        """
        Creates an instance of a dataset from the given dataset name.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of SR_benchmark with additional functionality
            >>> dataset = benchmark.create_dataset('I.16.6')
            >>> dataset.X.shape
            (10000, 3)

        Args:
            dataset_name: The name of the dataset to create.

        Returns:
            A SR_dataset instance containing the data, ground truth expression, and metadata for the given dataset.

        Raises:
            ValueError: If the dataset name is not found in the available datasets.
        """
        if dataset_name in self.datasets:
            if "sr_dataset" in self.datasets[dataset_name]:
                return self.datasets[dataset_name]["sr_dataset"]
            else:
                try:
                    return SR_dataset.from_dict(self.datasets[dataset_name])
                except Exception as e:
                    raise ValueError(
                        f"[SR_benchmark.create_dataset] Could not create SR_dataset from the given "
                        f"given dictionary. Original error: {e}"
                    )

        else:
            raise ValueError(f"Dataset {dataset_name} not found")

    def list_datasets(self, verbose: bool = True, num_variables: int = -1) -> List[str]:
        """
        Lists the available datasets.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of SR_benchmark with additional functionality
            >>> len(benchmark.list_datasets(num_variables=2, verbose=False))
            15
            >>> datasets_with_8_vars = benchmark.list_datasets(num_variables=8, verbose=False)
            >>> datasets_with_8_vars[0]
            'II.36.38'

        Args:
            verbose: If ``True``, also prints a description of each dataset.
            num_variables: If not ``-1``, only return datasets with this many input variables.

        Returns:
            A list of dataset names.
        """
        datasets = [
            dataset_name
            for dataset_name in self.datasets
            if num_variables < 0 or self.datasets[dataset_name]["num_variables"] == num_variables
        ]
        datasets = sorted(
            datasets,
            key=lambda dataset_name: (
                self.datasets[dataset_name]["num_variables"],
                dataset_name,
            ),
        )

        if verbose:
            part1 = []
            part2 = []
            part3 = []
            max_length_1 = 0
            max_length_2 = 0
            for d in datasets:
                if self.datasets[d]["num_variables"] == 1:
                    variable_str = "1 variable"
                elif self.datasets[d]["num_variables"] < 1:
                    variable_str = "Amount of variables unknown"
                else:
                    variable_str = f"{self.datasets[d]['num_variables']} variables"
                part1.append(d + ":")
                part2.append(variable_str)
                part3.append(self.datasets[d]["original_equation"])
                if len(d) + 1 > max_length_1:
                    max_length_1 = len(d) + 1
                if len(variable_str) > max_length_2:
                    max_length_2 = len(variable_str)

            for p1, p2, p3 in zip(part1, part2, part3):
                print(f"{p1:<{max_length_1}} {p2:<{max_length_2}}, Expression: {p3}")
        return datasets

    def save_benchmark(self):
        """
        Saves the benchmark to ``<base_dir>/dataset_info.json``.

        The JSON file stores dataset metadata and paths to data files; the data arrays themselves are
        not embedded. Use [load_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark.load_benchmark]
        to restore the benchmark.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of SR_benchmark with additional functionality
            >>> benchmark.save_benchmark()
        """
        datasets = []
        for dataset_name, dataset_info in self.datasets.items():
            if "sr_dataset" in dataset_info:
                datasets.append({"name": dataset_name, "info": dataset_info["sr_dataset"].to_dict(self.base_dir)})
            else:
                datasets.append({"name": dataset_name, "info": dataset_info})

        output = {"datasets": datasets, "metadata": self.metadata, "name": self.benchmark_name}

        with open(f"{self.base_dir}/dataset_info.json", "w") as f:
            json.dump(output, f)

    @staticmethod
    def load_benchmark(base_dir: str) -> "SR_benchmark":
        """
        Loads a benchmark stored at the base directory, returning an instance of SR_benchmark.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> b1 = Feynman("data/feynman") # Feynman is a specific instance of SR_benchmark with additional functionality
            >>> b1.save_benchmark()
            >>> b2 = SR_benchmark.load_benchmark('data/feynman')
            >>> len(b1.list_datasets(verbose=False))
            100
            >>> len(b2.list_datasets(verbose=False))
            100
            >>> dataset_name = b2.list_datasets(verbose=False)[0]
            >>> dataset = b2.create_dataset(dataset_name)
            >>> rmse = dataset.create_evaluator().evaluate_expr(dataset.ground_truth)
            >>> bool(rmse < dataset.success_threshold)
            True

        Args:
            base_dir: Directory containing the ``dataset_info.json`` file previously written by
                [save_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark.save_benchmark].

        Returns:
            An [SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark] instance with all datasets restored from the saved JSON.

        Raises:
            FileNotFoundError: If ``dataset_info.json`` does not exist in ``base_dir``.
            json.JSONDecodeError: If the JSON file is malformed.
        """
        with open(f"{base_dir}/dataset_info.json", "r") as f:
            data = json.load(f)

        datasets = {}
        for dataset_info in data["datasets"]:
            datasets[dataset_info["name"]] = dataset_info["info"]

        benchmark = SR_benchmark(data["name"], base_dir, metadata=data["metadata"])
        benchmark.datasets = datasets
        return benchmark


def download_benchmark_data(url: str, directory_path: str = user_data_dir("SRToolkit")) -> None:
    """
    Downloads and extracts a benchmark zip archive if the target directory is empty.

    Creates ``directory_path`` if it does not exist. If the directory is already non-empty,
    the download is skipped.

    Examples:
        >>> url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/feynman.zip"
        >>> dataset_directory = 'data/feynman'
        >>> download_benchmark_data(url, dataset_directory)

    Args:
        url: URL of the zip archive to download.
        directory_path: Local directory where the archive will be extracted. Defaults to the
            platform-appropriate user data directory (e.g. ``~/.local/share/SRToolkit`` on Linux).
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    if not os.listdir(directory_path):
        from io import BytesIO
        from urllib.request import urlopen
        from zipfile import ZipFile

        http_response = urlopen(url)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=directory_path)
