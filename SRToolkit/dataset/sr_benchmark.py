import copy
import os
from typing import List, Union, Tuple, Optional
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

import numpy as np

from SRToolkit.dataset import SR_dataset
from SRToolkit.evaluation import ResultAugmenter
from SRToolkit.utils import SymbolLibrary, Node, expr_to_executable_function


class SR_benchmark:
    def __init__(self, benchmark_name: str, base_dir: str,
                 datasets: List[Union[SR_dataset, Tuple[str, SR_dataset]]] = None,
                 metadata: dict = None):
        """
        Initializes an instance of the SR_benchmark class.

        Args:
            benchmark_name: The name of this benchmark.
            base_dir: The directory where the datasets will be stored.
            datasets: A list of SR_dataset instances or tuples containing the name of the dataset and an instance of
                SR_dataset. When name of the dataset is not provided, the dataset will be named
                'benchmark_name'_'index of dataset in the list + 1'
            metadata: An optional dictionary containing metadata about this benchmark. This could include information
                such as the name of the benchmark, a citation for the benchmark, number of datasets, etc.

        Raises:
            Exception: If elements in datasets argument are not instances of SR_dataset or tuples containing the name
                of the dataset and an instance of SR_dataset.
        """
        self.benchmark_name = benchmark_name
        self.base_dir = base_dir
        self.datasets = {}
        self.metadata = {} if metadata is None else metadata
        if datasets is not None:
            for i, dataset in enumerate(datasets):
                if isinstance(dataset, SR_dataset):
                    self.add_dataset_instance(benchmark_name + "_" + str(i+1), dataset)
                elif isinstance(dataset, tuple) and isinstance(dataset[0], str) and isinstance(dataset[1], SR_dataset):
                    self.add_dataset_instance(dataset[0], dataset[1])
                else:
                    raise ValueError("[SR_benchmark] Dataset inside the datasets argument must be either a tuple "
                                     "(name, SR_dataset) or a SR_dataset instance.")

    def add_dataset_instance(self, dataset_name: str, dataset: SR_dataset):
        """
        Adds an instance of the SR_dataset class to the benchmark.

        Args:
             dataset_name: The name of the dataset.
             dataset: An instance of the SR_dataset class.
        """
        self.datasets[dataset_name]["sr_dataset"] = dataset
        self.datasets[dataset_name]["num_variables"] = dataset.X.shape[1]

    def add_dataset(
        self,
        dataset: Union[str, np.array, Tuple[np.array, np.array]],
        symbol_library: SymbolLibrary,
        dataset_name: Optional[str] = None,
        ranking_function: str = "rmse",
        max_evaluations: int = -1,
        ground_truth: Optional[Union[List[str], Node, np.ndarray]] = None,
        original_equation: Optional[str] = None,
        success_threshold: Optional[float] = None,
        result_augmenters: Optional[List[ResultAugmenter]] = None,
        seed: Optional[int] = None,
        dataset_metadata: Optional[dict] = None,
        **kwargs
    ):
        """
        Adds a dataset to the benchmark.

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
            ground_truth: The ground truth expression. Can either a list of symbols, a SRToolkit.utils.Node, or a
                numpy array representing behavior of an expressions. When 'bed' is used as the ranking function,
                ground truth must be provided.
            original_equation: The original equation from which the ground truth expression was generated.
            success_threshold: The threshold below which the experiment is considered successful. If None, the
                threshold will be calculated automatically. See SRToolkit.evaluation.SR_evaluator for more details.
            dataset_metadata: An optional dictionary containing metadata about this dataset. This could include
                information such as the name of the dataset, a citation for the dataset, number of variables, etc.

        Keyword Arguments:
            method (str): The method to be used for minimization. Currently, only "L-BFGS-B" is supported/tested.
                Default is "L-BFGS-B".
            tol (float): The tolerance for termination. Default is 1e-6.
            gtol (float): The tolerance for the gradient norm. Default is 1e-3.
            max_iter (int): The maximum number of iterations. Default is 100.
            constant_bounds (Tuple[float, float]): A tuple of two elements, specifying the lower and upper bounds for
                the constant values. Default is (-5, 5).
            initialization (str): The method to use for initializing the constant values. Currently, only "random" and
                "mean" are supported. "random" creates a vector with random values sampled within the bounds. "mean"
                creates a vector where all values are calculated as (lower_bound + upper_bound)/2. Default is "random".
            max_constants (int): The maximum number of constants allowed in the expression. Default is 8.
            max_expr_length (int): The maximum length of the expression. Default is -1 (no limit).
            num_points_sampled (int): The number of points to sample when estimating the behavior of an expression.
                Default is 64. If num_points_sampled==-1, then the number of points sampled is equal to the number of
                points in the dataset.
            bed_X (Optional[np.ndarray]): Points used for BED evaluation. If None and domain_bounds are given, points
                are sampled from the domain. If None and domain_bounds are not givem, points are randomly selected
                from X. Default is None.
            num_consts_sampled (int): Number of constants sampled for BED evaluation. Default is 32.
            domain_bounds (Optional[List[Tuple[float, float]]]): Bounds for the domain to be used if bed_X is None to
                sample random points. Default is None.
        """
        if dataset_name is None:
            dataset_name = f"{self.benchmark_name}_{len(self.datasets)}+1"

        self.datasets[dataset_name] = {}
        self.datasets[dataset_name]["symbol_library"] = symbol_library
        self.datasets[dataset_name]["ranking_function"] = ranking_function
        self.datasets[dataset_name]["max_evaluations"] = max_evaluations

        self.datasets[dataset_name]["success_threshold"] = success_threshold
        self.datasets[dataset_name]["result_augmenters"] = result_augmenters
        self.datasets[dataset_name]["seed"] = seed
        self.datasets[dataset_name]["dataset_metadata"] = copy.deepcopy(self.metadata).update(dataset_metadata)

        self.datasets[dataset_name]["kwargs"] = kwargs
        self.datasets[dataset_name]["original_equation"] = original_equation
        self.datasets[dataset_name]["ground_truth"] = ground_truth

        if ground_truth is None:
            if ranking_function == "bed":
                raise ValueError("[SR_benchmark.add_dataset] For 'bed' ranking, the ground truth must be provided. ")
            else:
                print(f"[SR_benchmark.add_dataset] 'ground_truth' argument not provided. We recommend providing it "
                      f"for more transparent evaluation.")
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
                    f"using np.load. The file may be corrupt or not a valid NumPy archive (.npz, .npy). "
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

            num_variables = data['X'].shape[1]

        elif isinstance(dataset, np.ndarray):
            if ranking_function == "rmse" and ground_truth is not None:
                try:
                    expr = expr_to_executable_function(ground_truth, symbol_library)
                    y = expr(dataset, None)
                except Exception as e:
                    raise Exception(f"[SR_benchmark.add_dataset] Could not evaluate the ground truth. "
                                    f"Original error: {e}")
                if not os.path.isdir(self.base_dir):
                    os.makedirs(self.base_dir)
                np.savez(f"{self.base_dir}/{dataset_name}.npz", X=dataset, y=y, allow_pickle=False)
            elif ranking_function == "rmse" and ground_truth is None:
                raise ValueError("[SR_benchmark.add_dataset] For 'rmse' ranking, if the dataset argument is a numpy "
                                 "array, the ground truth must be provided in order for the target values to be "
                                 "calculated.")
            elif ranking_function == "bed":
                if not os.path.isdir(self.base_dir):
                    os.makedirs(self.base_dir)
                np.savez(f"{self.base_dir}/{dataset_name}.npz", X=dataset, allow_pickle=False)

            self.datasets[dataset_name]["dataset_path"] = f"{self.base_dir}/{dataset_name}.npz"
            num_variables = dataset.shape[1]

        elif isinstance(dataset, tuple):
            if not isinstance(dataset[0], np.ndarray) or not isinstance(dataset[1], np.ndarray):
                raise ValueError("[SR_benchmark.add_dataset] When dataset argument is provided as a tuple, both "
                                 "values must be a numpy array. The first array represents the features ('X'), "
                                 "the second array represents the targets ('y').")
            if ranking_function == "bed":
                print(f"[SR_benchmark.add_dataset] 'bed' ranking only utilizes the array with feature. Array with "
                      f"targets will be ignored.")
            if not os.path.isdir(self.base_dir):
                os.makedirs(self.base_dir)
            np.savez(f"{self.base_dir}/{dataset_name}.npz", X=dataset[0], y=dataset[1], allow_pickle=False)
            self.datasets[dataset_name]["dataset_path"] = f"{self.base_dir}/{dataset_name}.npz"
            num_variables = dataset[0].shape[1]

        else:
            raise ValueError("[SR_benchmark.add_dataset] The dataset argument must be a string, a numpy array, "
                             "or a tuple containing two numpy arrays.")

        self.datasets[dataset_name]["num_variables"] = num_variables

    def create_dataset(self, dataset_name: str) -> SR_dataset:
        """
        Creates an instance of a dataset from the given dataset name.

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
                if os.path.exists(self.datasets[dataset_name]["dataset_path"]):
                    data = np.load(self.datasets[dataset_name]["dataset_path"], allow_pickle=True)
                elif os.path.exists(self.datasets[dataset_name]["dataset_path"][:-4]):
                    data = np.load(self.datasets[dataset_name]["path"][:-4], allow_pickle=True)
                else:
                    raise ValueError(f"[SR_benchmark.create_dataset] Could not find dataset {dataset_name} at "
                                     f"{self.datasets[dataset_name]['dataset_path']}")

                if self.datasets[dataset_name]["ranking_function"] == "rmse":
                    X = data["X"]
                    y = data["y"]

                elif self.datasets[dataset_name]["ranking_function"] == "bed":
                    X = data["X"]
                    y = None
                else:
                    raise ValueError(f"The ranking function '{self.datasets[dataset_name]['ranking_function']}' "
                                 f"must be either 'rmse' or 'bed'.")

                return SR_dataset(X,
                                  symbol_library=self.datasets[dataset_name]["symbol_library"],
                                  ranking_function=self.datasets[dataset_name]["ranking_function"],
                                  y=y,
                                  max_evaluations= self.datasets[dataset_name]["max_evaluations"],
                                  ground_truth = self.datasets[dataset_name]["ground_truth"],
                                  original_equation = self.datasets[dataset_name]["original_equation"],
                                  success_threshold = self.datasets[dataset_name]["success_threshold"],
                                  result_augmenter = self.datasets[dataset_name]["result_augmenters"],
                                  seed = self.datasets[dataset_name]["seed"],
                                  dataset_metadata = self.datasets[dataset_name]["dataset_metadata"],
                                  **self.datasets[dataset_name]["kwargs"])

        else:
            raise ValueError(f"Dataset {dataset_name} not found")

    def list_datasets(self, verbose=True, num_variables: int = -1) -> List[str]:
        """
        Lists the available datasets.

        Args:
            verbose (bool): If True, also prints out a description of each dataset.
            num_variables (int): If not -1, only show datasets with the given number of variables.

        Returns:
            A list of dataset names.
        """
        datasets = [
            dataset_name
            for dataset_name in self.datasets
            if num_variables < 0
            or self.datasets[dataset_name]["num_variables"] == num_variables
        ]
        datasets = sorted(
            datasets,
            key=lambda dataset_name: (
                self.datasets[dataset_name]["num_variables"],
                dataset_name,
            ),
        )

        if verbose:
            # TODO: Make all names be of equal length for nicer output
            for d in datasets:
                if self.datasets[d]["num_variables"] == 1:
                    variable_str = "1 variable"
                elif self.datasets[d]["num_variables"] < 1:
                    variable_str = "Amount of variables unknown"
                else:
                    variable_str = f"{self.datasets[d]['num_variables']} variables"

                print(
                    f"{d}:\t{variable_str}, \tExpression: {self.datasets[d]['original_equation']}"
                )
        return datasets

    @staticmethod
    def download_benchmark_data(url, directory_path):
        # Check if directory_path exist
        """
        Downloads a benchmark dataset from the given url to the given directory path.

        This function will first check if the directory_path exists. If not, it will create it. Then it will check if the directory_path is empty. If it is not empty, it will not download the data. If it is empty, it will download the data from the given url and extract it to the directory_path.

        Args:
            url (str): The url of the benchmark dataset to download.
            directory_path (str): The path of the directory where the dataset should be downloaded.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Check if directory_path is empty
        if not os.listdir(directory_path):
            # Download data from the url to the directory_path
            http_response = urlopen(url)
            zipfile = ZipFile(BytesIO(http_response.read()))
            zipfile.extractall(path=directory_path)

    @staticmethod
    def feynman(dataset_directory: str, seed: Optional[int] = None) -> "SR_benchmark":
        """
        Downloads the Feynman benchmark dataset, sets up symbol libraries, and adds predefined datasets to the benchmark.

        This method downloads the Feynman benchmark dataset from a specified URL, initializes symbol libraries for
        symbolic regression with varying numbers of variables, and adds multiple predefined datasets to the benchmark
        with their respective equations and metadata.

        Examples:
            >>> benchmark = SR_benchmark.feynman('data/feynman')
            >>> for dataset in benchmark.list_datasets(verbose=False):
            ...     ds = benchmark.create_dataset(dataset)
            ...     rmse = ds.create_evaluator().evaluate_expr(ds.ground_truth)
            ...     if rmse > ds.success_threshold:
            ...         print(f'Failed dataset: {dataset} with RMSE {rmse}')

        Args:
            dataset_directory: The directory path where the benchmark dataset will be downloaded and stored or where
                it will be loaded from.
            seed: The seed to use for the random number generator. If None, the random number generation will not
                be seeded

        Returns:
            SR_benchmark: An instance of the SR_benchmark class containing the predefined datasets.
        """
        url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/feynman.zip"

        metadata = {"description": "Feynman benchmark containing 100 equations from the domain of physics. "
                                   "Expressions can contain up to 9 variables.",
                    "citation": """@article{Tegmark2020Feynman,
  title={{AI Feynman: A physics-inspired method for symbolic regression}},
  author={Udrescu, Silviu-Marian and Tegmark, Max},
  journal={Science Advances},
  volume={6},
  number={16},
  pages={eaay2631},
  year={2020},
  publisher={American Association for the Advancement of Science}
}
"""
                    }

        SR_benchmark.download_benchmark_data(url, dataset_directory)

        sl_1v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp",
                                                "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 1)
        sl_2v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp",
                                                "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 2)
        sl_3v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp",
                                                "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 3)
        sl_4v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp",
                                                "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 4)
        sl_5v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp",
                                                "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 5)
        sl_6v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp",
                                                "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 6)
        sl_8v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp",
                                                "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 8)
        sl_9v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp",
                                                "arcsin", "tanh", "ln", "^2", "^3", "pi", "C"], 9)

        benchmark = SR_benchmark("feynman", dataset_directory)
        benchmark.metadata = metadata
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.16.6",
            ranking_function="rmse",
            max_evaluations=100000,
            ground_truth = ["(", "X_2", "+","X_1",")","/","(","1","+","(","X_2","*","X_1",")","/","(","X_0","^2",")",")"], # noqa: F401
            original_equation="v1 = (u+v)/(1+u*v/c^2)",
            success_threshold=1e-7,
            result_augmenters=[],
            seed = seed,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            max_expression_length=50,
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.15.4",
            ranking_function="rmse",
            ground_truth = ["u-", "X_0", "*", "X_1", "*", "cos", "(", "X_2", ")"], # noqa: F401
            original_equation="E_n = -mom*B*cos(theta)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.27.16",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "X_2", "^2"], # noqa: F401
            original_equation="flux = epsilon*c*Ef^2",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_6v,
            dataset_name="I.11.19",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_3", "+", "X_1", "*", "X_4", "+", "X_2", "*", "X_5"], # noqa: F401
            original_equation="A = x1*y1+x2*y2+x3*y3",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.15.3x",
            ranking_function="rmse",
            ground_truth = ["(", "X_0", "-", "X_1", "*", "X_3", ")", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"], # noqa: F401
            original_equation="x1 = (x-u*t)/sqrt(1-u^2/c^2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.10.7",
            ranking_function="rmse",
            ground_truth = ["X_0", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"], # noqa: F401
            original_equation="m = m_0/sqrt(1-v^2/c^2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_9v,
            dataset_name="I.9.18",
            ranking_function="rmse",
            ground_truth = ["X_2", "*", "X_0", "*", "X_1", "/", "(", "(", "X_4", "-", "X_3", ")", "^2", "+", "(", "X_6", "-", "X_5", ")", "^2", "+", "(", "X_8", "-", "X_7", ")", "^2", ")"], # noqa: F401
            original_equation="F = G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.15.3t",
            ranking_function="rmse",
            ground_truth = ["(", "X_3", "-", "X_2", "*", "X_0", "/", "X_1", "^2", ")", "/", "sqrt", "(", "1", "-", "X_2", "^2", "/", "X_1", "^2", ")"], # noqa: F401
            original_equation="t1 = (t-u*x/c^2)/sqrt(1-u^2/c^2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_8v,
            dataset_name="II.36.38",
            ranking_function="rmse",
            ground_truth = ["(", "X_0", "*", "X_1", ")", "/", "(", "X_2", "*", "X_3", ")", "+", "(", "(", "X_0", "*", "X_4", ")", "/", "(", "X_5", "*", "X_6", "^2", "*", "X_2", "*", "X_3", ")", ")", "*", "X_7"], # noqa: F401
            original_equation="f = mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.43.43",
            ranking_function="rmse",
            ground_truth = ["(", "1", "/", "(", "X_0", "-", "1", ")", ")", "*", "X_1", "*", "X_3", "/", "X_2"], # noqa: F401
            original_equation="kappa = 1/(gamma-1)*kb*v/A",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.15.5",
            ranking_function="rmse",
            ground_truth = ["u-", "X_0", "*", "X_1", "*", "cos", "(", "X_2", ")"], # noqa: F401
            original_equation="E_n = -p_d*Ef*cos(theta)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.37.4",
            ranking_function="rmse",
            ground_truth = ["X_0", "+", "X_1", "+", "2", "*", "sqrt", "(", "X_0", "*", "X_1", ")", "*", "cos", "(", "X_2", ")"], # noqa: F401
            original_equation="Int = I1+I2+2*sqrt(I1*I2)*cos(delta)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="II.6.11",
            ranking_function="rmse",
            ground_truth = ["(", "1", "/", "(", "4", "*", "pi", "*", "X_0", ")", ")", "*", "X_1", "*", "cos", "(", "X_2", ")", "/", "X_3", "^2"], # noqa: F401
            original_equation="Volt = 1/(4*pi*epsilon)*p_d*cos(theta)/r^2",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="III.7.38",
            ranking_function="rmse",
            ground_truth = ["2", "*", "X_0", "*", "X_1", "/", "(", "X_2", "/", "(", "2", "*", "pi", ")", ")"], # noqa: F401
            original_equation="omega = 2*mom*B/(h/(2*pi))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.34.2a",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "/", "(", "2", "*", "pi", "*", "X_2", ")"], # noqa: F401
            original_equation="l = q*v/(2*pi*r)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.13.23",
            ranking_function="rmse",
            ground_truth = ["X_0", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"], # noqa: F401
            original_equation="rho_c = rho_c_0/sqrt(1-v^2/c^2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="I.29.4",
            ranking_function="rmse",
            ground_truth = ["X_0", "/", "X_1"], # noqa: F401
            original_equation="k = omega/c",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.38.12",
            ranking_function="rmse",
            ground_truth = ["4", "*", "pi", "*", "X_3", "*", "(", "X_2", "/", "(", "2", "*", "pi", ")", ")", "^2", "/", "(", "X_0", "*", "X_1", "^2", ")"], # noqa: F401
            original_equation="r = 4*pi*epsilon*(h/(2*pi))^2/(m*q^2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="III.15.27",
            ranking_function="rmse",
            ground_truth = ["2", "*", "pi", "*", "X_0", "/", "(", "X_1", "*", "X_2", ")"], # noqa: F401
            original_equation="k = 2*pi*alpha/(n*d)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="I.41.16",
            ranking_function="rmse",
            ground_truth = ["(", "X_2", "/", "(", "2", "*", "pi", ")", ")", "*", "X_0", "^3", "/", "(", "pi", "^2", "*", "X_4", "^2", "*", "(", "exp", "(", "(", "X_2", "/", "(", "2", "*", "pi", ")", ")", "*", "X_0", "/", "(", "X_3", "*", "X_1", ")", ")", "-", "1", ")", ")"], # noqa: F401
            original_equation="L_rad = h/(2*pi)*omega^3/(pi^2*c^2*(exp((h/(2*pi))*omega/(kb*T))-1))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.48.20",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_2", "^2", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"], # noqa: F401
            original_equation="E_n = m*c^2/sqrt(1-v^2/c^2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="II.11.20",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "^2", "*", "X_2", "/", "(", "3", "*", "X_3", "*", "X_4", ")"], # noqa: F401
            original_equation="Pol = n_rho*p_d^2*Ef/(3*kb*T)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="I.25.13",
            ranking_function="rmse",
            ground_truth = ["X_0", "/", "X_1"], # noqa: F401
            original_equation="Volt = q/C",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="III.15.12",
            ranking_function="rmse",
            ground_truth = ["2", "*", "X_0", "*", "(", "1", "-", "cos", "(", "X_1", "*", "X_2", ")", ")"], # noqa: F401
            original_equation="E_n = 2*U*(1-cos(k*d))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.24.6",
            ranking_function="rmse",
            ground_truth = ["0.25", "*", "X_0", "*", "(", "X_1", "^2", "+", "X_2", "^2", ")", "*", "X_3", "^2"], # noqa: F401
            original_equation="E_n = 1/2*m*(omega^2+omega_0^2)*1/2*x^2",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="I.34.27",
            ranking_function="rmse",
            ground_truth = ["(", "X_1", "/", "(", "2", "*", "pi", ")", ")", "*", "X_0"], # noqa: F401
            original_equation="E_n =(h/(2*pi))*omega",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.43.31",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_2", "*", "X_1"], # noqa: F401
            original_equation="D = mob*kb*T",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.29.16",
            ranking_function="rmse",
            ground_truth = ["sqrt", "(", "X_0", "^2", "+", "X_1", "^2", "-", "2", "*", "X_0", "*", "X_1", "*", "cos", "(", "X_2", "-", "X_3", ")", ")"], # noqa: F401
            original_equation="x = sqrt(x1^2+x2^2-2*x1*x2*cos(theta1-theta2))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.18.4",
            ranking_function="rmse",
            ground_truth = ["(", "X_0", "*", "X_2", "+", "X_1", "*", "X_3", ")", "/", "(", "X_0", "+", "X_1", ")"], # noqa: F401
            original_equation="r = (m1*r1+m2*r2)/(m1+m2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_6v,
            dataset_name="II.6.15a",
            ranking_function="rmse",
            ground_truth = ["(", "X_1", "/", "(", "4", "*", "pi", "*", "X_0", ")", ")", "*", "(", "3", "*", "X_5", "/", "(", "X_2", "^2", "*", "X_2", "^3", ")", ")", "*", "sqrt", "(", "X_3", "^2", "+", "X_4", "^2", ")"], # noqa: F401
            original_equation="Ef = p_d/(4*pi*epsilon)*3*z/r^5*sqrt(x^2+y^2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.30.3",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "sin", "(", "X_2", "*", "X_1", "/", "2", ")", "^2", "/", "sin", "(", "X_1", "/", "2", ")", "^2"], # noqa: F401
            original_equation="Int = Int_0*sin(n*theta/2)^2/sin(theta/2)^2",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_6v,
            dataset_name="III.9.52",
            ranking_function="rmse",
            ground_truth = ["(", "X_0", "*", "X_1", "*", "X_2", "/", "(", "X_3", "/", "(", "2", "*", "pi", ")", ")", ")", "*", "sin", "(", "(", "X_4", "-", "X_5", ")", "*", "X_2", "/", "2", ")", "^2", "/", "(", "(", "X_4", "-", "X_5", ")", "*", "X_2", "/", "2", ")", "^2"], # noqa: F401
            original_equation="prob = (p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)^2/((omega-omega_0)*t/2)^2",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.34.2",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "X_2", "/", "2"], # noqa: F401
            original_equation="mom = q*v*r/2",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.39.11",
            ranking_function="rmse",
            ground_truth = ["(", "1", "/", "(", "X_0", "-", "1", ")", ")", "*", "X_1", "*", "X_2"], # noqa: F401
            original_equation="E_n = (1/(gamma-1))*pr*V",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="II.11.28",
            ranking_function="rmse",
            ground_truth = ["1", "+", "X_0", "*", "X_1", "/", "(", "1", "-", "(", "X_0", "*", "X_1", "/", "3", ")", ")"], # noqa: F401
            original_equation="theta = 1+n*alpha/(1-(n*alpha/3))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="II.3.24",
            ranking_function="rmse",
            ground_truth = ["X_0", "/", "(", "4", "*", "pi", "*", "X_1", "^2", ")"], # noqa: F401
            original_equation="flux = Pwr/(4*pi*r^2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.24.17",
            ranking_function="rmse",
            ground_truth = ["sqrt", "(", "X_0", "^2", "/", "X_1", "^2", "-", "pi", "^2", "/", "X_2", "^2", ")"], # noqa: F401
            original_equation="k = sqrt(omega^2/c^2-pi^2/d^2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="II.13.17",
            ranking_function="rmse",
            ground_truth = ["(", "1", "/", "(", "4", "*", "pi", "*", "X_0", "*", "X_1", "^2", ")", ")", "*", "2", "*", "X_2", "/", "X_3"], # noqa: F401
            original_equation="B = 1/(4*pi*epsilon*c^2)*2*I/r",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="I.12.5",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1"], # noqa: F401
            original_equation="F = q2*Ef",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="II.35.18",
            ranking_function="rmse",
            ground_truth = ["X_0", "/", "(", "exp", "(", "X_3", "*", "X_4", "/", "(", "X_1", "*", "X_2", ")", ")", "+", "exp", "(", "u-", "X_3", "*", "X_4", "/", "(", "X_1", "*", "X_2", ")", ")", ")"], # noqa: F401
            original_equation="n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="II.34.11",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "X_2", "/", "(", "2", "*", "X_3", ")"], # noqa: F401
            original_equation="g_*q*B/(2*m)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.34.29a",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "/", "(", "4", "*", "pi", "*", "X_2", ")"], # noqa: F401
            original_equation="q*h/(4*pi*m)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_6v,
            dataset_name="I.32.17",
            ranking_function="rmse",
            ground_truth = ["(", "0.5", "*", "X_0", "*", "X_1", "*", "X_2", "^2", ")", "*", "(", "8", "*", "pi", "*", "X_3", "^2", "/", "3", ")", "*", "(", "(", "X_4", "^2", "*", "X_4", "^2", ")", "/", "(", "X_4", "^2", "-", "X_5", "^2", ")", "^2", ")"], # noqa: F401
            original_equation="(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="II.35.21",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "tanh", "(", "X_1", "*", "X_2", "/", "(", "X_3", "*", "X_4", ")", ")"], # noqa: F401
            original_equation="n_rho*mom*tanh(mom*B/(kb*T))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="I.44.4",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "X_2", "*", "ln", "(", "X_4", "/", "X_3", ")"], # noqa: F401
            original_equation="n*kb*T*ln(V2/V1)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="III.4.32",
            ranking_function="rmse",
            ground_truth = ["1", "/", "(", "exp", "(", "(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "*", "X_1", "/", "(", "X_2", "*", "X_3", ")", ")", "-", "1", ")"], # noqa: F401
            original_equation="1/(exp((h/(2*pi))*omega/(kb*T))-1)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.10.9",
            ranking_function="rmse",
            ground_truth = ["(", "X_0", "/", "X_1", ")", "*", "1", "/", "(", "1", "+", "X_2", ")"], # noqa: F401
            original_equation="sigma_den/epsilon*1/(1+chi)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="II.38.3",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "X_3", "/", "X_2"], # noqa: F401
            original_equation="Y*A*x/d",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.6.2b",
            ranking_function="rmse",
            ground_truth = ["exp", "(", "u-", "(", "(", "(", "X_1", "-", "X_2", ")", "/", "X_0", ")", "^2", ")", "/", "2", ")", "/", "(", "sqrt", "(", "2", "*", "pi", ")", "*", "X_0", ")"], # noqa: F401
            original_equation="exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="II.8.31",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "^2", "/", "2"], # noqa: F401
            original_equation="epsilon*Ef**2/2",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_1v,
            dataset_name="I.6.2a",
            ranking_function="rmse",
            ground_truth = ["exp", "(", "u-", "X_0", "^2", "/", "2", ")", "/", "sqrt", "(", "2", "*", "pi", ")"], # noqa: F401
            original_equation="exp(-theta**2/2)/sqrt(2*pi)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="III.12.43",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "(", "X_1", "/", "(", "2", "*", "pi", ")", ")"], # noqa: F401
            original_equation="n*(h/(2*pi))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="III.17.37",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "(", "1", "+", "X_1", "*", "cos", "(", "X_2", ")", ")"], # noqa: F401
            original_equation="beta*(1+alpha*cos(theta))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="III.10.19",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "sqrt", "(", "X_1", "^2", "+", "X_2", "^2", "+", "X_3", "^2", ")"], # noqa: F401
            original_equation="mom*sqrt(Bx**2+By**2+Bz**2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_6v,
            dataset_name="II.11.7",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "(", "1", "+", "X_4", "*", "X_5", "*", "cos", "(", "X_3", ")", "/", "(", "X_1", "*", "X_2", ")", ")"], # noqa: F401
            original_equation="n_0*(1+p_d*Ef*cos(theta)/(kb*T))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="I.39.1",
            ranking_function="rmse",
            ground_truth = ["1.5", "*", "X_0", "*", "X_1"], # noqa: F401
            original_equation="3/2*pr*V",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.37.1",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "(", "1", "+", "X_2", ")", "*", "X_1"], # noqa: F401
            original_equation="mom*(1+chi)*B",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.12.4",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_2", "/", "(", "4", "*", "pi", "*", "X_1", "*", "X_2", "^3", ")"], # noqa: F401
            original_equation="q1*r/(4*pi*epsilon*r**3)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="II.27.18",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "^2"], # noqa: F401
            original_equation="epsilon*Ef**2",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.12.2",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "X_3", "/", "(", "4", "*", "pi", "*", "X_2", "*", "X_3", "^3", ")"], # noqa: F401
            original_equation="q1*q2*r/(4*pi*epsilon*r**3)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="III.13.18",
            ranking_function="rmse",
            ground_truth = ["2", "*", "X_0", "*", "X_1", "^2", "*", "X_2", "/", "(", "X_3", "/", "(", "2", "*", "pi", ")", ")"], # noqa: F401
            original_equation="2*E_n*d**2*k/(h/(2*pi))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="II.11.3",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "/", "(", "X_2", "*", "(", "X_3", "^2", "-", "X_4", "^2", ")", ")"], # noqa: F401
            original_equation="q*Ef/(m*(omega_0**2-omega**2))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_6v,
            dataset_name="I.40.1",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "exp", "(", "u-", "X_1", "*", "X_4", "*", "X_2", "/", "(", "X_5", "*", "X_3", ")", ")"], # noqa: F401
            original_equation="n_0*exp(-m*g*x/(kb*T))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="III.21.20",
            ranking_function="rmse",
            ground_truth = ["u-", "X_0", "*", "X_1", "*", "X_2", "/", "X_3"], # noqa: F401
            original_equation="-rho_c_0*q*A_vec/m",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.43.16",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "X_2", "/", "X_3"], # noqa: F401
            original_equation="mu_drift*q*Volt/d",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.15.10",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"], # noqa: F401
            original_equation="m_0*v/sqrt(1-v**2/c**2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.30.5",
            ranking_function="rmse",
            ground_truth = ["arcsin", "(", "X_0", "/", "(", "X_2", "*", "X_1", ")", ")"], # noqa: F401
            original_equation="arcsin(lambd/(n*d))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.50.26",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "(", "cos", "(", "X_1", "*", "X_2", ")", "+", "X_3", "*", "cos", "(", "X_1", "*", "X_2", ")", "^2", ")"], # noqa: F401
            original_equation="x1*(cos(omega*t)+alpha*cos(omega*t)**2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="I.12.11",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "(", "X_1", "+", "X_2", "*", "X_3", "*", "sin", "(", "X_4", ")", ")"], # noqa: F401
            original_equation="q*(Ef+B*v*sin(theta))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="I.6.2",
            ranking_function="rmse",
            ground_truth = ["exp", "(", "u-", "(", "(", "X_1", "/", "X_0", ")", "^2", ")", "/", "2", ")", "/", "(", "sqrt", "(", "2", "*", "pi", ")", "*", "X_0", ")"], # noqa: F401
            original_equation="exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="I.14.4",
            ranking_function="rmse",
            ground_truth = ["0.5", "*", "X_0", "*", "X_1", "^2"], # noqa: F401
            original_equation="1/2*k_spring*x**2",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.47.23",
            ranking_function="rmse",
            ground_truth = ["sqrt", "(", "X_0", "*", "X_1", "/", "X_2", ")"], # noqa: F401
            original_equation="sqrt(gamma*pr/rho)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.8.7",
            ranking_function="rmse",
            ground_truth = ["0.6", "*", "X_0", "^2", "/", "(", "4", "*", "pi", "*", "X_1", "*", "X_2", ")"], # noqa: F401
            original_equation="3/5*q**2/(4*pi*epsilon*d)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="III.15.14",
            ranking_function="rmse",
            ground_truth = ["(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "^2", "/", "(", "2", "*", "X_1", "*", "X_2", "^2", ")"], # noqa: F401
            original_equation="(h/(2*pi))**2/(2*E_n*d**2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.34.14",
            ranking_function="rmse",
            ground_truth = ["(", "(", "1", "+", "(", "X_1", "/", "X_0", ")", ")", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_0", "^2", ")", ")", "*", "X_2"], # noqa: F401
            original_equation="((1+v/c)/sqrt(1-v**2/c**2))*omega_0",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="III.8.54",
            ranking_function="rmse",
            ground_truth = ["sin", "(", "X_0", "*", "X_1", "/", "(", "X_2", "/", "(", "2", "*", "pi", ")", ")", ")", "^2"], # noqa: F401
            original_equation="sin(E_n*t/(h/(2*pi)))**2",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="I.26.2",
            ranking_function="rmse",
            ground_truth = ["arcsin", "(", "X_0", "*", "sin", "(", "X_1", ")", ")"], # noqa: F401
            original_equation="arcsin(n*sin(theta2))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="III.19.51",
            ranking_function="rmse",
            ground_truth = ["(", "u-", "X_0", "*", "(", "X_1", "^2", "*", "X_1", "^2", ")", "/", "(", "(", "2", "*", "(", "4", "*", "pi", "*", "X_4", ")", "^2", ")", "*", "(", "X_2", "/", "(", "2", "*", "pi", ")", ")", "^2", ")", "*", "(", "1", "/", "X_3", "^2", ")", ")"], # noqa: F401
            original_equation="-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="III.4.33",
            ranking_function="rmse",
            ground_truth = ["(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "*", "X_1", "/", "(", "exp", "(", "(", "X_0", "/", "(", "2", "*", "pi", ")", ")", "*", "X_1", "/", "(", "X_2", "*", "X_3", ")", ")", "-", "1", ")"], # noqa: F401
            original_equation="(h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.34.1",
            ranking_function="rmse",
            ground_truth = ["X_2", "/", "(", "1", "-", "X_1", "/", "X_0", ")"], # noqa: F401
            original_equation="omega_0/(1-v/c)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="II.11.27",
            ranking_function="rmse",
            ground_truth = ["(", "X_0", "*", "X_1", "/", "(", "1", "-", "(", "X_0", "*", "X_1", "/", "3", ")", ")", ")", "*", "X_2", "*", "X_3"], # noqa: F401
            original_equation="n*alpha/(1-(n*alpha/3))*epsilon*Ef",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.13.34",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"], # noqa: F401
            original_equation="rho_c_0*v/sqrt(1-v**2/c**2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="II.4.23",
            ranking_function="rmse",
            ground_truth = ["X_0", "/", "(", "4", "*", "pi", "*", "X_1", "*", "X_2", ")"], # noqa: F401
            original_equation="q/(4*pi*epsilon*r)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.32.5",
            ranking_function="rmse",
            ground_truth = ["X_0", "^2", "*", "X_1", "^2", "/", "(", "6", "*", "pi", "*", "X_2", "*", "X_3", "^3", ")"], # noqa: F401
            original_equation="q**2*a**2/(6*pi*epsilon*c**3)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="I.13.12",
            ranking_function="rmse",
            ground_truth = ["X_4", "*", "X_0", "*", "X_1", "*", "(", "1", "/", "X_3", "-", "1", "/", "X_2", ")"], # noqa: F401
            original_equation="G*m1*m2*(1/r2-1/r1)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="II.2.42",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "(", "X_2", "-", "X_1", ")", "*", "X_3", "/", "X_4"], # noqa: F401
            original_equation="kappa*(T2-T1)*A/d",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.27.6",
            ranking_function="rmse",
            ground_truth = ["1", "/", "(", "1", "/", "X_0", "+", "X_2", "/", "X_1", ")"], # noqa: F401
            original_equation="1/(1/d1+n/d2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="III.14.14",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "(", "exp", "(", "X_1", "*", "X_2", "/", "(", "X_3", "*", "X_4", ")", ")", "-", "1", ")"], # noqa: F401
            original_equation="I_0*(exp(q*Volt/(kb*T))-1)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.18.12",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "sin", "(", "X_2", ")"], # noqa: F401
            original_equation="r*F*sin(theta)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.18.14",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "X_2", "*", "sin", "(", "X_3", ")"], # noqa: F401
            original_equation="m*r*v*sin(theta)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="II.21.32",
            ranking_function="rmse",
            ground_truth = ["X_0", "/", "(", "4", "*", "pi", "*", "X_1", "*", "X_2", "*", "(", "1", "-", "X_3", "/", "X_4", ")", ")"], # noqa: F401
            original_equation="q/(4*pi*epsilon*r*(1-v/c))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="II.38.14",
            ranking_function="rmse",
            ground_truth = ["X_0", "/", "(", "2", "*", "(", "1", "+", "X_1", ")", ")"], # noqa: F401
            original_equation="Y/(2*(1+sigma))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.34.8",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "X_2", "/", "X_3"], # noqa: F401
            original_equation="q*v*B/p",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.8.14",
            ranking_function="rmse",
            ground_truth = ["sqrt", "(", "(", "X_1", "-", "X_0", ")", "^2", "+", "(", "X_3", "-", "X_2", ")", "^2", ")"], # noqa: F401
            original_equation="sqrt((x2-x1)**2+(y2-y1)**2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="II.6.15b",
            ranking_function="rmse",
            ground_truth = ["(", "X_1", "/", "(", "4", "*", "pi", "*", "X_0", ")", ")", "*", "3", "*", "cos", "(", "X_2", ")", "*", "sin", "(", "X_2", ")", "/", "X_3", "^3"], # noqa: F401
            original_equation="p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_2v,
            dataset_name="I.12.1",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1"], # noqa: F401
            original_equation="mu*Nn",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_5v,
            dataset_name="II.34.29b",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_3", "*", "X_4", "*", "X_2", "/", "(", "X_1", "/", "(", "2", "*", "pi", ")", ")"], # noqa: F401
            original_equation="g_*mom*B*Jz/(h/(2*pi))",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.13.4",
            ranking_function="rmse",
            ground_truth = ["0.5", "*", "X_0", "*", "(", "X_1", "^2", "+", "X_2", "^2", "+", "X_3", "^2", ")"], # noqa: F401
            original_equation="1/2*m*(v**2+u**2+w**2)",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_4v,
            dataset_name="I.39.22",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_3", "*", "X_1", "/", "X_2"], # noqa: F401
            original_equation="n*kb*T/V",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )
        benchmark.add_dataset(
            "",
            sl_3v,
            dataset_name="I.14.3",
            ranking_function="rmse",
            ground_truth = ["X_0", "*", "X_1", "*", "X_2"], # noqa: F401
            original_equation="m*g*z",
            max_evaluations=100000,
            max_expression_length=50,
            success_threshold=1e-7,
            dataset_metadata=benchmark.metadata,
            constant_range=[-5.0, 5.0],
            result_augmenters=[],
            seed = seed
        )

        return benchmark

    @staticmethod
    def nguyen(dataset_directory: str, seed: Optional[int] = None) -> "SR_benchmark":
        """
        Downloads and initializes the Nguyen benchmark datasets for symbolic regression.

        This method downloads the Nguyen symbolic regression benchmark datasets from a specified URL
        and initializes a set of datasets using a provided dataset directory. It creates two symbol libraries
        for equations with one variable and two variables, respectively, and populates the benchmark with various
        Nguyen equations, each represented with its symbolic tokens and associated symbol library.

        Examples:
            >>> benchmark = SR_benchmark.nguyen('data/nguyen')
            >>> for dataset in benchmark.list_datasets(verbose=False):
            ...     ds = benchmark.create_dataset(dataset)
            ...     rmse = ds.create_evaluator().evaluate_expr(ds.ground_truth)
            ...     if rmse > ds.success_threshold:
            ...         print(f'Failed dataset: {dataset} with RMSE {rmse}')

        Args:
            dataset_directory: The directory path where the benchmark dataset will be downloaded and stored or where
                it will be loaded from.
            seed: The seed to use for the random number generator. If None, the random number generation will not
                be seeded

        Returns:
            SR_benchmark: An initialized SR_benchmark instance containing the Nguyen datasets.
        """
        url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/nguyen.zip"
        SR_benchmark.download_benchmark_data(url, dataset_directory)
        # we create a SymbolLibrary with 1 and with 2 variables
        # Each library contains +, -, *, /, sin, cos, exp, log, sqrt, ^2, ^3
        sl_1v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "^2", "^3"], 1)
        sl_2v = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "^2", "^3"], 2)

        metadata = {"description": "Symbolic regression benchmark with 10 expressions that don't contain constant "
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
}"""}

        # Add datasets to the benchmark
        benchmark = SR_benchmark("Nguyen", dataset_directory, metadata=metadata)
        benchmark.add_dataset(
			"",
			sl_1v,
			dataset_name="NG-1",
			ranking_function="rmse",
			ground_truth = ["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], # noqa: F401
			original_equation="x+x^2+x^3",
			max_evaluations=100000,
			max_expression_length=50,
			success_threshold=1e-7,
			dataset_metadata=benchmark.metadata,
			result_augmenters=[],
			seed = seed
		)

        benchmark.add_dataset(
			"",
			sl_1v,
			dataset_name="NG-2",
			ranking_function="rmse",
			ground_truth = ["X_0", "+", "X_0", "^2", "+", "X_0", "^3", "+", "X_0", "*", "X_0", "^3"], # noqa: F401
			original_equation="x+x^2+x^3+x^4",
			max_evaluations=100000,
			max_expression_length=50,
			success_threshold=1e-7,
			dataset_metadata=benchmark.metadata,
			result_augmenters=[],
			seed = seed
		)

        benchmark.add_dataset(
			"",
			sl_1v,
			dataset_name="NG-3",
			ranking_function="rmse",
			ground_truth = ["X_0", "+", "X_0", "^2", "+", "X_0", "^3", "+", "X_0", "*", "X_0", "^3", "+", "X_0", "^2", "*", "X_0", "^3"], # noqa: F401
			original_equation="x+x^2+x^3+x^4+x^5",
			max_evaluations=100000,
			max_expression_length=50,
			success_threshold=1e-7,
			dataset_metadata=benchmark.metadata,
			result_augmenters=[],
			seed = seed
		)

        benchmark.add_dataset(
			"",
			sl_1v,
			dataset_name="NG-4",
			ranking_function="rmse",
			ground_truth = ["X_0", "+", "X_0", "^2", "+", "X_0", "^3", "+", "X_0", "*", "X_0", "^3", "+", "X_0", "^2", "*", "X_0", "^3", "+", "X_0", "^3", "*", "X_0", "^3"], # noqa: F401
			original_equation="x+x^2+x^3+x^4+x^5+x^6",
			max_evaluations=100000,
			max_expression_length=50,
			success_threshold=1e-7,
			dataset_metadata=benchmark.metadata,
			result_augmenters=[],
			seed = seed
		)

        benchmark.add_dataset(
			"",
			sl_1v,
			dataset_name="NG-5",
			ranking_function="rmse",
			ground_truth = ["sin", "(", "X_0", "^2", ")", "*", "cos", "(", "X_0", ")", "-", "1"], # noqa: F401
			original_equation="sin(x^2)*cos(x)-1",
			max_evaluations=100000,
			max_expression_length=50,
			success_threshold=1e-7,
			dataset_metadata=benchmark.metadata,
			result_augmenters=[],
			seed = seed
		)

        benchmark.add_dataset(
			"",
			sl_1v,
			dataset_name="NG-6",
			ranking_function="rmse",
			ground_truth = ["sin", "(", "X_0", ")", "+", "sin", "(", "X_0", "+", "X_0", "^2", ")"], # noqa: F401
			original_equation="sin(x)+sin(x+x^2)",
			max_evaluations=100000,
			max_expression_length=50,
			success_threshold=1e-7,
			dataset_metadata=benchmark.metadata,
			result_augmenters=[],
			seed = seed
		)

        benchmark.add_dataset(
			"",
			sl_1v,
			dataset_name="NG-7",
			ranking_function="rmse",
			ground_truth = ["log", "(", "1", "+", "X_0", ")", "+", "log", "(", "1", "+", "X_0", "^2", ")"], # noqa: F401
			original_equation="log(1+x)+log(1+x^2)",
			max_evaluations=100000,
			max_expression_length=50,
			success_threshold=1e-7,
			dataset_metadata=benchmark.metadata,
			result_augmenters=[],
			seed = seed
		)

        benchmark.add_dataset(
			"",
			sl_1v,
			dataset_name="NG-8",
			ranking_function="rmse",
			ground_truth = ["sqrt", "(", "X_0", ")"], # noqa: F401
			original_equation="sqrt(x)",
			max_evaluations=100000,
			max_expression_length=50,
			success_threshold=1e-7,
			dataset_metadata=benchmark.metadata,
			result_augmenters=[],
			seed = seed
		)

        benchmark.add_dataset(
			"",
			sl_2v,
			dataset_name="NG-9",
			ranking_function="rmse",
			ground_truth = ["sin", "(", "X_0", ")", "+", "sin", "(", "X_1", "^2", ")"], # noqa: F401
			original_equation="sin(x)+sin(y^2)",
			max_evaluations=100000,
			max_expression_length=50,
			success_threshold=1e-7,
			dataset_metadata=benchmark.metadata,
			result_augmenters=[],
			seed = seed
		)

        benchmark.add_dataset(
			"",
			sl_2v,
			dataset_name="NG-10",
			ranking_function="rmse",
			ground_truth = ["2", "*", "sin", "(", "X_0", ")", "*", "cos", "(", "X_1", ")"], # noqa: F401
			original_equation="2*sin(x)*cos(y)",
			max_evaluations=100000,
			max_expression_length=50,
			success_threshold=1e-7,
			dataset_metadata=benchmark.metadata,
			result_augmenters=[],
			seed = seed
		)

        return benchmark
