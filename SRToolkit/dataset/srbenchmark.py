import copy
import os
from typing import List
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

import numpy as np

from SRToolkit.dataset import SRDataset

from SRToolkit.utils import SymbolLibrary


class SRBenchmark:
    def __init__(self, benchmark_name: str, base_dir: str, metadata: dict = None):
        """
        Initializes an instance of the SRBenchmark class.

        Args:
            benchmark_name: The name of this benchmark.
            base_dir: The directory where the datasets will be stored.
            metadata: An optional dictionary containing metadata about this benchmark. This could include information such as the name of the benchmark, a citation for the benchmark, number of datasets, etc.
        """
        self.benchmark_name = benchmark_name
        self.base_dir = base_dir
        self.datasets = {}
        self.metadata = {} if metadata is None else metadata

    def add_dataset(self, dataset_name: str, ground_truth: List[str],  symbol_library: SymbolLibrary,
                    original_equation: str = None, max_evaluations: int=-1, max_expression_length: int=-1,
                    max_constants: int=8, success_threshold: float=1e-7, constant_range: List[float]=None,
                    num_variables: int=-1, dataset_metadata: dict=None):

        """
        Adds a dataset to the benchmark.

        Args:
            dataset_name: The name of the dataset.
            ground_truth: The ground truth expression, represented as a list of tokens (strings) in the infix notation.
            symbol_library: The symbol library to use.
            original_equation: The original equation from which the ground truth expression was generated.
            max_evaluations: The maximum number of expressions to evaluate. Less than 0 means no limit.
            max_expression_length: The maximum length of the expression. Less than 0 means no limit.
            max_constants: The maximum number of constants allowed in the expression. Less than 0 means no limit.
            success_threshold: The RMSE threshold below which the experiment is considered successful.
            constant_range: A list of two floats, specifying the lower and upper bounds for the constant values.
                Default is [-5.0, 5.0]. If constant_range is None, we automatically set it to [-5.0, 5.0]
                if the symbol library contains a symbol for constants.
            num_variables: The number of variables in the expression. Default is -1, which means we don't know.
            dataset_metadata: An optional dictionary containing metadata about this dataset. This could include
                information such as the name of the dataset, a citation for the dataset, number of variables, etc.
        """
        if original_equation is None:
            original_equation = "".join(ground_truth)

        self.datasets[dataset_name] = {
            "path": self.base_dir + "/" + dataset_name + ".npy",
            "ground_truth": ground_truth,
            "original_equation": original_equation,
            "symbols": symbol_library,
            "max_evaluations": max_evaluations,
            "max_expression_length": max_expression_length,
            "max_constants": max_constants,
            "success_threshold": success_threshold,
            "constant_range": constant_range,
            "dataset_metadata": self.metadata.update(dataset_metadata),
            "num_variables": num_variables
        }

    def create_dataset(self, dataset_name: str):
        """
        Creates an instance of a dataset from the given dataset name.

        Args:
            dataset_name: The name of the dataset to create.

        Returns:
            A SRDataset instance containing the data, ground truth expression, and metadata for the given dataset.

        Raises:
            ValueError: If the dataset name is not found in the available datasets.
        """
        if dataset_name in self.datasets:
            # Check if dataset exists otherwise download it from an url
            if os.path.exists(self.datasets[dataset_name]["path"]):
                data = np.load(self.datasets[dataset_name]["path"])
            else:
                raise ValueError(f"Could not find dataset {dataset_name} at {self.datasets[dataset_name]['path']}")

            X = data[:, :-1]
            y = data[:, -1]

            return SRDataset(X, y, ground_truth=self.datasets[dataset_name]["ground_truth"],
                             original_equation=self.datasets[dataset_name]["original_equation"],
                             symbols=self.datasets[dataset_name]["symbols"],
                             max_evaluations=self.datasets[dataset_name]["max_evaluations"],
                             max_expression_length=self.datasets[dataset_name]["max_expression_length"],
                             max_constants=self.datasets[dataset_name]["max_constants"],
                             success_threshold=self.datasets[dataset_name]["success_threshold"],
                             constant_range=self.datasets[dataset_name]["constant_range"],
                             dataset_metadata=self.datasets[dataset_name]["dataset_metadata"])
        else:
            raise ValueError(f"Dataset {dataset_name} not found")

    def list_datasets(self, verbose=True, num_variables: int=-1):
        """
        Lists the available datasets.

        Args:
            verbose (bool): If True, also prints out a description of each dataset.
            num_variables (int): If not -1, only show datasets with the given number of variables.

        Returns:
            A list of dataset names.
        """
        datasets = [dataset_name for dataset_name in self.datasets if num_variables < 0 or self.datasets[dataset_name]["num_variables"] == num_variables]
        datasets = sorted(datasets, key= lambda dataset_name: (self.datasets[dataset_name]["num_variables"], dataset_name))

        if verbose:
            # TODO: Make all names be of equal length for nicer output
            for d in datasets:
                if self.datasets[d]["num_variables"] == 1:
                    variable_str = "1 variable"
                elif self.datasets[d]["num_variables"] < 1:
                    variable_str = "Amount of variables unknown"
                else:
                    variable_str = f"{self.datasets[d]['num_variables']} variables"

                print(f"{d}:\t{variable_str}, \tExpression: {self.datasets[d]['original_equation']}")
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

        Returns:
            None
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
    def feynman(dataset_directory: str):
        """
        Downloads the Feynman benchmark dataset, sets up symbol libraries, and adds predefined datasets to the benchmark.

        This method downloads the Feynman benchmark dataset from a specified URL, initializes symbol libraries for symbolic regression with varying numbers of variables, and adds multiple predefined datasets to the benchmark with their respective equations and metadata.

        Args:
            dataset_directory (str): The directory path where the benchmark dataset will be downloaded and stored.

        Returns:
            SRBenchmark: An instance of the SRBenchmark class containing the predefined datasets.
        """
        url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/feynman.zip"
        SRBenchmark.download_benchmark_data(url, dataset_directory)

        sl_1v = SymbolLibrary()
        sl_1v.add_symbol("+", symbol_type="op", precedence=0, np_fn="{} = {} + {}")
        sl_1v.add_symbol("-", symbol_type="op", precedence=0, np_fn="{} = {} - {}")
        sl_1v.add_symbol("*", symbol_type="op", precedence=1, np_fn="{} = {} * {}")
        sl_1v.add_symbol("/", symbol_type="op", precedence=1, np_fn="{} = {} / {}")
        sl_1v.add_symbol("sin", symbol_type="fn", precedence=5, np_fn="{} = np.sin({})")
        sl_1v.add_symbol("cos", symbol_type="fn", precedence=5, np_fn="{} = np.cos({})")
        sl_1v.add_symbol("exp", symbol_type="fn", precedence=5, np_fn="{} = np.exp({})")
        sl_1v.add_symbol("sqrt", symbol_type="fn", precedence=5, np_fn="{} = np.sqrt({})")
        sl_1v.add_symbol("^2", symbol_type="fn", precedence=5, np_fn="{} = np.power({}, 2)")
        sl_1v.add_symbol("^3", symbol_type="fn", precedence=5, np_fn="{} = np.power({}, 3)")
        sl_1v.add_symbol("C", "const", 5, "C[{}]")
        sl_1v.add_symbol("X_0", "var", 5, "X[:, 0]")

        sl_2v = copy.copy(sl_1v)
        sl_2v.add_symbol("X_1", "var", 5, "X[:, 1]")

        sl_3v = copy.copy(sl_2v)
        sl_3v.add_symbol("X_2", "var", 5, "X[:, 2]")

        sl_4v = copy.copy(sl_3v)
        sl_4v.add_symbol("X_3", "var", 5, "X[:, 3]")

        sl_5v = copy.copy(sl_4v)
        sl_5v.add_symbol("X_4", "var", 5, "X[:, 4]")

        sl_6v = copy.copy(sl_5v)
        sl_6v.add_symbol("X_5", "var", 5, "X[:, 5]")

        sl_8v = copy.copy(sl_6v)
        sl_8v.add_symbol("X_6", "var", 5, "X[:, 6]")
        sl_8v.add_symbol("X_7", "var", 5, "X[:, 7]")

        sl_9v = copy.copy(sl_8v)
        sl_9v.add_symbol("X_8", "var", 5, "X[:, 8]")

        benchmark = SRBenchmark("feynman", dataset_directory)
        benchmark.add_dataset("I.16.6", ["(", "X_2", "+", "X_1", ")", "/", "(", "1", "+", "X_2", "*", "X_1", "/", "X_1", "^2",")"], sl_3v, original_equation="v1 = (u+v)/(1+u*v/c^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.15.4", ["(", "0", "-", "X_0", ")", "*", "X_1", "*", "cos", "(", "X_2", ")"], sl_3v, original_equation="E_n = -mom*B*cos(theta)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.27.16", ["X_0", "*", "X_1", "*", "X_2", "^2"], sl_3v, original_equation="flux = epsilon*c*Ef^2", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.11.19", ["X_0", "*", "X_3", "+", "X_1", "*", "X_4", "+", "X_2", "*", "X_5"], sl_6v, original_equation="A = x1*y1+x2*y2+x3*y3", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=6,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.15.3x", ["(", "X_0", "-", "X_1", "*", "X_3", ")", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"], sl_4v, original_equation="x1 = (x-u*t)/sqrt(1-u^2/c^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.10.7", ["X_0", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"], sl_3v, original_equation="m = m_0/sqrt(1-v^2/c^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.9.18", ["X_2", "*", "X_0", "*", "X_1", "/", "(", "(", "X_4", "-", "X_3", ")", "^2", "+", "(", "X_6", "-", "X_5", ")", "^2", "+", "(", "X_8", "-", "X_7", ")", "^2",")"], sl_9v, original_equation="F = G*m1*m2/((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=9,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.15.3t", ["(", "X_3", "-", "X_2", "*", "X_0", "/", "X_1", "^2", ")", "/", "sqrt", "(", "1", "-", "X_2", "^2", "/", "X_1", "^2", ")"], sl_4v, original_equation="t1 = (t-u*x/c^2)/sqrt(1-u^2/c^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.36.38", ["X_0", "*", "X_1", "/", "(", "X_2", "*", "X_3",")", "+", "(", "X_0", "*", "X_4", ")", "/", "(", "X_5", "*", "X_6", "^2", "*", "X_2", "*", "X_3" ")", "*", "X_7"], sl_8v, original_equation="f = mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=8,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.43.43", ["1", "/", "(", "X_0", "-", "1", ")", "*", "X_1", "*", "X_3", "/", "X_2"], sl_4v, original_equation="kappa = 1/(gamma-1)*kb*v/A", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.15.5", ["(", "0", "-", "X_0", ")", "*", "X_1", "*", "cos", "(", "X_2", ")"], sl_3v, original_equation="E_n = -p_d*Ef*cos(theta)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.37.4", ["X_0", "+", "X_1", "+", "2", "*", "sqrt", "(", "X_0", "*", "X_1", ")", "*", "cos", "(", "X_2", ")"], sl_3v, original_equation="Int = I1+I2+2*sqrt(I1*I2)*cos(delta)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.6.11", ["1", "/", "(", "4", "*", "pi", "*", "X_0", ")", "*", "X_1", "*", "cos", "(", "X_2", ")", "/", "X_3", "^2"], sl_4v, original_equation="Volt = 1/(4*pi*epsilon)*p_d*cos(theta)/r^2", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.7.38", ["2", "*", "X_0", "*", "X_1", "/", "(", "X_2", "/", "(", "2", "*", "pi", ")", ")"], sl_3v, original_equation="omega = 2*mom*B/(h/(2*pi))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.34.2a", ["X_0", "*", "X_1", "/", "(", "2", "*", "pi", "*", "X_2", ")"], sl_3v, original_equation="l = q*v/(2*pi*r)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.13.23", ["X_0", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"], sl_3v, original_equation="rho_c = rho_c_0/sqrt(1-v^2/c^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.29.4", ["X_0", "/", "X_1"], sl_2v, original_equation="k = omega/c", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.38.12", ["4", "*", "pi", "*", "X_3", "*", "(", "X_2", "/", "(", "2", "*", "pi", ")", ")", "^2", "/", "(", "X_0", "*", "X_1", "^2", ")"], sl_4v, original_equation="r = 4*pi*epsilon*(h/(2*pi))^2/(m*q^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.15.27", ["2", "*", "pi", "*", "X_0", "/", "(", "X_1", "*", "X_2", ")"], sl_3v, original_equation="k = 2*pi*alpha/(n*d)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.41.16", ["X_2", "/", "(", "2", "*", "pi", ")", "*", "X_0", "^3", "/", "(", "pi", "^2", "*", "X_4", "^2", "*", "(", "exp", "(", "(", "X_2", "/", "(", "2", "*", "pi", ")", ")", "*", "X_0", "/", "(", "X_3", "*", "X_1", ")", ")", "-", "1", ")", ")"], sl_5v, original_equation="L_rad = h/(2*pi)*omega^3/(pi^2*c^2*(exp((h/(2*pi))*omega/(kb*T))-1))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.48.20", ["X_0", "*", "X_2", "^2", "/", "sqrt", "(", "1", "-", "X_1", "^2", "/", "X_2", "^2", ")"], sl_3v, original_equation="E_n = m*c^2/sqrt(1-v^2/c^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.11.20", ["X_0", "*", "X_1", "^2", "*", "X_2", "/", "(", "3", "*", "X_3", "*", "X_4", ")"], sl_5v, original_equation="Pol = n_rho*p_d^2*Ef/(3*kb*T)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.25.13", ["X_0", "/", "X_1"], sl_2v, original_equation="Volt = q/C", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.15.12", ["2", "*", "X_0", "*", "cos", "(", "X_1", "*", "X_2", ")"], sl_3v, original_equation="E_n = 2*U*(1-cos(k*d))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.24.6", ["0.25", "*", "X_0", "*", "(", "X_1", "^2", "+", "X_2", "^2", ")", "*", "X_3", "^2"], sl_4v, original_equation="E_n = 1/2*m*(omega^2+omega_0^2)*1/2*x^2", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.34.27", ["(", "X_1", "/", "(", "2", "*", "pi", ")", ")", "*", "X_0"], sl_2v, original_equation="E_n =(h/(2*pi))*omega", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.43.31", ["X_0", "*", "X_2", "*", "X_1"], sl_3v, original_equation="D = mob*kb*T", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.29.16", ["sqrt", "(", "X_0", "^2", "+", "X_1", "^2", "-", "2", "*", "X_0", "*", "X_1", "*", "cos", "(", "X_2", "-", "X_3", ")", ")"], sl_4v, original_equation="x = sqrt(x1^2+x2^2-2*x1*x2*cos(theta1-theta2))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.18.4", ["(","X_0", "*", "X_2", "+", "X_1", "*", "X_3", ")", "/", "(", "X_0", "+", "X_1", ")"], sl_4v, original_equation="r = (m1*r1+m2*r2)/(m1+m2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.6.15a", ["X_1", "*", "(", "4", "*", "pi", "*", "X_0", ")", "*", "3", "*", "X_5", "/", "(", "X_2", "^2", "*", "X_2", "^3", ")", "*", "sqrt", "(", "X_3", "^2", "+", "X_4", "^2", ")"], sl_6v, original_equation="Ef = p_d/(4*pi*epsilon)*3*z/r^5*sqrt(x^2+y^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=6,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.30.3", ["X_0", "*", "sin", "(", "X_2", "*", "X_1", "/", "2", ")", "^2", "/", "sin", "(", "X_1", "/", "2", ")", "^2"], sl_3v, original_equation="Int = Int_0*sin(n*theta/2)^2/sin(theta/2)^2", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.9.52", ["(", "X_0", "*", "X_1", "*", "X_2", "/", "(", "X_3", "/", "(", "2", "*", "pi", ")", ")", ")", "*", "sin", "(", "(", "X_4", "-", "X_5", ")", "*", "X_2", "/", "2", ")", "^2", "/", "(", "(", "X_4", "-", "X_5", ")", "*", "X_2", "/", "2", ")","^2"], sl_6v, original_equation="prob = (p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)^2/((omega-omega_0)*t/2)^2", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=6,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.34.2", ["X_0", "*", "X_1", "*", "X_2", "/", "2"], sl_3v, original_equation="mom = q*v*r/2", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.39.11", ["1", "/", "(", "X_0", "-", "1", ")", "*", "X_1", "*", "X_2"], sl_3v, original_equation="E_n = 1/(gamma-1)*pr*V", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.11.28", ["1", "+", "X_0", "*", "X_1", "/", "(", "1", "-", "(", "X_0", "*", "X_1", "/", "3", ")", ")"], sl_2v, original_equation="theta = 1+n*alpha/(1-(n*alpha/3))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.3.24", ["X_0", "/", "(", "4", "*", "pi", "*", "X_1", "^2", ")"], sl_2v, original_equation="flux = Pwr/(4*pi*r^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.24.17", ["sqrt", "(", "X_0", "^2", "/", "X_1", "^2", "-", "pi", "^2", "/", "X_2", "^2", ")"], sl_3v, original_equation="k = sqrt(omega^2/c^2-pi^2/d^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.13.17", ["1", "/", "(", "4", "*", "pi", "*", "X_0", "*", "X_1", "^2", ")", "*", "2", "*", "X_2", "/", "X_3"], sl_4v, original_equation="B = 1/(4*pi*epsilon*c^2)*2*I/r", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.12.5", ["X_0", "*", "X_1"], sl_2v, original_equation="F = q2*Ef", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.35.18", [], sl_5v, original_equation="n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.34.11", [], sl_4v, original_equation="g_*q*B/(2*m)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.34.29a", [], sl_3v, original_equation="q*h/(4*pi*m)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.32.17", [], sl_6v, original_equation="(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=6,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.35.21", [], sl_5v, original_equation="n_rho*mom*tanh(mom*B/(kb*T))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.44.4", [], sl_5v, original_equation="n*kb*T*ln(V2/V1)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.4.32", [], sl_4v, original_equation="1/(exp((h/(2*pi))*omega/(kb*T))-1)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.10.9", [], sl_3v, original_equation="sigma_den/epsilon*1/(1+chi)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.38.3", [], sl_4v, original_equation="Y*A*x/d", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.6.2b", [], sl_3v, original_equation="exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.8.31", [], sl_2v, original_equation="epsilon*Ef**2/2", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.6.2a", [], sl_1v, original_equation="exp(-theta**2/2)/sqrt(2*pi)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.12.43", [], sl_2v, original_equation="n*(h/(2*pi))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.17.37", [], sl_3v, original_equation="beta*(1+alpha*cos(theta))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.10.19", [], sl_4v, original_equation="mom*sqrt(Bx**2+By**2+Bz**2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.11.7", [], sl_6v, original_equation="n_0*(1+p_d*Ef*cos(theta)/(kb*T))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=6,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.39.1", [], sl_2v, original_equation="3/2*pr*V", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.37.1", [], sl_3v, original_equation="mom*(1+chi)*B", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.12.4", [], sl_3v, original_equation="q1*r/(4*pi*epsilon*r**3)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.27.18", [], sl_2v, original_equation="epsilon*Ef**2", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.12.2", [], sl_4v, original_equation="q1*q2*r/(4*pi*epsilon*r**3)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.13.18", [], sl_4v, original_equation="2*E_n*d**2*k/(h/(2*pi))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.11.3", [], sl_5v, original_equation="q*Ef/(m*(omega_0**2-omega**2))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.40.1", [], sl_6v, original_equation="n_0*exp(-m*g*x/(kb*T))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=6,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.21.20", [], sl_4v, original_equation="-rho_c_0*q*A_vec/m", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.43.16", [], sl_4v, original_equation="mu_drift*q*Volt/d", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.15.10", [], sl_3v, original_equation="m_0*v/sqrt(1-v**2/c**2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.30.5", [], sl_3v, original_equation="arcsin(lambd/(n*d))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.50.26", [], sl_4v, original_equation="x1*(cos(omega*t)+alpha*cos(omega*t)**2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.12.11", [], sl_5v, original_equation="q*(Ef+B*v*sin(theta))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.6.2", [], sl_2v, original_equation="exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.14.4", [], sl_2v, original_equation="1/2*k_spring*x**2", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.47.23", [], sl_3v, original_equation="sqrt(gamma*pr/rho)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.8.7", [], sl_3v, original_equation="3/5*q**2/(4*pi*epsilon*d)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.15.14", [], sl_3v, original_equation="(h/(2*pi))**2/(2*E_n*d**2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.34.14", [], sl_3v, original_equation="(1+v/c)/sqrt(1-v**2/c**2)*omega_0", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.8.54", [], sl_3v, original_equation="sin(E_n*t/(h/(2*pi)))**2", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.26.2", [], sl_2v, original_equation="arcsin(n*sin(theta2))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.19.51", [], sl_5v, original_equation="-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.4.33", [], sl_4v, original_equation="(h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.34.1", [], sl_3v, original_equation="omega_0/(1-v/c)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.11.27", [], sl_4v, original_equation="n*alpha/(1-(n*alpha/3))*epsilon*Ef", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.13.34", [], sl_3v, original_equation="rho_c_0*v/sqrt(1-v**2/c**2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.4.23", [], sl_3v, original_equation="q/(4*pi*epsilon*r)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.32.5", [], sl_4v, original_equation="q**2*a**2/(6*pi*epsilon*c**3)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.13.12", [], sl_5v, original_equation="G*m1*m2*(1/r2-1/r1)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.2.42", [], sl_5v, original_equation="kappa*(T2-T1)*A/d", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.27.6", [], sl_3v, original_equation="1/(1/d1+n/d2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("III.14.14", [], sl_5v, original_equation="I_0*(exp(q*Volt/(kb*T))-1)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.18.12", [], sl_3v, original_equation="r*F*sin(theta)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.18.14", [], sl_4v, original_equation="m*r*v*sin(theta)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.21.32", [], sl_5v, original_equation="q/(4*pi*epsilon*r*(1-v/c))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.38.14", [], sl_2v, original_equation="Y/(2*(1+sigma))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.34.8", [], sl_4v, original_equation="q*v*B/p", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.8.14", [], sl_4v, original_equation="sqrt((x2-x1)**2+(y2-y1)**2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.6.15b", [], sl_4v, original_equation="p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.12.1", [], sl_2v, original_equation="mu*Nn", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("II.34.29b", [], sl_5v, original_equation="g_*mom*B*Jz/(h/(2*pi))", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=5,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.13.4", [], sl_4v, original_equation="1/2*m*(v**2+u**2+w**2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.39.22", [], sl_4v, original_equation="n*kb*T/V", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=4,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])
        benchmark.add_dataset("I.14.3", [], sl_3v, original_equation="m*g*z", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=3,
                              dataset_metadata=benchmark.metadata, constant_range=[-5.0, 5.0])

        return benchmark

    @staticmethod
    def nguyen(dataset_directory: str):
        """
        Downloads and initializes the Nguyen benchmark datasets for symbolic regression.

        This method downloads the Nguyen symbolic regression benchmark datasets from a specified URL
        and initializes a set of datasets using a provided dataset directory. It creates two symbol libraries
        for equations with one variable and two variables, respectively, and populates the benchmark with various
        Nguyen equations, each represented with its symbolic tokens and associated symbol library.

        Args:
            dataset_directory (str): The directory where the benchmark datasets will be stored and accessed.

        Returns:
            SRBenchmark: An initialized SRBenchmark instance containing the Nguyen datasets.
        """
        url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/nguyen.zip"
        SRBenchmark.download_benchmark_data(url, dataset_directory)
        # we create a SymbolLibrary with 1 and with 2 variables
        # Each library contains +, -, *, /, sin, cos, exp, log, sqrt, ^2, ^3
        sl_1v = SymbolLibrary()
        sl_1v.add_symbol("+", symbol_type="op", precedence=0, np_fn="{} = {} + {}")
        sl_1v.add_symbol("-", symbol_type="op", precedence=0, np_fn="{} = {} - {}")
        sl_1v.add_symbol("*", symbol_type="op", precedence=1, np_fn="{} = {} * {}")
        sl_1v.add_symbol("/", symbol_type="op", precedence=1, np_fn="{} = {} / {}")
        sl_1v.add_symbol("sin", symbol_type="fn", precedence=5, np_fn="{} = np.sin({})")
        sl_1v.add_symbol("cos", symbol_type="fn", precedence=5, np_fn="{} = np.cos({})")
        sl_1v.add_symbol("exp", symbol_type="fn", precedence=5, np_fn="{} = np.exp({})")
        sl_1v.add_symbol("log", symbol_type="fn", precedence=5, np_fn="{} = np.log10({})")
        sl_1v.add_symbol("sqrt", symbol_type="fn", precedence=5, np_fn="{} = np.sqrt({})")
        sl_1v.add_symbol("^2", symbol_type="fn", precedence=5, np_fn="{} = np.power({}, 2)")
        sl_1v.add_symbol("^3", symbol_type="fn", precedence=5, np_fn="{} = np.power({}, 3)")
        sl_1v.add_symbol("X_0", "var", 5, "X[:, 0]")

        sl_2v = copy.copy(sl_1v)
        sl_2v.add_symbol("X_1", "var", 5, "X[:, 1]")

        # Add datasets to the benchmark
        benchmark = SRBenchmark("Nguyen", dataset_directory)
        benchmark.add_dataset("NG-1", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], sl_1v,
                              original_equation="x+x^2+x^3", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-2", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3", "+", "X_0","*", "X_0", "^3"], sl_1v,
                              original_equation="x+x^2+x^3+x^4", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-3", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3", "+", "X_0","*", "X_0", "^3", "+", "X_0","^2", "*", "X_0", "^3"], sl_1v,
                              original_equation="x+x^2+x^3+x^4+x^5", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-4", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3", "+", "X_0","*", "X_0", "^3", "+", "X_0","^2", "*", "X_0", "^3", "+", "X_0","^3", "*", "X_0", "^3"], sl_1v,
                              original_equation="x+x^2+x^3+x^4+x^5+x^6", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-5", ["sin", "(", "X_0", "^2", ")", "*", "cos", "(", "X_0", ")", "-", "1"], sl_1v,
                              original_equation="sin(x^2)*cos(x)-1", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-6", ["sin", "(", "X_0", ")", "+", "sin", "(", "X_0", "+", "X_0", "^2", ")"], sl_1v,
                              original_equation="sin(x)+sin(x+x^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-7", ["log", "(", "1", "+", "X_0", ")", "+", "log", "(", "1", "+", "X_0", "^2", ")"], sl_1v,
                              original_equation="log(1+x)+log(1+x^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-8", ["sqrt", "(", "X_0", ")"], sl_1v,
                              original_equation="sqrt(x)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-9", ["sin", "(", "X_0", ")", "+", "sin", "(", "X_1", "^2", ")"], sl_2v,
                              original_equation="sin(x)+sin(y^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-10", ["2", "*", "sin", "(", "X_0", ")", "*", "cos", "(", "X_1", ")"], sl_2v,
                              original_equation="2*sin(x)*cos(y)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata)

        return benchmark


if __name__ == '__main__':
    benchmark = SRBenchmark.feynman("../../data/feynman")
    benchmark.list_datasets()
