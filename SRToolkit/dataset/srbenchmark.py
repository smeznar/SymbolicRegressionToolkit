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
        self.benchmark_name = benchmark_name
        self.base_dir = base_dir
        self.datasets = {}
        self.metadata = {} if metadata is None else metadata

    def add_dataset(self, dataset_name: str, ground_truth: List[str],  symbol_library: SymbolLibrary,
                    original_equation: str = None, max_evaluations: int=-1, max_expression_length: int=-1,
                    max_constants: int=8, success_threshold: float=1e-7, constant_range: List[float]=None,
                    num_variables: int=-1, dataset_metadata: dict=None):

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
        if dataset_name in self.datasets:
            # Check if dataset exists otherwise download it from an url
            if os.path.exists(self.datasets[dataset_name]["path"]):
                data = np.load(self.datasets[dataset_name]["path"] + ".npy")
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

    @staticmethod
    def download_benchmark_data(url, directory_path):
        # Check if directory_path exist
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
        url = "https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/master/data/feynman.zip"

    @staticmethod
    def nguyen(dataset_directory: str):
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
        sl_1v.add_symbol("ln", symbol_type="fn", precedence=5, np_fn="{} = np.ln({})")
        sl_1v.add_symbol("sqrt", symbol_type="fn", precedence=5, np_fn="{} = np.sqrt({})")
        sl_1v.add_symbol("^2", symbol_type="fn", precedence=5, np_fn="{} = np.pow({}, 2)")
        sl_1v.add_symbol("^3", symbol_type="fn", precedence=5, np_fn="{} = np.pow({}, 3)")
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
    # benchmark = SRBenchmark.nguyen("../../data/nguyen")
    # a = 0
    from SRToolkit.utils.expression_compiler import expr_to_executable_function

    equations = [["X_0", "+", "X_0", "^2", "+", "X_0", "^3"],
                 ["X_0", "+", "X_0", "^2", "+", "X_0", "^3", "+", "X_0", "*", "X_0", "^3"],
                 ["X_0", "+", "X_0", "^2", "+", "X_0", "^3", "+", "X_0", "*", "X_0", "^3", "+", "X_0", "^2", "*", "X_0", "^3"],
                 ["X_0", "+", "X_0", "^2", "+", "X_0", "^3", "+", "X_0", "*", "X_0", "^3", "+", "X_0", "^2", "*", "X_0", "^3", "+", "X_0", "^3", "*", "X_0", "^3"],
                 ["sin", "(", "X_0", "^2", ")", "*", "cos", "(", "X_0", ")", "-", "1"],
                 ["sin", "(", "X_0", ")", "+", "sin", "(", "X_0", "+", "X_0", "^2", ")"],
                 ["log", "(", "1", "+", "X_0", ")", "+", "log", "(", "1", "+", "X_0", "^2", ")"],
                 ["sqrt", "(", "X_0", ")"],
                 ["sin", "(", "X_0", ")", "+", "sin", "(", "X_1", "^2", ")"],
                 ["2", "*", "sin", "(", "X_0", ")", "*", "cos", "(", "X_1", ")"]]

    bounds = [(-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20), (1, 100), (0, 100), (-20, 20), (-20, 20)]

    for i, eq in enumerate(equations):
        exec_fun = expr_to_executable_function(eq)
        if i < 8:
            x = np.random.random((10000, 1)) * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        else:
            x = np.random.random((10000, 2)) * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        y = exec_fun(x, None)

        np.save(f"../../data/Nguyen/NG-{i+1}.npy", np.concatenate([x, y[:, np.newaxis]], axis=1))
