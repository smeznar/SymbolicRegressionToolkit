import copy
import os
from typing import List

import numpy as np

from SRToolkit.dataset import SRDataset
from SRToolkit.utils import SymbolLibrary


class SRBenchmark:
    def __init__(self, benchmark_name: str, base_dir: str, base_url: str = None, metadata: dict = None):
        self.benchmark_name = benchmark_name
        self.base_dir = base_dir
        self.base_url = base_url
        self.datasets = {}
        self.metadata = metadata

    def add_dataset(self, dataset_name: str, ground_truth: List[str],  symbol_library: SymbolLibrary,
                    original_equation: str = None, max_evaluations: int=-1, max_expression_length: int=-1,
                    max_constants: int=8, success_threshold: float=1e-7, constant_range: List[float]=None,
                    num_variables: int=-1, dataset_metadata: dict=None):

        if original_equation is None:
            original_equation = "".join(ground_truth)

        group = "other"

        if num_variables > 0:
            group = str(num_variables)

        if group not in self.datasets:
            self.datasets[group] = {}

        self.datasets[group][dataset_name] = {
            "path": os.path.join(self.base_dir, dataset_name),
            "ground_truth": ground_truth,
            "original_equation": original_equation,
            "symbols": symbol_library,
            "max_evaluations": max_evaluations,
            "max_expression_length": max_expression_length,
            "max_constants": max_constants,
            "success_threshold": success_threshold,
            "constant_range": constant_range,
            "dataset_metadata": dataset_metadata
        }

    def create_dataset(self, dataset_name: str):
        if dataset_name in self.datasets:
            # Check if dataset exists otherwise download it from an url
            if os.path.exists(self.datasets[dataset_name]["path"]):
                X = np.load(self.datasets[dataset_name]["path"] + "/X.npy")
                y = np.load(self.datasets[dataset_name]["path"] + "/y.npy")
            elif self.base_url is not None:
                # Download data from the url
                X = np.load(self.base_url + self.datasets[dataset_name]["path"] + "/X.npy")
                y = np.load(self.base_url + self.datasets[dataset_name]["path"] + "/y.npy")
            else:
                raise ValueError(f"Could not find dataset {dataset_name} at {self.datasets[dataset_name]['path']}"
                                 f"base_url is None")

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
    def nguyen(dataset_directory: str):
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
        sl_1v.add_symbol(f"X_0", "var", 5, "X[:, 0]")

        sl_2v = copy.copy(sl_1v)
        sl_2v.add_symbol(f"X_1", "var", 5, "X[:, 1]")

        benchmark = SRBenchmark("Nguyen", dataset_directory)
        benchmark.add_dataset("NG-1", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], sl_1v,
                              original_equation="x+x^2+x^3", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-2", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], sl_1v,
                              original_equation="x+x^2+x^3", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-3", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], sl_1v,
                              original_equation="x+x^2+x^3", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-4", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], sl_1v,
                              original_equation="x+x^2+x^3", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-5", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], sl_1v,
                              original_equation="x+x^2+x^3", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-6", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], sl_1v,
                              original_equation="x+x^2+x^3", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-7", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], sl_1v,
                              original_equation="ln()", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-8", ["sqrt", "(", "X_0", ")"], sl_1v,
                              original_equation="sqrt(x)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=1,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-9", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], sl_1v,
                              original_equation="sin(x)+sin(y^2)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata)
        benchmark.add_dataset("NG-10", ["X_0", "+", "X_0", "^2", "+", "X_0", "^3"], sl_1v,
                              original_equation="2*sin(x)*cos(y)", max_evaluations=100000,
                              max_expression_length=50, success_threshold=1e-7, num_variables=2,
                              dataset_metadata=benchmark.metadata)

