import numpy as np

from .symbol_library import SymbolLibrary
from .parameter_estimator import ParameterEstimator


class SR_evaluator:
    def __init__(self, X, y, metadata=None, symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()):
        self.models = {}
        self.metadata = metadata
        self.symbol_library = symbol_library
        self.total_expressions = 0
        self.parameter_estimator = ParameterEstimator(X, y, symbol_library=symbol_library)

    def evaluate_expr(self, expr: list[str]):
        self.total_expressions += 1

        expr_str = ''.join(expr)
        if expr_str in self.models:
            return self.models[expr_str]["rmse"]
        else:
            rmse, parameters = self.parameter_estimator.estimate_parameters(expr)
            self.models[expr_str] = {"rmse": rmse, "parameters": parameters, "expr": expr}
            return rmse

    def get_results(self, top_k=20):
        # print(self.parameter_estimator.stats)

        if top_k > len(self.models) or top_k < 0:
            top_k = len(self.models)

        models = list(self.models.values())
        best_indices = np.argsort([v["rmse"] for v in models])

        results = {"metadata": self.metadata,
                   "min_rmse": models[best_indices[0]]["rmse"],
                   "best_expr": "".join(models[best_indices[0]]["expr"]),
                   "num_evaluated": len(models),
                   "total_expressions": self.total_expressions,
                   "results": list()}

        for i in best_indices[:top_k]:
            results["results"].append(models[i])

        return results