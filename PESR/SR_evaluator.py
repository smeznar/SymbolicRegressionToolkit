from symbol_library import SymbolLibrary
from parameter_estimator import ParameterEstimator

class SR_evaluator:
    def __init__(self, X, y, metadata=None, symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()):
        self.models = {}
        self.metadata = metadata
        self.symbol_library = symbol_library
        self.parameter_estimator = ParameterEstimator(X, y, symbol_library=symbol_library)

    def evaluate_expr(self, expr: list[str]):
        expr_str = ''.join(expr)
        # TODO: Popravi, ko se odlocis kako bos shranjeval izraze
        if expr_str in self.models:
            return self.models[expr_str]
        else:
            rmse, _ = self.parameter_estimator.estimate_parameters(expr)
            self.models[expr_str] = rmse
            return rmse