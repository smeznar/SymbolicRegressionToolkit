"""
This module contains the ResultAugmenter class and the result augmentation implementations that inherit from it.
"""
from typing import List

import numpy as np

from SRToolkit.utils import simplify


class ResultAugmenter:
    def __init__(self):
        """
        Generic class that defines the interface for result augmentation.
        """
        pass

    def augment_results(self, results: dict, models: List[dict], evaluator: "SR_evaluator") -> dict:
        """
        Augments the results dictionary with additional information. The model variable contains all models, for only
        top models, results["top_models"] should be used.

        Args:
            results: The dictionary containing the results to augment.
            models: The list of models that were evaluated. Variable models[i]["expr"] contains the expression model i,
                models[i]["error"] contains the error for model i, ...
            evaluator: The evaluator used to evaluate the models.

        Returns:
            The augmented results dictionary.
        """
        pass


class ExpressionSimplifier(ResultAugmenter):
    def __init__(self, only_best_expression: bool=False, verbose: bool=False):
        """
        Simplifies the expressions inside the results dictionary if possible.

        Args:
            only_best_expression: If True, only the best expression is simplified. If False, all top expressions are
                simplified.
            verbose: If True, warns the user if simplification fails for a given expression.
        """
        super().__init__()
        self.only_best_expression = only_best_expression
        self.verbose = verbose

    def augment_results(self, results: dict, models: List[dict], evaluator: "SR_evaluator") -> dict:
        """
        Simplifies the expressions inside the results dictionary if possible.

        Args:
            results: The dictionary containing the results to augment.
            models: The list of models that were evaluated. Variable models[i]["expr"] contains the expression model i, ...
            evaluator: The evaluator used to evaluate the models.

        Returns:
            The augmented results dictionary. The results dictionary contains an additional key "simplified_best_expr"
            if simplification was successful for the best expression, and similarly keys "simplified_expr" inside the
            top_models list if only_best_expression is False.
        """
        try:
            simplified_expr = simplify(models[0]["expr"], evaluator.symbol_library)
            results["simplified_best_expr"] = "".join(simplified_expr)
        except Exception as e:
            if self.verbose:
                print(f"Unable to simplify {results['best_expr']}: {e}")

        for model in results["top_models"]:
            try:
                simplified_expr = simplify(model["expr"], evaluator.symbol_library)
                model["simplified_expr"] = "".join(simplified_expr)
            except Exception as e:
                if self.verbose:
                    print(f"Unable to simplify {model['expr']}: {e}")

        return results


class RMSE(ResultAugmenter):
    def __init__(self, evaluator: "SR_evaluator"):
        """
        Computes the RMSE for the top models in the results dictionary.

        Args:
             evaluator: The evaluator used to evaluate the models (e.g., evaluator defined with test set data)
        """
        super().__init__()
        self.evaluator = evaluator

    def augment_results(self, results: dict, models: List[dict], evaluator: "SR_evaluator") -> dict:
        """
        Computes the RMSE for the top models in the results dictionary.

        Args:
            results: The dictionary containing the results to augment.
            models: The list of models that were evaluated. Variable models[i]["expr"] contains the expression model i, ...
            evaluator: The evaluator used to evaluate the models.

        Returns:
            The augmented results dictionary. The results dictionary contains an additional key "best_expr_rmse" with the
            RMSE of the best expression, and keys "rmse" and "parameters_rmse" for each of the top_models inside the
            results["top_models"] list.
        """
        expr = models[0]["expr"]
        error = self.evaluator.evaluate_expr(expr)
        results["min_rmse"] = error
        for model in results["top_models"]:
            error = self.evaluator.evaluate_expr(model["expr"])
            model["rmse"] = error
            model["parameters_rmse"] = self.evaluator.models["".join(model["expr"])]["parameters"]
        return results
