"""
This module contains the ResultAugmenter class and the result augmentation implementations that inherit from it.
"""

from typing import List, Optional

import numpy as np

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import simplify, tokens_to_tree


class ResultAugmenter:
    def __init__(self):
        """
        Generic class that defines the interface for result augmentation. For examples, see the implementations of
        this class.
        """
        pass

    def augment_results(
        self,
        results: dict,
        models: List[dict],
        evaluator: "SR_evaluator",  # noqa: F821
    ) -> dict:
        """
        Augments the results dictionary with additional information. The model variable contains all models, for only
        top models, results["top_models"] should be used.

        Args:
            results: The dictionary containing the results to augment.
            models: A list of dictionaries describing the performance of expressions using the base ranking function.
                Keyword expr contains the expression, error contains the error of the expression. The list is sorted
                by error.
            evaluator: The evaluator used to evaluate the models.

        Returns:
            The augmented results dictionary.
        """
        pass

    def to_dict(self, base_path, name) -> dict:
        """
        Transforms the augmenter into a dictionary. This is used for saving the augmenter to disk.

        Args:
            base_path: The base path used for saving the data inside the augmenter, if needed.
            name: The name/identifier used by the augmenter for saving.

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """

    @staticmethod
    def from_dict(data: dict, augmenter_map: Optional[dict] = None) -> "ResultAugmenter":
        """
        Creates an instance of the ResultAugmenter class from the dictionary with the relevant data.

        Args:
            data: the dictionary containing the data needed to recreate the augmenter.
            augmenter_map: A dictionary mapping augmenter names to their classes.

        Returns:
            An instance of the ResultAugmenter class with the same configuration as in the data dictionary.
        """
        pass


class ExpressionToLatex(ResultAugmenter):
    def __init__(self, only_best_expression: bool = False, verbose: bool = False):
        """
        Transforms the expressions inside the results dictionary into LaTeX strings.

        Args:
            only_best_expression: If True, only the best expression is transformed. If False, all top expressions are
                transformed.
            verbose: If True, warns the user if LaTeX conversion fails for a given expression.
        """
        super().__init__()
        self.only_best_expression = only_best_expression
        self.verbose = verbose

    def augment_results(
        self,
        results: dict,
        models: List[dict],
        evaluator: "SR_evaluator",  # noqa: F821
    ) -> dict:
        """
        Transforms the expressions inside the results dictionary into LaTeX strings.

        Args:
            results: The dictionary containing the results to augment.
            models: A list of dictionaries describing the performance of expressions using the base ranking function.
                Keyword expr contains the expression, error contains the error of the expression. The list is sorted
                by error.
            evaluator: The evaluator used to evaluate the models.

        Returns:
            The augmented results dictionary. The results dictionary contains an additional key "best_expr_latex" with
                the LaTeX representation of the best expression, and similarly keys "expr_latex" for expressions inside
                the top_models list if only_best_expression is False.
        """
        try:
            results["best_expr_latex"] = tokens_to_tree(
                models[0]["expr"], evaluator.symbol_library
            ).to_latex(evaluator.symbol_library)
        except Exception as e:
            if self.verbose:
                print(f"Unable to convert best expression to LaTeX: {e}")
        if not self.only_best_expression:
            for model in results["top_models"]:
                try:
                    model["expr_latex"] = tokens_to_tree(
                        model["expr"], evaluator.symbol_library
                    ).to_latex(evaluator.symbol_library)
                except Exception as e:
                    if self.verbose:
                        print(
                            f"Unable to convert expression {''.join(model['expr'])} to LaTeX: {e}"
                        )

        return results

    def to_dict(self, base_path, name) -> dict:
        """
        Creates a dictionary representation of the ExpressionToLatex augmenter.

        Args:
            base_path: Unused and ignored
            name: Unused and ignored

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        return {"type": "ExpressionToLatex", "only_best_expression": self.only_best_expression, "verbose": self.verbose}

    @staticmethod
    def from_dict(data: dict, augmenter_map: Optional[dict] = None) -> "ExpressionToLatex":
        """
        Creates an instance of the ExpressionToLatex augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.
            augmenter_map: Unused and ignored

        Returns:
            An instance of the ExpressionToLatex augmenter.
        """
        return ExpressionToLatex(only_best_expression=data["only_best_expression"], verbose=data["verbose"])


class ExpressionSimplifier(ResultAugmenter):
    def __init__(self, only_best_expression: bool = False, verbose: bool = False):
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

    def augment_results(
        self,
        results: dict,
        models: List[dict],
        evaluator: "SR_evaluator",  # noqa: F821
    ) -> dict:
        """
        Simplifies the expressions inside the results dictionary if possible.

        Args:
            results: The dictionary containing the results to augment.
            models: A list of dictionaries describing the performance of expressions using the base ranking function.
                Keyword expr contains the expression, error contains the error of the expression. The list is sorted
                by error.
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

    def to_dict(self, base_path, name) -> dict:
        """
        Creates a dictionary representation of the ExpressionSimplifier augmenter.

        Args:
            base_path: Unused and ignored
            name: Unused and ignored

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        return {"type": "ExpressionSimplifier", "only_best_expression": self.only_best_expression, "verbose": self.verbose}

    @staticmethod
    def from_dict(data: dict, augmenter_map: Optional[dict] = None) -> "ExpressionSimplifier":
        """
        Creates an instance of the ExpressionSimplifier augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.
            augmenter_map: Unused and ignored
        Returns:
            An instance of the ExpressionSimplifier augmenter.
        """
        return ExpressionSimplifier(only_best_expression=data["only_best_expression"], verbose=data["verbose"])


class RMSE(ResultAugmenter):
    def __init__(self, evaluator: "SR_evaluator"):  # noqa: F821
        """
        Computes the RMSE for the top models in the results dictionary.

        Args:
            evaluator: The evaluator used to evaluate the models (e.g., evaluator defined with test set data). This
                evaluator must be initialized with ranking_function = "rmse"

        Raises:
            Exception: If the evaluator is not initialized with ranking_function = "rmse" or if y in the evaluator is None.
        """
        super().__init__()
        self.evaluator = evaluator
        if self.evaluator.ranking_function != "rmse":
            raise Exception(
                "[RMSE augmenter] Ranking function of the evaluator must be set to 'rmse' to compute RMSE."
            )
        if self.evaluator.y is None:
            raise Exception(
                "[RMSE augmenter] y in the evaluator must not be None to compute RMSE."
            )

    def augment_results(
        self,
        results: dict,
        models: List[dict],
        evaluator: "SR_evaluator",  # noqa: F821
    ) -> dict:
        """
        Computes the RMSE for the top models in the results dictionary.

        Args:
            results: The dictionary containing the results to augment.
            models: A list of dictionaries describing the performance of expressions using the base ranking function.
                Keyword expr contains the expression, error contains the error of the expression. The list is sorted
                by error.
            evaluator: The evaluator used to evaluate the models.

        Returns:
            The augmented results dictionary. The results dictionary contains an additional key "best_expr_rmse" with the
            RMSE of the best expression, and keys "rmse" and "parameters_rmse" for each of the top_models inside the
            results["top_models"] list.
        """
        expr = models[0]["expr"]
        error = self.evaluator.evaluate_expr(expr)
        results["best_expr_rmse"] = error
        for model in results["top_models"]:
            error = self.evaluator.evaluate_expr(model["expr"])
            model["rmse"] = error
            model["parameters_rmse"] = self.evaluator.models["".join(model["expr"])][
                "parameters"
            ]
        return results

    def to_dict(self, base_path, name) -> dict:
        """
        Creates a dictionary representation of the RMSE augmenter.

        Args:
            base_path: Used to save the data of the evaluator to disk.
            name: Used to save the data of the evaluator to disk.

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        # Save SR_evaluator
        return {"type": "RMSE", "evaluator": self.evaluator.to_dict(base_path, name)}

    @staticmethod
    def from_dict(data: dict, augmenter_map: Optional[dict] = None) -> "RMSE":
        """
        Creates an instance of the RMSE augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.
            augmenter_map: A dictionary mapping augmenter names to their classes.

        Returns:
            An instance of the RMSE augmenter.
        """
        evaluator = SR_evaluator.from_dict(data["evaluator"], augmenter_map=augmenter_map)
        return RMSE(evaluator)


class BED(ResultAugmenter):
    def __init__(self, evaluator: "SR_evaluator"):  # noqa: F821
        """
        Computes BED for the top models in the results dictionary.

        Args:
            evaluator: The evaluator used to evaluate the models. This evaluator must be initialized with
                ranking_function = "bed"

        Raises:
            Exception: If the evaluator is not initialized with ranking_function = "bed".
        """
        super().__init__()
        self.evaluator = evaluator
        if self.evaluator.ranking_function != "bed":
            raise Exception(
                "[BED augmenter] Ranking function of the evaluator must be set to 'bed' to compute BED."
            )

    def augment_results(
        self,
        results: dict,
        models: List[dict],
        evaluator: "SR_evaluator",  # noqa: F821
    ) -> dict:
        """
        Computes BED for the top models in the results dictionary.

        Args:
            results: The dictionary containing the results to augment.
            models: A list of dictionaries describing the performance of expressions using the base ranking function.
                Keyword expr contains the expression, error contains the error of the expression. The list is sorted
                by error.
            evaluator: The evaluator used to evaluate the models.

        Returns:
            The augmented results dictionary. The results dictionary contains an additional key "best_expr_bed" with
            BED of the best expression, and key "bed" for each of the top_models inside the results["top_models"] list.
        """
        expr = models[0]["expr"]
        error = self.evaluator.evaluate_expr(expr)
        results["best_expr_bed"] = error
        for model in results["top_models"]:
            model["bed"] = self.evaluator.evaluate_expr(model["expr"])
        return results

    def to_dict(self, base_path, name) -> dict:
        """
        Creates a dictionary representation of the BED augmenter.

        Args:
            base_path: Used to save the data of the evaluator to disk.
            name: Used to save the data of the evaluator to disk.

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        # Save SR_evaluator
        return {"type": "BED", "evaluator": self.evaluator.to_dict(base_path, name)}

    @staticmethod
    def from_dict(data: dict, augmenter_map: Optional[dict] = None) -> "BED":
        """
        Creates an instance of the BED augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.
            augmenter_map: A dictionary mapping augmenter names to their classes.

        Returns:
            An instance of the BED augmenter.
        """
        evaluator = SR_evaluator.from_dict(data["evaluator"], augmenter_map=augmenter_map)
        return BED(evaluator)


class R2(ResultAugmenter):
    def __init__(self, evaluator: "SR_evaluator"):  # noqa: F821
        """
        Computes the R^2 for the top models in the results dictionary.

        Args:
            evaluator: The evaluator used to evaluate the models (e.g., evaluator defined with test set data). This
                evaluator must be initialized with ranking_function = "rmse". If you're also using the RMSE augmenter,
                they the same one can be used for both.

        Raises:
            Exception: If the evaluator is not initialized with ranking_function = "rmse" or if y in the evaluator is None.
        """
        super().__init__()
        self.evaluator = evaluator
        if self.evaluator.ranking_function != "rmse":
            raise Exception(
                "[R2 augmenter] Ranking function of the evaluator must be set to 'rmse' to compute R^2."
            )
        if self.evaluator.y is None:
            raise Exception(
                "[R2 augmenter] y in the evaluator must not be None to compute R^2."
            )
        self.ss_tot = np.sum((self.evaluator.y - np.mean(self.evaluator.y)) ** 2)

    def augment_results(
        self,
        results: dict,
        models: List[dict],
        evaluator: "SR_evaluator",  # noqa: F821
    ) -> dict:
        """
        Computes the R^2 for the top models in the results dictionary.

        Args:
            results: The dictionary containing the results to augment.
            models: A list of dictionaries describing the performance of expressions using the base ranking function.
                Keyword expr contains the expression, error contains the error of the expression. The list is sorted
                by error.
            evaluator: The evaluator used to evaluate the models.

        Returns:
            The augmented results dictionary. The results dictionary contains an additional key "best_expr_r^2" with the
            R^2 of the best expression, and keys "r^2" and "parameters_r^2" for each of the top_models inside the
            results["top_models"] list.
        """
        results["best_expr_r^2"] = self._compute_r2(models[0])
        for model in results["top_models"]:
            r2 = self._compute_r2(model)
            model["r^2"] = r2
            model["parameters_r^2"] = (
                self.evaluator.models["".join(model["expr"])]["parameters"]
                if "parameters" in self.evaluator.models["".join(model["expr"])]
                else ""
            )
        return results

    def _compute_r2(self, model: dict):
        ss_res = (
            self.evaluator.y.shape[0] * self.evaluator.evaluate_expr(model["expr"]) ** 2
        )
        return max(0, 1 - ss_res / self.ss_tot)

    def to_dict(self, base_path, name) -> dict:
        """
        Creates a dictionary representation of the R2 augmenter.

        Args:
            base_path: Used to save the data of the evaluator to disk.
            name: Used to save the data of the evaluator to disk.

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        # Save SR_evaluator
        return {"type": "R2", "evaluator": self.evaluator.to_dict(base_path, name)}

    @staticmethod
    def from_dict(data: dict, augmenter_map: Optional[dict] = None) -> "R2":
        """
        Creates an instance of the R2 augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.
            augmenter_map: A dictionary mapping augmenter names to their classes.

        Returns:
            An instance of the R2 augmenter.
        """
        evaluator = SR_evaluator.from_dict(data["evaluator"], augmenter_map=augmenter_map)
        return R2(evaluator)
