"""
This module contains the implementations of the ResultAugmenter class. These implementations augment the results
dictionary returned by the SRToolkit.evaluate function with additional information, such as the LaTeX representation
of the best expression, or RMSE on the test set, ...
"""

import warnings
from typing import Any, Dict, Type

import numpy as np

from SRToolkit.evaluation.sr_evaluator import EvalResult, ModelResult, ResultAugmenter, SR_evaluator
from SRToolkit.utils import Node, SymbolLibrary, simplify, tokens_to_tree


class ExpressionToLatex(ResultAugmenter):
    _type = "ExpressionToLatex"

    def __init__(
        self,
        symbol_library: SymbolLibrary,
        scope: str = "top",
        verbose: bool = False,
        name: str = "ExpressionToLatex",
    ) -> None:
        """
        Transforms the expressions inside the results into LaTeX strings.

        Args:
            symbol_library: The symbol library used to convert tokens into LaTeX symbols.
            scope: Which expressions to convert. Can be:
                - "best": only the best expression is converted
                - "top": the best expression and expressions in the top models are converted
                - "all": everything in "top" plus all evaluated expressions
            verbose: If True, warns the user if LaTeX conversion fails for a given expression.
            name: Key used in :attr:`EvalResult.augmentations` and :attr:`ModelResult.augmentations`.
        """
        super().__init__(name)
        self.symbol_library = symbol_library

        if scope not in ["best", "top", "all"]:
            raise Exception(f"[RMSE augmenter] Invalid scope: {scope}. Must be one of 'best', 'top', 'all'.")
        self.scope = scope

        self.verbose = verbose

    def write_results(self, results: EvalResult) -> None:
        """
        Writes LaTeX representations into *results* and its top models.

        Stores ``{"best_expr_latex": ...}`` in ``results.augmentations[self.name]``.
        If ``only_best_expression`` is False, also stores ``{"expr_latex": ...}`` in
        each top model's augmentations.

        Args:
            results: The :class:`EvalResult` to augment..
        """
        eval_data: Dict[str, Any] = {}
        try:
            eval_data["best_expr_latex"] = tokens_to_tree(results.top_models[0].expr, self.symbol_library).to_latex(
                self.symbol_library
            )
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Unable to convert best expression to LaTeX: {e}")
        results.add_augmentation(self.name, eval_data, self._type)

        if self.scope == "top" or self.scope == "all":
            for model in results.top_models:
                try:
                    model.add_augmentation(
                        self.name,
                        {"expr_latex": tokens_to_tree(model.expr, self.symbol_library).to_latex(self.symbol_library)},
                        self._type,
                    )
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Unable to convert expression {''.join(model.expr)} to LaTeX: {e}")

        if self.scope == "all":
            for model in results.all_models:
                try:
                    model.add_augmentation(
                        self.name,
                        {"expr_latex": tokens_to_tree(model.expr, self.symbol_library).to_latex(self.symbol_library)},
                        self._type,
                    )
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Unable to convert expression {''.join(model.expr)} to LaTeX: {e}")

    @classmethod
    def format_eval_result(cls, data: Dict[str, Any]) -> str:
        latex = data.get("best_expr_latex", "")
        return f"LaTeX of the best expression: {latex}" if latex else ""

    @classmethod
    def format_model_result(cls, data: Dict[str, Any]) -> str:
        latex = data.get("expr_latex", "")
        return f"LaTeX: {latex}" if latex else ""

    def to_dict(self, base_path: str, name: str) -> dict:
        """
        Creates a dictionary representation of the ExpressionToLatex augmenter.

        Args:
            base_path: Unused and ignored
            name: Unused and ignored

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        return {
            "format_version": 1,
            "type": "ExpressionToLatex",
            "name": self.name,
            "symbol_library": self.symbol_library.to_dict(),
            "scope": self.scope,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_dict(data: dict) -> "ExpressionToLatex":
        """
        Creates an instance of the ExpressionToLatex augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.

        Returns:
            An instance of the ExpressionToLatex augmenter.
        """
        if data.get("format_version", 1) != 1:
            raise ValueError(
                f"[ExpressionToLatex.from_dict] Unsupported format_version: {data.get('format_version')!r}. Expected 1."
            )
        return ExpressionToLatex(
            symbol_library=data["symbol_library"],
            scope=data["scope"],
            verbose=data["verbose"],
            name=data["name"],
        )


class ExpressionSimplifier(ResultAugmenter):
    _type = "ExpressionSimplifier"

    def __init__(
        self,
        symbol_library: SymbolLibrary,
        scope: str = "top",
        verbose: bool = False,
        name: str = "ExpressionSimplifier",
    ) -> None:
        """
        Simplifies the expressions inside the results if possible.

        Args:
            symbol_library: The symbol library used to simplify the expressions.
            scope: Which expressions to convert. Can be:
                - "best": only the best expression is converted
                - "top": the best expression and expressions in the top models are converted
                - "all": everything in "top" plus all evaluated expressions
            verbose: If True, warns the user if simplification fails for a given expression.
            name: Key used in :attr:`EvalResult.augmentations` and :attr:`ModelResult.augmentations`.
        """
        super().__init__(name)
        self.symbol_library = symbol_library

        if scope not in ["best", "top", "all"]:
            raise Exception(f"[RMSE augmenter] Invalid scope: {scope}. Must be one of 'best', 'top', 'all'.")
        self.scope = scope

        self.verbose = verbose

    def write_results(self, results: EvalResult) -> None:
        """
        Writes simplified expressions into *results* and its top models.

        Stores ``{"simplified_best_expr": ...}`` in ``results.augmentations[self.name]`` if
        simplification succeeds. Also stores ``{"simplified_expr": ...}`` for each top model
        if ``only_best_expression`` is False.

        Args:
            results: The :class:`EvalResult` to augment.
        """
        eval_data: Dict[str, Any] = {}
        try:
            simplified_expr = simplify(results.top_models[0].expr, self.symbol_library)
            if isinstance(simplified_expr, list):
                eval_data["simplified_best_expr"] = "".join(simplified_expr)
            elif isinstance(simplified_expr, Node):
                eval_data["simplified_best_expr"] = "".join(simplified_expr.to_list(self.symbol_library))
            else:
                raise Exception(f"Simplified expression is not a list or Node: {simplified_expr}")
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Unable to simplify {results.best_expr}: {e}")
        results.add_augmentation(self.name, eval_data, self._type)

        if self.scope == "top" or self.scope == "all":
            for model in results.top_models:
                top_model_data: Dict[str, Any] = {}
                try:
                    simplified_expr = simplify(model.expr, self.symbol_library)
                    if isinstance(simplified_expr, list):
                        top_model_data["simplified_expr"] = "".join(simplified_expr)
                    elif isinstance(simplified_expr, Node):
                        top_model_data["simplified_expr"] = "".join(simplified_expr.to_list(self.symbol_library))
                    else:
                        raise Exception(f"Simplified expression is not a list or Node: {simplified_expr}")
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Unable to simplify {''.join(model.expr)}: {e}")
                model.add_augmentation(self.name, top_model_data, self._type)

        if self.scope == "all":
            for model in results.all_models:
                all_model_data: Dict[str, Any] = {}
                try:
                    simplified_expr = simplify(model.expr, self.symbol_library)
                    if isinstance(simplified_expr, list):
                        all_model_data["simplified_expr"] = "".join(simplified_expr)
                    elif isinstance(simplified_expr, Node):
                        all_model_data["simplified_expr"] = "".join(simplified_expr.to_list(self.symbol_library))
                    else:
                        raise Exception(f"Simplified expression is not a list or Node: {simplified_expr}")
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Unable to simplify {''.join(model.expr)}: {e}")
                model.add_augmentation(self.name, all_model_data, self._type)

    @classmethod
    def format_eval_result(cls, data: Dict[str, Any]) -> str:
        simplified = data.get("simplified_best_expr", "")
        return f"Simplified: {simplified}" if simplified else ""

    @classmethod
    def format_model_result(cls, data: Dict[str, Any]) -> str:
        simplified = data.get("simplified_expr", "")
        return f"Simplified: {simplified}" if simplified else ""

    def to_dict(self, base_path: str, name: str) -> dict:
        """
        Creates a dictionary representation of the ExpressionSimplifier augmenter.

        Args:
            base_path: Unused and ignored
            name: Unused and ignored

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        return {
            "format_version": 1,
            "type": "ExpressionSimplifier",
            "name": self.name,
            "symbol_library": self.symbol_library.to_dict(),
            "scope": self.scope,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_dict(data: dict) -> "ExpressionSimplifier":
        """
        Creates an instance of the ExpressionSimplifier augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.
        Returns:
            An instance of the ExpressionSimplifier augmenter.
        """
        if data.get("format_version", 1) != 1:
            raise ValueError(
                f"[ExpressionSimplifier.from_dict] Unsupported format_version: {data.get('format_version')!r}. Expected 1."
            )
        return ExpressionSimplifier(
            symbol_library=data["symbol_library"],
            scope=data["scope"],
            verbose=data["verbose"],
            name=data["name"],
        )


class RMSE(ResultAugmenter):
    _type = "RMSE"

    def __init__(self, evaluator: SR_evaluator, scope: str = "top", name: str = "RMSE") -> None:  # noqa: F821
        """
        Computes the RMSE for the top models using a separate evaluator (e.g. a test-set evaluator).

        Args:
            evaluator: The evaluator used to evaluate the models. Must be initialized with
                ``ranking_function = "rmse"``.
            scope: Which expressions to convert. Can be:
                - "best": only the best expression is converted
                - "top": the best expression and expressions in the top models are converted
                - "all": everything in "top" plus all evaluated expressions
            name: Key used in :attr:`EvalResult.augmentations` and :attr:`ModelResult.augmentations`.

        Raises:
            Exception: If the evaluator is not initialized with ranking_function = "rmse" or if y in the evaluator is None.
        """
        super().__init__(name)
        self.evaluator = evaluator

        if scope not in ["best", "top", "all"]:
            raise Exception(f"[RMSE augmenter] Invalid scope: {scope}. Must be one of 'best', 'top', 'all'.")
        self.scope = scope

        if self.evaluator.ranking_function != "rmse":
            raise Exception("[RMSE augmenter] Ranking function of the evaluator must be set to 'rmse' to compute RMSE.")
        if self.evaluator.y is None:
            raise Exception("[RMSE augmenter] y in the evaluator must not be None to compute RMSE.")

    def write_results(self, results: EvalResult) -> None:
        """
        Writes RMSE scores into *results* and its top models.

        Stores ``{"min_error": ...}`` in ``results.augmentations[self.name]`` and
        ``{"error": ..., "parameters": ...}`` in each top model's augmentations.

        Args:
            results: The :class:`EvalResult` to augment.
        """
        eval_data: Dict[str, Any] = {"min_error": self.evaluator.evaluate_expr(results.top_models[0].expr)}
        results.add_augmentation(self.name, eval_data, self._type)

        if self.scope == "top" or self.scope == "all":
            for model in results.top_models:
                key = "".join(model.expr)
                top_model_data: Dict[str, Any] = {
                    "error": self.evaluator.evaluate_expr(model.expr),
                    "parameters": self.evaluator.models[key].parameters,
                }
                model.add_augmentation(self.name, top_model_data, self._type)

        if self.scope == "all":
            for model in results.all_models:
                key = "".join(model.expr)
                all_model_data: Dict[str, Any] = {
                    "error": self.evaluator.evaluate_expr(model.expr),
                    "parameters": self.evaluator.models[key].parameters,
                }
                model.add_augmentation(self.name, all_model_data, self._type)

    @classmethod
    def format_eval_result(cls, data: Dict[str, Any]) -> str:
        val = data.get("min_error", "")
        return f"Test RMSE: {val}" if val != "" else ""

    @classmethod
    def format_model_result(cls, data: Dict[str, Any]) -> str:
        parts = [f"RMSE={data['error']:.6g}"]
        if "parameters" in data and data["parameters"] is not None:
            parts.append(f"params={np.round(data['parameters'], 4).tolist()}")
        return ", ".join(parts)

    def to_dict(self, base_path: str, name: str) -> dict:
        """
        Creates a dictionary representation of the RMSE augmenter.

        Args:
            base_path: Used to save the data of the evaluator to disk.
            name: Used to save the data of the evaluator to disk.

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        return {
            "format_version": 1,
            "name": self.name,
            "type": "RMSE",
            "scope": self.scope,
            "evaluator": self.evaluator.to_dict(base_path, name + "_RMSE_augmenter"),
        }

    @staticmethod
    def from_dict(data: dict) -> "RMSE":
        """
        Creates an instance of the RMSE augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.

        Returns:
            An instance of the RMSE augmenter.
        """
        if data.get("format_version", 1) != 1:
            raise ValueError(
                f"[RMSE.from_dict] Unsupported format_version: {data.get('format_version')!r}. Expected 1."
            )
        evaluator = SR_evaluator.from_dict(data["evaluator"])
        return RMSE(evaluator, scope=data["scope"], name=data["name"])


class BED(ResultAugmenter):
    _type = "BED"

    def __init__(self, evaluator: SR_evaluator, scope: str = "top", name: str = "BED") -> None:  # noqa: F821
        """
        Computes BED for the top models using a separate evaluator.

        Args:
            evaluator: The evaluator used to evaluate the models. Must be initialized with
                ``ranking_function = "bed"``.
            scope: Which expressions to convert. Can be:
                - "best": only the best expression is converted
                - "top": the best expression and expressions in the top models are converted
                - "all": everything in "top" plus all evaluated expressions
            name: Key used in :attr:`EvalResult.augmentations` and :attr:`ModelResult.augmentations`.

        Raises:
            Exception: If the evaluator is not initialized with ranking_function = "bed".
        """
        super().__init__(name)
        self.evaluator = evaluator

        if scope not in ["best", "top", "all"]:
            raise Exception(f"[RMSE augmenter] Invalid scope: {scope}. Must be one of 'best', 'top', 'all'.")
        self.scope = scope

        if self.evaluator.ranking_function != "bed":
            raise Exception("[BED augmenter] Ranking function of the evaluator must be set to 'bed' to compute BED.")

    def write_results(
        self,
        results: EvalResult,
    ) -> None:
        """
        Writes BED scores into *results* and its top models.

        Stores ``{"best_expr_bed": ...}`` in ``results.augmentations[self.name]`` and
        ``{"bed": ...}`` in each top model's augmentations.

        Args:
            results: The :class:`EvalResult` to augment.
        """
        eval_data: Dict[str, Any] = {"best_expr_bed": self.evaluator.evaluate_expr(results.top_models[0].expr)}
        results.add_augmentation(self.name, eval_data, self._type)

        if self.scope == "top" or self.scope == "all":
            for model in results.top_models:
                top_model_data: Dict[str, Any] = {"bed": self.evaluator.evaluate_expr(model.expr)}
                model.add_augmentation(self.name, top_model_data, self._type)

        if self.scope == "all":
            for model in results.all_models:
                all_model_data: Dict[str, Any] = {"bed": self.evaluator.evaluate_expr(model.expr)}
                model.add_augmentation(self.name, all_model_data, self._type)

    @classmethod
    def format_eval_result(cls, data: Dict[str, Any]) -> str:
        val = data.get("best_expr_bed", "")
        return f"Test BED: {val}" if val != "" else ""

    @classmethod
    def format_model_result(cls, data: Dict[str, Any]) -> str:
        val = data.get("bed", "")
        return f"BED={val}" if val != "" else ""

    def to_dict(self, base_path: str, name: str) -> dict:
        """
        Creates a dictionary representation of the BED augmenter.

        Args:
            base_path: Used to save the data of the evaluator to disk.
            name: Used to save the data of the evaluator to disk.

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        return {
            "format_version": 1,
            "name": self.name,
            "type": "BED",
            "scope": self.scope,
            "evaluator": self.evaluator.to_dict(base_path, name + "_BED_augmenter"),
        }

    @staticmethod
    def from_dict(data: dict) -> "BED":
        """
        Creates an instance of the BED augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.

        Returns:
            An instance of the BED augmenter.
        """
        if data.get("format_version", 1) != 1:
            raise ValueError(f"[BED.from_dict] Unsupported format_version: {data.get('format_version')!r}. Expected 1.")
        evaluator = SR_evaluator.from_dict(data["evaluator"])
        return BED(evaluator, scope=data["scope"], name=data["name"])


class R2(ResultAugmenter):
    _type = "R2"

    def __init__(self, evaluator: SR_evaluator, scope: str = "top", name: str = "R2") -> None:  # noqa: F821
        """
        Computes the R^2 for the top models using a separate evaluator.

        Args:
            evaluator: The evaluator used to evaluate the models (e.g., evaluator defined with test set data). Must be
                initialized with ``ranking_function = "rmse"``. The same evaluator instance can be shared with
                :class:`RMSE`.
            scope: Which expressions to convert. Can be:
                - "best": only the best expression is converted
                - "top": the best expression and expressions in the top models are converted
                - "all": everything in "top" plus all evaluated expressions
            name: Key used in :attr:`EvalResult.augmentations` and :attr:`ModelResult.augmentations`.

        Raises:
            Exception: If the evaluator is not initialized with ranking_function = "rmse" or if y in the evaluator is None.
        """
        super().__init__(name)

        if scope not in ["best", "top", "all"]:
            raise Exception(f"[RMSE augmenter] Invalid scope: {scope}. Must be one of 'best', 'top', 'all'.")
        self.scope = scope

        self.evaluator = evaluator
        if self.evaluator.ranking_function != "rmse":
            raise Exception("[R2 augmenter] Ranking function of the evaluator must be set to 'rmse' to compute R^2.")
        if self.evaluator.y is None:
            raise Exception("[R2 augmenter] y in the evaluator must not be None to compute R^2.")
        self.ss_tot = np.sum((self.evaluator.y - np.mean(self.evaluator.y)) ** 2)

    def write_results(self, results: EvalResult) -> None:
        """
        Writes R^2 scores into *results* and its top models.

        Stores ``{"best_expr_r^2": ...}`` in ``results.augmentations[self.name]`` and
        ``{"r^2": ..., "parameters_r^2": ...}`` in each top model's augmentations.

        Args:
            results: The :class:`EvalResult` to augment.
        """
        eval_data: Dict[str, Any] = {"best_expr_r^2": self._compute_r2(results.top_models[0])}
        results.add_augmentation(self.name, eval_data, self._type)

        if self.scope == "top" or self.scope == "all":
            for model in results.top_models:
                key = "".join(model.expr)
                top_model_data: Dict[str, Any] = {
                    "r^2": self._compute_r2(model),
                    "parameters_r^2": self.evaluator.models[key].parameters,
                }
                model.add_augmentation(self.name, top_model_data, self._type)

        if self.scope == "all":
            for model in results.all_models:
                key = "".join(model.expr)
                all_model_data: Dict[str, Any] = {
                    "r^2": self._compute_r2(model),
                    "parameters_r^2": self.evaluator.models[key].parameters,
                }
                model.add_augmentation(self.name, all_model_data, self._type)

    def _compute_r2(self, model: ModelResult) -> float:
        assert self.evaluator.y is not None, "y in the evaluator must not be None to compute R^2."
        ss_res = self.evaluator.y.shape[0] * self.evaluator.evaluate_expr(model.expr) ** 2
        return max(0, 1 - ss_res / self.ss_tot)

    @classmethod
    def format_eval_result(cls, data: Dict[str, Any]) -> str:
        val = data.get("best_expr_r^2", "")
        return f"Test R²: {val}" if val != "" else ""

    @classmethod
    def format_model_result(cls, data: Dict[str, Any]) -> str:
        parts = [f"R²={data['r^2']:.4g}"]
        if "parameters_r^2" in data and data["parameters_r^2"] is not None:
            parts.append(f"params={np.round(data['parameters_r^2'], 4).tolist()}")
        return ", ".join(parts)

    def to_dict(self, base_path: str, name: str) -> dict:
        """
        Creates a dictionary representation of the R2 augmenter.

        Args:
            base_path: Used to save the data of the evaluator to disk.
            name: Used to save the data of the evaluator to disk.

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        return {
            "format_version": 1,
            "name": self.name,
            "type": "R2",
            "scope": self.scope,
            "evaluator": self.evaluator.to_dict(base_path, name + "_R2_augmenter"),
        }

    @staticmethod
    def from_dict(data: dict) -> "R2":
        """
        Creates an instance of the R2 augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.

        Returns:
            An instance of the R2 augmenter.
        """
        if data.get("format_version", 1) != 1:
            raise ValueError(f"[R2.from_dict] Unsupported format_version: {data.get('format_version')!r}. Expected 1.")
        evaluator = SR_evaluator.from_dict(data["evaluator"])
        return R2(evaluator, scope=data["scope"], name=data["name"])


RESULT_AUGMENTERS: Dict[str, Type[ResultAugmenter]] = {
    "ExpressionToLatex": ExpressionToLatex,
    "RMSE": RMSE,
    "BED": BED,
    "R2": R2,
    "ExpressionSimplifier": ExpressionSimplifier,
}
"""A mapping of augmentation names to their corresponding ResultAugmenter classes.

This constant defines the library of available result augmentation classes used across the benchmarking framework.

The dictionary keys are the unique string identifiers for the augmentor found under the 'type' value in the to_dict
function. The values are the uninstantiated class objects, all of which inherit from ResultAugmenter.
"""


def register_augmenter(name: str, cls: Type[ResultAugmenter]) -> None:
    """
    Registers a custom ResultAugmenter class in the global registry.

    This allows users to add custom augmenters that can be discovered and used
    alongside the built-in ones.

    Examples:
        >>> from SRToolkit.evaluation import ResultAugmenter, register_augmenter
        >>> class MyAugmenter(ResultAugmenter):
        ...     def __init__(self):
        ...         super().__init__("MyAugmenter")
        ...     def write_results(self, results):
        ...         pass
        ...     def to_dict(self, base_path, name):
        ...         return {"type": "MyAugmenter"}
        >>> register_augmenter("MyAugmenter", MyAugmenter)
        >>> "MyAugmenter" in RESULT_AUGMENTERS
        True

    Args:
        name: The string identifier for the augmenter.
        cls: The ResultAugmenter subclass to register.
    """
    RESULT_AUGMENTERS[name] = cls
