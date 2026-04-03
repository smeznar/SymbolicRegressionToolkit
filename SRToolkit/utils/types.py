"""
Shared type definitions and constants for the SRToolkit package.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict, Union

import numpy as np

from .serialization import _from_json_safe, _to_json_safe

VAR = "var"
CONST = "const"
FN = "fn"
OP = "op"
LIT = "lit"
VALID_SYMBOL_TYPES: Set[str] = {VAR, CONST, FN, OP, LIT}


class EstimationSettings(TypedDict, total=False):
    """
    Shared settings for parameter estimation and BED evaluation.

    Passed as ``**kwargs`` to ``SR_dataset``, ``SR_evaluator``, and
    ``ParameterEstimator``. All fields are optional.

    Attributes:
        method: Optimization algorithm for parameter fitting. Default: ``"L-BFGS-B"``.
        tol: Termination tolerance for the optimizer. Default: ``1e-6``.
        gtol: Gradient-norm termination tolerance. Default: ``1e-3``.
        max_iter: Maximum optimizer iterations. Default: ``100``.
        constant_bounds: ``(lower, upper)`` bounds for sampled constant values.
            Default: ``(-5, 5)``.
        initialization: Constant initialization strategy — ``"random"`` samples
            uniformly within ``constant_bounds``; ``"mean"`` sets all constants
            to the midpoint. Default: ``"random"``.
        max_constants: Maximum number of free constants permitted in a single
            expression. Expressions exceeding this limit score ``NaN``.
            Default: ``8``.
        max_expr_length: Maximum expression length in tokens. ``-1`` disables the
            limit. Default: ``-1``.
        num_points_sampled: Number of domain points used when evaluating expression
            behavior for BED. ``-1`` uses all points in ``X``. Default: ``64``.
        bed_X: Fixed evaluation points for BED. If ``None``, points are sampled from
            ``domain_bounds`` or selected randomly from ``X``. Default: ``None``.
        num_consts_sampled: Number of constant vectors sampled per expression for
            BED. Default: ``32``.
        domain_bounds: Per-variable ``(lower, upper)`` bounds used to sample
            ``bed_X`` when it is ``None``. Default: ``None``.
    """

    method: str
    tol: float
    gtol: float
    max_iter: int
    constant_bounds: Union[Tuple[float, float]]
    initialization: str
    max_constants: int
    max_expr_length: int
    num_points_sampled: int
    bed_X: Optional[np.ndarray]
    num_consts_sampled: int
    domain_bounds: Optional[List[Tuple[float, float]]]


@dataclass
class ModelResult:
    """
    A single model entry in ``EvalResult.top_models`` and ``EvalResult.all_models``.

    Attributes:
        expr: Token list representing the expression, e.g. ``["C", "*", "X_0"]``.
        error: Numeric error under the ranking function (RMSE or BED).
        parameters: Fitted constant values. Present for RMSE ranking only, ``None`` otherwise.
        augmentations: Per-augmenter data keyed by augmenter name. Populated by
            :class:`ResultAugmenter` subclasses via :meth:`add_augmentation`.
    """

    expr: List[str]
    error: float
    parameters: Optional["np.ndarray"] = None
    augmentations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_augmentation(self, name: str, data: Dict[str, Any], aug_type: str) -> None:
        resolved = name
        counter = 1
        while resolved in self.augmentations:
            resolved = f"{name}_{counter}"
            counter += 1
        data["_type"] = aug_type
        self.augmentations[resolved] = data

    def to_dict(self) -> dict:
        """Serializes this model result to a JSON-safe dictionary."""
        return {
            "expr": self.expr,
            "error": float(self.error),
            "parameters": _to_json_safe(self.parameters),
            "augmentations": _to_json_safe(self.augmentations),
        }

    @staticmethod
    def from_dict(data: dict) -> "ModelResult":
        """Creates a :class:`ModelResult` from a dictionary produced by :meth:`to_dict`."""
        return ModelResult(
            expr=data["expr"],
            error=data["error"],
            parameters=_from_json_safe(data["parameters"]),
            augmentations=_from_json_safe(data["augmentations"]),
        )


@dataclass
class EvalResult:
    """
    Result for a single SR experiment, as returned by ``SR_results[i]``.

    Attributes:
        min_error: Lowest error achieved across all evaluated expressions.
        best_expr: String representation of the best expression found.
        num_evaluated: Number of unique expressions evaluated.
        evaluation_calls: Number of times ``evaluate_expr`` was called (includes cache hits).
        top_models: Top-*k* models sorted by error.
        all_models: All evaluated models sorted by error.
        approach_name: Name of the SR approach, or empty string if not provided.
        success: Whether ``min_error`` is below the configured ``success_threshold``.
        dataset_name: Name of the dataset, extracted from metadata. ``None`` if not provided.
        metadata: Remaining metadata dict after ``dataset_name`` is popped. ``None`` if empty.
        augmentations: Per-augmenter data keyed by augmenter name. Populated by
            :class:`ResultAugmenter` subclasses via :meth:`add_augmentation`.
    """

    min_error: float
    best_expr: str
    num_evaluated: int
    evaluation_calls: int
    top_models: List[ModelResult]
    all_models: List[ModelResult]
    approach_name: str
    success: bool
    dataset_name: Optional[str] = None
    metadata: Optional[dict] = None
    augmentations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_augmentation(self, name: str, data: Dict[str, Any], aug_type: str) -> None:
        resolved = name
        counter = 1
        while resolved in self.augmentations:
            resolved = f"{name}_{counter}"
            counter += 1
        data["_type"] = aug_type
        self.augmentations[resolved] = data

    def to_dict(self) -> dict:
        """Serializes this evaluation result to a JSON-safe dictionary."""
        return {
            "min_error": float(self.min_error),
            "best_expr": self.best_expr,
            "num_evaluated": int(self.num_evaluated),
            "evaluation_calls": int(self.evaluation_calls),
            "top_models": [m.to_dict() for m in self.top_models],
            "all_models": [m.to_dict() for m in self.all_models],
            "approach_name": self.approach_name,
            "success": bool(self.success),
            "dataset_name": self.dataset_name,
            "metadata": self.metadata,
            "augmentations": _to_json_safe(self.augmentations),
        }

    @staticmethod
    def from_dict(data: dict) -> "EvalResult":
        """Creates an :class:`EvalResult` from a dictionary produced by :meth:`to_dict`."""
        return EvalResult(
            min_error=data["min_error"],
            best_expr=data["best_expr"],
            num_evaluated=data["num_evaluated"],
            evaluation_calls=data["evaluation_calls"],
            top_models=[ModelResult.from_dict(m) for m in data["top_models"]],
            all_models=[ModelResult.from_dict(m) for m in data["all_models"]],
            approach_name=data["approach_name"],
            success=data["success"],
            dataset_name=data.get("dataset_name"),
            metadata=data.get("metadata"),
            augmentations=_from_json_safe(data["augmentations"]),
        )
