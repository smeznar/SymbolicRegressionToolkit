"""
Shared type definitions, constants, and result dataclasses for the SRToolkit package.

Defines symbol-type constants (``VAR``, ``CONST``, ``FN``, ``OP``, ``LIT``),
``EstimationSettings`` for parameter estimation configuration, and ``ModelResult`` /
``EvalResult`` for representing SR experiment outcomes.
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

    Examples:
        >>> settings: EstimationSettings = {"method": "L-BFGS-B", "max_iter": 200}
        >>> settings.get("method")
        'L-BFGS-B'
        >>> settings.get("tol", 1e-6)
        1e-06

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

    Examples:
        >>> result = ModelResult(expr=["C", "*", "X_0"], error=0.42)
        >>> result.expr
        ['C', '*', 'X_0']
        >>> result.error
        0.42
        >>> result.parameters is None
        True

    Attributes:
        expr: Token list representing the expression, e.g. ``["C", "*", "X_0"]``.
        error: Numeric error under the ranking function (RMSE or BED).
        parameters: Fitted constant values. Present for RMSE ranking only, ``None`` otherwise.
        augmentations: Per-augmenter data keyed by augmenter name. Populated by
            [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] subclasses via
            [add_augmentation][SRToolkit.utils.types.ModelResult.add_augmentation].
    """

    expr: List[str]
    error: float
    parameters: Optional["np.ndarray"] = None
    augmentations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_augmentation(self, name: str, data: Dict[str, Any], aug_type: str) -> None:
        """
        Attach augmentation data produced by a :class:`ResultAugmenter` to this result.

        If ``name`` is already present in :attr:`augmentations`, a numeric suffix is
        appended (``name_1``, ``name_2``, …) to avoid overwriting existing data.

        Examples:
            >>> result = ModelResult(expr=["X_0"], error=0.1)
            >>> result.add_augmentation("latex", {"value": "$X_0$"}, "LaTeXAugmenter")
            >>> result.augmentations["latex"]["value"]
            '$X_0$'
            >>> result.add_augmentation("latex", {"value": "$X_0$"}, "LaTeXAugmenter")
            >>> "latex_1" in result.augmentations
            True

        Args:
            name: Key under which the augmentation is stored in :attr:`augmentations`.
                A suffix is added automatically if the key already exists.
            data: Arbitrary dict of augmentation data. A ``"_type"`` key is injected
                automatically and should not be included.
            aug_type: Augmenter class name, stored as ``data["_type"]``.
        """
        resolved = name
        counter = 1
        while resolved in self.augmentations:
            resolved = f"{name}_{counter}"
            counter += 1
        data["_type"] = aug_type
        self.augmentations[resolved] = data

    def to_dict(self) -> dict:
        """
        Serialize this model result to a JSON-safe dictionary.

        NumPy arrays and scalars are converted to native Python types so the
        result can be passed directly to ``json.dump``.

        Examples:
            >>> result = ModelResult(expr=["X_0", "+", "C"], error=0.25)
            >>> d = result.to_dict()
            >>> d["expr"]
            ['X_0', '+', 'C']
            >>> d["error"]
            0.25
            >>> d["parameters"] is None
            True

        Returns:
            A JSON-safe dictionary suitable for passing to :meth:`from_dict`.
        """
        return {
            "expr": self.expr,
            "error": float(self.error),
            "parameters": _to_json_safe(self.parameters),
            "augmentations": _to_json_safe(self.augmentations),
        }

    @staticmethod
    def from_dict(data: dict) -> "ModelResult":
        """
        Reconstruct a :class:`ModelResult` from a dictionary produced by :meth:`to_dict`.

        Examples:
            >>> result = ModelResult(expr=["X_0", "+", "C"], error=0.25)
            >>> result2 = ModelResult.from_dict(result.to_dict())
            >>> result2.expr
            ['X_0', '+', 'C']
            >>> result2.error
            0.25

        Args:
            data: Dictionary representation of a :class:`ModelResult`, as produced
                by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`ModelResult`.
        """
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

    Examples:
        >>> model = ModelResult(expr=["X_0"], error=0.05)
        >>> result = EvalResult(
        ...     min_error=0.05,
        ...     best_expr="X_0",
        ...     num_evaluated=500,
        ...     evaluation_calls=612,
        ...     top_models=[model],
        ...     all_models=[model],
        ...     approach_name="MyApproach",
        ...     success=True,
        ... )
        >>> result.min_error
        0.05
        >>> result.success
        True
        >>> result.dataset_name is None
        True

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
            [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] subclasses via
            [add_augmentation][SRToolkit.utils.types.EvalResult.add_augmentation].
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
        """
        Attach augmentation data produced by a :class:`ResultAugmenter` to this result.

        If ``name`` is already present in :attr:`augmentations`, a numeric suffix is
        appended (``name_1``, ``name_2``, …) to avoid overwriting existing data.

        Examples:
            >>> model = ModelResult(expr=["X_0"], error=0.05)
            >>> result = EvalResult(
            ...     min_error=0.05, best_expr="X_0", num_evaluated=10,
            ...     evaluation_calls=10, top_models=[model], all_models=[model],
            ...     approach_name="MyApproach", success=True,
            ... )
            >>> result.add_augmentation("complexity", {"value": 3}, "ComplexityAugmenter")
            >>> result.augmentations["complexity"]["value"]
            3
            >>> result.add_augmentation("complexity", {"value": 5}, "ComplexityAugmenter")
            >>> "complexity_1" in result.augmentations
            True

        Args:
            name: Key under which the augmentation is stored in :attr:`augmentations`.
                A suffix is added automatically if the key already exists.
            data: Arbitrary dict of augmentation data. A ``"_type"`` key is injected
                automatically and should not be included.
            aug_type: Augmenter class name, stored as ``data["_type"]``.
        """
        resolved = name
        counter = 1
        while resolved in self.augmentations:
            resolved = f"{name}_{counter}"
            counter += 1
        data["_type"] = aug_type
        self.augmentations[resolved] = data

    def to_dict(self) -> dict:
        """
        Serialize this evaluation result to a JSON-safe dictionary.

        NumPy arrays and scalars within nested :class:`ModelResult` entries are
        converted to native Python types so the result can be passed directly
        to ``json.dump``.

        Examples:
            >>> model = ModelResult(expr=["X_0"], error=0.05)
            >>> result = EvalResult(
            ...     min_error=0.05, best_expr="X_0", num_evaluated=10,
            ...     evaluation_calls=10, top_models=[model], all_models=[model],
            ...     approach_name="MyApproach", success=True,
            ... )
            >>> d = result.to_dict()
            >>> d["min_error"]
            0.05
            >>> d["approach_name"]
            'MyApproach'
            >>> len(d["top_models"])
            1

        Returns:
            A JSON-safe dictionary suitable for passing to :meth:`from_dict`.
        """
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
        """
        Reconstruct an :class:`EvalResult` from a dictionary produced by :meth:`to_dict`.

        Examples:
            >>> model = ModelResult(expr=["X_0"], error=0.05)
            >>> result = EvalResult(
            ...     min_error=0.05, best_expr="X_0", num_evaluated=10,
            ...     evaluation_calls=10, top_models=[model], all_models=[model],
            ...     approach_name="MyApproach", success=True,
            ... )
            >>> result2 = EvalResult.from_dict(result.to_dict())
            >>> result2.min_error
            0.05
            >>> result2.best_expr
            'X_0'
            >>> len(result2.top_models)
            1

        Args:
            data: Dictionary representation of an :class:`EvalResult`, as produced
                by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`EvalResult`.
        """
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
