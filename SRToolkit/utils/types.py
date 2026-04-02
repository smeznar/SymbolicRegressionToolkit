"""
Shared type definitions and constants for the SRToolkit package.
"""

from typing import List, Optional, Set, Tuple, TypedDict, Union

import numpy as np

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
