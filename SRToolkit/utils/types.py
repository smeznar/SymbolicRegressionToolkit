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
    TypedDict for parameter estimation and BED evaluation settings.

    These settings are passed as ``**kwargs`` through ``SR_dataset``, ``SR_evaluator``,
    and ``ParameterEstimator``. All fields are optional and have defaults.

    Attributes:
        method: Minimization method. Default ``"L-BFGS-B"``.
        tol: Tolerance for termination. Default ``1e-6``.
        gtol: Tolerance for the gradient norm. Default ``1e-3``.
        max_iter: Maximum number of iterations. Default ``100``.
        constant_bounds: Lower and upper bounds for constant values. Default ``(-5, 5)``.
        initialization: How to initialize constants: ``"random"`` or ``"mean"``. Default ``"random"``.
        max_constants: Maximum number of free constants. Default ``8``.
        max_expr_length: Maximum expression length (``-1`` = no limit). Default ``-1``.
        num_points_sampled: Number of points for behavior estimation. Default ``64``.
        bed_X: Points used for BED evaluation. Default ``None``.
        num_consts_sampled: Number of constants sampled for BED. Default ``32``.
        domain_bounds: Domain bounds for sampling BED points. Default ``None``.
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
