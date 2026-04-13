"""
Constant parameter estimation for symbolic regression expressions using numerical optimization.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from typing_extensions import Unpack

from SRToolkit.utils.expression_compiler import expr_to_error_function
from SRToolkit.utils.expression_tree import Node
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import EstimationSettings


class ParameterEstimator:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
        seed: Optional[int] = None,
        **kwargs: Unpack[EstimationSettings],
    ) -> None:
        """
        Fits free constants in symbolic expressions by minimizing RMSE against target values.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> pe = ParameterEstimator(X, y)
            >>> rmse, constants = pe.estimate_parameters(["C", "*", "X_1", "-", "X_0"])
            >>> print(rmse < 1e-6)
            True
            >>> print(1.99 < constants[0] < 2.01)
            True

        Args:
            X: Input data of shape ``(n_samples, n_features)`` used to evaluate expressions.
            y: Target values of shape ``(n_samples,)``.
            symbol_library: Symbol library defining the token vocabulary.
                Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].
            seed: Random seed for reproducible constant initialization. Default ``None``.
            **kwargs: Optional estimation settings from
                [EstimationSettings][SRToolkit.utils.types.EstimationSettings].
                Supported keys: ``method``, ``tol``, ``gtol``, ``max_iter``,
                ``constant_bounds``, ``initialization``, ``max_constants``,
                ``max_expr_length``.

        Attributes:
            symbol_library: The symbol library used.
            X: Input data.
            y: Target values.
            seed: Random seed.
            estimation_settings: Active settings dict, merged from defaults and ``**kwargs``.
        """
        self.symbol_library = symbol_library
        self.X = X
        self.y = y
        self.seed = seed

        self.estimation_settings = {
            "method": "L-BFGS-B",
            "tol": 1e-6,
            "gtol": 1e-3,
            "max_iter": 100,
            "constant_bounds": (-5, 5),
            "initialization": "random",  # random, mean
            "max_constants": 8,
            "max_expr_length": -1,
        }

        if kwargs:
            for k in self.estimation_settings.keys():
                if k in kwargs:
                    self.estimation_settings[k] = kwargs[k]  # type: ignore[literal-required]

        self._rng = np.random.default_rng(self.seed)

    def estimate_parameters(self, expr: Union[List[str], Node]) -> Tuple[float, np.ndarray]:
        """
        Fit free constants in *expr* by minimizing RMSE against the target values.

        Expressions that exceed ``max_constants`` or ``max_expr_length`` immediately
        return ``(NaN, [])``. Expressions with no free constants are evaluated directly
        without running the optimizer.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> pe = ParameterEstimator(X, y)
            >>> rmse, constants = pe.estimate_parameters(["C", "*", "X_1", "-", "X_0"])
            >>> print(rmse < 1e-6)
            True
            >>> print(1.99 < constants[0] < 2.01)
            True
            >>> # Constant-free expressions are evaluated directly
            >>> rmse, constants = pe.estimate_parameters(["X_1", "-", "X_0"])
            >>> constants.size
            0

        Args:
            expr: Expression as a token list in infix notation or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.

        Returns:
            A 2-tuple ``(rmse, parameters)`` where ``rmse`` is the root-mean-square error of the fitted expression and ``parameters`` is a 1-D array of optimized constant values. Returns ``(NaN, [])`` if the expression violates ``max_constants`` or ``max_expr_length``.
        """
        if isinstance(expr, Node):
            expr_str = expr.to_list(notation="prefix")
            num_constants = sum([1 for t in expr_str if self.symbol_library.get_type(t) == "const"])
        else:
            num_constants = sum([1 for t in expr if self.symbol_library.get_type(t) == "const"])
        if (
            isinstance(self.estimation_settings["max_constants"], int)
            and 0 <= self.estimation_settings["max_constants"] < num_constants
        ):
            return np.nan, np.array([])

        if isinstance(self.estimation_settings["max_expr_length"], int) and 0 <= self.estimation_settings[
            "max_expr_length"
        ] < len(expr):
            return np.nan, np.array([])

        executable_error_fn = expr_to_error_function(expr, self.symbol_library)

        if num_constants == 0:
            rmse = executable_error_fn(self.X, np.array([]), self.y)
            return rmse, np.array([])
        else:
            return self._optimize_parameters(executable_error_fn, num_constants)

    def _optimize_parameters(self, executable_error_fn: Callable, num_constants: int) -> Tuple[float, np.ndarray]:
        """
        Run L-BFGS-B to minimize *executable_error_fn* over *num_constants* constant values.

        Args:
            executable_error_fn: Compiled error function ``f(X, C, y)`` returning scalar RMSE.
            num_constants: Number of free constants to optimize.

        Returns:
            A 2-tuple ``(rmse, parameters)`` — the achieved minimum RMSE and the
            corresponding optimized constant vector.
        """
        if not (
            isinstance(self.estimation_settings["constant_bounds"], tuple)
            and len(self.estimation_settings["constant_bounds"]) == 2
        ):
            raise ValueError("constant_bounds must be a tuple of two elements")
        if self.estimation_settings["initialization"] == "random":
            x0 = (
                self._rng.random(num_constants)
                * (
                    self.estimation_settings["constant_bounds"][1]
                    - self.estimation_settings["constant_bounds"][0]
                    - 1e-8
                )
                + self.estimation_settings["constant_bounds"][0]
            )
        else:
            x0 = np.array([np.mean(self.estimation_settings["constant_bounds"]) for _ in range(num_constants)])

        res = minimize(
            lambda c: executable_error_fn(self.X, c, self.y),
            x0,
            method=self.estimation_settings["method"],
            tol=self.estimation_settings["tol"],
            options={
                "maxiter": self.estimation_settings["max_iter"],
                "gtol": self.estimation_settings["gtol"],
            },
            bounds=[
                (
                    self.estimation_settings["constant_bounds"][0],
                    self.estimation_settings["constant_bounds"][1],
                )
                for _ in range(num_constants)
            ],
        )
        return res.fun, res.x
