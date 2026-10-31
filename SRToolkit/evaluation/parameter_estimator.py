"""
Constant parameter estimation for symbolic regression expressions using numerical optimization.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from typing_extensions import Unpack

from SRToolkit.utils.expression_compiler import compile_expr_rmse
from SRToolkit.utils.expression_tree import Node
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import RankingSettings

# The settings ParameterEstimator reads, with their defaults. Any other keys passed in are ignored.
DEFAULT_ESTIMATION_SETTINGS: Dict[str, Any] = {
    "method": "L-BFGS-B",
    "tol": 1e-6,
    "gtol": 1e-3,
    "max_iter": 100,
    "constant_bounds": (-5, 5),
    "initialization": "random",  # random, mean
    "max_constants": 8,
    "max_expr_length": -1,
    "backend": "stack",
}


class ParameterEstimator:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        symbol_library: Optional[SymbolLibrary] = None,
        seed: Optional[int] = None,
        **kwargs: Unpack[RankingSettings],
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
                [RankingSettings][SRToolkit.utils.types.RankingSettings].
                Supported keys: ``method``, ``tol``, ``gtol``, ``max_iter``,
                ``constant_bounds``, ``initialization``, ``max_constants``,
                ``max_expr_length``, ``backend``.

        Attributes:
            symbol_library: The symbol library used.
            X: Input data.
            y: Target values.
            seed: Random seed.
            estimation_settings: Active settings dict, merged from defaults and ``**kwargs``.
        """
        if symbol_library is None:
            symbol_library = SymbolLibrary.get_or_default()
        self.symbol_library = symbol_library
        self.X = X
        self.y = y
        self.seed = seed

        self.estimation_settings = dict(DEFAULT_ESTIMATION_SETTINGS)

        if kwargs:
            for k in self.estimation_settings.keys():
                if k in kwargs:
                    self.estimation_settings[k] = kwargs[k]  # type: ignore[literal-required]

        bounds = self.estimation_settings["constant_bounds"]
        if not (isinstance(bounds, (tuple, list, np.ndarray)) and len(bounds) == 2):
            raise ValueError(f"constant_bounds must be a (lower, upper) pair, got {bounds!r}")
        self.estimation_settings["constant_bounds"] = (bounds[0], bounds[1])

        if self.estimation_settings["backend"] not in ("stack", "codegen", "stack_py"):
            raise ValueError(
                f"backend must be 'stack', 'codegen', or 'stack_py', got {self.estimation_settings['backend']!r}"
            )

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
        rmse, parameters, _ = self.estimate_parameters_with_info(expr)
        return rmse, parameters

    def estimate_parameters_with_info(self, expr: Union[List[str], Node]) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """
        Like [estimate_parameters][SRToolkit.evaluation.parameter_estimator.ParameterEstimator.estimate_parameters],
        but also reports how the optimizer ended.

        The returned error is the lowest finite RMSE the optimizer evaluated, with the constants
        that gave it, rather than the optimizer's final value. When a step leaves the region where
        the expression is defined (e.g. ``sqrt(C - X_0)`` with ``C < X_0``), L-BFGS-B aborts and
        reports ``NaN`` even though it had already found good constants.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> pe = ParameterEstimator(X, y, seed=0)
            >>> rmse, constants, info = pe.estimate_parameters_with_info(["C", "*", "X_1", "-", "X_0"])
            >>> sorted(info)
            ['converged', 'evaluations', 'iterations']
            >>> info["converged"], info["evaluations"] >= info["iterations"]
            (True, True)
            >>> pe.estimate_parameters_with_info(["X_1", "-", "X_0"])[2]
            {}

        Args:
            expr: Expression as a token list in infix notation or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.

        Returns:
            A 3-tuple ``(rmse, parameters, info)``. ``info`` is empty when the optimizer did not
            run (no free constants, or a violated limit). Otherwise it holds:

            - ``converged``: ``scipy``'s ``success`` flag;
            - ``evaluations``: how often the RMSE was computed (``nfev``), including the extra
              evaluations for the finite-difference gradient, so the cost of the fit;
            - ``iterations``: optimizer iterations (``nit``);
            - ``message``: the optimizer's message, only when ``converged`` is ``False``.

            ``converged=False`` does not mean a bad fit: RMSE has no gradient at a
            perfect fit, so L-BFGS-B often ends with ``"ABNORMAL"`` right at the optimum (about 1
            in 10 random starts for an exact fit). Judge a fit by its error.
        """
        # Count on the infix token list, brackets included, as SR_evaluator does, so both apply the
        # limits identically. (For a Node, len() would count tree nodes, without brackets.)
        tokens = expr.to_list(symbol_library=self.symbol_library) if isinstance(expr, Node) else expr
        num_constants = sum([1 for t in tokens if self.symbol_library.get_type(t) == "const"])
        if (
            isinstance(self.estimation_settings["max_constants"], int)
            and 0 <= self.estimation_settings["max_constants"] < num_constants
        ):
            return np.nan, np.array([]), {}

        if isinstance(self.estimation_settings["max_expr_length"], int) and 0 <= self.estimation_settings[
            "max_expr_length"
        ] < len(tokens):
            return np.nan, np.array([]), {}

        executable_error_fn = compile_expr_rmse(
            expr,
            self.symbol_library,
            backend=str(self.estimation_settings["backend"]),
            X=self.X if num_constants > 0 else None,
        )

        if num_constants == 0:
            rmse = executable_error_fn(self.X, np.array([]), self.y)
            return rmse, np.array([]), {}
        else:
            return self._optimize_parameters(executable_error_fn, num_constants)

    def _optimize_parameters(
        self, executable_error_fn: Callable, num_constants: int
    ) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """
        Run L-BFGS-B to minimize *executable_error_fn* over *num_constants* constant values.

        Args:
            executable_error_fn: Compiled error function ``f(X, C, y)`` returning scalar RMSE.
            num_constants: Number of free constants to optimize.

        Returns:
            A 3-tuple ``(rmse, parameters, info)``: the lowest finite RMSE evaluated and its
            constants (the optimizer's final point if none was finite), and how the optimizer
            ended.
        """
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

        # Remember the best finite value evaluated: L-BFGS-B reports NaN as its result when a step
        # lands where the expression is undefined, discarding what it had already found.
        best_value = np.inf
        best_constants: Optional[np.ndarray] = None

        def objective(c: np.ndarray) -> float:
            nonlocal best_value, best_constants
            value = executable_error_fn(self.X, c, self.y)
            if np.isfinite(value) and value < best_value:
                best_value = float(value)
                best_constants = np.array(c, copy=True)
            return value

        res = minimize(
            objective,
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
        info: Dict[str, Any] = {
            "converged": bool(res.success),
            "evaluations": int(res.nfev),
            "iterations": int(res.nit),
        }
        if not res.success:
            info["message"] = str(res.message)
        if best_constants is None:
            return res.fun, res.x, info
        return best_value, best_constants, info
