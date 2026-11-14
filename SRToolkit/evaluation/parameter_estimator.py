"""Constant parameter estimation for symbolic regression expressions using numerical optimization."""

import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from typing_extensions import Unpack

from SRToolkit.utils.expression_compiler import compile_expr_rmse
from SRToolkit.utils.expression_constraints import _validate_limit
from SRToolkit.utils.expression_tree import Node, _as_tokens, _count_constants
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
    """Fits free constants in symbolic expressions by minimizing RMSE against target values.

    With ``initialization="random"``, each fit's starting point is drawn from a generator seeded
    with a hash of ``seed`` and the expression's tokens (a tree is seeded as its ``to_list`` tokens,
    a token list as given). An expression's fit therefore depends only on ``seed``, its tokens,
    ``X``, ``y`` and the settings, not on earlier fits, and is the same in every process, on every
    platform and Python version. With ``seed=None`` every fit draws a fresh random starting point.

    Args:
        X: Input data of shape ``(n_samples, n_features)`` used to evaluate expressions.
        y: Target values of shape ``(n_samples,)``.
        symbol_library: Symbol library defining the token vocabulary.
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].
        seed: Seed of the random starting points (only with ``initialization="random"``),
            combined with each expression's tokens. ``None`` gives a fresh random starting point
            per fit. Default ``None``.
        **kwargs: Optional estimation settings from
            [RankingSettings][SRToolkit.utils.types.RankingSettings].
            Supported keys: ``method``, ``tol``, ``gtol``, ``max_iter``,
            ``constant_bounds``, ``initialization``, ``max_constants``,
            ``max_expr_length``, ``backend``.

    Raises:
        ValueError: If ``constant_bounds`` is not a ``(lower, upper)`` pair, ``backend`` is not
            one of the supported values, or ``max_constants`` / ``max_expr_length`` is not an
            ``int`` or numpy integer (``bool`` is rejected).

    Attributes:
        symbol_library: The symbol library used.
        X: Input data.
        y: Target values.
        seed: The ``seed`` argument.
        estimation_settings: Active settings dict, merged from defaults and ``**kwargs``.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        symbol_library: Optional[SymbolLibrary] = None,
        seed: Optional[int] = None,
        **kwargs: Unpack[RankingSettings],
    ) -> None:
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

        for key in ("max_constants", "max_expr_length"):
            value = self.estimation_settings[key]  # type: ignore[literal-required]
            self.estimation_settings[key] = _validate_limit(value, key, allow_negative=True)  # type: ignore[literal-required]

        bounds = self.estimation_settings["constant_bounds"]
        if not (isinstance(bounds, (tuple, list, np.ndarray)) and len(bounds) == 2):
            raise ValueError(f"constant_bounds must be a (lower, upper) pair, got {bounds!r}")
        self.estimation_settings["constant_bounds"] = (bounds[0], bounds[1])

        if self.estimation_settings["backend"] not in ("stack", "codegen", "stack_py"):
            raise ValueError(
                f"backend must be 'stack', 'codegen', or 'stack_py', got {self.estimation_settings['backend']!r}"
            )

    def estimate_parameters(self, expr: Union[List[str], Node]) -> Tuple[float, np.ndarray]:
        """Fit free constants in *expr* by minimizing RMSE against the target values.

        Expressions that exceed a non-negative ``max_constants`` or ``max_expr_length``
        immediately return ``(NaN, [])``; the counts are those of
        [MaxConstants][SRToolkit.utils.expression_constraints.MaxConstants] and
        [MaxTokens][SRToolkit.utils.expression_constraints.MaxTokens] (a tree is written out with
        ``to_list`` before its tokens are counted). Expressions with no free constants are
        evaluated directly without running the optimizer.

        Args:
            expr: Expression as a token list in infix notation or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.

        Returns:
            A 2-tuple ``(rmse, parameters)``: the lowest finite RMSE evaluated and the 1-D array of
            constants that gave it (empty if the expression has none). If no evaluation was finite
            (e.g. the expression is undefined on ``X``), ``rmse`` is non-finite (``nan`` or
            ``inf``) and ``parameters`` is the optimizer's final point. ``(NaN, [])`` if the
            expression violates ``max_constants`` or ``max_expr_length``.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> pe = ParameterEstimator(X, y)
            >>> rmse, constants = pe.estimate_parameters(["C", "*", "X_1", "-", "X_0"])
            >>> print(rmse < 1e-6)
            True
            >>> print(1.99 < constants[0] < 2.01)
            True
        """
        rmse, parameters, _ = self.estimate_parameters_with_info(expr)
        return rmse, parameters

    def estimate_parameters_with_info(self, expr: Union[List[str], Node]) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """Fit free constants in *expr* and report how the optimizer ended.

        Like [estimate_parameters][SRToolkit.evaluation.parameter_estimator.ParameterEstimator.estimate_parameters].
        The returned error is the lowest finite RMSE the optimizer evaluated, with the constants
        that gave it, rather than the optimizer's final value. When a step leaves the region where
        the expression is defined (e.g. ``sqrt(C - X_0)`` with ``C < X_0``), L-BFGS-B aborts and
        reports ``NaN`` even though it had already found good constants.

        Args:
            expr: Expression as a token list in infix notation or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.

        Returns:
            A 3-tuple ``(rmse, parameters, info)``. ``rmse`` and ``parameters`` are as in
            ``estimate_parameters``, so ``rmse`` is non-finite (``nan`` or ``inf``) if no
            evaluation was finite. ``info`` is empty when the optimizer did not run (no free
            constants, or a violated limit). Otherwise it holds:

            - ``converged``: ``scipy``'s ``success`` flag;
            - ``evaluations``: RMSE computations (``nfev``), finite-difference gradient included;
            - ``iterations``: optimizer iterations (``nit``);
            - ``message``: the optimizer's message, only when ``converged`` is ``False``.

            ``converged=False`` does not mean a bad fit: RMSE has no gradient at a perfect fit, so
            L-BFGS-B often ends with ``"ABNORMAL"`` right at the optimum. Judge a fit by its error.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> pe = ParameterEstimator(X, y, seed=0)
            >>> rmse, constants, info = pe.estimate_parameters_with_info(["C", "*", "X_1", "-", "X_0"])
            >>> sorted(info)
            ['converged', 'evaluations', 'iterations']
            >>> info["converged"], info["evaluations"] >= info["iterations"]
            (True, True)
        """
        # Count on the infix token list, brackets included, as SR_evaluator does, so both apply the
        # limits identically. (For a Node, len() would count tree nodes, without brackets.)
        tokens = _as_tokens(expr, self.symbol_library)
        num_constants = _count_constants(tokens, self.symbol_library)
        if 0 <= self.estimation_settings["max_constants"] < num_constants:
            return np.nan, np.array([]), {}
        if 0 <= self.estimation_settings["max_expr_length"] < len(tokens):
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
            return self._optimize_parameters(executable_error_fn, num_constants, tokens)

    def _optimize_parameters(
        self, executable_error_fn: Callable, num_constants: int, tokens: List[str]
    ) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """Run L-BFGS-B to minimize *executable_error_fn* over *num_constants* constant values.

        Args:
            executable_error_fn: Compiled error function ``f(X, C, y)`` returning scalar RMSE.
            num_constants: Number of free constants to optimize.
            tokens: The expression's infix tokens, which seed its random starting point.

        Returns:
            A 3-tuple ``(rmse, parameters, info)``: the lowest finite RMSE evaluated and its
            constants (the optimizer's final point if none was finite), and how the optimizer
            ended.
        """
        if self.estimation_settings["initialization"] == "random":
            # Seed from a stable hash of (seed, tokens); Python's hash() is salted per process. The
            # separator is a control character no real token contains, so different token lists
            # get different seeds.
            fit_seed = None
            if self.seed is not None:
                key = f"{int(self.seed)}\x1f" + "\x1f".join(tokens)
                fit_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
            rng = np.random.default_rng(fit_seed)
            x0 = (
                rng.random(num_constants)
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
