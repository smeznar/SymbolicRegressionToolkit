"""Fitting of free constants by minimizing RMSE.

[ParameterEstimator][SRToolkit.evaluation.parameter_estimator.ParameterEstimator] fits the
parameters of an expression (its free constants, in the order of
[parameter_labels][SRToolkit.utils.expression_tree.parameter_labels]) to data ``(X, y)`` with
``scipy.optimize.minimize``.
[RMSERanking][SRToolkit.evaluation.ranking.RMSERanking] scores expressions with it.

**Specs.** Each parameter is fitted under a spec (see
[free_constants][SRToolkit.utils.free_constants]). Specs given for an expression's labelled
constants are completed by
[resolve_constants][SRToolkit.utils.free_constants.resolve_constants]; every other parameter uses
the default spec: the ``default_constant`` setting (a complete template) if given, else
[uniform_default(constant_bounds)][SRToolkit.utils.free_constants.uniform_default], with
``constant_bounds`` the setting of that name. A fixed parameter keeps its
``value`` and is not fitted; the returned vector holds it in place.

**Starting points.** A free parameter starts at its spec's ``value`` if given. Otherwise:

- ``initialization="random"``: a draw from its prior (the prior's ``ppf`` of a uniform draw, or a
  direct draw when the prior does not implement [ppf][SRToolkit.dataset.sampling.Sampler.ppf]);
- ``initialization="mean"``: the prior's ``ppf(0.5)`` (the midpoint for a uniform prior).

A start outside the spec's bounds is clipped to the nearest bound. Each parameter is fitted within
its bounds; an infinite end leaves that side unbounded.

**Seeding.** With ``initialization="random"``, each fit's starting point is drawn from a generator
seeded with a hash of ``seed``, the expression's tokens (a tree is seeded as its ``to_list`` tokens,
a token list as given) and the specs passed for it (in any order; the order does not affect the
seed). An expression's fit therefore depends only on ``seed``, its tokens, its specs, ``X``, ``y``
and the settings, not on earlier fits, and is the same in every process, on every platform and
Python version. With ``seed=None`` every fit draws a fresh random starting point.
"""

import hashlib
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from typing_extensions import Unpack

from SRToolkit.utils.expression_compiler import compile_expr_rmse
from SRToolkit.utils.expression_tree import Node, _as_tokens
from SRToolkit.utils.free_constants import Constant, _spec_key, resolve_constants, uniform_default
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import RankingSettings

# The settings ParameterEstimator reads, with their defaults. Any other keys passed in are ignored.
DEFAULT_ESTIMATION_SETTINGS: dict[str, Any] = {
    "method": "L-BFGS-B",
    "tol": 1e-6,
    "gtol": 1e-3,
    "max_iter": 100,
    "constant_bounds": (-5, 5),
    "default_constant": None,
    "initialization": "random",  # random, mean
    "backend": "stack",
}


class ParameterEstimator:
    """Fits free constants in symbolic expressions by minimizing RMSE against target values.

    Each parameter is fitted under its spec, or under the default spec ``default_constant`` if
    it has none. See the
    [module docstring][SRToolkit.evaluation.parameter_estimator] for specs, starting points, bounds
    and seeding.

    Args:
        X: Input data of shape ``(n_samples, n_features)`` used to evaluate expressions.
        y: Target values of shape ``(n_samples,)``.
        symbol_library: Symbol library defining the token vocabulary.
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].
        seed: Seed of the random starting points (only with ``initialization="random"``),
            combined with each expression's tokens and the specs passed for it. ``None`` gives a
            fresh random starting point per fit. Default ``None``.
        **kwargs: Optional estimation settings from
            [RankingSettings][SRToolkit.utils.types.RankingSettings].
            Supported keys: ``method``, ``tol``, ``gtol``, ``max_iter``,
            ``constant_bounds``, ``default_constant``, ``initialization``, ``backend``. ``None``
            means the default.

    Raises:
        TypeError: If ``default_constant`` is not given and ``constant_bounds`` is not a pair of real
            numbers, or ``default_constant`` is neither ``None`` nor a
            [Constant][SRToolkit.utils.free_constants.Constant].
        ValueError: If ``default_constant`` is not given and ``constant_bounds`` is not finite with
            ``lower < upper`` (``constant_bounds`` is not used otherwise); ``default_constant`` is
            not a complete template
            (``prior`` and ``bounds`` set, no ``label``, no ``value``, ``fixed=False``; this
            setting is not completed from ``constant_bounds``); ``initialization`` is not
            ``"random"`` or ``"mean"``; or ``backend`` is not one of the supported values. Errors
            about ``default_constant`` name it in the message.

    Attributes:
        symbol_library: The symbol library used.
        X: Input data.
        y: Target values.
        seed: The ``seed`` argument.
        estimation_settings: Active settings dict, merged from defaults and ``**kwargs``.
        default_constant: The default spec, for parameters without a spec: the
            ``default_constant`` setting, else
            [uniform_default(constant_bounds)][SRToolkit.utils.free_constants.uniform_default].
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        symbol_library: SymbolLibrary | None = None,
        seed: int | None = None,
        **kwargs: Unpack[RankingSettings],
    ) -> None:
        if symbol_library is None:
            symbol_library = SymbolLibrary.get_or_default()
        self.symbol_library = symbol_library
        self.X = X
        self.y = y
        self.seed = seed

        self.estimation_settings = dict(DEFAULT_ESTIMATION_SETTINGS)

        for k in self.estimation_settings.keys():
            if kwargs.get(k) is not None:  # None means the default
                self.estimation_settings[k] = kwargs[k]  # type: ignore[literal-required]

        given = self.estimation_settings["default_constant"]
        if given is None:
            self.default_constant: Constant = uniform_default(self.estimation_settings["constant_bounds"])
        else:
            if not isinstance(given, Constant):
                raise TypeError(
                    f"[ParameterEstimator] default_constant must be a Constant or None, got {type(given).__name__}."
                )
            try:
                resolve_constants([], None, given)  # raises ValueError unless it is a complete template
            except ValueError as e:
                raise ValueError(f"[ParameterEstimator] {e}") from None
            self.default_constant = given

        if self.estimation_settings["initialization"] not in ("random", "mean"):
            raise ValueError(
                f"[ParameterEstimator] initialization must be 'random' or 'mean', got {self.estimation_settings['initialization']!r}"
            )

        if self.estimation_settings["backend"] not in ("stack", "codegen", "stack_py"):
            raise ValueError(
                f"[ParameterEstimator] backend must be 'stack', 'codegen', or 'stack_py', got {self.estimation_settings['backend']!r}"
            )

    def estimate_parameters(
        self, expr: list[str] | Node, free_constants: Sequence[Constant] | None = None
    ) -> tuple[float, np.ndarray]:
        """Fit free constants in *expr* by minimizing RMSE against the target values.

        Expressions without free parameters (no constants, or only fixed ones) are evaluated directly,
        without running the optimizer. Limits on the number of constants or tokens are not checked
        here; [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] enforces them as
        required constraints.

        Args:
            expr: Expression as a token list in infix notation or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.
            free_constants: Specs for labelled constants of ``expr`` (see
                [Constant][SRToolkit.utils.free_constants.Constant]); parameters without one use
                ``default_constant``. In any order; ``None`` or empty: all use it.

        Returns:
            A 2-tuple ``(rmse, parameters)``: the lowest finite RMSE evaluated and the parameters
            that gave it, one per parameter in the order of
            [parameter_labels][SRToolkit.utils.expression_tree.parameter_labels], fixed values
            included. If no evaluation was finite, ``rmse`` is ``nan`` or ``inf`` and the free
            entries of ``parameters`` are the optimizer's final point.

        Raises:
            ValueError: If a spec in ``free_constants`` has no label, two specs share a label, a
                label is not a labelled constant of ``expr``, or a spec that is not fixed has a
                ``value`` outside the bounds it takes from ``default_constant``.
            NotImplementedError: If ``initialization="mean"`` and a free parameter without a
                ``value`` has a prior that does not implement
                [ppf][SRToolkit.dataset.sampling.Sampler.ppf].

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> pe = ParameterEstimator(X, y)
            >>> rmse, constants = pe.estimate_parameters(["C", "*", "X_1", "-", "X_0"])
            >>> print(rmse < 1e-6)
            True
            >>> print(1.99 < constants[0] < 2.01)
            True
            >>> from SRToolkit.dataset import UniformSampling
            >>> from SRToolkit.utils.free_constants import Constant
            >>> spec = Constant(label="C_0", prior=UniformSampling(0, 10), bounds=(0, float("inf")))
            >>> rmse, constants = pe.estimate_parameters(["C_0", "*", "X_1", "-", "X_0"], [spec])
            >>> print(rmse < 1e-6, 1.99 < constants[0] < 2.01)
            True True
        """
        rmse, parameters, _ = self.estimate_parameters_with_info(expr, free_constants)
        return rmse, parameters

    def estimate_parameters_with_info(
        self, expr: list[str] | Node, free_constants: Sequence[Constant] | None = None
    ) -> tuple[float, np.ndarray, dict[str, Any]]:
        """Fit free constants in *expr* and report how the optimizer ended.

        Like [estimate_parameters][SRToolkit.evaluation.parameter_estimator.ParameterEstimator.estimate_parameters],
        and also reports how the optimizer ended.

        Args:
            expr: Expression as a token list in infix notation or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.
            free_constants: Specs for labelled constants of ``expr``, as in ``estimate_parameters``.

        Returns:
            A 3-tuple ``(rmse, parameters, info)``. ``rmse`` and ``parameters`` are as in
            ``estimate_parameters``, so ``rmse`` is non-finite (``nan`` or ``inf``) if no
            evaluation was finite. ``info`` is empty when the optimizer did not run (no free
            constants, or only fixed ones). Otherwise it holds:

            - ``converged``: ``scipy``'s ``success`` flag;
            - ``evaluations``: RMSE computations (``nfev``), finite-difference gradient included;
            - ``iterations``: optimizer iterations (``nit``);
            - ``message``: the optimizer's message, only when ``converged`` is ``False``.

            ``converged=False`` does not mean a bad fit: RMSE has no gradient at a perfect fit, so
            L-BFGS-B often ends with ``"ABNORMAL"`` right at the optimum. Judge a fit by its error.

        Raises:
            ValueError: As for ``estimate_parameters``.
            NotImplementedError: As for ``estimate_parameters``.

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
        # Infix tokens: the specs are resolved against them, and they seed the random starting point.
        tokens = _as_tokens(expr, self.symbol_library)
        specs = resolve_constants(tokens, free_constants, self.default_constant, self.symbol_library)

        executable_error_fn = compile_expr_rmse(
            expr,
            self.symbol_library,
            backend=str(self.estimation_settings["backend"]),
            X=self.X if len(specs) > 0 else None,
        )

        # The full parameter vector: fixed values in place, free entries filled in by the fit.
        parameters = np.array([s.value if s.fixed else np.nan for s in specs], dtype=np.float64)
        free = [i for i, s in enumerate(specs) if not s.fixed]
        if not free:
            return executable_error_fn(self.X, parameters, self.y), parameters, {}
        return self._optimize_parameters(
            executable_error_fn, specs, free, parameters, tokens, _spec_key(free_constants)
        )

    def _optimize_parameters(
        self,
        executable_error_fn: Callable,
        specs: Sequence[Constant],
        free: list[int],
        parameters: np.ndarray,
        tokens: list[str],
        spec_key: str,
    ) -> tuple[float, np.ndarray, dict[str, Any]]:
        """Run the optimizer over the free parameters of *executable_error_fn*.

        Args:
            executable_error_fn: Compiled error function ``f(X, C, y)`` returning scalar RMSE.
            specs: One resolved spec per parameter.
            free: Indices of the parameters to fit (those not fixed).
            parameters: The full parameter vector with the fixed values in place.
            tokens: The expression's infix tokens, which seed its random starting point.
            spec_key: ``_spec_key`` of the specs given, also part of the seed (``""`` adds nothing).

        Returns:
            A 3-tuple ``(rmse, parameters, info)``: the lowest finite RMSE evaluated and its full
            parameter vector (the optimizer's final point if none was finite), and how the
            optimizer ended.
        """
        free_specs = [specs[i] for i in free]
        rng = None
        if self.estimation_settings["initialization"] == "random":
            # Seed from a stable hash of (seed, tokens, specs); Python's hash() is salted per process.
            # The separator is a control character no real token contains, so different token lists
            # get different seeds. Without specs the key is as before specs existed.
            fit_seed = None
            if self.seed is not None:
                key = f"{int(self.seed)}\x1f" + "\x1f".join(tokens) + (f"\x1f{spec_key}" if spec_key else "")
                fit_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
            rng = np.random.default_rng(fit_seed)
        x0 = self._starting_point(free_specs, rng)

        def full(c: np.ndarray) -> np.ndarray:
            vector = parameters.copy()
            vector[free] = c
            return vector

        # Remember the best finite value evaluated: L-BFGS-B reports NaN as its result when a step
        # lands where the expression is undefined, discarding what it had already found.
        best_value = np.inf
        best_constants: np.ndarray | None = None

        def objective(c: np.ndarray) -> float:
            nonlocal best_value, best_constants
            vector = full(c)
            value = executable_error_fn(self.X, vector, self.y)
            if np.isfinite(value) and value < best_value:
                best_value = float(value)
                best_constants = vector
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
            # An infinite end is passed as None: no bound on that side.
            bounds=[tuple(None if math.isinf(b) else b for b in s.bounds) for s in free_specs],  # type: ignore[union-attr]
        )
        info: dict[str, Any] = {
            "converged": bool(res.success),
            "evaluations": int(res.nfev),
            "iterations": int(res.nit),
        }
        if not res.success:
            info["message"] = str(res.message)
        if best_constants is None:
            return res.fun, full(res.x), info
        return best_value, best_constants, info

    def _starting_point(self, specs: Sequence[Constant], rng: np.random.Generator | None) -> np.ndarray:
        """Return the starting point of a fit, one value per spec (all free, prior and bounds set).

        A spec's ``value`` if given, else a point from its prior: drawn with ``rng``
        (``initialization="random"``) or its ``ppf(0.5)`` (``rng`` is ``None``, ``"mean"``).
        Clipped to the nearest bound of each spec.
        """
        # Imported here: SRToolkit.dataset imports this package.
        from SRToolkit.dataset.sampling import UniformSampling

        u = None if rng is None else rng.random(len(specs))
        x0 = np.empty(len(specs))
        for j, spec in enumerate(specs):
            prior = spec.prior
            assert prior is not None and spec.bounds is not None  # set by resolve_constants
            if spec.value is not None:
                x0[j] = spec.value
            elif isinstance(prior, UniformSampling) and not prior.mirror:
                # The formulas used before specs existed, so default specs keep their exact starts.
                low, high = prior.min_value, prior.max_value
                x0[j] = np.mean((low, high)) if u is None else u[j] * (high - low - 1e-8) + low
            elif u is None:
                x0[j] = prior.ppf(np.array([0.5]))[0]
            else:
                try:
                    x0[j] = prior.ppf(u[j : j + 1])[0]
                except NotImplementedError:
                    x0[j] = prior(1, rng=rng)[0]
            x0[j] = min(max(x0[j], spec.bounds[0]), spec.bounds[1])
        return x0
