"""Fitting of free constants by minimizing RMSE.

[ParameterEstimator][SRToolkit.evaluation.parameter_estimator.ParameterEstimator] fits the
parameters of an expression (its free constants, in the order of
[parameter_labels][SRToolkit.utils.expression.Expression.parameter_labels]) to data ``(X, y)`` with
``scipy.optimize.minimize``, configured by the ``minimize`` settings section.
[RMSERanking][SRToolkit.evaluation.ranking.RMSERanking] scores expressions with it.

**Specs.** Each parameter is fitted under a spec (see
[free_constants][SRToolkit.utils.free_constants]). Specs given for an expression's labelled
constants, carried by an [Expression][SRToolkit.utils.expression.Expression], are completed by
[Expression.resolve][SRToolkit.utils.expression.Expression.resolve]; every other parameter uses
the default spec ``default_constant`` (a complete template), by default
[uniform_default((-5, 5))][SRToolkit.utils.free_constants.uniform_default]. A fixed parameter
keeps its ``value`` and is not fitted; the returned vector holds it in place.

**Starting points.** A free parameter starts at its spec's ``value`` if given. Otherwise, by the
setting ``estimation.initialization``:

- ``"random"``: a draw from its prior (the prior's ``ppf`` of a uniform draw, or a
  direct draw when the prior does not implement [ppf][SRToolkit.dataset.sampling.Sampler.ppf]);
- ``"mean"``: the prior's ``ppf(0.5)`` (the midpoint for a uniform prior).

A start outside the spec's bounds is clipped to the nearest bound. Each parameter is fitted within
its bounds; an infinite end leaves that side unbounded.

**Seeding.** With ``"random"`` initialization, each fit's starting point is drawn from a
generator seeded with a hash of ``seed``, the expression's tokens (a token list as given, an
``Expression`` by its ``tokens``) and its specs (in any order; the order does not affect the seed). An expression's fit therefore depends only on ``seed``, its tokens, its
specs, ``default_constant``, ``X``, ``y`` and the ``estimation`` and ``minimize`` settings, not on
earlier fits, and is the same in every process, on every platform and Python version. With
``seed=None`` every fit draws a fresh random starting point.
"""

import hashlib
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize

from SRToolkit.utils.expression import Expression, _as_expression
from SRToolkit.utils.expression_compiler import compile_expr_rmse
from SRToolkit.utils.free_constants import Constant, _check_default_constant, _spec_key, uniform_default
from SRToolkit.utils.settings import Settings
from SRToolkit.utils.symbol_library import SymbolLibrary


class ParameterEstimator:
    """Fits free constants in symbolic expressions by minimizing RMSE against target values.

    Each parameter is fitted under its spec, or under the default spec ``default_constant`` if
    it has none. See the
    [module docstring][SRToolkit.evaluation.parameter_estimator] for specs, starting points, bounds
    and seeding.

    Args:
        X: Input data of shape ``(n_samples, n_features)`` used to evaluate expressions.
        y: Target values of shape ``(n_samples,)``.
        symbol_library: Symbol library defining the token vocabulary. ``None``: the active
            library ([get_or_default][SRToolkit.utils.symbol_library.SymbolLibrary.get_or_default]).
        seed: Seed of the random starting points (only with ``"random"`` initialization),
            combined with each expression's tokens and the specs passed for it. ``None`` gives a
            fresh random starting point per fit. Default ``None``.
        settings: The run's [Settings][SRToolkit.utils.settings.Settings]; the estimator reads
            the ``estimation`` and ``minimize`` sections and ``evaluator.backend``. ``None``: the
            defaults.
        default_constant: The default spec, a complete
            [Constant][SRToolkit.utils.free_constants.Constant] template. ``None``:
            [uniform_default((-5, 5))][SRToolkit.utils.free_constants.uniform_default].

    Raises:
        TypeError: If ``settings`` is not a ``Settings`` or ``default_constant`` is not a
            ``Constant``.
        ValueError: If ``default_constant`` is not a complete template (``prior`` and ``bounds``
            set, no ``label``, no ``value``, ``fixed=False``); the message names it.

    Attributes:
        symbol_library: The symbol library used.
        X: Input data.
        y: Target values.
        seed: The ``seed`` argument.
        settings: The settings used.
        default_constant: The default spec used.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        symbol_library: SymbolLibrary | None = None,
        seed: int | None = None,
        settings: Settings | None = None,
        default_constant: Constant | None = None,
    ) -> None:
        if symbol_library is None:
            symbol_library = SymbolLibrary.get_or_default()
        if settings is None:
            settings = Settings()
        if not isinstance(settings, Settings):
            raise TypeError(f"[ParameterEstimator] settings must be a Settings, got {type(settings).__name__}.")
        if default_constant is None:
            default_constant = uniform_default((-5, 5))
        if not isinstance(default_constant, Constant):
            raise TypeError(
                f"[ParameterEstimator] default_constant must be a Constant, got {type(default_constant).__name__}."
            )
        try:
            _check_default_constant(default_constant)
        except ValueError as e:
            raise ValueError(f"[ParameterEstimator] {e}") from None
        self.symbol_library = symbol_library
        self.X = X
        self.y = y
        self.seed = seed
        self.settings = settings
        self.default_constant = default_constant

    def estimate_parameters(self, expr: list[str] | Expression) -> tuple[float, np.ndarray]:
        """Fit free constants in *expr* by minimizing RMSE against the target values.

        Expressions without free parameters (no constants, or only fixed ones) are evaluated directly,
        without running the optimizer. Limits on the number of constants or tokens are not checked
        here; [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] enforces them as
        required constraints.

        Args:
            expr: The expression, as an infix token list (no specs) or an
                [Expression][SRToolkit.utils.expression.Expression], whose specs of labelled
                constants are used. Parameters without a spec use ``default_constant``.

        Returns:
            A 2-tuple ``(rmse, parameters)``: the lowest finite RMSE evaluated and the parameters
            that gave it, one per parameter in the order of
            [parameter_labels][SRToolkit.utils.expression.Expression.parameter_labels], fixed values
            included. If no evaluation was finite, ``rmse`` is ``nan`` or ``inf`` and the free
            entries of ``parameters`` are the optimizer's final point.

        Raises:
            TypeError: If ``expr`` is neither a token list nor an ``Expression``.
            ExpressionParseError: If ``expr`` is not a well-formed expression.
            ValueError: If a spec of ``expr`` that is not fixed has a ``value`` outside the bounds
                it takes from ``default_constant``.
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
            >>> from SRToolkit.utils import Constant, Expression
            >>> spec = Constant(label="C_0", prior=UniformSampling(0, 10), bounds=(0, float("inf")))
            >>> expr = Expression(["C_0", "*", "X_1", "-", "X_0"], free_constants=[spec])
            >>> rmse, constants = pe.estimate_parameters(expr)
            >>> print(rmse < 1e-6, 1.99 < constants[0] < 2.01)
            True True
        """
        rmse, parameters, _ = self.estimate_parameters_with_info(expr)
        return rmse, parameters

    def estimate_parameters_with_info(self, expr: list[str] | Expression) -> tuple[float, np.ndarray, dict[str, Any]]:
        """Fit free constants in *expr* and report how the optimizer ended.

        Like [estimate_parameters][SRToolkit.evaluation.parameter_estimator.ParameterEstimator.estimate_parameters],
        and also reports how the optimizer ended.

        Args:
            expr: The expression, as in ``estimate_parameters``.

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
        # The tokens and specs also seed the random starting point.
        expr = _as_expression(expr, self.symbol_library)
        specs = expr.resolve(self.default_constant)

        executable_error_fn = compile_expr_rmse(
            expr,
            self.symbol_library,
            backend=self.settings.evaluator.backend,
            X=self.X if len(specs) > 0 else None,
        )

        # The full parameter vector: fixed values in place, free entries filled in by the fit.
        parameters = np.array([s.value if s.fixed else np.nan for s in specs], dtype=np.float64)
        free = [i for i, s in enumerate(specs) if not s.fixed]
        if not free:
            return executable_error_fn(self.X, parameters, self.y), parameters, {}
        return self._optimize_parameters(
            executable_error_fn, specs, free, parameters, list(expr.tokens), _spec_key(expr.free_constants)
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
        if self.settings.estimation.initialization == "random":
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
            method=self.settings.minimize.method,
            tol=self.settings.minimize.tol,
            options={
                "maxiter": self.settings.minimize.max_iter,
                "gtol": self.settings.minimize.gtol,
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
            assert prior is not None and spec.bounds is not None  # set by resolve
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
