"""Shared Hypothesis strategies and test doubles for the test suite.

Sizes are bounded so that property tests stay fast.
"""

import os
import random
import signal
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from SRToolkit.approaches.sr_approach import ApproachConfig, SR_approach
from SRToolkit.evaluation.callbacks import SR_callback

# --- Expressions -----------------------------------------------------------------------------

EXPR_LEAVES = ["X_0", "X_1", "C", "1", "2"]
EXPR_BINARY = ["+", "-", "*", "/"]
EXPR_UNARY = ["sin", "cos", "sqrt"]


def _extend(children):
    binary = st.tuples(children, st.sampled_from(EXPR_BINARY), children).map(
        lambda t: ["("] + t[0] + [t[1]] + t[2] + [")"]
    )
    unary = st.tuples(st.sampled_from(EXPR_UNARY), children).map(lambda t: [t[0], "("] + t[1] + [")"])
    return binary | unary


def infix_expressions(max_leaves: int = 6):
    """Fully parenthesised infix token lists over the default symbol library.

    Leaves are ``X_0``, ``X_1``, ``C``, ``1`` and ``2``; operators ``+ - * /``; functions
    ``sin``, ``cos``, ``sqrt``. Every subtree is bracketed, so precedence never matters.
    """
    return st.recursive(st.sampled_from(EXPR_LEAVES).map(lambda s: [s]), _extend, max_leaves=max_leaves)


EXPR_CONSTANT_LABELS = ["C", "C_0", "C_1", "C_7"]


@st.composite
def labelled_infix_expressions(draw, max_leaves: int = 6):
    """``infix_expressions`` with every ``C`` leaf replaced by a draw from ``EXPR_CONSTANT_LABELS``.

    Labels can repeat, so the expressions have shared (labelled) and bare parameters.
    """
    tokens = draw(infix_expressions(max_leaves))
    return [draw(st.sampled_from(EXPR_CONSTANT_LABELS)) if t == "C" else t for t in tokens]


# --- Behavior matrices -----------------------------------------------------------------------

_FINITE = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


@st.composite
def matrix_pairs(draw, max_rows: int = 5, max_cols: int = 6):
    """Two finite float64 matrices of the same shape ``(n, k)``, ``1 <= n <= max_rows``, ``1 <= k <= max_cols``."""
    shape = (draw(st.integers(1, max_rows)), draw(st.integers(1, max_cols)))
    return draw(hnp.arrays(np.float64, shape, elements=_FINITE)), draw(hnp.arrays(np.float64, shape, elements=_FINITE))


# --- Regression data -------------------------------------------------------------------------


@st.composite
def regression_data(draw, min_n: int = 2, max_n: int = 20):
    """``(X, y, y_pred)`` with finite values in ``[-1e3, 1e3]``; ``X`` has one column and ``std(y) > 1e-3``."""
    n = draw(st.integers(min_n, max_n))
    X = draw(hnp.arrays(np.float64, (n, 1), elements=_FINITE))
    y = draw(hnp.arrays(np.float64, (n,), elements=_FINITE).filter(lambda y: np.std(y) > 1e-3))
    y_pred = draw(hnp.arrays(np.float64, (n,), elements=_FINITE))
    return X, y, y_pred


# --- JSON-serialisable values ----------------------------------------------------------------

_SERIAL_TEXT = st.text(alphabet=st.characters(codec="utf-8", exclude_characters="\x00"), max_size=4)
_SERIAL_KEY_SCALARS = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False)
    | _SERIAL_TEXT
    | st.sampled_from(["__ndarray__", "__tuple__", "__dict__"])
)
_SERIAL_KEYS = st.recursive(_SERIAL_KEY_SCALARS, lambda k: st.tuples(k) | st.tuples(k, k), max_leaves=4)
_SERIAL_ARRAY_DTYPES = [np.bool_, np.int8, np.uint8, np.int64, np.float32, np.float64]


@st.composite
def _serial_arrays(draw):
    shape = draw(hnp.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=3))
    if draw(st.booleans()):
        return draw(hnp.arrays(np.dtype("<U3"), shape, elements=st.text(alphabet="abé", max_size=3)))
    return draw(hnp.arrays(draw(st.sampled_from(_SERIAL_ARRAY_DTYPES)), shape))


def serializable_values(max_leaves: int = 12):
    """Values supported by ``SRToolkit.utils.serialization`` without numpy scalars or subclasses.

    Leaves: ``None``, bool, int, float (``nan``/``inf`` included), text, and arrays of dtype bool, int8,
    uint8, int64, float32, float64 or ``<U3`` with 0-3 dimensions (zero-length sides allowed).
    Containers: lists, tuples and dicts whose keys are scalars, reserved marker names or nested tuples.
    """
    leaves = st.none() | st.booleans() | st.integers() | st.floats() | _SERIAL_TEXT | _serial_arrays()
    return st.recursive(
        leaves,
        lambda c: (
            st.lists(c, max_size=3) | st.lists(c, max_size=3).map(tuple) | st.dictionaries(_SERIAL_KEYS, c, max_size=3)
        ),
        max_leaves=max_leaves,
    )


# --- Physical units --------------------------------------------------------------------------

BASE_UNITS = ["m", "kg", "s", "A", "K", "mol", "cd"]
DERIVED_UNITS = ["N", "J", "W", "Pa", "Hz", "C", "V", "Ohm", "F", "T"]
CUSTOM_DIMENSIONS = ["bit", "usd"]

_EXPONENTS = st.builds(Fraction, st.integers(-3, 3), st.integers(1, 4))


@st.composite
def units(draw, custom: bool = False):
    """A ``Unit`` from up to three names with exponents ``p/q``, ``|p| <= 3``, ``1 <= q <= 4``.

    Names are base and derived units (derived ones expand to base dimensions), plus ``bit`` /
    ``usd`` as confirmed custom units when ``custom``. ``units.py`` is imported lazily so that
    modules using the other strategies do not depend on it.
    """
    from SRToolkit.utils.units import Unit

    names = BASE_UNITS + DERIVED_UNITS + (CUSTOM_DIMENSIONS if custom else [])
    exponents = draw(st.dictionaries(st.sampled_from(names), _EXPONENTS, max_size=3))
    return Unit(exponents, custom_units=CUSTOM_DIMENSIONS if custom else ())


# --- Test doubles for checkpoint and signal tests --------------------------------------------


# None of them fits y = 2*X_0 or y = X_0*X_0 exactly, so no run stops on its success threshold.
SCRIPTED_CALLS = [["X_0"], ["X_0", "+", "C"], ["X_0"], ["X_0", "-", "C"], ["C"], ["X_0", "+", "X_0", "+", "X_0"]]


def result_dict(result):
    """``result.to_dict()`` without ``wall_time`` and ``limits``, which differ between two runs that are
    otherwise equal (e.g. an interrupted run records the callback that interrupted it)."""
    return {**result.to_dict(), "wall_time": None, "limits": None}


def result_answers(result, k, budgets=range(-1, 12), thresholds=(0.05, 0.1, 0.3, 0.5, 1.0, 2.0, float("nan"))):
    """Every public answer of an ``EvalResult`` that a compact result (one with a ``summary``
    keeping the top ``k`` models) must give like the full one. Compare with
    ``np.testing.assert_equal``, which treats NaN as equal."""
    return {
        "num_evaluated": result.num_evaluated,
        "evaluation_calls": result.evaluation_calls,
        "top_models": [result.top_models(j) for j in range(1, k + 1)],
        "min_error": result.min_error,
        "best_expr": result.best_expr,
        "success": result.success,
        "status_counts": result.status_counts(),
        "call_status_counts": result.status_counts(per="call"),
        "best_so_far": result.best_so_far(),
        "best_error_within": result.best_error_within(list(budgets)),
        "calls_to_threshold": [result.calls_to_threshold(t) for t in thresholds],
        "pareto_front": result.pareto_front(),
    }


# --- Model store runs --------------------------------------------------------------------------

# Lengths 1, 1, 1, 3, 3, 5, 5: ties in length.
STORE_POOL = [
    ["X_0"],
    ["X_1"],
    ["C"],
    ["C", "*", "X_0"],
    ["X_0", "+", "C"],
    ["X_0", "*", "X_0", "+", "C"],
    ["X_1", "-", "X_0", "*", "C"],
]


def store_calls(max_calls: int = 25):
    """Lists of ``(pool_index, error)``: one counted call each, of ``STORE_POOL[pool_index]``. A
    repeated expression is a cache hit (its error is ignored). Errors come from a small set so
    ties are frequent; ``inf`` is a failed model."""
    entry = st.tuples(st.integers(0, len(STORE_POOL) - 1), st.sampled_from([0.1, 0.5, 1.0, float("inf")]))
    return st.lists(entry, max_size=max_calls)


def send_signal(sig: int) -> None:
    """Send ``sig`` to this process, but only if a handler other than the default is installed, so
    a missing handler fails the test instead of interrupting or killing the test run."""
    if signal.getsignal(sig) in (signal.SIG_DFL, signal.default_int_handler):
        raise RuntimeError(f"no signal handler installed for {signal.Signals(sig).name}")
    os.kill(os.getpid(), sig)


@dataclass
class ScriptedConfig(ApproachConfig):
    name: str = "scripted"
    kill_seed: int | None = None
    raise_seed: int | None = None
    at_call: int = 3
    sig: int = int(signal.SIGINT)


class ScriptedApproach(SR_approach):
    """Evaluates SCRIPTED_CALLS in order while ``should_stop`` is False, so a run is reproducible.

    In the experiment seeded ``kill_seed`` it sends ``sig`` to the process after call ``at_call``;
    in the one seeded ``raise_seed`` it raises ``RuntimeError("boom")`` there. Records each search's
    seed and the SIGINT handler installed during it.
    """

    def __init__(self, **kwargs):
        super().__init__(ScriptedConfig(**kwargs))
        self.seeds = []
        self.handlers = []

    @classmethod
    def from_config(cls, config):
        return cls(**{k: config[k] for k in ("kill_seed", "raise_seed", "at_call", "sig")})

    def search(self, sr_evaluator, seed=None):
        self.seeds.append(seed)
        self.handlers.append(signal.getsignal(signal.SIGINT))
        for k, expr in enumerate(SCRIPTED_CALLS, start=1):
            if sr_evaluator.should_stop:
                break
            sr_evaluator.evaluate_expr(expr)
            if k == self.config.at_call and seed is not None and seed == self.config.kill_seed:
                send_signal(self.config.sig)
            if k == self.config.at_call and seed is not None and seed == self.config.raise_seed:
                raise RuntimeError("boom")


class EventLog(SR_callback):
    """Appends each evaluation event to the class-level ``events``, so it can be read even when a job
    rebuilds its callbacks from their dicts. Clear ``EventLog.events`` before use."""

    events: list = []
    ended: list = []  # the stop reason of each experiment that ended
    references: list = []  # the checkpoint reference of each experiment that ended

    def on_expr_evaluated(self, event):
        EventLog.events.append(event.expression)

    def on_experiment_end(self, event, results):
        EventLog.ended.append(results.stop_reason)
        EventLog.references.append(results.checkpoint)


class SignalAfter(SR_callback):
    """Sends ``sig`` to the process at the ``n``-th evaluation event (inside the step that made it),
    only in the experiment seeded ``seed`` if that is set."""

    def __init__(self, n, sig=int(signal.SIGINT), seed=None):
        self.n, self.sig, self.seed = n, sig, seed
        self.count = 0
        self.current_seed = None

    def on_experiment_start(self, event):
        self.current_seed, self.count = event.seed, 0

    def on_expr_evaluated(self, event):
        if self.seed is not None and self.current_seed != self.seed:
            return None
        self.count += 1
        if self.count == self.n:
            send_signal(self.sig)
        return None

    def to_dict(self):
        return {**super().to_dict(), "n": self.n, "sig": self.sig, "seed": self.seed}

    @classmethod
    def from_dict(cls, d):
        return cls(d["n"], d["sig"], d["seed"])


# None of them fits y = 2*X_0 or y = X_0*X_0 exactly.
STEPPED_POOL = [
    ["X_0"],
    ["X_0", "+", "C"],
    ["X_0", "-", "C"],
    ["C"],
    ["X_0", "+", "X_0", "+", "X_0"],
    ["C", "-", "X_0"],
]


@dataclass
class SteppedConfig(ApproachConfig):
    name: str = "stepped"
    batch: int = 2
    stop_after_asks: int | None = None
    stop_with: str | None = None


class SteppedApproach(SR_approach):
    """A step-interface approach whose proposals depend on the global random states.

    ``start`` seeds Python's ``random`` and NumPy; each ``ask`` draws ``batch`` token lists from
    STEPPED_POOL with both. After ``stop_after_asks`` asks it returns no candidate, first calling
    ``se.stop(stop_with)`` if that is set. ``tell`` logs what it gets; ``started`` records ``start``
    and ``evaluator`` the evaluator it got.
    The search state is ``{"asks": number of asks}``, pickled by the default state methods.
    """

    def __init__(self, **kwargs):
        super().__init__(SteppedConfig(**kwargs))
        self.told = []
        self.started = False
        self.evaluator = None

    @classmethod
    def from_config(cls, config):
        return cls(**{k: config[k] for k in ("batch", "stop_after_asks", "stop_with")})

    def start(self, sr_evaluator, seed=None):
        self.started = True
        self.evaluator = sr_evaluator
        random.seed(seed)
        np.random.seed(seed)
        return {"asks": 0}

    def ask(self, sr_evaluator, state):
        if self.config.stop_after_asks is not None and state["asks"] >= self.config.stop_after_asks:
            if self.config.stop_with is not None:
                sr_evaluator.stop(self.config.stop_with)
            return []
        state["asks"] += 1
        n = len(STEPPED_POOL)
        return [STEPPED_POOL[(int(np.random.randint(n)) + random.randrange(n)) % n] for _ in range(self.config.batch)]

    def tell(self, sr_evaluator, state, candidates, results):
        self.told.append((list(candidates), list(results)))
