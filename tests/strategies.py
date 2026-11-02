"""Shared Hypothesis strategies for the test suite.

Sizes are bounded so that property tests stay fast.
"""

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

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
