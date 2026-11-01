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
