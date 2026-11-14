"""Shared pytest fixtures and Hypothesis configuration for the test suite."""

import sys
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, settings

from SRToolkit.utils.symbol_library import SymbolLibrary

# mutmut runs pytest in its own process and calls the same tests repeatedly (stats, clean run, each
# mutant). Under mutmut, use fixed examples (so a mutant is killed or not deterministically), no
# example database, no deadline (timing is noisy under mutation), and allow a @given test method
# to be called from several test-class instances.
settings.register_profile(
    "mutmut",
    derandomize=True,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.differing_executors],
)
if "mutmut" in sys.modules:
    settings.load_profile("mutmut")


@pytest.fixture
def fit_starts():
    """Record the starting point of every constant fit run during the test, in order."""
    from scipy.optimize import minimize as real_minimize

    starts = []

    def capture_x0(fn, x0, **kwargs):
        starts.append(x0.copy())
        return real_minimize(fn, x0, **kwargs)

    with patch("SRToolkit.evaluation.parameter_estimator.minimize", side_effect=capture_x0):
        yield starts


@pytest.fixture(autouse=True)
def _reset_default_symbol_library():
    """Clear the module-level default SymbolLibrary before and after every test."""
    SymbolLibrary.set_default(None)
    yield
    SymbolLibrary.set_default(None)
