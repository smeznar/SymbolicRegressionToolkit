"""Shared pytest fixtures and Hypothesis configuration for the test suite."""

import sys

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


@pytest.fixture(autouse=True)
def _reset_default_symbol_library():
    """Clear the module-level default SymbolLibrary before and after every test."""
    SymbolLibrary.set_default(None)
    yield
    SymbolLibrary.set_default(None)
