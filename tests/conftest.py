"""Shared pytest fixtures and Hypothesis configuration for the test suite."""

import os
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


@pytest.fixture
def http_dir(tmp_path):
    """Serve a fresh directory over HTTP on localhost; yield ``(directory, base_url)``."""
    import functools
    import threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class _QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    served = tmp_path / "served"
    served.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(served)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield served, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _keep_egraphs_alive() -> None:
    # egglog frees an e-graph through its global rayon pool, whose threads a forked process does not
    # have, so freeing one after a fork hangs (mutmut forks a worker per mutant). Forked processes
    # keep every e-graph instead; mutmut's workers end with os._exit, which frees nothing.
    import egglog

    kept = []
    post_init = egglog.EGraph.__post_init__

    def keep(self, *args, **kwargs):
        kept.append(self)
        post_init(self, *args, **kwargs)

    egglog.EGraph.__post_init__ = keep
    # Keeping them alive makes simplifying safe here, so simplify need not warn.
    from SRToolkit.utils import expression_simplifier

    expression_simplifier._forked = False


os.register_at_fork(after_in_child=_keep_egraphs_alive)
