"""
This module contains implementations of symbolic regression approaches. Class SR_approach is the base class for all
symbolic regression approaches.

Modules:
    sr_approach: The base class for all symbolic regression approaches.
    ProGED: The ProGED approach - Probabilistic grammar-based equation discovery.
"""

from .sr_approach import SR_approach
from .ProGED import ProGED

__all__ = ["SR_approach", "ProGED"]

try:
    from .EDHiE import EDHiE
except ImportError:
    class EDHiE:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "EDHiE requires PyTorch. Please install the required dependencies either"
                "manually or using the command: pip install 'symbolic-regression-toolkit[approaches]'")
finally:
    __all__.append("EDHiE")

