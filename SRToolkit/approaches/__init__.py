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
