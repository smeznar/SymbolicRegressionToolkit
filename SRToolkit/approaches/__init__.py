"""
This module contains implementations of symbolic regression approaches. Class SR_approach is the base class for all
symbolic regression approaches.

Modules:
    sr_approach: The base class for all symbolic regression approaches.
    ProGED: The ProGED approach - Probabilistic grammar-based equation discovery.
"""

from .ProGED import ProGED
from .sr_approach import ApproachConfig, SR_approach

__all__ = ["SR_approach", "ProGED", "ApproachConfig", "EDHiEConfig"]

try:
    from .EDHiE import EDHiE, EDHiEConfig
except ImportError:

    class EDHiE:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "EDHiE requires PyTorch. Please install the required dependencies either"
                "manually or using the command: pip install 'symbolic-regression-toolkit[approaches]'"
            )

    # TODO: add a fake EDHiEConfig
finally:
    __all__.append("EDHiE")
    __all__.append("EDHiEConfig")
