"""
Symbolic regression approaches and the base class for implementing new ones.

Modules:
    sr_approach: [SR_approach][SRToolkit.approaches.sr_approach.SR_approach] — abstract base class
        for all SR approaches; [ApproachConfig][SRToolkit.approaches.sr_approach.ApproachConfig] —
        serialisable configuration dataclass.
    ProGED: [ProGED][SRToolkit.approaches.ProGED.ProGED] — probabilistic grammar-based equation
        discovery by Brence et al.
    EDHiE: [EDHiE][SRToolkit.approaches.EDHiE.EDHiE] — equation discovery with hierarchical
        variational autoencoders by Mežnar et al. Requires PyTorch and pymoo.
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
