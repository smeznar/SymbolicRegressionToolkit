"""Symbolic regression approaches and the base class for implementing new ones.

Modules:
    sr_approach: [SR_approach][SRToolkit.approaches.sr_approach.SR_approach] — abstract base class
        for all SR approaches; [ApproachConfig][SRToolkit.approaches.sr_approach.ApproachConfig] —
        serialisable configuration dataclass.
    ProGED: [ProGED][SRToolkit.approaches.ProGED.ProGED] — probabilistic grammar-based equation
        discovery by Brence et al.
    EDHiE: [EDHiE][SRToolkit.approaches.EDHiE.EDHiE] — equation discovery with hierarchical
        variational autoencoders by Mežnar et al. Requires PyTorch and pymoo, which are imported
        only when ``EDHiE`` or ``EDHiEConfig`` is first accessed, so ``import SRToolkit`` stays light.
"""

import importlib as _importlib
import sys as _sys
import types as _types
from typing import Any, Dict

from .ProGED import ProGED
from .sr_approach import ApproachConfig, SR_approach

__all__ = ["SR_approach", "ProGED", "ApproachConfig", "EDHiE", "EDHiEConfig"]

_LAZY_EDHIE = ("EDHiE", "EDHiEConfig")
_INSTALL_HINT = (
    "requires PyTorch and pymoo. Please install the required dependencies either manually or using "
    "the command: pip install 'symbolic-regression-toolkit[approaches]'"
)


def _missing_dependency_stubs(error: ImportError) -> Dict[str, Any]:
    """Return placeholder classes that raise the original ``ImportError`` when instantiated."""

    class EDHiE:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(f"EDHiE {_INSTALL_HINT}") from error

    class EDHiEConfig(ApproachConfig):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(f"EDHiEConfig {_INSTALL_HINT}") from error

    return {"EDHiE": EDHiE, "EDHiEConfig": EDHiEConfig}


def __getattr__(name: str) -> Any:
    """Import ``EDHiE`` and ``EDHiEConfig`` on first access (PEP 562).

    Returns:
        The requested class, or a placeholder that raises ``ImportError`` on instantiation if
        PyTorch or pymoo is not installed.

    Raises:
        AttributeError: If ``name`` is not a lazily imported attribute of this package.
    """
    if name not in _LAZY_EDHIE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = _importlib.import_module(".EDHiE", __name__)
        values = {attr: getattr(module, attr) for attr in _LAZY_EDHIE}
    except ImportError as error:
        values = _missing_dependency_stubs(error)
    globals().update(values)
    return values[name]


class _ApproachesPackage(_types.ModuleType):
    """Package module type that keeps ``EDHiE`` bound to the class, not to its submodule.

    Importing the submodule ``SRToolkit.approaches.EDHiE`` (e.g. when a saved config is resolved by
    its import path) makes Python set the package attribute ``EDHiE`` to that submodule, which would
    shadow the class. Ignoring that assignment lets the module-level ``__getattr__``
    resolve ``EDHiE`` to the class instead.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _LAZY_EDHIE and isinstance(value, _types.ModuleType):
            return
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _ApproachesPackage
