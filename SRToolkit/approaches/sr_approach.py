"""
This module contains the SR_approach class, which is the base class for all symbolic regression approaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import SymbolLibrary
from SRToolkit.utils.serialization import _from_json_safe, _to_json_safe


def check_dependencies(packages: List[str]):
    if "pytorch" in packages:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "This approach requires PyTorch. Install dependencies either manually or with"
                "the command: pip install 'symbolic-regression-toolkit[approaches]'"
            )
    if "pymoo" in packages:
        try:
            import pymoo  # noqa: F401
        except ImportError:
            raise ImportError(
                "This approach requires pymoo. Install dependencies either manually or with"
                "the command: pip install 'symbolic-regression-toolkit[approaches]'"
            )


@dataclass
class ApproachConfig:
    """
    Base configuration for SR approaches.

    Each approach should define its own config dataclass inheriting from this.
    """

    name: str = "base"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return _to_json_safe(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "ApproachConfig":
        """Deserialize from dictionary."""
        return cls(**_from_json_safe(d))


class SR_approach(ABC):
    def __init__(self, name: str, config: Optional[ApproachConfig] = None):
        """
        The base class for all symbolic regression approaches. Any symbolic regression approach should inherit from
        this class.

        If the approach requires pretraining on an external corpus, this should be done inside ``__init__``.
        The approach is responsible for defining what corpus it expects and how to load it.

        Args:
            name: The name of the approach.
            config: Optional configuration dataclass for the approach.
        """
        self.name = name
        self._config = config

    @property
    def config(self) -> Optional[ApproachConfig]:
        """Returns the approach configuration, if set."""
        return self._config

    @config.setter
    def config(self, value: ApproachConfig) -> None:
        """Set the approach configuration."""
        self._config = value

    @property
    def adaptation_scope(self) -> str:
        """
        Controls when ``adapt()`` is called by the framework, and whether
        ``save_adapted_state()`` / ``load_adapted_state()`` are used.

        - ``"never"``: ``adapt()`` is never called (default). ``save_adapted_state()`` and ``load_adapted_state()``
                       do not need to be implemented.
        - ``"once"``: ``adapt()`` is called once; the framework caches the result via ``save_adapted_state()`` and
                      restores it via ``load_adapted_state()`` on subsequent experiments with the same library. Both
                      methods must be implemented when this scope is used.
        - ``"experiment"``: ``adapt()`` is called before every ``search()``. ``save_adapted_state()``
                            is called only if ``save_adapted_model`` is also ``True``.

        For neural approaches, we recommend you save weights to disk with ``save_adapted_state()`` and
        return an identifier (e.g. a file path). ``load_adapted_state()`` receives that identifier and loads the
        weights from the disk, keeping the in-memory cache small.

        Returns:
            One of ``"never"``, ``"once"``, or ``"experiment"``.
        """
        return "never"

    @property
    def save_adapted_model(self) -> bool:
        """
        Whether the framework should call ``save_adapted_state()`` when ``adaptation_scope`` is ``"experiment"``.
        Ignored for ``"never"`` and ``"once"`` scopes (the latter always saves). Default is ``False``.

        Returns:
            ``True`` if the adapted model should be saved after each adaptation.
        """
        return False

    @abstractmethod
    def prepare(self) -> None:
        """
        Reset the approach's per-experiment state in preparation for a new run.

        Called by the framework before ``adapt()`` and ``search()`` for every
        experiment, so any state accumulated during a previous run is cleared
        before the next one starts.

        Three cases:

        - **Stateless** (e.g. random sampling): implement as ``pass``.
        - **Stateful search** (e.g. GP): reset the population or any other
          mutable search state to its initial configuration.
        - **Pretrained weights + per-run search state** (e.g. neural): reset
          only the search state; leave pretrained weights untouched.
        """
        raise NotImplementedError

    def adapt(self, X: np.ndarray, symbol_library: SymbolLibrary) -> None:
        """
        Adapt the approach to the target dataset's input space and symbols in the symbol library.

        Called by the framework after ``prepare()`` and before ``search()``, according
        to ``adaptation_scope``. This method must NOT access target values (y) or evaluate
        expressions (e.g. by calling ``sr_evaluator.evaluate_expr()``) during adaptation —
        only the points from the domain and the symbol library may be used.

        The default implementation does nothing. Override when ``adaptation_scope`` is
        ``"once"`` or ``"experiment"``.

        Args:
            X: input variables from the domain, shape ``(n_samples, n_variables)``.
            symbol_library: The symbol library defining the available symbols/tokens.
        """
        pass

    @abstractmethod
    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None) -> None:
        """
        Run the symbolic regression search.

        Implementations must call ``sr_evaluator.evaluate_expr(expr)`` to score
        candidate expressions. The evaluator accumulates all results internally;
        do not call ``sr_evaluator.get_results()`` — that is handled by the
        framework after ``search`` finishes.

        Stop conditions to respect:
            - ``sr_evaluator.total_evaluations >= sr_evaluator.max_evaluations``
              (when ``max_evaluations > 0``)
            - The current best error has dropped below ``sr_evaluator.success_threshold``

        The symbol library is available in ``sr_evaluator.symbol_library``.

        Args:
            sr_evaluator: Evaluator used to score expressions. All results are
                stored inside this object.
            seed: Optional random seed for reproducible expression generation.

        Returns:
            None
        """
        raise NotImplementedError

    def save_adapted_state(self) -> Any:
        """
        Saves the adapted model/approach and returns the saved state.

        For neural approaches we suggest you save the model's weights to disk (e.g. via
        ``torch.save``) and return the file path. The returned value is stored
        in memory by the framework and passed back to ``load_adapted_state()``
        when the cached state is needed.

        Must be implemented when ``adaptation_scope == "symbol_library"``, or
        when ``adaptation_scope == "experiment"`` and ``save_adapted_model`` is
        ``True``.

        The default raises ``NotImplementedError``. Override when needed.

        Returns:
            The saved state (e.g., a file path or the model).
        """
        raise NotImplementedError

    def load_adapted_state(self, state: Any) -> None:
        """
        Restore the previously saved state.

        For neural approaches we suggest that this function loads weights from the disk
        using the identifier returned by ``save_adapted_state()`` (e.g. a file path
        passed to ``torch.load``).

        Must be implemented when ``adaptation_scope == "symbol_library"``

        The default raises ``NotImplementedError``. Override when needed.

        Args:
            state: The identifier previously returned by ``save_adapted_state()``.
        """
        raise NotImplementedError
