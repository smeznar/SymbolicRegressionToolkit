"""
Abstract base class and configuration dataclass for symbolic regression approaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import SymbolLibrary
from SRToolkit.utils.serialization import _from_json_safe, _to_json_safe


def check_dependencies(packages: List[str]) -> None:
    """
    Verify that optional heavy dependencies are installed, raising a descriptive error if not.

    Args:
        packages: Names of packages to check. Recognised values: ``"pytorch"``, ``"pymoo"``.

    Raises:
        ImportError: If any requested package cannot be imported.
    """
    if "pytorch" in packages:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "This approach requires PyTorch. Install dependencies either manually or with "
                "the command: pip install 'symbolic-regression-toolkit[approaches]'"
            )
    if "pymoo" in packages:
        try:
            import pymoo  # noqa: F401
        except ImportError:
            raise ImportError(
                "This approach requires pymoo. Install dependencies either manually or with "
                "the command: pip install 'symbolic-regression-toolkit[approaches]'"
            )


@dataclass
class ApproachConfig:
    """
    Serialisable base configuration for SR approaches.

    Subclass this dataclass to define approach-specific hyperparameters that can be
    saved and restored via [to_dict][SRToolkit.approaches.sr_approach.ApproachConfig.to_dict]
    and [from_dict][SRToolkit.approaches.sr_approach.ApproachConfig.from_dict].

    The ``approach_class`` field is populated automatically by
    [SR_approach][SRToolkit.approaches.sr_approach.SR_approach] and contains
    the fully-qualified importlib path of the concrete approach class. This makes a serialised
    config dict self-sufficient for reconstruction without any additional metadata.

    Examples:
        >>> cfg = ApproachConfig(name="my_approach")
        >>> d = cfg.to_dict()
        >>> ApproachConfig.from_dict(d).name
        'my_approach'
    """

    name: str = "base"
    approach_class: str = ""

    def to_dict(self) -> dict:
        """
        Serialise this config to a JSON-safe dictionary.

        Returns:
            A dictionary representation with all values converted to JSON-compatible types.
        """
        return _to_json_safe(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "ApproachConfig":
        """
        Restore a config from a dictionary produced by
        [to_dict][SRToolkit.approaches.sr_approach.ApproachConfig.to_dict].

        Args:
            d: Dictionary representation of the config.

        Returns:
            A new instance of this config class populated with the values from ``d``.
        """
        return cls(**_from_json_safe(d))


class SR_approach(ABC):
    def __init__(self, config: ApproachConfig) -> None:
        """
        Abstract base class for all symbolic regression approaches.

        Subclasses must implement [prepare][SRToolkit.approaches.sr_approach.SR_approach.prepare]
        and [search][SRToolkit.approaches.sr_approach.SR_approach.search], and optionally override
        [adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt],
        [save_adapted_state][SRToolkit.approaches.sr_approach.SR_approach.save_adapted_state], and
        [load_adapted_state][SRToolkit.approaches.sr_approach.SR_approach.load_adapted_state]
        according to their [adaptation_scope][SRToolkit.approaches.sr_approach.SR_approach.adaptation_scope].

        The ``config.approach_class`` field is set automatically to the fully-qualified class path
        of the concrete subclass, so the config dict alone is sufficient to reconstruct the approach.

        Args:
            config: Configuration dataclass for the approach. Every concrete approach defines its
                own [ApproachConfig][SRToolkit.approaches.sr_approach.ApproachConfig] subclass
                and passes an instance here from its own ``__init__``.

        Attributes:
            name: Name of this approach, read from ``config.name``.
            config: The approach configuration.
        """
        cls = type(self)
        config.approach_class = f"{cls.__module__}.{cls.__qualname__}"
        self._config = config

    @property
    def name(self) -> str:
        """Name of this approach, as stored in ``config.name``."""
        return self._config.name

    @property
    def config(self) -> ApproachConfig:
        """Returns the approach configuration."""
        return self._config

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

    def save_adapted_state(self, path: str) -> None:
        """
        Save the adapted model/approach state to ``path``.

        The approach is free to choose any serialization format (e.g. ``torch.save``,
        ``numpy.save``, JSON, pickle).  The framework passes ``path`` as a base path
        without an extension; the approach may append its own extension (e.g.
        ``path + ".pt"``), but must be consistent between ``save_adapted_state`` and
        ``load_adapted_state``.

        Must be implemented when ``adaptation_scope == "once"``, or when
        ``adaptation_scope == "experiment"`` and ``save_adapted_model`` is ``True``.

        The default raises ``NotImplementedError``. Override when needed.

        Args:
            path: Base file path to save the adapted state to.
        """
        raise NotImplementedError

    def load_adapted_state(self, path: str) -> None:
        """
        Restore the previously saved adapted state from ``path``.

        Must be implemented when ``adaptation_scope == "once"``.

        The default raises ``NotImplementedError``. Override when needed.

        Args:
            path: Base file path previously passed to ``save_adapted_state()``.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> "SR_approach":
        """
        Reconstruct an approach instance from a configuration dictionary.

        This classmethod is the counterpart to saving an
        [ApproachConfig][SRToolkit.approaches.sr_approach.ApproachConfig] via
        [to_dict][SRToolkit.approaches.sr_approach.ApproachConfig.to_dict].  It is used by
        [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] when loading an approach from a
        ``grid.json`` manifest for CLI or HPC execution.

        The default raises ``NotImplementedError``.  Override this classmethod in subclasses
        that need to support CLI/grid loading.

        Examples:
            >>> class MyApproach(SR_approach):
            ...     @classmethod
            ...     def from_config(cls, config: dict) -> "MyApproach":
            ...         return cls(param=config["param"])

        Args:
            config: Dictionary previously produced by
                [ApproachConfig.to_dict][SRToolkit.approaches.sr_approach.ApproachConfig.to_dict]
                or any JSON-serialisable dict describing the approach's hyperparameters.

        Returns:
            A new instance of this approach class.

        Raises:
            NotImplementedError: If the subclass has not implemented this method.
        """
        raise NotImplementedError(
            f"[{cls.__name__}] from_config() is not implemented. "
            "Override this classmethod to support loading from a grid manifest (CLI/HPC use)."
        )
