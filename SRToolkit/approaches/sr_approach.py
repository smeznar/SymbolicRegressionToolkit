"""Abstract base class and configuration dataclass for symbolic regression approaches."""

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

from SRToolkit.bundle._relocate import _auto_bind
from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils.expression_constraints import ExpressionConstraint
from SRToolkit.utils.serialization import _from_json_safe, _to_json_safe

if TYPE_CHECKING:
    # Imported for annotations only: SRToolkit.dataset imports this module.
    from SRToolkit.dataset.background_knowledge import BackgroundKnowledge


def check_dependencies(packages: List[str]) -> None:
    """Verify that optional heavy dependencies are installed, raising a descriptive error if not.

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
    """Serialisable base configuration for SR approaches.

    Subclass this dataclass to define approach-specific hyperparameters that can be
    saved and restored via [to_dict][SRToolkit.approaches.sr_approach.ApproachConfig.to_dict]
    and [from_dict][SRToolkit.approaches.sr_approach.ApproachConfig.from_dict]. Fields must
    hold values that can be saved (numbers, strings, lists, dicts, numpy arrays, …; see
    [serialization][SRToolkit.utils.serialization]).

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
        """Serialise this config to a JSON-safe dictionary.

        Returns:
            A dictionary representation with all values converted to JSON-compatible types.

        Raises:
            TypeError: If a field holds a value that cannot be saved (see
                [serialization][SRToolkit.utils.serialization]); the message names the field.
        """
        return _to_json_safe(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "ApproachConfig":
        """Restore a config from a dictionary produced by
        [to_dict][SRToolkit.approaches.sr_approach.ApproachConfig.to_dict].

        Args:
            d: Dictionary representation of the config.

        Returns:
            A new instance of this config class populated with the values from ``d``.
        """
        return cls(**_from_json_safe(d))


class SR_approach(ABC):
    def __init__(self, config: ApproachConfig) -> None:
        """Abstract base class for all symbolic regression approaches.

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
        """When the framework calls [adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt].

        - ``"never"`` (default): never.
        - ``"experiment"``: before every ``search``.
        - ``"once"``: before the first experiment of each
          [evaluate_approach][SRToolkit.dataset.sr_dataset.SR_dataset.evaluate_approach] call or grid
          job. With a state path (``adaptation_path`` or ``adapted_states``), what happens depends on
          whether that file exists: if it does, the state is loaded from it with
          [load_adapted_state][SRToolkit.approaches.sr_approach.SR_approach.load_adapted_state] and
          ``adapt`` is not called; if not, the approach adapts and saves the state there with
          [save_adapted_state][SRToolkit.approaches.sr_approach.SR_approach.save_adapted_state].
          Without a path, it adapts every time.

        A loaded state is not checked against the dataset: an existing file is used as is. To adapt
        again after the dataset changed, delete the file or call
        [adapt_one][SRToolkit.experiments.ExperimentGrid.adapt_one] with ``force=True``.

        Returns:
            ``"never"``, ``"once"`` or ``"experiment"``.
        """
        return "never"

    @property
    def save_adapted_model(self) -> bool:
        """Whether the adapted state should be saved after each adaptation.

        The framework does not read it; an approach may use it in its own code.

        Returns:
            ``False`` by default.
        """
        return False

    def prepare(self) -> None:
        """Reset the approach's per-experiment state in preparation for a new run.

        Called by the framework before ``adapt()`` and ``search()`` for every
        experiment, so any state accumulated during a previous run is cleared
        before the next one starts.

        Three example cases:

        - **Stateless** (e.g. random sampling): implement as ``pass``.
        - **Stateful search** (e.g. GP): reset the population or any other
          mutable search state to its initial configuration.
        - **Pretrained weights + per-run search state** (e.g. neural): reset
          only the search state; leave pretrained weights untouched.
        """
        pass

    def adapt(
        self,
        X: np.ndarray,
        background_knowledge: "BackgroundKnowledge",
        required_constraints: Sequence[ExpressionConstraint],
    ) -> None:
        """Adapt the approach to a task before searching, without seeing the targets.

        The framework calls it after ``prepare()`` and before ``search()``, as
        [adaptation_scope][SRToolkit.approaches.sr_approach.SR_approach.adaptation_scope] says. Use
        what you need and ignore the rest. It must not fit or evaluate expressions on data (e.g.
        through an evaluator kept from an earlier run); calling ``pre_check`` of a constraint is
        allowed, since it only looks at the expression.

        The default does nothing. Override it when ``adaptation_scope`` is ``"once"`` or
        ``"experiment"``.

        Args:
            X: Inputs of shape ``(n_samples, n_variables)``, columns in the symbol library's
                variable order.
            background_knowledge: What the task tells the approach besides the data (see
                [BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge]).
            required_constraints: The constraints the evaluator will enforce.
        """
        pass

    @abstractmethod
    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None) -> None:
        """Run the symbolic regression search.

        Implementations must call ``sr_evaluator.evaluate_expr(expr)`` to score
        candidate expressions. The evaluator accumulates all results internally;
        do not call ``sr_evaluator.get_results()`` — that is handled by the
        framework after ``search`` finishes.

        Stop conditions to respect:
            - ``sr_evaluator.total_evaluations >= sr_evaluator.max_evaluations``
              (when ``max_evaluations > 0``)
            - The current best error has dropped below ``sr_evaluator.success_threshold``

        The symbol library is available in ``sr_evaluator.symbol_library``. ``sr_evaluator.describe()``
        gives the task as text (e.g. for an LLM prompt) and ``sr_evaluator.background_knowledge`` as
        objects. Do not read ``sr_evaluator.ground_truth``: it is the answer.
        [get_model][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_model] and
        [check_constraints][SRToolkit.evaluation.sr_evaluator.SR_evaluator.check_constraints] do not
        use the budget: use them to skip expressions that were already evaluated or would be rejected.

        Args:
            sr_evaluator: Evaluator used to score expressions. All results are
                stored inside this object.
            seed: Optional random seed for reproducible expression generation.

        Returns:
            None
        """
        raise NotImplementedError

    def save_adapted_state(self, path: str) -> None:
        """Save the adapted model/approach state to ``path``.

        The approach is free to choose any serialization format (e.g. ``torch.save``,
        ``numpy.save``, JSON, pickle), but must write to exactly ``path``: the framework restores
        a state only when a file exists at ``path``.

        Should be implemented when ``adaptation_scope == "once"``.

        The default raises ``NotImplementedError``. Override when needed.

        Args:
            path: File path to save the adapted state to.
        """
        raise NotImplementedError

    def load_adapted_state(self, path: str) -> None:
        """Restore the previously saved adapted state from ``path``.

        Should be implemented when ``adaptation_scope == "once"``.

        The default raises ``NotImplementedError``. Override when needed.

        Args:
            path: File path previously passed to ``save_adapted_state()``.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> "SR_approach":
        """Reconstruct an approach instance from a configuration dictionary.

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

    @classmethod
    def from_config_dict(cls, config: dict) -> "SR_approach":
        """Reconstruct an approach from a config dict that includes an ``approach_class`` path.

        This is the self-dispatching counterpart to
        [from_config][SRToolkit.approaches.sr_approach.SR_approach.from_config]: it binds the
        config to any installed ``.srtk`` bundle, resolves the concrete approach class named
        by ``approach_class``, and delegates to that class's
        [from_config][SRToolkit.approaches.sr_approach.SR_approach.from_config]. Mirrors
        [SR_dataset.from_dict][SRToolkit.dataset.sr_dataset.SR_dataset.from_dict], which
        resolves its own class the same way.

        Args:
            config: A serialised approach config containing an ``approach_class`` key, as
                produced by
                [ApproachConfig.to_dict][SRToolkit.approaches.sr_approach.ApproachConfig.to_dict].

        Returns:
            A new instance of the concrete approach class.

        Raises:
            ImportError: If ``approach_class`` cannot be imported (e.g. its ``.srtk`` bundle
                is not installed, or the config has no ``_bundle`` key and was never bound).
        """
        config = _auto_bind(config)
        class_path = config["approach_class"]
        module_path, cls_name = class_path.rsplit(".", 1)
        try:
            target_cls = getattr(importlib.import_module(module_path), cls_name)
        except (ImportError, AttributeError):
            raise ImportError(
                f"Cannot import approach class {class_path!r}. "
                "If this is a bundle class, install the bundle first. "
                "If the config has no '_bundle' key, call bind_config(config) manually."
            ) from None
        return target_cls.from_config(config)
