"""Base class and configuration of symbolic regression approaches.

## Lifecycle

One approach object can run many experiments. Each part of it sets up one kind of state:

- ``__init__``: the hyperparameters, from the config.
- [adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt]: what the task alone determines, e.g. a
  grammar for its symbol library or a trained model; how often it runs is set by
  [adaptation_scope][SRToolkit.approaches.sr_approach.SR_approach.adaptation_scope]. A resumed run
  adapts again or loads the adapted state.
- the search: everything a run learns, set up anew for each run from its seed.

## Two ways to implement the search

An approach implements either [search][SRToolkit.approaches.sr_approach.SR_approach.search], its own
loop that calls ``evaluate_expr`` until ``should_stop``, or the **step interface**, in which the
framework runs the loop:

- [start][SRToolkit.approaches.sr_approach.SR_approach.start]``(se, seed)`` seeds the approach's
  random generators and resets the search state to empty (e.g. no population yet);
- [ask][SRToolkit.approaches.sr_approach.SR_approach.ask]``(se)`` proposes the next expressions: the
  first call creates them from scratch (e.g. a random population), later ones build on what
  ``tell`` stored;
- [tell][SRToolkit.approaches.sr_approach.SR_approach.tell]``(se, candidates, results)`` updates the
  search state from their [ModelResult][SRToolkit.utils.types.ModelResult]s (e.g. selects the
  next population); an approach that learns nothing from the results (e.g. random sampling) need
  not implement it;
- optionally, [get_search_state][SRToolkit.approaches.sr_approach.SR_approach.get_search_state] and
  [set_search_state][SRToolkit.approaches.sr_approach.SR_approach.set_search_state] save and restore
  the search state between two steps.

One **step** is one ``ask``, the evaluation of its candidates and one ``tell``. What it covers (one
sample, a batch, a generation of a genetic algorithm) is the approach's choice; document it. Only
the framework evaluates: ``start``, ``ask`` and ``tell`` never call ``evaluate_expr``, so every
evaluation falls in a step, which an interrupt lets finish and a snapshot never splits.

## Which one to implement

``search`` is the simpler choice: write the loop as you would anyway, or call a library that runs
its own. Its runs write checkpoints like any other, so their results survive a crash, but a
stopped run starts again from the beginning.

The step interface takes more work, since the search state lives in the object between calls.
With the state methods, a stopped or crashed run then continues from its last snapshot and finds
the same models as a run that was never stopped (see
[Checkpoints][SRToolkit.evaluation.sr_evaluator]).

## The step loop

For a step approach, the framework calls ``start`` and then runs this loop (this is what the
inherited ``search`` does). An approach that overrides ``search`` runs its own loop instead, and
``start``, ``ask`` and ``tell`` are never called. ``snapshots`` is true when the evaluator has a
checkpoint and the approach has the state methods:

```python
stepped, complete = False, True
while not se.should_stop:
    stepped = True
    if snapshots and se.snapshot_due:
        se.save_snapshot(approach.get_search_state())
    candidates = approach.ask(se)
    if not candidates:
        break  # as if search had returned
    with se.step():
        results = []
        for candidate in candidates:
            if se.should_stop:
                break
            if candidate is None:  # not an expression: neither looked up nor evaluated
                results.append(None)
                continue
            if not isinstance(candidate, Candidate):  # a token list or a Node
                candidate = Candidate(candidate)
            if candidate.evaluate:
                results.append(se.evaluate_expr(candidate.expr, candidate.free_constants,
                                                candidate.simplify_expr, candidate.canonicalise_constants))
            else:  # a lookup, without spending budget; None if never evaluated
                results.append(se.get_model(candidate.expr, candidate.free_constants,
                                            candidate.simplify_expr, candidate.canonicalise_constants))
    approach.tell(se, candidates[: len(results)], results)
    complete = len(results) == len(candidates)
if snapshots and stepped and complete:  # when the loop ends normally, unless its last step was cut
    se.save_snapshot(approach.get_search_state())
```

Snapshots fall between steps, where nothing is half done. None is taken after an exception; after
a step that a stop cut short (its ``tell`` saw only part of its results), so that an
[extended run][SRToolkit.dataset.sr_dataset.SR_dataset.extend_run] redoes that step in full; or at
the end of a session that ran no step (e.g. a finished run resumed), which keeps the snapshot it
has.
[run][SRToolkit.approaches.sr_approach.SR_approach.run] continues this loop from the evaluator's
snapshot when resuming, and calls ``search`` otherwise.
"""

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from SRToolkit.bundle._relocate import _auto_bind, _import_class
from SRToolkit.evaluation import SR_evaluator
from SRToolkit.evaluation.model_store import _restore_rngs
from SRToolkit.utils.expression_constraints import ExpressionConstraint
from SRToolkit.utils.serialization import _from_json_safe, _to_json_safe

if TYPE_CHECKING:
    # Imported for annotations only: SRToolkit.dataset imports this module.
    from SRToolkit.dataset.background_knowledge import BackgroundKnowledge
    from SRToolkit.utils.expression_tree import Node
    from SRToolkit.utils.free_constants import Constant
    from SRToolkit.utils.types import ModelResult


def check_dependencies(packages: list[str]) -> None:
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


@dataclass(frozen=True)
class Candidate:
    """An expression an approach proposes from [ask][SRToolkit.approaches.sr_approach.SR_approach.ask],
    with the arguments the framework passes to
    [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr].

    Attributes:
        expr: The expression, as a token list or a [Node][SRToolkit.utils.expression_tree.Node].
        free_constants: Specs for its labelled constants, or ``None``.
        simplify_expr: Simplify it before evaluating (see
            [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr]); a lookup
            simplifies it the same way. Default ``False``.
        canonicalise_constants: Canonicalise its constants before evaluating. Default ``False``.
        evaluate: ``False`` only looks up the stored model of an expression evaluated before (see
            [get_model][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_model]), spending no
            budget; ``tell`` gets ``None`` for it if there is none. Default ``True``.
    """

    expr: "list[str] | Node"
    free_constants: "Sequence[Constant] | None" = None
    simplify_expr: bool = False
    canonicalise_constants: bool = False
    evaluate: bool = True


class SR_approach(ABC):
    def __init__(self, config: ApproachConfig) -> None:
        """Abstract base class for all symbolic regression approaches.

        Subclasses implement either [search][SRToolkit.approaches.sr_approach.SR_approach.search]
        or the step interface (see the [module docs][SRToolkit.approaches.sr_approach]), and
        optionally override
        [adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt],
        [save_adapted_state][SRToolkit.approaches.sr_approach.SR_approach.save_adapted_state], and
        [load_adapted_state][SRToolkit.approaches.sr_approach.SR_approach.load_adapted_state]
        according to their [adaptation_scope][SRToolkit.approaches.sr_approach.SR_approach.adaptation_scope].

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
          job, through [adapt_or_load][SRToolkit.dataset.sr_dataset.SR_dataset.adapt_or_load] with
          the state path (``adaptation_path`` or ``adapted_states``), if any.

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

    def adapt(
        self,
        X: np.ndarray,
        background_knowledge: "BackgroundKnowledge",
        required_constraints: Sequence[ExpressionConstraint],
    ) -> None:
        """Adapt the approach to a task before searching, without seeing the targets.

        The framework calls it before ``search()``, as
        [adaptation_scope][SRToolkit.approaches.sr_approach.SR_approach.adaptation_scope] says. Use
        what you need and ignore the rest. It must not fit or evaluate expressions on data (e.g.
        through an evaluator kept from an earlier run); calling ``pre_check`` of a constraint is
        allowed, since it only looks at the expression.

        The default does nothing. Override it when ``adaptation_scope`` is ``"once"`` or
        ``"experiment"``. With ``"experiment"``, a resumed run adapts again, so ``adapt`` should
        give the same result for the same inputs (e.g. seed its own random generators) for a
        resumed run to equal an uninterrupted one.

        Args:
            X: Inputs of shape ``(n_samples, n_variables)``, columns in the symbol library's
                variable order.
            background_knowledge: What the task tells the approach besides the data (see
                [BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge]).
            required_constraints: The constraints the evaluator will enforce.
        """
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: int | None = None) -> None:
        """Run the search of one experiment.

        The default runs the step interface: ``start``, then the step loop shown in the
        [module docs][SRToolkit.approaches.sr_approach]. Approaches without the step interface
        override it and set up their search state anew from ``seed`` (see the lifecycle in the
        module docs). Score expressions with ``evaluate_expr``; the framework collects the results.
        ``evaluate_expr`` returns the expression's
        [ModelResult][SRToolkit.utils.types.ModelResult], with its error, fitted ``parameters``
        and ``status``.

        Search while ``sr_evaluator.should_stop`` is ``False``; it turns ``True`` when the success
        threshold or the budget is reached, a callback stops the run, or the run is interrupted
        (Ctrl-C or SIGTERM). To end the search early (e.g. the search space is exhausted), call
        [sr_evaluator.stop][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stop] with a reason, so
        the results explain why; returning without it is recorded as ``"returned"``.

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

        Raises:
            NotImplementedError: If the approach implements neither ``search`` nor the step
                interface.
        """
        if not self.supports_steps:
            raise NotImplementedError(
                f"[{type(self).__name__}] implements neither search() nor the step interface (start/ask/tell)."
            )
        self.start(sr_evaluator, seed)
        self._step_loop(sr_evaluator)

    @property
    def supports_steps(self) -> bool:
        """Whether the approach implements the step interface (it overrides ``ask``)."""
        return type(self).ask is not SR_approach.ask

    @property
    def supports_snapshots(self) -> bool:
        """Whether a run of the approach can be snapshotted and resumed: it implements the step
        interface and overrides both ``get_search_state`` and ``set_search_state``.
        """
        cls = type(self)
        return (
            self.supports_steps
            and cls.get_search_state is not SR_approach.get_search_state
            and cls.set_search_state is not SR_approach.set_search_state
        )

    def run(self, sr_evaluator: SR_evaluator, seed: int | None = None) -> None:
        """Run the approach on ``sr_evaluator``: continue from its snapshot, or search anew.

        If ``sr_evaluator.resumed_from`` holds a
        [Snapshot][SRToolkit.evaluation.model_store.Snapshot], the approach state is restored with
        [set_search_state][SRToolkit.approaches.sr_approach.SR_approach.set_search_state], then the
        global random states, and the step loop continues (``start`` is not called).
        Otherwise it calls [search][SRToolkit.approaches.sr_approach.SR_approach.search]. If the
        search returns while evaluation goes on, ``run`` stops it with the reason ``"returned"``,
        so the checkpoint records the run as finished.
        [evaluate_approach][SRToolkit.dataset.sr_dataset.SR_dataset.evaluate_approach] and
        [ExperimentJob.run][SRToolkit.experiments.ExperimentJob.run] call it.

        Args:
            sr_evaluator: The evaluator, new or resumed.
            seed: The seed passed to ``search`` for a new run; ignored when continuing.

        Raises:
            ValueError: If the evaluator holds a snapshot but the approach cannot continue from it
                (see [supports_snapshots][SRToolkit.approaches.sr_approach.SR_approach.supports_snapshots]).
        """
        snapshot = sr_evaluator.resumed_from
        if snapshot is None:
            self.search(sr_evaluator, seed)
        else:
            if not self.supports_snapshots:
                raise ValueError(
                    f"[{type(self).__name__}] cannot continue from a snapshot: it lacks the step interface "
                    "(start/ask/tell) or get_search_state/set_search_state."
                )
            self.set_search_state(snapshot.state)
            _restore_rngs(snapshot.rngs)
            self._step_loop(sr_evaluator)
        if not sr_evaluator.should_stop:
            sr_evaluator.stop("returned")

    def _step_loop(self, se: SR_evaluator) -> None:
        # The loop of the step interface (shown in the module docstring), used by search and run.
        snapshots = se.checkpoint is not None and self.supports_snapshots
        stepped = False  # whether this session ran a step
        complete = True  # whether the last step got all its results
        while not se.should_stop:
            stepped = True
            if snapshots and se.snapshot_due:
                se.save_snapshot(self.get_search_state())
            candidates = self.ask(se)
            if not candidates:
                break
            results: list[ModelResult | None] = []
            with se.step():
                for item in candidates:
                    if se.should_stop:
                        break
                    if item is None:
                        results.append(None)
                        continue
                    c = item if isinstance(item, Candidate) else Candidate(item)
                    if c.evaluate:
                        results.append(
                            se.evaluate_expr(c.expr, c.free_constants, c.simplify_expr, c.canonicalise_constants)
                        )
                    else:
                        results.append(
                            se.get_model(c.expr, c.free_constants, c.simplify_expr, c.canonicalise_constants)
                        )
            self.tell(se, candidates[: len(results)], results)
            complete = len(results) == len(candidates)
        if snapshots and stepped and complete:
            se.save_snapshot(self.get_search_state())

    def start(self, sr_evaluator: SR_evaluator, seed: int | None = None) -> None:
        """Seed the approach's random generators and reset its search state to empty (step
        interface).

        Called once per run before the first ``ask``, but not when a run resumes (see
        [run][SRToolkit.approaches.sr_approach.SR_approach.run]). Create nothing to evaluate here;
        the first ``ask`` does that.

        Args:
            sr_evaluator: The evaluator of the run.
            seed: The run's seed, or ``None``.

        Raises:
            NotImplementedError: Unless overridden.
        """
        raise NotImplementedError(
            f"[{type(self).__name__}] implements ask but not start; start must seed the approach's "
            "random generators and set up its search state."
        )

    def ask(self, sr_evaluator: SR_evaluator) -> "list[Candidate | list[str] | Node | None]":
        """Propose the next expressions to evaluate (step interface).

        The first call after ``start`` creates them from scratch (e.g. a random population); later
        calls build on what ``tell`` stored. It may use the unbilled ``get_model`` and
        ``check_constraints`` of ``sr_evaluator``, but not ``evaluate_expr``. An empty list ends the
        search, as if ``search`` had returned; to record why, call ``sr_evaluator.stop(reason)``
        first. Only evaluations spend the budget, so an approach that may propose nothing but
        lookups (``evaluate=False``) step after step must stop the run itself.

        Args:
            sr_evaluator: The evaluator of the run.

        Returns:
            The expressions, in the order they are to be evaluated: token lists or
            [Node][SRToolkit.utils.expression_tree.Node]s, or
            [Candidate][SRToolkit.approaches.sr_approach.Candidate]s to pass free-constant specs,
            ``canonicalise_constants`` or ``evaluate=False``. ``None`` marks a proposal that is not an
            expression (e.g. a latent vector that does not decode): it is neither looked up nor
            evaluated, and ``tell`` gets ``None`` for it.

        Raises:
            NotImplementedError: Unless overridden.
        """
        raise NotImplementedError

    def tell(
        self,
        sr_evaluator: SR_evaluator,
        candidates: "list[Candidate | list[str] | Node | None]",
        results: "list[ModelResult | None]",
    ) -> None:
        """Update the search state from the results of the evaluated candidates (step interface),
        e.g. select the next population. Like ``ask``, it does not call ``evaluate_expr``.

        If evaluation stopped in the middle of a step, ``candidates`` holds only those evaluated
        before it stopped, the first ones that ``ask`` returned; both lists may be empty. The
        default does nothing.

        Args:
            sr_evaluator: The evaluator of the run.
            candidates: The evaluated candidates, in order, as ``ask`` returned them.
            results: Their models, with the error, status, fitted constants, etc. A model's
                ``expr`` is a token list, canonicalised if its candidate asked for it. ``None`` for
                a ``None`` entry and for a candidate with ``evaluate=False`` that was never
                evaluated.
        """
        pass

    def get_search_state(self) -> bytes:
        """Return the search state between two steps (step interface).

        What ``start`` set up and ``tell`` changed, including the approach's own random generators
        and anything the search fine-tunes (e.g. model weights). Leave out the global random states
        (Python, NumPy, PyTorch), which the framework saves itself, and what adaptation set up,
        which it restores before ``set_search_state``.

        Returns:
            The state, e.g. pickled.

        Raises:
            NotImplementedError: Unless overridden.
        """
        raise NotImplementedError

    def set_search_state(self, state: bytes) -> None:
        """Restore a state returned by
        [get_search_state][SRToolkit.approaches.sr_approach.SR_approach.get_search_state]
        (step interface).

        Called after adaptation, so it overrides what that set up.

        Args:
            state: The saved state.

        Raises:
            NotImplementedError: Unless overridden.
        """
        raise NotImplementedError

    def save_adapted_state(self, path: str) -> None:
        """Save the adapted model/approach state to ``path``.

        The approach is free to choose any serialization format (e.g. ``torch.save``,
        ``numpy.save``, JSON, pickle), but must write to exactly ``path``: the framework restores
        a state only when a file exists at ``path``.

        Needed whenever [adapt_or_load][SRToolkit.dataset.sr_dataset.SR_dataset.adapt_or_load] gets a
        state path.

        The default raises ``NotImplementedError``. Override when needed.

        Args:
            path: File path to save the adapted state to.
        """
        raise NotImplementedError

    def load_adapted_state(self, path: str) -> None:
        """Restore the previously saved adapted state from ``path``.

        Needed whenever [adapt_or_load][SRToolkit.dataset.sr_dataset.SR_dataset.adapt_or_load] gets a
        state path.

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
            TypeError: If ``approach_class`` is not an SR_approach subclass.
        """
        config = _auto_bind(config)
        target_cls = _import_class(config["approach_class"], "approach", SR_approach)
        return target_cls.from_config(config)
