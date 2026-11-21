"""Results of symbolic regression experiments and their post-processing.

Contains [SR_results][SRToolkit.evaluation.results.SR_results], a container for the
[EvalResult][SRToolkit.utils.types.EvalResult] of each experiment, and
[ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter], the base class for augmenters that
add post-hoc information to results. Concrete augmenters are in
[result_augmentation][SRToolkit.evaluation.result_augmentation].
"""

import json
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, Literal

import numpy as np

from SRToolkit.bundle._relocate import _import_class
from SRToolkit.utils.free_constants import Constant
from SRToolkit.utils.types import EvalResult, ModelResult


class ResultAugmenter(ABC):
    _type: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._type = f"{cls.__module__}.{cls.__qualname__}"

    def __init__(self, name: str):
        """Base class for result augmenters. Subclasses implement
        [write_results][SRToolkit.evaluation.results.ResultAugmenter.write_results] to compute
        and store additional data in an [EvalResult][SRToolkit.utils.types.EvalResult] via
        [add_augmentation][SRToolkit.utils.types.EvalResult.add_augmentation].

        For concrete implementations, see
        [result_augmentation][SRToolkit.evaluation.result_augmentation].

        Args:
            name: Identifier used as the key in
                ``augmentations`` dict of [EvalResult][SRToolkit.utils.types.EvalResult] and
                [ModelResult][SRToolkit.utils.types.ModelResult].
                If two augmenters share the same name,
                [add_augmentation][SRToolkit.utils.types.EvalResult.add_augmentation] appends a
                numeric suffix automatically.
        """
        self.name = name

    @abstractmethod
    def write_results(
        self,
        results: "EvalResult",
    ) -> None:
        """Compute and write augmentation data into *results* and its models.

        Call ``results.add_augmentation(self.name, data, self._type)`` for experiment-level
        data and ``model.add_augmentation(self.name, data, self._type)`` for per-model data.
        ``data`` may hold only values that can be saved (see
        [serialization][SRToolkit.utils.serialization]), or saving the results raises ``TypeError``.

        Args:
            results: The [EvalResult][SRToolkit.utils.types.EvalResult] to augment.
        """

    @abstractmethod
    def to_dict(self, base_path: str, name: str) -> dict:
        """Transforms the augmenter into a dictionary. This is used for saving the augmenter to disk.

        Args:
            base_path: The base path used for saving the data inside the augmenter, if needed.
            name: The name/identifier used by the augmenter for saving.

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """

    @classmethod
    def format_eval_result(cls, data: dict[str, Any]) -> str:
        """Returns a formatted string for experiment-level augmentation data.

        Subclasses override this for custom formatting. The *data* dict is the inner
        augmentation dictionary (includes ``_type``).

        Args:
            data: The augmentation data dictionary.

        Returns:
            A formatted string, or empty string if no relevant data exists.
        """
        return "\n".join(f"  {k}: {v}" for k, v in data.items() if k != "_type")

    @classmethod
    def format_model_result(cls, data: dict[str, Any]) -> str:
        """Returns a formatted string for a single model's augmentation data.

        Subclasses override this for custom formatting. The *data* dict is the inner
        augmentation dictionary (includes ``_type``).

        Args:
            data: The augmentation data dictionary.

        Returns:
            A formatted string, or empty string if no relevant data exists.
        """
        parts = [f"{k}={v}" for k, v in data.items() if k != "_type"]
        return ", ".join(parts)

    @staticmethod
    def from_dict(data: dict) -> "ResultAugmenter":
        """Creates an instance of the ResultAugmenter class from the dictionary with the relevant data.

        Subclasses should override this method if they support serialization. The default
        implementation raises ``NotImplementedError``, allowing custom augmenters to skip
        serialization if not needed.

        Args:
            data: the dictionary containing the data needed to recreate the augmenter.

        Returns:
            An instance of the ResultAugmenter class with the same configuration as in the data dictionary.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "from_dict is not implemented for this augmenter. "
            "Override this method if your augmenter supports serialization."
        )


class SR_results:
    def __init__(self):
        """Container for SR experiment results, typically obtained via
        [SR_evaluator.get_results][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_results].

        Examples:
            >>> from SRToolkit.evaluation import SR_evaluator
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> _ = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> print(results[0].best_expr)
            C*X_1-X_0
            >>> print(results[0].min_error < 1e-6)
            True
            >>> len(results)
            1

        Attributes:
            results: List of [EvalResult][SRToolkit.utils.types.EvalResult] instances,
                one per experiment.
        """
        self.results = list()

    def add_results(
        self,
        models: Sequence[ModelResult],
        trace: Sequence[int],
        top_k: int,
        success_threshold: float | None,
        approach_name: str,
        metadata: dict | None = None,
        default_constant: Constant | None = None,
        stop_reason: str | None = None,
    ) -> None:
        """Adds the results of an evaluation to the results object.

        Args:
            models: Evaluated models in the order they were first evaluated.
            trace: The evaluation trace: the index into ``models`` requested by each counted call.
            top_k: Default number of models returned by ``top_models()``. Negative means all.
            success_threshold: The success threshold used to determine whether the evaluation was successful.
            approach_name: The name of the approach used to discover the equations.
            metadata: A dictionary containing additional metadata about the evaluation.
            default_constant: The default spec of free constants in the evaluation, stored as
                [EvalResult.default_constant][SRToolkit.utils.types.EvalResult].
            stop_reason: Why the run ended, stored as
                [EvalResult.stop_reason][SRToolkit.utils.types.EvalResult].
        """
        dataset_name = None
        remaining_metadata = None
        if metadata is not None and "dataset_name" in metadata:
            dataset_name = metadata["dataset_name"]
            remaining_metadata = {key: value for key, value in metadata.items() if key != "dataset_name"}
            if len(remaining_metadata) == 0:
                remaining_metadata = None
        elif metadata is not None:
            remaining_metadata = metadata

        results_obj = EvalResult(
            all_models=list(models),
            trace=list(trace),
            approach_name=approach_name,
            success_threshold=success_threshold,
            top_k=top_k,
            dataset_name=dataset_name,
            metadata=remaining_metadata,
            default_constant=default_constant,
            stop_reason=stop_reason,
        )

        self.results.append(results_obj)

    def print_results(
        self,
        experiment_number: int | None = None,
        detailed: bool = False,
        model_scope: Literal["best", "top", "all"] = "top",
        augmentations: list[str] | None = None,
    ):
        r"""Prints the results of the SR_evaluator.

        Displays the minimum error, best expression, evaluation counts, success status, stop
        reason, metadata, and approach name. When *detailed* is ``True``, also prints per-model
        information. Augmentation data is formatted by the corresponding
        [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter] subclass,
        looked up from the global registry via the ``_type`` field stored in each
        augmentation entry.

        Examples:
            >>> from SRToolkit.evaluation import SR_evaluator
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> results.print_results()  # doctest: +ELLIPSIS
            === Experiment 1/1 ===
            Best expression: C*X_1-X_0
            Error: ...
            Evaluated: 1 expressions | Calls: 1 | Success: ...
            Stopped: ...
            <BLANKLINE>
            >>> results.print_results(detailed=True, experiment_number=0)  # doctest: +ELLIPSIS
            Best expression: C*X_1-X_0
            Error: ...
            Evaluated: 1 expressions | Calls: 1 | Success: ...
            Stopped: ...
            <BLANKLINE>
            Models:
              C*X_1-X_0  (error=..., params=...)
            <BLANKLINE>

        Args:
            experiment_number: Number of the experiment to print. If None, prints all.
            detailed: If True, prints per-model information.
            model_scope: Which models to show when *detailed* is True.
                ``"best"`` shows only the top model, ``"top"`` shows the top-k,
                ``"all"`` shows all evaluated models.
            augmentations: Filter which augmenters to display by name.
                If None, all augmentations present in the data are shown.

        Raises:
            TypeError: If ``experiment_number`` is not an integer.
            IndexError: If ``experiment_number`` is out of range. Negative numbers count from the end.
        """
        if experiment_number is None:
            for i, result in enumerate(self.results):
                print(f"=== Experiment {i + 1}/{len(self.results)} ===")
                SR_results._print_result_(result, detailed, model_scope, augmentations)
                print()
        else:
            SR_results._print_result_(self[experiment_number], detailed, model_scope, augmentations)

    @staticmethod
    def _print_result_(
        result: EvalResult,
        detailed: bool = False,
        model_scope: Literal["best", "top", "all"] = "top",
        augmentations_filter: list[str] | None = None,
    ):
        def _augmenter_class(data: dict) -> type[ResultAugmenter] | None:
            type_str = data.get("_type")
            if not isinstance(type_str, str):
                return None
            try:
                return _import_class(type_str, "augmenter", ResultAugmenter)
            except (ImportError, TypeError):
                return None

        # --- Core info ---
        if result.dataset_name is not None:
            print(f"Dataset: {result.dataset_name}")
        if result.approach_name != "":
            print(f"Approach: {result.approach_name}")
        print(f"Best expression: {result.best_expr}")
        print(f"Error: {result.min_error}")
        print(
            f"Evaluated: {result.num_evaluated} expressions | Calls: {result.evaluation_calls} | Success: {result.success}"
        )
        if result.stop_reason is not None:
            print(f"Stopped: {result.stop_reason}")
        if result.wall_time is not None:
            print(f"Wall time: {result.wall_time:.3f}s")
        if result.metadata is not None and len(result.metadata) > 0:
            print("Metadata:")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")
        print()

        # --- Resolve which augmenter keys to show ---
        eval_aug_keys = list(result.augmentations.keys())
        if augmentations_filter is not None:
            eval_aug_keys = [k for k in eval_aug_keys if k in augmentations_filter]

        # --- Experiment-level augmentation sections ---
        for key in eval_aug_keys:
            data = result.augmentations[key]
            cls = _augmenter_class(data)
            print(f"--- {key} ---")
            if cls is not None:
                line = cls.format_eval_result(data)
                if line:
                    print(line)
            else:
                for k, v in data.items():
                    if k != "_type":
                        print(f"  {k}: {v}")

        # --- Per-model output ---
        if detailed:
            # Pick model list based on scope
            if model_scope == "best":
                models = result.top_models(1)
            elif model_scope == "top":
                models = result.top_models()
            else:
                models = result.all_models

            if models:
                print()
                print("Models:")
                for model in models:
                    parts = [f"error={model.error:.6g}"]
                    if model.parameters is not None:
                        parts.append(f"params={np.round(model.parameters, 4).tolist()}")
                    expr_str = "".join(model.expr)
                    print(f"  {expr_str}  ({', '.join(parts)})")

                    # Model-level augmentations
                    model_aug_keys = list(model.augmentations.keys())
                    if augmentations_filter is not None:
                        model_aug_keys = [k for k in model_aug_keys if k in augmentations_filter]
                    for key in model_aug_keys:
                        data = model.augmentations[key]
                        cls = _augmenter_class(data)
                        if cls is not None:
                            line = cls.format_model_result(data)
                            if line:
                                print(f"    {key}: {line}")
                        else:
                            parts = [f"{k}={v}" for k, v in data.items() if k != "_type"]
                            if parts:
                                print(f"    {key}: {', '.join(parts)}")

    def augment(
        self, augmenters: list[ResultAugmenter] | ResultAugmenter, experiment_number: int | None = None
    ) -> None:
        r"""Applies the given [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter]
        instances to the stored results. Augmenters add post-hoc information such as LaTeX
        representations, simplified expressions, or R² scores.

        Examples:
            >>> from SRToolkit.evaluation import SR_evaluator
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> from SRToolkit.evaluation.result_augmentation import ExpressionToLatex
            >>> from SRToolkit.utils import SymbolLibrary
            >>> results.augment([ExpressionToLatex(SymbolLibrary.default_symbols(2))])
            >>> results[0].augmentations["ExpressionToLatex"]["best_expr_latex"]  # doctest: +ELLIPSIS
            '$C_{0} \\cdot X_{1} - X_{0}$'

        Args:
            augmenters: A [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter] or a list of [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter] objects to apply to the results.
            experiment_number: If provided, apply augmenters only to this experiment's result.
                If None, apply to all results.
        """
        if isinstance(augmenters, ResultAugmenter):
            augmenters = [augmenters]

        if experiment_number is not None:
            assert experiment_number < len(self.results), "[SR_results.augment] experiment number out of bounds"
            for augmenter in augmenters:
                try:
                    augmenter.write_results(self.results[experiment_number])
                except Exception as e:
                    warnings.warn(f"Error augmenting results with {augmenter.name}, skipping: {e}")
        else:
            for result in self.results:
                for augmenter in augmenters:
                    try:
                        augmenter.write_results(result)
                    except Exception as e:
                        warnings.warn(f"Error augmenting results with {augmenter.name}, skipping: {e}")

    def __add__(self, other: "SR_results") -> "SR_results":
        """Returns a new SR_results object that is the concatenation of the current SR_results object with the other SR_results object.

        Args:
            other: SR_results object to concatenate with the current SR_results object.

        Returns:
            A new SR_results object containing the concatenated results.
        """
        new = SR_results()
        new.results = self.results + other.results
        return new

    def __iadd__(self, other: "SR_results") -> "SR_results":
        """In-place concatenation of SR_results objects.

        Args:
            other: SR_results object to concatenate with the current SR_results object.

        Returns:
            self
        """
        self.results += other.results
        return self

    def __getitem__(self, item: int) -> EvalResult:
        """Returns the results of the experiment with the given index.

        Examples:
            >>> from SRToolkit.evaluation import SR_evaluator
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> result_of_first_experiment = results[0]

        Args:
            item: the index of the experiment. Negative indices count from the end, as for a list.

        Returns:
            The results of the experiment with the given index.

        Raises:
            TypeError: If ``item`` is not an integer.
            IndexError: If ``item`` is out of range.
        """
        if not isinstance(item, int):
            raise TypeError(f"[SR_results.__getitem__] Index must be an integer, got {type(item).__name__}.")
        if not -len(self.results) <= item < len(self.results):
            raise IndexError(f"[SR_results.__getitem__] Index {item} out of range for {len(self.results)} results.")
        return self.results[item]

    def __iter__(self) -> Iterator[EvalResult]:
        """Iterates over the results stored in the results object, in the order they were added. Usually, each result corresponds to a single experiment.

        Examples:
            >>> from SRToolkit.evaluation import SR_evaluator
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> for r in results:
            ...     print(r.best_expr)
            C*X_1-X_0

        Returns:
            An iterator over the [EvalResult][SRToolkit.utils.types.EvalResult] objects stored in the results object.
        """
        return iter(self.results)

    def __len__(self) -> int:
        """Returns the number of results stored in the results object. Usually, each result corresponds to a single experiment.

        Examples:
            >>> from SRToolkit.evaluation import SR_evaluator
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> len(results)
            1

        Returns:
            The number of results stored in the results object.
        """
        return len(self.results)

    def save(self, path: str) -> None:
        """Saves the results to a specific file or directory as JSON.

        If *path* is an existing directory, it writes ``results.json`` inside it.
        If *path* is a file path, it must end with the ``.json`` extension.

        Examples:
            >>> import tempfile
            >>> from SRToolkit.evaluation import SR_evaluator
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> _ = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     results.save(tmpdir + "/my_results/results.json")
            ...     loaded = SR_results.load(tmpdir + "/my_results/results.json")
            ...     print(loaded[0].best_expr)
            C*X_1-X_0

        Args:
            path: Directory path or specific .json file path.

        Raises:
            ValueError: If the path is a file with an extension other than .json.
            OSError: If the directory cannot be created.
            TypeError: If a result holds augmentation or ranking data that cannot be saved (see
                [serialization][SRToolkit.utils.serialization]).
        """
        if os.path.isdir(path):
            target_file = os.path.join(path, "results.json")
        else:
            _, extension = os.path.splitext(path)

            if extension == "":
                target_file = os.path.join(path, "results.json")
            elif extension.lower() == ".json":
                target_file = path
            else:
                raise ValueError(f"Invalid file extension '{extension}'. Results must be saved to a '.json' file.")

        target_dir = os.path.dirname(target_file)
        if target_dir and not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        # 3. Prepare data
        output = {
            "format_version": 1,
            "type": "SR_results",
            "results": [r.to_dict() for r in self.results],
        }

        with open(target_file, "w") as f:
            json.dump(output, f, indent=2)

    @staticmethod
    def load(path: str) -> "SR_results":
        """Load results previously saved with [save][SRToolkit.evaluation.results.SR_results.save].

        If *path* is a directory, it looks for ``results.json`` inside it.
        If *path* is a file, it must end with the ``.json`` extension.

        Args:
            path: Directory path containing ``results.json`` or path to a specific .json file.

        Returns:
            A new [SR_results][SRToolkit.evaluation.results.SR_results] instance with the loaded data.

        Raises:
            FileNotFoundError: If the specified file or directory does not exist.
            ValueError: If the file extension is not .json or if ``format_version`` is not ``1``.
        """
        if os.path.isdir(path):
            results_path = os.path.join(path, "results.json")
        else:
            _, extension = os.path.splitext(path)
            if extension.lower() != ".json":
                raise ValueError(
                    f"Invalid file extension '{extension}'. SR_results can only be loaded from '.json' files."
                )
            results_path = path

        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Could not find results file at: {results_path}")

        with open(results_path, "r") as f:
            data = json.load(f)

        if data.get("format_version", 1) != 1:
            raise ValueError(
                f"[SR_results.load] Unsupported format_version: {data.get('format_version')!r}. Expected 1."
            )

        sr_results = SR_results()
        sr_results.results = [EvalResult.from_dict(r) for r in data["results"]]
        return sr_results
