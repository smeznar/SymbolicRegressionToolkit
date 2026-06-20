"""
Data-source descriptors for symbolic regression datasets.

A [DataSource][SRToolkit.dataset.data_source.DataSource] describes the *origin* of a
dataset's data (where the ``X`` / ``y`` arrays come from). This is a separate
concern from the dataset's *input-distribution spec* (its ``samplers``): the samplers
define what the inputs of the problem look like and power
[resample][SRToolkit.dataset.sr_dataset.SR_dataset.resample], and remain available no
matter where the data originated.

Two concrete sources ship with the toolkit:

- [UrlSource][SRToolkit.dataset.data_source.UrlSource] — download a ``.zip`` from a URL
  and extract it into the cache.
- [SampleSource][SRToolkit.dataset.data_source.SampleSource] — generate the data by
  drawing ``n_samples`` points from the dataset's ``samplers``.

A ``data_source`` of ``None`` means the data was supplied directly (e.g. as arrays) and
already lives in the cache; nothing needs to be materialised.

Sources serialize via a ``"source_class"`` key holding the fully-qualified class path,
mirroring other custom classes, e.g., [Sampler][SRToolkit.dataset.sampling.Sampler]. As a result
[DataSource.from_dict][SRToolkit.dataset.data_source.DataSource.from_dict] can reconstruct any
subclass — **including user-defined ones** — via ``importlib`` without a central
registry: subclass [DataSource][SRToolkit.dataset.data_source.DataSource], implement
``to_dict`` / ``from_dict`` / ``materialize``, and it round-trips. Custom sources only
travel to other machines via the bundle mechanism (their code must be importable on the
other end). Because reconstruction imports the class named in the config, only load
configs you trust.
"""

from __future__ import annotations

import importlib
import json
import warnings
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

from SRToolkit.bundle._relocate import _auto_bind
from SRToolkit.dataset.sampling import Sampler
from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.symbol_library import SymbolLibrary


class DataSource(ABC):
    """
    Abstract base class describing the origin of a dataset's cached data.

    Concrete subclasses must implement
    [to_dict][SRToolkit.dataset.data_source.DataSource.to_dict],
    [from_dict][SRToolkit.dataset.data_source.DataSource.from_dict], and
    [materialize][SRToolkit.dataset.data_source.DataSource.materialize]. The dict produced
    by ``to_dict`` must include a ``"source_class"`` key holding the fully-qualified class
    path (e.g. ``"SRToolkit.dataset.data_source.UrlSource"``) so that
    [DataSource.from_dict][SRToolkit.dataset.data_source.DataSource.from_dict] can reconstruct it
    via ``importlib`` without a central registry.

    The cache layer stores a hash of every source's ``data_source`` + ``samplers`` config
    and warns when it drifts from what the cache was built with — regardless of this flag —
    so a changed ``url``, changed ``n_samples``/``seed``, or a switch between source kinds is
    always surfaced.

    Attributes:
        is_volatile: Informational hint: ``True`` when the materialised output depends on
            parameters that can change between runs (e.g. sampler ranges or seed), as opposed
            to a fixed external artifact (a pinned ``url``). Defaults to ``False``. Does not
            affect drift detection, which runs for all sources.
    """

    is_volatile: bool = False

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Serialize this source to a JSON-compatible dictionary.

        The returned dict **must** include ``"source_class"`` set to the fully-qualified
        class path of this source.
        """

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["DataSource"]:
        """
        Reconstruct a source from a dict produced by
        [to_dict][SRToolkit.dataset.data_source.DataSource.to_dict].

        When called on the base [DataSource][SRToolkit.dataset.data_source.DataSource]
        class, dispatches to the concrete subclass named by the ``"source_class"`` key
        using ``importlib`` — both built-in and user-defined subclasses round-trip without
        a central registry. When called on a concrete subclass, that subclass must override
        this method.

        Args:
            d: Dictionary with a ``"source_class"`` key (fully-qualified class path) and the
                source's parameters, or ``None`` (data is already cached / supplied directly).

        Returns:
            A reconstructed [DataSource][SRToolkit.dataset.data_source.DataSource] instance,
            or ``None`` if ``d`` is ``None``.

        Raises:
            KeyError: If ``"source_class"`` is missing from ``d`` (dispatch path).
            ImportError: If the class cannot be imported (dispatch path).
            NotImplementedError: If called on a subclass that has not overridden this method.
        """
        if cls is DataSource:
            if d is None:
                return None
            d = _auto_bind(d)
            class_path = d["source_class"]
            module_path, cls_name = class_path.rsplit(".", 1)
            try:
                resolved = getattr(importlib.import_module(module_path), cls_name)
            except (ImportError, AttributeError):
                raise ImportError(
                    f"Cannot import data source class {class_path!r}. "
                    "If this is a bundle class, install the bundle first. "
                    "If the config has no '_bundle' key, call bind_config(config) manually."
                ) from None
            return resolved.from_dict(d)
        raise NotImplementedError(f"{cls.__name__}.from_dict is not implemented.")

    @abstractmethod
    def materialize(self, cache_path: Path, config: dict) -> None:
        """
        Produce the dataset's data at ``cache_path`` (an ``.npz`` file with ``X`` and,
        for RMSE datasets, ``y``).

        Args:
            cache_path: Target ``.npz`` path inside the data cache.
            config: The full serialised dataset config (see
                [SR_dataset.to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict]).
                Sources that generate data (e.g.
                [SampleSource][SRToolkit.dataset.data_source.SampleSource]) read the
                dataset's ``"samplers"``, ``"ground_truth"``, ``"symbol_library"``, and
                ``"ranking_function"`` from it.
        """


class UrlSource(DataSource):
    """
    Data downloaded as a ``.zip`` archive from a URL and extracted into the cache.

    Two archive layouts are accepted transparently: a *flat* zip whose
    ``<dataset_name>.npz`` files sit at the root (as served by the built-in benchmarks),
    or a ``to_archive`` archive whose data lives under a ``data/`` prefix (in which case
    the prefix is stripped and the bundled ``benchmark.json`` / ``dataset.json`` is
    ignored). Either way, the expected ``<dataset_name>.npz`` must end up in the version directory.

    Args:
        url: URL of a ``.zip`` archive.
    """

    is_volatile = False

    def __init__(self, url: str):
        self.url = url

    def to_dict(self) -> dict:
        """Serialize this source to a JSON-compatible dictionary."""
        return {"source_class": "SRToolkit.dataset.data_source.UrlSource", "url": self.url}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "UrlSource":
        """Deserialize a [UrlSource][SRToolkit.dataset.data_source.UrlSource] from a dictionary."""
        if d is None:
            raise ValueError("[UrlSource.from_dict] requires a dictionary, got None.")
        return cls(d["url"])

    def materialize(self, cache_path: Path, config: dict) -> None:
        """Download the archive and extract its data files into the version directory."""
        from SRToolkit.dataset import data_cache

        version_dir = cache_path.parent
        version_dir.mkdir(parents=True, exist_ok=True)

        http_response = urlopen(self.url)
        with ZipFile(BytesIO(http_response.read())) as zf:
            data_cache.extract_zip_into_version_dir(zf, version_dir)

        if not cache_path.exists():
            raise RuntimeError(
                f"[UrlSource.materialize] After downloading from '{self.url}', the expected "
                f"file '{cache_path}' still does not exist."
            )


class SampleSource(DataSource):
    """
    Data generated by drawing ``n_samples`` points from the dataset's ``samplers``.

    This source does **not** own the samplers — it carries only the generation parameters
    and reads the samplers from the dataset config at materialisation time. The dataset
    must therefore define ``samplers`` (one per input variable). For RMSE datasets with a
    token-list ground truth, the targets ``y`` are produced by evaluating that expression
    on the generated inputs.

    Because the output depends on the samplers and seed, this source is *volatile*: the
    cache layer records a hash and warns when the configuration drifts (call
    [refresh][SRToolkit.dataset.sr_dataset.SR_dataset.refresh] to regenerate).

    Args:
        n_samples: Number of input rows to generate. Defaults to ``10000``.
        seed: Random seed for reproducible generation. ``None`` means no seed is set.
    """

    is_volatile = True

    def __init__(self, n_samples: int = 10000, seed: Optional[int] = None):
        self.n_samples = n_samples
        self.seed = seed

    def to_dict(self) -> dict:
        """Serialize this source to a JSON-compatible dictionary."""
        return {
            "source_class": "SRToolkit.dataset.data_source.SampleSource",
            "n_samples": self.n_samples,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "SampleSource":
        """Deserialize a [SampleSource][SRToolkit.dataset.data_source.SampleSource] from a dictionary."""
        if d is None:
            raise ValueError("[SampleSource.from_dict] requires a dictionary, got None.")
        return cls(n_samples=d.get("n_samples", 10000), seed=d.get("seed"))

    def materialize(self, cache_path: Path, config: dict) -> None:
        """Generate ``X`` (and ``y`` for RMSE) from the dataset's samplers and save them."""
        samplers_raw = config.get("samplers")
        if not samplers_raw:
            raise ValueError(
                "[SampleSource.materialize] Cannot generate data: the dataset defines no "
                "'samplers'. A SampleSource requires samplers (one per input variable)."
            )

        if self.seed is not None:
            np.random.seed(self.seed)

        samplers = [Sampler.from_dict(s) for s in samplers_raw]
        X = np.column_stack([s(self.n_samples) for s in samplers])

        ranking_function = config.get("ranking_function", "rmse")
        ground_truth = config.get("ground_truth")
        y = None

        if ranking_function == "rmse":
            if ground_truth is None:
                raise ValueError(
                    "[SampleSource.materialize] Cannot generate data: ranking_function is "
                    "'rmse' but the dataset has no 'ground_truth'. A SampleSource produces "
                    "targets 'y' by evaluating the ground-truth expression on the sampled "
                    "inputs, so a ground truth is required. Provide one, or supply X/y "
                    "directly (data_source=None)."
                )
            if isinstance(ground_truth, np.ndarray):
                raise ValueError(
                    "[SampleSource.materialize] ranking_function is 'rmse' but 'ground_truth' "
                    "is a numpy array (a behaviour matrix). Behaviour matrices are a 'bed' "
                    "concept and cannot be evaluated to produce targets 'y'. Provide the "
                    "ground truth as a token list or Node expression for RMSE datasets."
                )
            sl_dict = config.get("symbol_library")
            if sl_dict is not None:
                sl = SymbolLibrary.from_dict(sl_dict)
            else:
                sl = SymbolLibrary.default_symbols(X.shape[1])
            f = compile_expr(ground_truth, sl)
            y = f(X, np.array([]))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if y is not None:
            np.savez(str(cache_path), X=X, y=y)
        else:
            np.savez(str(cache_path), X=X)


class FallbackSource(DataSource):
    """
    A chain of data sources tried in order until one succeeds.

    [materialize][SRToolkit.dataset.data_source.FallbackSource.materialize] attempts each
    source's ``materialize`` in turn; if one raises, a warning is emitted and the next is
    tried. The first to succeed wins; if all fail, a ``RuntimeError`` chaining the last
    error is raised.

    The built-in benchmarks use this to prefer canonical downloaded data while keeping a
    network-free fallback —
    ``FallbackSource([UrlSource(archive), SampleSource(seed=...)])`` downloads the
    authoritative archive when reachable and regenerates from the dataset's samplers
    otherwise. Because the whole chain is serialised in the config, the preference
    **travels**: a grid recipe, an [export][SRToolkit.experiments.ExperimentGrid.export], or
    a cold worker each download the canonical data first and only sample if it is
    unavailable — unlike a bare ``SampleSource``, which would always regenerate.

    Args:
        sources: Ordered, non-empty list of
            [DataSource][SRToolkit.dataset.data_source.DataSource] to try.
    """

    def __init__(self, sources: List[DataSource]):
        if not sources:
            raise ValueError("[FallbackSource] requires at least one source.")
        self.sources = list(sources)
        # ``True`` if any source in the chain is volatile (e.g. a ``SampleSource``).
        self.is_volatile = any(s.is_volatile for s in self.sources)

    def to_dict(self) -> dict:
        """Serialize this source and its whole chain to a JSON-compatible dictionary."""
        return {
            "source_class": "SRToolkit.dataset.data_source.FallbackSource",
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "FallbackSource":
        """Deserialize a [FallbackSource][SRToolkit.dataset.data_source.FallbackSource], reconstructing each child source."""
        if d is None:
            raise ValueError("[FallbackSource.from_dict] requires a dictionary, got None.")
        sources = []
        for s in d["sources"]:
            ds = DataSource.from_dict(s)
            if ds is not None:
                sources.append(ds)
        if len(sources) > 0:
            return cls(sources)
        else:
            raise ValueError(
                f"[FallbackSource.from_dict] Could not construct at least one DataSource for dictionary: {json.dumps(d)}"
            )

    def materialize(self, cache_path: Path, config: dict) -> None:
        """Try each source in order; warn and continue on failure, re-raising if all fail."""
        last_exc: Optional[Exception] = None
        for i, source in enumerate(self.sources):
            try:
                source.materialize(cache_path, config)
                return
            except Exception as exc:  # noqa: BLE001 - fall through to the next source in the chain
                last_exc = exc
                if i + 1 < len(self.sources):
                    warnings.warn(
                        f"[FallbackSource] {type(source).__name__} failed to materialise "
                        f"({exc}); trying the next fallback.",
                        stacklevel=2,
                    )
        raise RuntimeError(
            f"[FallbackSource] All {len(self.sources)} sources failed to materialise; last error: {last_exc}"
        ) from last_exc
