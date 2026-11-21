"""Data-source descriptors for symbolic regression datasets.

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

import json
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

from SRToolkit.bundle._relocate import _auto_bind, _import_class
from SRToolkit.dataset.sampling import Sampler, _sample_inputs
from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.free_constants import Constant, _ground_truth_parameters
from SRToolkit.utils.symbol_library import SymbolLibrary

_URL_TIMEOUT = 60  # seconds without data before a download fails


def _config_samplers(config: dict) -> list | None:
    """The saved samplers of a dataset config, from ``background_knowledge`` (format 4) or the top level (2/3)."""
    bk = config.get("background_knowledge")
    if isinstance(bk, dict):
        return bk.get("samplers")
    return config.get("samplers")


class DataSource(ABC):
    """Abstract base class describing the origin of a dataset's cached data.

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
        """Serialize this source to a JSON-compatible dictionary.

        The returned dict **must** include ``"source_class"`` set to the fully-qualified
        class path of this source.
        """

    @classmethod
    def from_dict(cls, d: dict | None) -> "DataSource" | None:
        """Reconstruct a source from a dict produced by
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
            TypeError: If the class is not a DataSource subclass (dispatch path).
            NotImplementedError: If called on a subclass that has not overridden this method.
        """
        if cls is DataSource:
            if d is None:
                return None
            d = _auto_bind(d)
            resolved = _import_class(d["source_class"], "data source", DataSource)
            return resolved.from_dict(d)
        raise NotImplementedError(f"{cls.__name__}.from_dict is not implemented.")

    @abstractmethod
    def materialize(self, cache_path: Path, config: dict) -> None:
        """Produce the dataset's data at ``cache_path``.

        The file is an ``.npz`` with ``X`` and, for RMSE datasets, ``y``.

        Args:
            cache_path: Target ``.npz`` path inside the data cache.
            config: The full serialised dataset config (see
                [SR_dataset.to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict]).
                Sources that generate data (e.g.
                [SampleSource][SRToolkit.dataset.data_source.SampleSource]) read the
                dataset's samplers, symbol library, ``"ground_truth"`` and
                ``"ranking_function"`` from it. The samplers and symbol library are read from
                ``"background_knowledge"`` (format 4) or from the top level (formats 2 and 3).
        """


class UrlSource(DataSource):
    """Data downloaded as a ``.zip`` archive from a URL and extracted into the cache.

    The archive holds ``<key>.npz`` files (``<key>`` is ``<name>``, ``<name>/test`` or
    ``<name>/aux/<k>``) at its root or under ``data/`` (as the built-in benchmarks host them),
    or it is an archive written by [share.save][SRToolkit.share.save], of which only the part
    for the dataset's own benchmark and version is taken. Every data file in the taken part is
    written to the cache (overwriting files of the same name), so the other datasets in it load
    without another download.

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
    def from_dict(cls, d: dict | None) -> "UrlSource":
        """Deserialize a [UrlSource][SRToolkit.dataset.data_source.UrlSource] from a dictionary produced by its ``to_dict``.

        Args:
            d: The serialized source (its ``url``).

        Returns:
            The reconstructed source.

        Raises:
            ValueError: If ``d`` is ``None``.
        """
        if d is None:
            raise ValueError("[UrlSource.from_dict] requires a dictionary, got None.")
        return cls(d["url"])

    def materialize(self, cache_path: Path, config: dict) -> None:
        """Download the archive and extract its data files into the dataset's version directory.

        Args:
            cache_path: Target path, as given by
                [dataset_path][SRToolkit.dataset.data_cache.dataset_path]. The archive is extracted
                into the version folder of its benchmark, also for test and aux keys.
            config: The dataset's config dict (unused).

        Raises:
            urllib.error.URLError: If the archive cannot be downloaded.
            TimeoutError: If the server sends nothing for 60 seconds.
            RuntimeError: If the archive has no file for this dataset's key.
            ValueError: If an archive member would land outside the version folder; nothing is
                written then.
        """
        from SRToolkit.dataset import data_cache

        # <data_root>/<benchmark>/<version_slug>, also for keys with a "/" (test and aux splits).
        benchmark, slug = cache_path.relative_to(data_cache.data_root()).parts[:2]
        version_dir = data_cache.data_root() / benchmark / slug

        # Download to a temporary file rather than into memory; the timeout keeps a dead server from
        # hanging the call.
        with tempfile.TemporaryFile() as archive:
            with urlopen(self.url, timeout=_URL_TIMEOUT) as response:
                shutil.copyfileobj(response, archive)
            archive.seek(0)
            with ZipFile(archive) as zf:
                names = zf.namelist()
                if "manifest.json" in names:  # written by share.save
                    prefix = f"data/{benchmark}/{slug}/"
                elif any(name.startswith("data/") for name in names):  # the built-in benchmarks' hosted archives
                    prefix = "data/"
                else:
                    prefix = ""
                # The archive may come from an untrusted URL: check every destination before writing any.
                targets = {}
                for name in names:
                    if not name.startswith(prefix) or name.endswith("/"):
                        continue
                    target = version_dir / name[len(prefix) :]
                    if not target.resolve().is_relative_to(version_dir.resolve()):
                        raise ValueError(
                            f"[UrlSource.materialize] Refusing to extract {name!r}: it leads outside {str(version_dir)!r}."
                        )
                    targets[name] = target
                for name, target in targets.items():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(name))
                    # Its drift sidecar described the replaced data (resolve writes a new one for cache_path).
                    data_cache._meta_path(target).unlink(missing_ok=True)

        if not cache_path.exists():
            raise RuntimeError(
                f"[UrlSource.materialize] After downloading from '{self.url}', the expected "
                f"file '{cache_path}' still does not exist."
            )


class SampleSource(DataSource):
    """Data generated by drawing ``n_samples`` points from the dataset's ``samplers``.

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

    def __init__(self, n_samples: int = 10000, seed: int | None = None):
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
    def from_dict(cls, d: dict | None) -> "SampleSource":
        """Deserialize a [SampleSource][SRToolkit.dataset.data_source.SampleSource] from a dictionary produced by its ``to_dict``.

        Args:
            d: The serialized source (its ``n_samples`` and ``seed``).

        Returns:
            The reconstructed source.

        Raises:
            ValueError: If ``d`` is ``None``.
        """
        if d is None:
            raise ValueError("[SampleSource.from_dict] requires a dictionary, got None.")
        return cls(n_samples=d.get("n_samples", 10000), seed=d.get("seed"))

    def materialize(self, cache_path: Path, config: dict) -> None:
        """Generate ``X`` (and ``y`` for RMSE) from the dataset's samplers and save them.

        Args:
            cache_path: Target ``.npz`` path inside the data cache.
            config: The dataset's config dict; its samplers, ``ranking_function``,
                ``ground_truth``, ``ground_truth_constants`` and symbol library are used to generate
                the data. The samplers
                and symbol library are read from ``"background_knowledge"`` (format 4) or from
                the top level (formats 2 and 3).

        Raises:
            ValueError: If the config has no samplers, or ``ranking_function`` is ``"rmse"`` and
                ``ground_truth`` is missing, an ndarray, or has a free constant without a fixed value
                in ``ground_truth_constants``, or a ``ground_truth_constants`` label is not a
                labelled constant of it.
        """
        samplers_raw = _config_samplers(config)
        if not samplers_raw:
            raise ValueError(
                "[SampleSource.materialize] Cannot generate data: the dataset defines no "
                "'samplers'. A SampleSource requires samplers (one per input variable)."
            )

        if self.seed is not None:
            np.random.seed(self.seed)

        samplers = [Sampler.from_dict(s) for s in samplers_raw]
        X = _sample_inputs(samplers, self.n_samples)

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
            # Format 4 keeps the symbol library in "background_knowledge"; formats 2 and 3 at the top level.
            bk = config.get("background_knowledge")
            sl_dict = bk.get("symbol_library") if isinstance(bk, dict) else config.get("symbol_library")
            if sl_dict is not None:
                sl = SymbolLibrary.from_dict(sl_dict)
            else:
                sl = SymbolLibrary.default_symbols(X.shape[1])
            specs = [Constant.from_dict(c) for c in config.get("ground_truth_constants") or ()]
            parameters = _ground_truth_parameters(
                ground_truth, specs, sl, f"[SampleSource.materialize] Dataset '{config.get('dataset_name', 'unnamed')}'"
            )
            y = compile_expr(ground_truth, sl)(X, parameters)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if y is not None:
            np.savez(str(cache_path), X=X, y=y)
        else:
            np.savez(str(cache_path), X=X)


class FallbackSource(DataSource):
    """A chain of data sources tried in order until one succeeds.

    [materialize][SRToolkit.dataset.data_source.FallbackSource.materialize] attempts each
    source's ``materialize`` in turn; if one raises, a warning is emitted and the next is
    tried. The first to succeed wins; if all fail, a ``RuntimeError`` chaining the last
    error is raised.

    The built-in benchmarks use this to prefer canonical downloaded data while keeping a
    network-free fallback —
    ``FallbackSource([UrlSource(archive), SampleSource(seed=...)])`` downloads the
    authoritative archive when reachable and regenerates from the dataset's samplers
    otherwise. Because the whole chain is serialised in the config, the preference
    **travels**: a grid recipe, a [share.save][SRToolkit.share.save] archive, or
    a cold worker each download the canonical data first and only sample if it is
    unavailable — unlike a bare ``SampleSource``, which would always regenerate.

    Args:
        sources: Ordered, non-empty list of
            [DataSource][SRToolkit.dataset.data_source.DataSource] to try.
    """

    def __init__(self, sources: list[DataSource]):
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
    def from_dict(cls, d: dict | None) -> "FallbackSource":
        """Deserialize a [FallbackSource][SRToolkit.dataset.data_source.FallbackSource] from a dictionary produced by its ``to_dict``.

        Args:
            d: The serialized source (its list of ``sources``, each reconstructed in turn).

        Returns:
            The reconstructed source.

        Raises:
            ValueError: If ``d`` is ``None``.
        """
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
        """Try each source in order; warn and continue on failure, re-raising if all fail.

        Args:
            cache_path: Target ``.npz`` path inside the data cache.
            config: The dataset's config dict, passed to each source.
        """
        last_exc: Exception | None = None
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
