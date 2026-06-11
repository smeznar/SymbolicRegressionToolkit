"""
Persistent cache for symbolic regression benchmark datasets.

Datasets are stored as ``.npz`` files in a platform-appropriate data directory,
keyed by benchmark name, version, and dataset name. This module is the public
interface for locating, listing, refreshing, and garbage-collecting cached
datasets, and also houses the materialisation engine used internally by the
``dataset`` subpackage.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from platformdirs import user_data_dir

if TYPE_CHECKING:
    import zipfile

    from SRToolkit.dataset.data_source import DataSource


def data_root() -> Path:
    """Return ``<user_data_dir>/SRToolkit/data`` as a Path."""
    return Path(user_data_dir("SRToolkit")) / "data"


def _version_slug(version: str) -> str:
    """Replace dots and hyphens with underscores (mirrors bundle _store.py)."""
    return version.replace(".", "_").replace("-", "_")


def dataset_path(benchmark: str, version: str, key: str) -> Path:
    """
    Return the expected cache path for a dataset (the file may not exist yet).

    Args:
        benchmark: Benchmark name (e.g. ``"feynman"``).
        version: Version string (e.g. ``"1.0.0"``).
        key: Dataset key / name (e.g. ``"I.16.6"``).

    Returns:
        The ``Path`` to the ``.npz`` file that would hold this dataset.
    """
    slug = _version_slug(version)
    return data_root() / benchmark / slug / f"{key}.npz"


def _drift_hash(config: dict) -> str:
    """
    Return a 16-char SHA-256 digest of the parts of ``config`` that affect generated
    data: the ``data_source`` descriptor and the ``samplers``. Used to warn when a
    volatile source's configuration has drifted since the cache was built.
    """
    payload = {
        "data_source": config.get("data_source"),
        "samplers": config.get("samplers"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _meta_path(cache_path: Path) -> Path:
    # Use with_name (not chained with_suffix) so dataset keys containing dots — e.g.
    # Feynman's "I.16.6" — map to "I.16.6.meta.json" rather than "I.16.meta.json"
    # (which would also collide across keys like "I.16.6" and "I.16.7").
    return cache_path.with_name(cache_path.stem + ".meta.json")


def _read_meta(cache_path: Path) -> Optional[dict]:
    mp = _meta_path(cache_path)
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _write_meta(cache_path: Path, config: dict) -> None:
    mp = _meta_path(cache_path)
    mp.write_text(json.dumps({"source_hash": _drift_hash(config)}))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve(
    benchmark: str,
    version: str,
    key: str,
    config: Optional[dict] = None,
    *,
    force: bool = False,
) -> Path:
    """
    Return the cache path for ``<benchmark>/<version>/<key>.npz``, materialising
    it first if necessary.

    The data source is reconstructed from ``config["data_source"]`` via
    [source_from_dict][SRToolkit.dataset.data_source.source_from_dict]. Passing
    ``config=None`` (or a config whose ``"data_source"`` is ``None``) is valid only when
    the cache entry already exists — it asserts presence without triggering
    materialisation.

    On a cache hit (file exists and ``force`` is ``False``): if the source is *volatile*
    (e.g. [SampleSource][SRToolkit.dataset.data_source.SampleSource]), the stored sidecar
    ``.meta.json`` hash is compared against the current ``data_source`` + ``samplers``
    config. A mismatch emits a warning telling the user to call ``refresh()``.

    On a cache miss (or ``force=True``): the data is materialised by calling the source's
    [materialize][SRToolkit.dataset.data_source.DataSource.materialize].

    Args:
        benchmark: Benchmark name (e.g. ``"feynman"``).
        version: Version string (e.g. ``"1.0.0"``).
        key: Dataset key / name (e.g. ``"I.16.6"``).
        config: Full dataset config dict containing a ``"data_source"`` key.
            ``None`` (or a config with ``data_source=None``) is valid only when
            the cache entry already exists.
        force: If ``True``, re-materialise even when the cache entry exists.

    Returns:
        Path to the ``.npz`` file in the cache.

    Raises:
        FileNotFoundError: If the cache entry is absent and no source is available.
    """
    from SRToolkit.dataset.data_source import source_from_dict

    source_dict = config.get("data_source") if config is not None else None
    source = source_from_dict(source_dict)
    cache_path = dataset_path(benchmark, version, key)

    if cache_path.exists() and not force:
        # Drift detection: if the data_source / samplers config no longer matches what
        # the cache was built from, the cached bytes may be stale. This applies to any
        # source — a SampleSource whose n_samples/seed changed, a UrlSource whose url
        # changed, or a switch between source kinds (e.g. UrlSource ↔ SampleSource).
        if source is not None and config is not None:
            meta = _read_meta(cache_path)
            if meta is not None and meta.get("source_hash") != _drift_hash(config):
                warnings.warn(
                    f"[dataset.data_cache.resolve] The data_source config for '{benchmark}/{key}' has changed "
                    f"since the cache was built. Call refresh() to regenerate the data.",
                    stacklevel=3,
                )
        return cache_path

    # Need to materialise
    if source is None or config is None:  # Second condition is just for the type checker
        raise FileNotFoundError(
            f"[dataset.data_cache.resolve] No cached data found for '{benchmark}/{version}/{key}' "
            f"and data_source is None. Provide a data_source or import the data manually."
        )

    source.materialize(cache_path, config)
    _write_meta(cache_path, config)

    return cache_path


def extract_zip_into_version_dir(zf: zipfile.ZipFile, version_dir: Path, *, flat_fallback: bool = True) -> None:
    """
    Extract an open ``ZipFile`` into ``version_dir``, accepting both archive layouts.

    - **``to_archive`` layout**: data files live under a ``data/`` prefix (alongside a
      ``benchmark.json`` / ``dataset.json`` that is *not* a data file). The ``data/``
      prefix is stripped so the ``.npz`` (and ``_gt.npy``) files land directly in the
      version directory; non-``data/`` members are ignored.
    - **flat layout**: the ``.npz`` files sit at the archive root (as served by the
      built-in benchmarks). When ``flat_fallback`` is ``True`` (the default) and no
      ``data/``-prefixed member is present, the whole archive is extracted as-is.

    The layout is detected automatically by the presence of any ``data/``-prefixed
    member, so a single hosted ``to_archive`` archive can be consumed both by
    [SR_benchmark.from_url][SRToolkit.dataset.sr_benchmark.SR_benchmark.from_url] and by a
    config carrying a [UrlSource][SRToolkit.dataset.data_source.UrlSource].

    Args:
        zf: An open ``ZipFile`` to extract.
        version_dir: Destination cache version directory.
        flat_fallback: If ``True``, an archive with no ``data/`` prefix is extracted in
            full (flat layout). If ``False``, such an archive yields nothing — only
            ``data/``-prefixed members are ever extracted. [import_archive][SRToolkit.dataset.data_cache.import_archive]
            uses ``False`` so stray metadata files never land in the cache version directory.
    """
    members = zf.infolist()
    has_data_prefix = any(m.filename.startswith("data/") for m in members)

    if not has_data_prefix:
        if flat_fallback:
            zf.extractall(path=str(version_dir))
        return

    for member in members:
        name = member.filename
        if not name.startswith("data/"):
            continue
        # Strip the "data/" prefix
        relative = name[len("data/") :]
        if not relative:
            continue  # Skip the directory entry itself
        target = version_dir / relative
        # Guard against zip-slip: the archive may be downloaded from an untrusted URL,
        # so refuse members that resolve outside the destination version directory.
        resolved_dir = version_dir.resolve()
        resolved_target = target.resolve()
        if resolved_dir != resolved_target and resolved_dir not in resolved_target.parents:
            raise ValueError(
                f"[extract_zip_into_version_dir] Refusing to extract '{name}': resolves outside the "
                f"destination directory (possible zip-slip)."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(member.filename))


def import_archive(archive_path: Path, benchmark: str, version: str) -> None:
    """
    Extract a ``.zip`` archive into the cache version directory
    ``<data_root>/<benchmark>/<version_slug>/``.

    Both the ``to_archive`` layout (data under a ``data/`` prefix) and a flat layout
    (``.npz`` files at the archive root) are handled — see
    [extract_zip_into_version_dir][SRToolkit.dataset.data_cache.extract_zip_into_version_dir].

    Args:
        archive_path: Path to a ``.zip`` file (e.g. a benchmark ``.zip``).
        benchmark: Benchmark name used to determine the cache directory.
        version: Version string used to determine the cache directory.
    """
    import zipfile

    slug = _version_slug(version)
    version_dir = data_root() / benchmark / slug
    version_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(archive_path), "r") as zf:
        extract_zip_into_version_dir(zf, version_dir, flat_fallback=False)


def list() -> List[Dict[str, Any]]:
    """
    List all datasets currently stored in the cache.

    Walks ``data_root()`` and returns one dict per cached ``.npz``.

    Returns:
        A list of dicts, each with keys ``benchmark``, ``version``, ``key``,
        ``path``, and ``size_bytes``.
    """
    root = data_root()
    entries: List[Dict[str, Any]] = []
    if not root.exists():
        return entries

    for benchmark_dir in sorted(root.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        benchmark = benchmark_dir.name
        for version_dir in sorted(benchmark_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            version_slug = version_dir.name
            # Convert slug back to version string (underscores → dots)
            version = version_slug.replace("_", ".")
            for npz_file in sorted(version_dir.glob("*.npz")):
                entries.append(
                    {
                        "benchmark": benchmark,
                        "version": version,
                        "key": npz_file.stem,
                        "path": str(npz_file),
                        "size_bytes": npz_file.stat().st_size,
                    }
                )
    return entries


def gc(keep_latest: bool = True) -> List[Path]:
    """
    Garbage-collect the data cache.

    Args:
        keep_latest: If ``True`` (default), retain only the latest version
            (by semantic version ordering) per benchmark and delete all older
            versions. If ``False``, wipes the entire cache.

    Returns:
        A list of ``Path`` objects that were removed.
    """
    import shutil

    from packaging.version import Version as PkgVersion

    root = data_root()
    removed: List[Path] = []
    if not root.exists():
        return removed

    for benchmark_dir in sorted(root.iterdir()):
        if not benchmark_dir.is_dir():
            continue

        version_dirs = [d for d in benchmark_dir.iterdir() if d.is_dir()]
        if not version_dirs:
            continue

        if not keep_latest:
            for vdir in version_dirs:
                shutil.rmtree(vdir)
                removed.append(vdir)
            continue

        # Determine latest version
        def _slug_to_version(slug: str) -> Any:
            ver_str = slug.replace("_", ".")
            if PkgVersion is not None:
                try:
                    return PkgVersion(ver_str)
                except Exception:
                    return ver_str
            return ver_str

        sorted_dirs = sorted(version_dirs, key=lambda d: _slug_to_version(d.name))
        for vdir in sorted_dirs[:-1]:
            shutil.rmtree(vdir)
            removed.append(vdir)

    return removed


def remove(benchmark: str, version: Optional[str] = None, key: Optional[str] = None) -> List[Path]:
    """
    Remove a specific cached benchmark, version, or dataset.

    The deletion granularity widens as fewer arguments are given:

    - ``remove("feynman")`` — delete every cached version of the benchmark.
    - ``remove("feynman", "1.0.0")`` — delete just that version directory.
    - ``remove("feynman", "1.0.0", "I.16.6")`` — delete a single dataset's ``.npz`` plus
      its ``_gt.npy`` and ``.meta.json`` sidecars.

    Targeting something that is not in the cache is a no-op (returns an empty list);
    nothing is raised.

    Args:
        benchmark: Benchmark name (e.g. ``"feynman"``).
        version: Version string. If ``None``, the whole benchmark is removed.
        key: Dataset key / name. If ``None``, the whole version directory is removed.

    Returns:
        List of paths that were removed.
    """
    import shutil

    removed: List[Path] = []
    root = data_root()
    if not root.exists():
        return removed

    benchmark_dir = root / benchmark
    if not benchmark_dir.is_dir():
        return removed

    if version is None:
        shutil.rmtree(benchmark_dir)
        removed.append(benchmark_dir)
        return removed

    version_dir = benchmark_dir / _version_slug(version)
    if not version_dir.is_dir():
        return removed

    if key is None:
        shutil.rmtree(version_dir)
        removed.append(version_dir)
    else:
        npz_path = version_dir / f"{key}.npz"
        for target in (npz_path, version_dir / f"{key}_gt.npy", _meta_path(npz_path)):
            if target.exists():
                target.unlink()
                removed.append(target)

    return removed


def refresh(benchmark: str, version: str, key: str, source: "DataSource") -> None:
    """
    Force-refresh a cached dataset entry by re-materialising it from ``source``.

    Args:
        benchmark: Benchmark name.
        version: Version string.
        key: Dataset key / name.
        source: A [DataSource][SRToolkit.dataset.data_source.DataSource] describing the
            origin of the data (e.g. [UrlSource][SRToolkit.dataset.data_source.UrlSource]).
            For a [SampleSource][SRToolkit.dataset.data_source.SampleSource] — which needs the
            dataset's samplers — use
            [SR_dataset.refresh][SRToolkit.dataset.sr_dataset.SR_dataset.refresh] instead.
    """
    resolve(benchmark, version, key, {"data_source": source.to_dict()}, force=True)
