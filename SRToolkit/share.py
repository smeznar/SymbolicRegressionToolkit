"""Save a dataset, benchmark or experiment grid as one archive, and load it on another machine.

[save][SRToolkit.share.save] writes an object, with its data and the custom code it uses, into a
single ``.zip``; [load][SRToolkit.share.load] reads it back from a path or URL and returns the
object. Use it to hand someone a whole object. To share only the settings, use the object's
``to_dict`` / ``from_dict``; to share only code, use [pack][SRToolkit.bundle.pack] /
[install][SRToolkit.bundle.install].

Examples:
    >>> from SRToolkit import share
    >>> share.save(benchmark, "my_benchmark.zip")                # doctest: +SKIP
    >>> benchmark = share.load("my_benchmark.zip")               # doctest: +SKIP
    >>> grid = share.load("https://example.org/run.zip", install_bundles=True)  # doctest: +SKIP

**Archive contents.**

| Member | Contents |
|---|---|
| ``manifest.json`` | ``format_version`` (the archive format, currently ``1``), ``kind`` (``"dataset"``, ``"benchmark"`` or ``"grid"``), ``srtk_version``, plus bookkeeping for ``load`` and the README (shipped and referenced data, bundles, uncovered classes, ...). Of the manifest, only ``format_version`` and ``kind`` are a stable contract. |
| ``config.json`` | The object's ``to_dict()``. Each dict that names a class from a shipped bundle in a ``*_class`` key (an approach, a sampler, a data source, ...) also gets ``_bundle`` / ``_version`` naming that bundle. |
| ``data/<benchmark>/<version_slug>/<key>.npz`` | Shipped data, at the same relative path as in the [data cache][SRToolkit.dataset.data_cache] (``<key>`` is ``<name>``, ``<name>/test`` or ``<name>/aux/<k>``). This layout is stable: a [UrlSource][SRToolkit.dataset.data_source.UrlSource] can point at an archive's URL. |
| ``bundles/<name>_<version>.srtk`` | Bundles holding the custom classes the object uses. |
| ``results/<dataset>/<approach>/exp_*.json`` | Grids saved with ``results=True``: the finished runs. |
| ``README.md`` | What the archive holds and how to load it, for a person; never read by ``load``. |

**Which data is shipped.** Each split of each dataset (train, test, each aux split) is decided
on its own, by ``data``:

- ``"all"``: every split.
- ``"auto"``: only splits the recipient cannot reproduce from the config. Splits whose
  ``data_source`` is a [UrlSource][SRToolkit.dataset.data_source.UrlSource], a
  [SampleSource][SRToolkit.dataset.data_source.SampleSource] with a seed, or a
  [FallbackSource][SRToolkit.dataset.data_source.FallbackSource] with at least one such source
  are referenced, and the recipient downloads or regenerates them. Every other split (no
  ``data_source``, a ``SampleSource`` without a seed, a custom source, ...) is shipped.
- ``"none"``: no split.

Configs always keep their ``data_source``, so a loaded dataset can still be
[refreshed][SRToolkit.dataset.dataset.Dataset.refresh]. Datasets of a grid or benchmark
with the same benchmark, version and name share one copy of their data; they must have the same
data (the same ``data_source`` and samplers, and equal arrays where shipped).

**Custom code.** Every class outside SRToolkit that the object's configs name (approaches,
callbacks, samplers, data sources, constraints, rankings, ...) must come from a bundle to load
elsewhere. ``save`` ships the bundles given in ``bundles`` that define such a class, and re-packs
installed bundles for the rest: the one the class is imported from, or the one its config names in
``_bundle`` (as a loaded benchmark or grid keeps it). ``load`` never runs shipped code unless ``install_bundles=True``.
After installing, class paths point to the installed bundle, so ``to_dict()`` of a loaded object
that uses custom classes differs from the saved one.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import secrets
import shutil
import tempfile
import warnings
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

from SRToolkit.bundle import _store, install, list_installed, read_manifest
from SRToolkit.bundle._install import _repack
from SRToolkit.bundle._relocate import _class_refs, bundle_class_index
from SRToolkit.dataset import data_cache
from SRToolkit.dataset.data_source import _URL_TIMEOUT

if TYPE_CHECKING:
    from SRToolkit.dataset import Benchmark, Dataset
    from SRToolkit.experiments import ExperimentGrid

_FORMAT_VERSION = 1
_SOURCES = "SRToolkit.dataset.data_source."


def save(
    obj: Dataset | Benchmark | ExperimentGrid,
    path: str | Path,
    *,
    data: Literal["all", "auto", "none"] = "all",
    bundles: Sequence[str | Path] = (),
    strict: bool = False,
    results: bool = False,
) -> Path:
    """Write ``obj`` with its data and custom code to a ``.zip`` archive.

    Datasets held as configs (benchmark entries, grid datasets) are materialised (read from the
    cache, downloaded or sampled) when their data is shipped. A custom class that neither
    ``bundles`` nor an installed bundle defines is not shipped, and a custom class nested in the
    config of a class from another bundle cannot load on the other end: both give a warning, or an
    error with ``strict``. An existing file at ``path`` is replaced; if ``save`` raises, ``path`` is
    left unchanged. A split that cannot be materialised raises.

    Args:
        obj: The object to save.
        path: Destination file; parent directories are created. A suffix other than ``.zip`` gives
            a warning.
        data: Which data to ship: ``"all"``, ``"auto"`` or ``"none"`` (see the module docs).
            With ``"none"``, splits without a ``data_source`` give a warning, since the recipient
            cannot load them.
        bundles: ``.srtk`` files built with [pack][SRToolkit.bundle.pack] for custom classes that
            are not installed as bundles. A file is matched to the classes it defines by class
            name and takes precedence over an installed bundle defining the same class; one that
            matches no class gives a warning.
        strict: Raise instead of warning when a custom class is not covered by a bundle or is
            nested in the config of a class from another bundle.
        results: Also ship a grid's finished runs.

    Returns:
        ``path`` as a ``Path``.

    Raises:
        TypeError: If ``obj`` is not an ``Dataset``, ``Benchmark`` or ``ExperimentGrid``.
        ValueError: If ``data`` is not one of the three values; ``results`` is set for an object
            that is not a grid; a dataset has no ``benchmark`` or ``version``; two datasets with
            the same benchmark, version and name have different data; ``strict`` is set and a
            custom class is not covered or is nested in another bundle's config; a file in ``bundles`` is missing or is not a ``.srtk``
            bundle; two files in ``bundles`` are the same bundle name and version; or two files in
            ``bundles`` define a class (by name) that the object uses.
        FileNotFoundError: If a shipped split has no ``data_source`` and is missing from the cache.
    """
    from SRToolkit import __version__
    from SRToolkit.dataset import Benchmark, Dataset
    from SRToolkit.experiments import ExperimentGrid

    if isinstance(obj, Dataset):
        kind = "dataset"
    elif isinstance(obj, Benchmark):
        kind = "benchmark"
    elif isinstance(obj, ExperimentGrid):
        kind = "grid"
    else:
        raise TypeError(
            f"[share.save] Cannot save a {type(obj).__name__}; expected a Dataset, Benchmark or ExperimentGrid."
        )
    if data not in ("all", "auto", "none"):
        raise ValueError(f"[share.save] data must be 'all', 'auto' or 'none', got {data!r}.")
    if results and kind != "grid":
        raise ValueError(f"[share.save] results=True needs an ExperimentGrid, got a {kind}.")
    path = Path(path)
    if path.suffix.lower() != ".zip":
        warnings.warn(f"[share.save] {str(path)!r} does not end in '.zip'; a zip archive is written.", stacklevel=2)

    supplied = []  # (path, manifest, class names) of each file in ``bundles``
    for bundle_path in dict.fromkeys(map(Path, bundles)):
        try:
            supplied.append((bundle_path, read_manifest(bundle_path), bundle_class_index(bundle_path)))
        except Exception as exc:
            raise ValueError(f"[share.save] {str(bundle_path)!r} is not a .srtk bundle ({exc}).") from exc
        same = [
            str(b[0]) for b in supplied if (b[1].name, b[1].version) == (supplied[-1][1].name, supplied[-1][1].version)
        ]
        if len(same) > 1:
            raise ValueError(
                f"[share.save] These files in `bundles` are all bundle {supplied[-1][1].name!r} "
                f"{supplied[-1][1].version}: {', '.join(same)}."
            )

    recipe = obj.to_dict()  # raises for a dataset without benchmark / version
    if isinstance(obj, Dataset):
        pairs: list[tuple[dict, Dataset | None]] = [(recipe, obj)]
    elif isinstance(obj, Benchmark):
        pairs = [(recipe["datasets"][name], entry.get("sr_dataset")) for name, entry in obj.datasets.items()]
    else:
        pairs = [(cfg, None) for cfg in recipe["datasets"].values()]

    # Write into a temporary file next to ``path``, moved into place at the end, so ``path`` is left
    # unchanged if anything below raises. Data is written one split at a time. Created with os.open so
    # the file gets the usual permissions (the umask applies), unlike mkstemp's owner-only ones.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = path.with_name(f".{path.name}.{secrets.token_hex(8)}.part")
    fd = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
    try:
        with os.fdopen(fd, "wb") as fh, zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as zf:
            # Data: decide each split. Datasets in memory give their arrays directly; configs read
            # them through the cache (materialising if needed).
            seen: dict[tuple[str, str, str], tuple[str, str | None]] = {}  # split -> (data hash, arrays hash)
            manifest_data, manifest_referenced, unreachable = [], [], []
            for cfg, instance in pairs:
                benchmark, version = cfg["benchmark"], cfg["version"]
                in_memory = dict(instance._splits()) if instance is not None else {}
                for key, split_source in _split_sources(cfg):
                    split_cfg = {**cfg, "data_source": split_source}
                    arrays = None
                    if data == "all" or (data == "auto" and not _reproducible(split_source)):
                        if key in in_memory:
                            arrays = in_memory[key].arrays
                        else:
                            with np.load(data_cache.resolve(benchmark, version, key, split_cfg)) as npz:
                                arrays = {k: npz[k] for k in npz.files}
                    arrays_hash = None
                    if arrays is not None:
                        digest = hashlib.sha256()
                        for array_name in sorted(arrays):
                            a = np.asarray(arrays[array_name])
                            digest.update(f"{array_name}{a.dtype}{a.shape}".encode())
                            digest.update(np.ascontiguousarray(a).tobytes())
                        arrays_hash = digest.hexdigest()
                    signature = (data_cache._drift_hash(split_cfg), arrays_hash)
                    label = f"{benchmark}/{version}/{key}"
                    if (benchmark, version, key) in seen:
                        if seen[benchmark, version, key] != signature:
                            raise ValueError(f"[share.save] Two datasets save different data as {label!r}.")
                        continue
                    seen[benchmark, version, key] = signature
                    entry = {"benchmark": benchmark, "version": version, "key": key}
                    if arrays is not None:
                        rel = data_cache.dataset_path(benchmark, version, key).relative_to(data_cache.data_root())
                        buf = io.BytesIO()
                        np.savez(buf, **arrays)  # type: ignore[arg-type]
                        zf.writestr(f"data/{rel.as_posix()}", buf.getvalue())
                        manifest_data.append(entry)
                    else:
                        manifest_referenced.append(entry)
                        if split_source is None:
                            unreachable.append(label)
            if unreachable:
                warnings.warn(
                    "[share.save] These splits have no data_source and are not shipped (data='none'), so the "
                    "archive cannot load them:\n  - " + "\n  - ".join(unreachable),
                    stacklevel=2,
                )

            # Code: match each custom class to a supplied bundle (by class name), an installed bundle
            # it is imported from, or the installed bundle its config names in ``_bundle`` (a loaded
            # benchmark or grid keeps the original class paths in its dataset configs).
            installed = list_installed()
            used: set[Path] = set()
            bundle_files: dict[str, dict[str, Any]] = {}  # archive member -> manifest entry, with "source"
            class_to_bundle: dict[str, tuple[str, str]] = {}
            uncovered: list[str] = []
            class_notes: dict[str, tuple[str, str | None] | None] = {}  # class -> innermost (_bundle, _version)
            for class_path, notes in _class_refs(recipe):
                if not class_path.startswith("SRToolkit.") and class_notes.get(class_path) is None:
                    class_notes[class_path] = notes[-1] if notes else None
            for class_path, note in sorted(class_notes.items()):
                class_name = class_path.rpartition(".")[2]
                matches = [b for b in supplied if class_name in b[2]]
                if len(matches) > 1:
                    raise ValueError(
                        f"[share.save] Class {class_name} ({class_path}) is defined by several bundles: "
                        + ", ".join(str(b[0]) for b in matches)
                        + "."
                    )
                bundle_source: Path | None  # the supplied .srtk, or None to re-pack the installed bundle
                if matches:
                    bundle_path, bundle_manifest, _ = matches[0]
                    used.add(bundle_path)
                    bundle_name, bundle_version, bundle_source = (
                        bundle_manifest.name,
                        bundle_manifest.version,
                        bundle_path,
                    )
                else:
                    installed_entry: dict[str, Any] | None = next(
                        (e for e in installed if class_path.startswith(e["import_prefix"] + ".")), None
                    )
                    if installed_entry is None and note is not None:
                        try:  # the bundle its config names; without a _version, the latest (as bind_config)
                            installed_entry = _store.lookup(*note)
                        except KeyError:
                            pass
                    if installed_entry is None:
                        uncovered.append(class_path)
                        continue
                    bundle_name, bundle_version, bundle_source = (
                        installed_entry["name"],
                        installed_entry["version"],
                        None,
                    )
                member = f"bundles/{bundle_name}_{bundle_version}.srtk"
                bundle_files.setdefault(
                    member,
                    {
                        "file": member,
                        "name": bundle_name,
                        "version": bundle_version,
                        "classes": [],
                        "source": bundle_source,
                    },
                )
                bundle_files[member]["classes"].append(class_path)
                class_to_bundle[class_path] = (bundle_name, bundle_version)
            if uncovered:
                message = (
                    "[share.save] These custom classes are not covered by an installed bundle or a file in "
                    "`bundles`, so the recipient cannot load them:\n  - "
                    + "\n  - ".join(uncovered)
                    + "\nPack their files with SRToolkit.bundle.pack and pass the .srtk in `bundles`."
                )
                if strict:
                    raise ValueError(message)
                warnings.warn(message, stacklevel=2)
            # bind_config on an annotated dict relocates every class inside it, so a class nested in a
            # dict annotated with another bundle (or uncovered) cannot load on the other end.
            annotated = _annotate_bundles(recipe, class_to_bundle)
            conflicts = sorted(
                {
                    class_path
                    for class_path, notes in _class_refs(annotated)
                    if not class_path.startswith("SRToolkit.")
                    and any(n != class_to_bundle.get(class_path) for n in notes)
                }
            )
            if conflicts:
                message = (
                    "[share.save] These custom classes sit inside a config whose class comes from another bundle, "
                    "so they cannot load on the other end:\n  - "
                    + "\n  - ".join(conflicts)
                    + "\nPut the classes one config uses into one bundle."
                )
                if strict:
                    raise ValueError(message)
                warnings.warn(message, stacklevel=2)
            unused = [str(b[0]) for b in supplied if b[0] not in used]
            if unused:
                warnings.warn(
                    "[share.save] These bundles define no class the object uses and are not shipped:\n  - "
                    + "\n  - ".join(unused),
                    stacklevel=2,
                )
            with tempfile.TemporaryDirectory() as tmp:
                for member, b in bundle_files.items():
                    if b["source"] is not None:
                        zf.write(b["source"], member)
                    else:
                        zf.write(_repack(b["name"], b["version"], Path(tmp) / Path(member).name), member)

            if results and isinstance(obj, ExperimentGrid):
                for dataset_name in obj.datasets:
                    for approach in obj.approach_configs:
                        run_dir = Path(obj.results_dir) / dataset_name / approach["name"]
                        for file in sorted(run_dir.glob("exp_*.json")):
                            zf.write(file, f"results/{dataset_name}/{approach['name']}/{file.name}")

            manifest = {
                "format_version": _FORMAT_VERSION,
                "kind": kind,
                "srtk_version": __version__,
                "data_policy": data,
                "data": manifest_data,
                "referenced": manifest_referenced,
                "bundles": [{k: v for k, v in b.items() if k != "source"} for b in bundle_files.values()],
                "uncovered": uncovered,
                "results": results,
            }
            readme = [
                f"# SRToolkit {kind}",
                "",
                f"Written with SRToolkit {__version__}. Load it with:",
                "",
                "```python",
                "from SRToolkit import share",
                f'obj = share.load("<this file>"{", install_bundles=True" if bundle_files else ""})',
                "```",
                "",
                f"Shipped data: {len(manifest_data)} file(s) under `data/`. "
                f"Referenced data (downloaded or regenerated on use): {len(manifest_referenced)} file(s).",
            ]
            if bundle_files:
                readme += ["", "Custom code, installed by `install_bundles=True`:", ""]
                readme += [f"- `{b['file']}`: {', '.join(b['classes'])}" for b in bundle_files.values()]
            if uncovered:
                readme += ["", "Custom classes not included (get their code from the sender):", ""]
                readme += [f"- `{c}`" for c in uncovered]
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("config.json", json.dumps(annotated, indent=2))
            zf.writestr("README.md", "\n".join(readme) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def load(
    source: str | Path,
    *,
    install_bundles: bool = False,
    results_dir: str | Path | None = None,
) -> Dataset | Benchmark | ExperimentGrid:
    """Load an object from an archive written by [save][SRToolkit.share.save].

    Shipped data is copied into the [data cache][SRToolkit.dataset.data_cache], replacing a cached
    file with different arrays (with a warning), and is used as shipped (no download, regeneration
    or stale-cache warning). Referenced splits are reached through their
    ``data_source``, as after ``from_dict``. Benchmarks and grids stay lazy: their datasets are
    built when used.

    A bundle counts as installed when a bundle with its name and version is. Bundles that are not
    installed give one warning naming them and the call that installs them
    (``share.load(source, install_bundles=True)``). Classes the archive lists as uncovered (no
    bundle shipped them) give a warning too. A custom class from such a bundle, or an uncovered
    one, raises when it is needed: on load for a dataset archive, and when the dataset or approach
    is built for a benchmark or grid.

    Args:
        source: Path of the archive, or an ``http://`` / ``https://`` URL to download it from.
        install_bundles: Install the archive's bundles that are not installed yet, with
            [install][SRToolkit.bundle.install], which asks for confirmation. A bundle the user
            declines stays uninstalled and gives the warning above.
        results_dir: Grids only: where the grid reads and writes results, and where shipped results
            are copied; existing result files are kept, with a warning. Defaults to the archive's
            file name without its suffix (``run.zip`` → ``run/``), next to the archive for a path
            and in the current directory for a URL; required when that name has no suffix.

    Returns:
        The saved object: an ``Dataset``, ``Benchmark`` or ``ExperimentGrid``.

    Raises:
        FileNotFoundError: If ``source`` is a path that does not exist.
        urllib.error.URLError: If ``source`` is a URL that cannot be downloaded.
        TimeoutError: If the server of a URL sends nothing for 60 seconds.
        ValueError: If ``source`` ends in ``.srtk`` (install it with
            [install][SRToolkit.bundle.install]); is not a zip file or has no ``manifest.json``;
            has a ``format_version`` this SRToolkit does not read or an unknown ``kind``; has a member whose path leads outside
            its destination (checked before anything is copied); or if ``results_dir`` is given
            for an object that is not a grid, or is missing for a grid whose archive name has no
            suffix.
    """
    from SRToolkit.dataset import Benchmark, Dataset
    from SRToolkit.experiments import ExperimentGrid

    is_url = str(source).startswith(("http://", "https://"))
    name = Path(urlparse(str(source)).path).name if is_url else Path(source).name
    if name.lower().endswith(".srtk"):
        raise ValueError(f"[share.load] {str(source)!r} is a code bundle; install it with SRToolkit.bundle.install.")
    with contextlib.ExitStack() as stack:
        if is_url:
            # Download to a temporary file rather than into memory; the timeout keeps a dead server from
            # hanging the call.
            archive = Path(stack.enter_context(tempfile.TemporaryDirectory())) / "archive.zip"
            with urlopen(str(source), timeout=_URL_TIMEOUT) as response, open(archive, "wb") as f:
                shutil.copyfileobj(response, f)
            default_results_dir: Path | None = Path.cwd() / Path(name).stem if Path(name).suffix else None
        else:
            archive = Path(source)
            if not archive.is_file():
                raise FileNotFoundError(f"[share.load] No archive at {str(source)!r}.")
            default_results_dir = archive.with_suffix("") if archive.suffix else None
        if not zipfile.is_zipfile(archive):
            raise ValueError(f"[share.load] {str(source)!r} is not a zip archive.")
        zf = stack.enter_context(zipfile.ZipFile(archive))
        members = zf.namelist()
        if "manifest.json" not in members:
            raise ValueError(f"[share.load] {str(source)!r} has no manifest.json; it was not written by share.save.")
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                f"[share.load] {str(source)!r} has format_version {manifest.get('format_version')!r}; this SRToolkit "
                f"reads format_version {_FORMAT_VERSION} (a newer one needs a newer SRToolkit)."
            )
        kind = manifest.get("kind")
        if kind not in ("dataset", "benchmark", "grid"):
            raise ValueError(
                f"[share.load] {str(source)!r} holds an unknown kind {kind!r}; expected 'dataset', 'benchmark' or 'grid'."
            )
        config = json.loads(zf.read("config.json"))
        if results_dir is not None and kind != "grid":
            raise ValueError(f"[share.load] results_dir is only for grids; this archive holds a {kind}.")
        if kind == "grid" and results_dir is None and default_results_dir is None:
            raise ValueError(
                f"[share.load] {str(source)!r} has no file suffix to name the results directory after; "
                "pass results_dir."
            )
        results_root = Path(results_dir or default_results_dir or Path.cwd()).absolute()

        # Check every member's destination before anything is copied.
        targets: dict[str, Path] = {}
        for member in members:
            prefix, _, rel = member.partition("/")
            root = {"data": data_cache.data_root(), "results": results_root}.get(prefix)
            if root is None or not rel or member.endswith("/"):
                continue
            target = root / rel
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"[share.load] Refusing member {member!r}: it leads outside {str(root)!r}.")
            targets[member] = target

        installed = {(e["name"], e["version"]) for e in list_installed()}
        missing = [b for b in manifest["bundles"] if (b["name"], b["version"]) not in installed]
        if missing and install_bundles:
            with tempfile.TemporaryDirectory() as tmp:
                for b in missing:
                    bundle_path = Path(tmp) / Path(b["file"]).name
                    bundle_path.write_bytes(zf.read(b["file"]))
                    install(bundle_path)
            installed = {(e["name"], e["version"]) for e in list_installed()}
            missing = [b for b in missing if (b["name"], b["version"]) not in installed]
        if missing:
            warnings.warn(
                "[share.load] The archive's code bundles are not installed, so their classes cannot be used:\n  - "
                + "\n  - ".join(f"{b['name']} {b['version']}: {', '.join(b['classes'])}" for b in missing)
                + f"\nInstall them with share.load({str(source)!r}, install_bundles=True).",
                stacklevel=2,
            )
        if manifest["uncovered"]:
            warnings.warn(
                "[share.load] The archive does not include the code of these classes; get it from the sender:\n  - "
                + "\n  - ".join(manifest["uncovered"]),
                stacklevel=2,
            )

        # Data: copy into the cache with a fresh drift sidecar, so the cache matches the config.
        split_configs = {}
        for cfg in [config] if kind == "dataset" else config["datasets"].values():
            for key, split_source in _split_sources(cfg):
                split_configs[cfg["benchmark"], cfg["version"], key] = {**cfg, "data_source": split_source}
        replaced = []
        for entry in manifest["data"]:
            benchmark, version, key = entry["benchmark"], entry["version"], entry["key"]
            target = data_cache.dataset_path(benchmark, version, key)
            content = zf.read(f"data/{target.relative_to(data_cache.data_root()).as_posix()}")
            if target.exists():
                with np.load(target) as old, np.load(io.BytesIO(content)) as new:
                    same = sorted(old.files) == sorted(new.files) and all(
                        np.array_equal(old[k], new[k], equal_nan=new[k].dtype.kind in "fc") for k in new.files
                    )
                if not same:
                    replaced.append(f"{benchmark}/{version}/{key}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            split_cfg = split_configs[benchmark, version, key]
            if split_cfg["data_source"] is not None:
                data_cache._write_meta(target, split_cfg)
            else:
                data_cache._meta_path(target).unlink(missing_ok=True)
        if replaced:
            warnings.warn(
                "[share.load] Replaced cached data that differed from the archive's:\n  - " + "\n  - ".join(replaced),
                stacklevel=2,
            )

        if kind == "dataset":
            return Dataset.from_dict(config)
        if kind == "benchmark":
            return Benchmark.from_dict(config)
        grid = ExperimentGrid.from_dict(config, results_dir=str(results_root))
        kept = []
        for member, target in targets.items():
            if not member.startswith("results/"):
                continue
            if target.exists():
                kept.append(str(target.relative_to(results_root)))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member))
        if kept:
            warnings.warn(
                "[share.load] Kept these existing result files instead of the archive's:\n  - " + "\n  - ".join(kept),
                stacklevel=2,
            )
        return grid


def _split_sources(cfg: dict) -> list[tuple[str, dict | None]]:
    """Return ``(cache key, data_source dict or None)`` of each split of a dataset config: train, test, aux."""
    name = cfg["dataset_name"]
    splits = [(name, cfg.get("data_source"))]
    if cfg.get("test"):
        splits.append((f"{name}/test", cfg["test"].get("data_source")))
    for key, entry in (cfg.get("aux") or {}).items():
        splits.append((f"{name}/aux/{key}", entry.get("data_source")))
    return splits


def _reproducible(source: dict | None) -> bool:
    """Whether a split with this ``data_source`` dict can be reproduced from the config alone."""
    if source is None:
        return False
    source_class = source.get("source_class")
    if source_class == _SOURCES + "SampleSource":
        return source.get("seed") is not None
    if source_class == _SOURCES + "FallbackSource":
        return any(_reproducible(s) for s in source["sources"])
    return source_class == _SOURCES + "UrlSource"


def _annotate_bundles(obj: Any, class_to_bundle: dict[str, tuple[str, str]]) -> Any:
    """Copy ``obj``, replacing ``_bundle`` / ``_version`` with those of the bundles shipped for its ``*_class``es."""
    if isinstance(obj, list):
        return [_annotate_bundles(v, class_to_bundle) for v in obj]
    if not isinstance(obj, dict):
        return obj
    annotated = {k: _annotate_bundles(v, class_to_bundle) for k, v in obj.items() if k not in ("_bundle", "_version")}
    for k, v in obj.items():
        if k.endswith("_class") and v in class_to_bundle:
            annotated["_bundle"], annotated["_version"] = class_to_bundle[v]
    return annotated
