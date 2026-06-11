"""
Bundle packing: zip a list of Python source files into a ``.srtk`` archive.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Union

from ._manifest import BundleManifest, _sha256


def pack(
    files: Sequence[Union[str, Path]],
    out_path: Path,
    name: str,
    version: str,
    author: str = "",
    python_deps: List[str] | None = None,
    srtk_min_version: str = "",
    configs: Optional[Sequence[Union[str, Path]]] = None,
) -> None:
    """
    Pack a list of Python source files into a ``.srtk`` bundle.

    Each file is stored flat under ``src/`` using its basename, so all filenames
    must be unique. Configs are intentionally excluded from the archive — share
    them separately as plain JSON. Class paths are rewritten at use-time by
    calling [bind_config][SRToolkit.bundle.bind_config].

    If ``configs`` is provided, each JSON config file is annotated with
    ``_bundle`` and ``_version`` metadata and written next to the original with
    a ``.srtk.json`` suffix (e.g. ``settings.json`` → ``settings.srtk.json``).
    These annotated copies are the files to share alongside the ``.srtk`` bundle;
    the originals are left untouched.

    The archive layout is::

        manifest.json
        src/
            <basename of each file>

    Args:
        files: Paths to the ``.py`` files to include. Basenames must be unique.
        out_path: Destination path for the ``.srtk`` file (created or overwritten).
        name: Bundle name (alphanumeric, hyphens allowed).
        version: Semantic version string, e.g. ``"0.2.1"``.
        author: Optional author identifier.
        python_deps: List of PEP 508 dependency specifiers, e.g. ``["torch>=2.0"]``.
        srtk_min_version: Minimum ``SRToolkit`` version required to run this bundle.
        configs: Optional list of JSON config file paths to annotate. Each is
            written to ``<stem>.srtk.json`` in the same directory.

    Raises:
        FileNotFoundError: If any source or config file does not exist.
        ValueError: If a source file is not ``.py``, two source files share the
            same basename, or a config already contains ``_bundle`` or
            ``_version`` keys (indicates an already-annotated config).
    """
    out_path = Path(out_path)
    python_deps = python_deps or []

    resolved: list[tuple[Path, str]] = []
    seen_basenames: dict[str, Path] = {}
    for raw in files:
        p = Path(raw).resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix != ".py":
            raise ValueError(f"Only .py files are supported: {p}")
        if p.name in seen_basenames:
            raise ValueError(f"Duplicate basename {p.name!r}: {seen_basenames[p.name]} and {p}")
        seen_basenames[p.name] = p
        resolved.append((p, f"src/{p.name}"))

    archive_files: dict[str, str] = {}
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path, archive_name in resolved:
            zf.write(file_path, archive_name)
            archive_files[archive_name] = _sha256(file_path)

        manifest = BundleManifest(
            name=name,
            version=version,
            author=author,
            srtk_min_version=srtk_min_version,
            python_deps=python_deps,
            files=archive_files,
        )
        zf.writestr("manifest.json", json.dumps(manifest.to_dict(), indent=2))

    if configs:
        for cfg_raw in configs:
            cfg_path = Path(cfg_raw).resolve()
            if not cfg_path.exists():
                raise FileNotFoundError(f"Config file not found: {cfg_path}")
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if "_bundle" in cfg or "_version" in cfg:
                raise ValueError(
                    f"{cfg_path.name} already contains '_bundle' or '_version' — "
                    "pass the original config, not an already-annotated one."
                )
            annotated = {"_bundle": name, "_version": version, **cfg}
            out = cfg_path.parent / (cfg_path.stem + ".srtk.json")
            out.write_text(json.dumps(annotated, indent=2), encoding="utf-8")
