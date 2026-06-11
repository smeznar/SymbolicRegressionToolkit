"""
Utilities for rewriting ``*_class`` dotted paths in serialized configs.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _scan_classes(base_dir_path: Path) -> Dict[str, List[str]]:
    """
    Scan ``base_dir_path`` and return a mapping ``class_name -> [module_stem, ...]``.

    Bundles use a flat ``src/`` layout, so only the top-level directory is read
    and each module is identified by its file stem. Uses ``ast`` to find
    top-level class definitions without importing the files.
    """
    index: Dict[str, List[str]] = {}
    root = Path(base_dir_path).resolve()
    for file_path in root.iterdir():
        if file_path.suffix != ".py" or file_path.name == "__init__.py":
            continue
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        module_stem = file_path.stem
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                index.setdefault(node.name, []).append(module_stem)
    return index


def _rewrite_one(class_path: str, base_dir: str, index: Dict[str, List[str]]) -> str:
    if class_path.startswith("SRToolkit."):
        return class_path
    if "." not in class_path:
        raise ValueError(f"Class path {class_path!r} is not fully qualified.")
    original_module, cls_name = class_path.rsplit(".", 1)
    candidates = index.get(cls_name)
    if not candidates:
        raise LookupError(f"Class {cls_name!r} (from {class_path!r}) not found under bundle directory.")
    original_stem = original_module.rsplit(".", 1)[-1]
    chosen = next((c for c in candidates if c == original_stem), candidates[0])
    return f"{base_dir}.{chosen}.{cls_name}"


def _walk(obj: Any, base_dir: str, index: Dict[str, List[str]]) -> Any:
    if isinstance(obj, dict):
        return {
            k: (
                _rewrite_one(v, base_dir, index)
                if isinstance(v, str) and k.endswith("_class")
                else _walk(v, base_dir, index)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_walk(v, base_dir, index) for v in obj]
    return obj


def _relocate_class_paths(
    config: dict,
    base_dir: str,
    base_dir_path: str | os.PathLike,
) -> dict:
    """
    Rewrite every ``*_class`` dotted path in ``config`` to live under ``base_dir``.

    Low-level helper used by [bind_config][SRToolkit.bundle.bind_config]. Walks
    ``config`` recursively and, for every dict value whose key ends in ``_class``,
    replaces the module portion with ``base_dir`` plus the stem of the file
    (under ``base_dir_path``) that defines a matching top-level class.

    Resolution rules:

    - If exactly one file under ``base_dir_path`` defines a class with the given
      name, that file's stem is used.
    - If multiple files match, the one whose stem equals the last segment of the
      original module path is preferred; otherwise the first match wins.
    - Paths already starting with ``SRToolkit.`` are treated as built-in and left
      unchanged.

    Discovery is done via ``ast`` parsing — no user code is imported or executed.

    Args:
        config: JSON-style configuration dictionary, potentially nested.
        base_dir: Importable dotted prefix the rewritten paths should sit under
            (e.g. ``"srtk_bundles.alice_bed_approach_0_2_1"``).
        base_dir_path: On-disk location corresponding to ``base_dir``.

    Returns:
        A new dictionary with rewritten ``*_class`` references. ``config`` is not
        modified.

    Raises:
        LookupError: If a referenced class name cannot be found under
            ``base_dir_path``.
        ValueError: If a ``*_class`` value is not a fully-qualified dotted path.
    """
    index = _scan_classes(Path(base_dir_path))
    return _walk(config, base_dir, index)


def _auto_bind(config: dict) -> dict:
    """Return bind_config(config) if ``_bundle`` metadata is present, else config unchanged."""
    if "_bundle" not in config:
        return config
    return bind_config(config)


def bind_config(
    config: dict,
    bundle_name: Optional[str] = None,
    version: Optional[str] = None,
) -> dict:
    """
    Rewrite ``*_class`` paths in ``config`` to point at an installed bundle.

    Looks up ``bundle_name`` in the local bundle index and rewrites every
    ``*_class`` value so that it resolves to the installed copy's import prefix.
    Paths already starting with ``SRToolkit.`` are left unchanged.

    If ``bundle_name`` is not provided, the ``_bundle`` and ``_version`` keys
    embedded in the config by [pack][SRToolkit.bundle.pack] are used as
    fallbacks. This allows calling ``bind_config(config)`` directly when the
    config was annotated at pack time.

    This is the recipient-side complement to [pack][SRToolkit.bundle.pack] for configs:
    the author shares the ``.srtk`` file (code) and an annotated ``.srtk.json``
    config; the recipient installs the bundle once and then calls ``bind_config``
    to make the config usable on their machine.

    Args:
        config: JSON-style configuration dictionary, optionally containing
            ``_bundle`` and ``_version`` metadata added by ``pack``.
        bundle_name: Name of an installed bundle. If ``None``, ``config["_bundle"]``
            is used.
        version: Version to bind against. If ``None``, ``config["_version"]`` is
            tried first, then the latest installed version.

    Returns:
        A new dictionary with rewritten ``*_class`` references ready for use
        with ``from_dict`` dispatchers. ``config`` is not modified and ``_bundle``/
        ``_version`` metadata keys are stripped from the result.

    Raises:
        ValueError: If ``bundle_name`` is not provided and ``config`` has no
            ``_bundle`` key.
        KeyError: If the bundle (or requested version) is not installed.
        LookupError: If a ``*_class`` value references a class that cannot be
            found in the installed bundle.
    """
    from . import _store

    resolved_name = bundle_name or config.get("_bundle")
    if not resolved_name:
        raise ValueError(
            "bundle_name must be provided or the config must contain a '_bundle' key "
            "(added automatically by pack(..., configs=[...]))."
        )
    resolved_version = version or config.get("_version")

    entry = _store.lookup(resolved_name, resolved_version)
    _store.enable_bundle_imports()

    stripped = {k: v for k, v in config.items() if k not in ("_bundle", "_version")}
    return _relocate_class_paths(stripped, entry["import_prefix"], entry["path"])
