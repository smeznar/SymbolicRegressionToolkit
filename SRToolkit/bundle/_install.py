"""
Bundle install, uninstall, list operations.
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

from packaging.requirements import Requirement
from packaging.version import Version

from . import _store
from ._manifest import BundleManifest


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def _check_srtk_version(required: str) -> None:
    if not required:
        return
    from SRToolkit import __version__

    if Version(__version__) < Version(required):
        raise RuntimeError(f"Bundle requires SRToolkit>={required} but {__version__} is installed.")


def _check_deps(python_deps: List[str]) -> List[str]:
    """Return the subset of ``python_deps`` that cannot be imported."""
    missing = []
    for spec in python_deps:
        pkg_name = Requirement(spec).name
        try:
            importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(spec)
    return missing


def read_manifest(srtk_path: Path) -> BundleManifest:
    """
    Read a ``.srtk`` bundle's manifest without installing it.

    Reads only the ``manifest.json`` entry from the archive — the bundle's source files
    are not extracted, no checksums are verified, and the global bundle store is not
    touched. Use this to inspect a not-yet-installed bundle's ``name`` / ``version`` /
    ``import_prefix`` (e.g. to match referenced class paths against a bundle a user
    supplied to [ExperimentGrid.export][SRToolkit.experiments.ExperimentGrid.export]).

    Args:
        srtk_path: Path to the ``.srtk`` archive.

    Returns:
        The bundle's [BundleManifest][SRToolkit.bundle._manifest.BundleManifest].

    Raises:
        FileNotFoundError: If ``srtk_path`` does not exist.
        ValueError: If the archive is not a valid bundle (no ``manifest.json``).
    """
    srtk_path = Path(srtk_path)
    if not srtk_path.is_file():
        raise FileNotFoundError(f"No bundle archive at {str(srtk_path)!r}.")
    try:
        with zipfile.ZipFile(srtk_path) as zf:
            raw = zf.read("manifest.json")
    except KeyError as exc:
        raise ValueError(f"{str(srtk_path)!r} is not a valid .srtk bundle (no manifest.json).") from exc
    return BundleManifest.from_dict(json.loads(raw.decode("utf-8")))


def _extract_and_verify(srtk_path: Path, tmp_dir: Path) -> BundleManifest:
    with zipfile.ZipFile(srtk_path) as zf:
        zf.extractall(tmp_dir)
    manifest = BundleManifest.from_dict(json.loads((tmp_dir / "manifest.json").read_text(encoding="utf-8")))
    manifest.verify(tmp_dir)
    _check_srtk_version(manifest.srtk_min_version)
    return manifest


def install(srtk_path: Path) -> None:
    """
    Install a ``.srtk`` bundle onto this machine.

    Steps:

    1. Unzip to a temporary directory and load the manifest.
    2. Verify per-file checksums declared in the manifest.
    3. Check the required ``SRToolkit`` version.
    4. Check Python dependencies — list missing ones and suggest a ``pip install``
       command; the user decides whether to continue.
    5. Prompt the user to confirm that arbitrary user code will be executable.
    6. Copy ``src/`` to the managed bundle directory.
    7. Register the bundle in the local index.

    Args:
        srtk_path: Path to a ``.srtk`` bundle file produced by
            [pack][SRToolkit.bundle.pack].

    Raises:
        ValueError: On checksum mismatch.
        RuntimeError: If the installed SRToolkit is too old.
    """
    srtk_path = Path(srtk_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        manifest = _extract_and_verify(srtk_path, tmp_dir)

        missing_deps = _check_deps(manifest.python_deps)
        if missing_deps:
            print(f"Bundle '{manifest.name}' declares missing dependencies:")
            for dep in missing_deps:
                print(f"  {dep}")
            print(f"\nInstall them with:\n  pip install {' '.join(missing_deps)}\n")
            if not _confirm("Continue installation without them?"):
                print("Installation cancelled.")
                return

        print(
            f"\nBundle '{manifest.name}' v{manifest.version}"
            + (f" by {manifest.author!r}" if manifest.author else "")
            + " contains user-defined code that will run when loaded."
        )
        if not _confirm("Install?"):
            print("Installation cancelled.")
            return

        install_path = _store.bundle_path(manifest.name, manifest.version)
        if install_path.exists():
            shutil.rmtree(install_path)
        install_path.mkdir(parents=True)

        src_tmp = tmp_dir / "src"
        if src_tmp.exists():
            shutil.copytree(src_tmp, install_path, dirs_exist_ok=True)

        _store.register(manifest, install_path)
        _store.enable_bundle_imports()
        print(f"Installed '{manifest.name}' v{manifest.version} → {install_path}")


def uninstall(name: str, version: Optional[str] = None) -> None:
    """
    Remove an installed bundle from this machine.

    Args:
        name: Bundle name.
        version: Version to remove. If ``None``, removes the latest installed version.
    """
    entry = _store.lookup(name, version)
    install_path = Path(entry["path"])
    if install_path.exists():
        shutil.rmtree(install_path)
    _store.deregister(entry["name"], entry["version"])
    print(f"Uninstalled '{entry['name']}' v{entry['version']}")


def list_installed() -> list:
    """
    Return a list of all installed bundle index entries.

    Each entry is a dict with keys ``name``, ``version``, ``author``,
    ``srtk_min_version``, ``python_deps``, ``path``, and ``import_prefix``.
    """
    return _store.all_entries()


def _repack(name: str, version: Optional[str], out_path: Path) -> Path:
    """
    Rebuild a ``.srtk`` archive from an already-installed bundle.

    The original ``.srtk`` is not retained at install time, so re-shipping a bundle's code
    (e.g. from [ExperimentGrid.export][SRToolkit.experiments.ExperimentGrid.export]) means
    reconstructing it from the installed source files plus the index metadata. Keeping the
    install-layout details (which files to include, which manifest fields to carry) here
    means callers don't reach into bundle internals.

    Args:
        name: Installed bundle name.
        version: Version to re-pack. ``None`` selects the latest installed.
        out_path: Destination ``.srtk`` path (parent directories are created).

    Returns:
        ``out_path``.

    Raises:
        KeyError: If the bundle (or requested version) is not installed.
        ValueError: If the installed bundle directory has no source files.
    """
    from ._pack import pack

    entry = _store.lookup(name, version)
    src_dir = Path(entry["path"])
    py_files = [p for p in src_dir.glob("*.py") if p.name != "__init__.py"]
    if not py_files:
        raise ValueError(f"Installed bundle {name!r} has no source files at {src_dir}.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pack(
        files=py_files,
        out_path=out_path,
        name=entry["name"],
        version=entry["version"],
        author=entry.get("author", ""),
        python_deps=entry.get("python_deps", []),
        srtk_min_version=entry.get("srtk_min_version", ""),
    )
    return out_path
