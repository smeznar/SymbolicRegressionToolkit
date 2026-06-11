"""
Bundle sharing utilities for the Symbolic Regression Toolkit.

Provides tools for packing, installing, and using bundles of user-defined Python
code as ``.srtk`` archives. Bundles contain **code only** — configs (the settings
a user chose) are intentionally separate so they can be shared, versioned, and
tweaked independently of the implementation.

## Workflow

**Author side** — pack the code:

```python
from SRToolkit.bundle import pack

pack(
    files=["my_approach/approach.py", "my_approach/ops.py"],
    out_path="meznar_gp.srtk",
    name="meznar-gp",
    version="0.1.0",
    author="meznar",
    python_deps=["torch>=2.0"],
)
```

Share the ``.srtk`` file and a plain JSON config separately.

**Recipient side** — install and use:

```python
import json
from SRToolkit.bundle import install

install("meznar_gp.srtk")   # one-time, interactive

raw = json.load(open("meznar_settings.srtk.json"))  # config shared separately
# The config is annotated with _bundle/_version by pack(..., configs=[...]),
# so it can be passed straight to any project `from_dict` consumer
# (SR_dataset, Sampler, the experiment grid, ...) — they call bind_config
# internally to repoint every *_class path at the installed bundle.
benchmark = SR_benchmark.from_dict(raw)
```

If the config was *not* annotated by ``pack`` (no ``_bundle`` key), bind it
explicitly first, passing the bundle name:

```python
from SRToolkit.bundle import bind_config

config = bind_config(raw, "meznar-gp")
```

**Listing and removing**:

```python
from SRToolkit.bundle import list_installed, uninstall

for entry in list_installed():
    print(entry["name"], entry["version"])

uninstall("meznar-gp", version="0.1.0")
```
"""

from ._install import install, list_installed, uninstall
from ._pack import pack
from ._relocate import bind_config
from ._store import enable_bundle_imports

__all__ = [
    "pack",
    "install",
    "uninstall",
    "list_installed",
    "bind_config",
    "enable_bundle_imports",
]
