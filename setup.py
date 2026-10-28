"""
Build script for the optional Cython extension.

Run ``pip install -e .`` or ``python setup.py build_ext --inplace`` to compile
the Cython stack machine.  When Cython or a C compiler is not available the
package installs without the extension and falls back to the pure-Python
implementation in ``SRToolkit/utils/_eval_cython.py``; a warning is printed
during the build and again when the evaluator is first used.

Set ``SRTOOLKIT_REQUIRE_CYTHON=1`` to make a failed build an error instead
(used in CI and when building wheels, so a wheel never ships without the
extension by accident).
"""

import os

from setuptools import Extension, setup

REQUIRE_CYTHON = os.environ.get("SRTOOLKIT_REQUIRE_CYTHON", "") not in ("", "0")

try:
    import numpy as np
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [
            Extension(
                name="SRToolkit.utils._eval_cython",
                sources=["SRToolkit/utils/_eval_cython.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=["-O3", "-ffast-math", "-fno-finite-math-only"],
            )
        ],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
    # A failed compile (e.g. no C compiler) is then a warning, not an error. Set after
    # cythonize, which drops ``optional`` when it rebuilds the Extension objects.
    for ext in ext_modules:
        ext.optional = not REQUIRE_CYTHON
except Exception as e:
    if REQUIRE_CYTHON:
        raise
    print(
        f"WARNING: SRToolkit is installed without the Cython extension ({e!r}); using the slower pure-Python evaluator."
    )
    ext_modules = []

setup(ext_modules=ext_modules)
