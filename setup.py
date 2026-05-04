"""
Build script for the optional Cython extension.

Run ``pip install -e .`` or ``python setup.py build_ext --inplace`` to compile
the Cython stack machine.  When Cython or a C compiler is not available the
package installs without the extension and falls back to the pure-Python
implementation in ``SRToolkit/utils/_eval_cython.py``.
"""

from setuptools import Extension, setup

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
except Exception:
    # Cython or C compiler unavailable — install without the extension.
    ext_modules = []

setup(ext_modules=ext_modules)
