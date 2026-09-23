"""
Soft-import helpers for optional third-party dependencies.

Some `analogy` functions were written against ``pims`` (image sequence
loading), ``trackpy`` (particle linking -- referenced implicitly via the
``trajs``/``tp_df`` DataFrames these functions consume), and ``pycpt``
(loading GMT/CPT colormaps for plotting). None of these are required just
to *import* analogy -- only to call the handful of functions that use them.
"""

import importlib


def optional_import(name, pip_name=None):
    """
    Import [name], raising a friendly ImportError (only when actually
    needed) if it isn't installed, instead of failing on `import analogy`.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(
            f"This function requires the optional dependency '{name}'. "
            f"Install it with: pip install {pip_name or name}"
        ) from exc


def get_pims():
    return optional_import("pims")


def get_pycpt():
    return optional_import("pycpt")


def get_trackpy():
    return optional_import("trackpy")
