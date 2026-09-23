"""
analogy
=======

A toolkit for post-processing analog (sandbox) tectonic model experiments:
particle image velocimetry/particle-tracking velocimetry (PIV/PTV),
surface & deformation front extraction, particle burial/exhumation thermal
history and kinematics, plate velocity & strain, and the histograms and
other plots used to summarize them.

The name is a small pun: this package analyzes *analog* models -- hence
"analogy". It was consolidated from a set of standalone analysis scripts
(temperatures.py, subsurface_slip.py, SPTVcode.py, short_vertical.py,
ratecomp.py, map_view_exhum_collection.py, erosional_thickness_and_time.py)
that had accumulated duplicate copies of the same helper functions. Those
scripts are unified here into one de-duplicated, importable package.

Submodules
----------
utils       General-purpose numeric/array helpers (gridding, filling NaNs,
            nearest-neighbor lookups, curve fitting, colormaps, monotonicity
            checks, time-axis resets, etc.)
io          Reading PIVlab-style output files and frame numbers.
surfaces    Working with model topography: masking data by surface,
            deformation-front detection, structural/topographic divides,
            volume change.
particles   Single/multi-particle tracking: depth-below-surface
            calculation, burial/exhumation statistics, particle
            displacement fields, particle-line extraction.
kinematics  Plate velocity and strain-rate / vorticity calculations.
histograms  Gridded histogram summaries of particle depth, burial,
            exhumation, and displacement fields.
plotting    Trajectory plotting and PIV sub-pixel bias diagnostics.

Optional dependencies
----------------------
A few functions rely on the optional packages ``pims``, ``trackpy``, and
``pycpt``. These are not required to import ``analogy``; if you call a
function that needs one and it isn't installed, you'll get a clear
``ImportError`` at call time (see ``analogy._optdeps``).

Quick start
-----------
>>> import analogy as an
>>> an.gridize(x, y, z)
>>> an.depth_calc(img_h, trajs, frame_spacing, surfname)
"""

from importlib import metadata as _metadata

from . import utils
from . import io
from . import surfaces
from . import particles
from . import kinematics
from . import histograms
from . import plotting

from .utils import *        # noqa: F401,F403
from .io import *           # noqa: F401,F403
from .surfaces import *     # noqa: F401,F403
from .particles import *    # noqa: F401,F403
from .kinematics import *   # noqa: F401,F403
from .histograms import *   # noqa: F401,F403
from .plotting import *     # noqa: F401,F403

try:
    __version__ = _metadata.version("analogy")
except Exception:
    __version__ = "0.1.0"

__all__ = []
for _mod in (utils, io, surfaces, particles, kinematics, histograms, plotting):
    __all__.extend(getattr(_mod, "__all__", []))
del _mod
