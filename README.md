# analogy

*A toolkit for post-processing **analog** tectonic sandbox models — hence the pun.*

`analogy` consolidates seven standalone PIV and PTV analysis scripts
(`temperatures.py`, `subsurface_slip.py`, `SPTVcode.py`, `short_vertical.py`,
`ratecomp.py`, `map_view_exhum_collection.py`, `erosional_thickness_and_time.py`)
into a single, de-duplicated, importable Python package. Those scripts had
accumulated many copy-pasted versions of the same helper functions over
time; this package keeps the most complete/most recent version of each and
organizes everything by what it does rather than which script happened to
need it that day.

## Install

```bash
pip install -e .
# or, with the optional heavier deps (image sequences, particle linking,
# GMT colormaps):
pip install -e ".[all]"
```

## Layout

| Module | What's in it |
|---|---|
| `analogy.utils` | Array helpers: `fill` (NaN nearest-neighbor fill), `gridize`/`degridize`, `do_kdtree`/`first_closest`/`k_closest`/`first_last`/`last_closest`, `line_fit`, `cmap_discretize`, `div_proportional_colormap`, `masked_array_shift_zero`, monotonicity checks (`strictly_increasing`, etc.), `find_roots`, `reset_time`/`reset_time_2d`, `to_slices`, `subtract_staggered`, `find_comp` |
| `analogy.io` | `roi_geom`, `PIV_framenumbers` — reading PIVlab output files |
| `analogy.surfaces` | Masking by model topography (`mask_from_surface` & friends), deformation-front detection (`deformation_front`, `deformation_front_dv`, `deformation_front_mode`), `structural_divide`, `topographic_divide`, `curvature_calc`, `slope_deffront`, `foreland_mask_surf`, `volume_calc` |
| `analogy.particles` | Particle depth/burial/exhumation (`depth_calc`, `cumulative_depth_stats`, `starting_max_depth_redux`, `depth_stats_compilation`, ...), dense displacement-field extraction (`particle_displacer*`), line-of-particles extraction (`line_particles*`) |
| `analogy.kinematics` | `plate_velocity_calc`, `strain_component_calc`, `vorticity` |
| `analogy.histograms` | Gridded summaries: `depth_synthetic_hist`, `displace_hist`, `exhum_burial_hist`, `depth_burial_hist` |
| `analogy.plotting` | `tp_plot_traj`, `test_subpix_bias` |

Everything is also re-exported at the top level, so `import analogy as an`
gives you `an.gridize`, `an.depth_calc`, `an.deformation_front_mode`, etc.

## Optional dependencies

A few original scripts imported `pims`, `trackpy`, and `pycpt` at the top
of the file, but the functions kept here don't actually call into them
directly (they consume plain DataFrames/arrays produced elsewhere in your
pipeline). `analogy._optdeps` provides a small soft-import helper in case
future additions need one of these — importing `analogy` itself never
requires them.
