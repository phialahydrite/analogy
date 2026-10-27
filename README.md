# analogy

*A toolkit for post-processing **analog** tectonic sandbox models — hence the pun.*

`analogy` consolidates seven standalone analysis scripts written for my [dissertation](https://docs.lib.purdue.edu/dissertations/AAI30506060/)
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

## Notes on consolidation

* Where the same function existed nearly identically across multiple
  scripts (e.g. `fill`, `gridize`, `PIV_framenumbers`, `depth_calc`,
  `mask_from_surface`), the most complete version (generally from
  `map_view_exhum_collection.py`) was kept.
* A few functions only existed in one of the smaller scripts
  (`find_comp`, `reset_time`, `reset_time_2d`, `cumulative_depth_stats`,
  `exhum_burial_hist`, `to_slices`, `subtract_staggered`,
  `foreland_mask_surf`) and were pulled in from there.
* One duplicate was dropped outright: `cumuative_depth_stats` (note the
  typo) in `map_view_exhum_collection.py` is an earlier, misspelled copy
  of `cumulative_depth_stats` — same logic, kept the correctly-named one.

## Example

`examples/synthetic_particle_displacement.py` builds an analytic velocity field
with a shortening zone, writes it as PIVlab-style files, advects markers with
`particle_displacer`, and checks them against an RK4 solution (~0.3 px RMS).

See [here](https://phialahydrite.github.io/synthetic_particle_displacement.html) for it in action.

## Known limitations

Several functions inherited from the original scripts read variables that
were globals in the scripts (e.g. `images`, `pixelmm_scale`, `im_h`, `xmin`,
`ymax`, `scale`) or rely on imports that were not carried over (`peakutils`,
`rdp`, `medfilt`, `savgol_filter`, ...). These need explicit parameters before
they can be called from outside the original scripts: `particle_displacer_temperatures*`,
`deformation_front*`, `topographic_divide`, `structural_divide`,
`slope_deffront`, `volume_calc`, `find_below_all`, `tp_plot_traj`,
`test_subpix_bias`, `depth_diff`, `depth_rate`, `depth_stats_compilation`,
`rdp_burial_calc_forlambda`, `extract_near_surf_dilation`, `foreland_mask_surf`,
`reset_time_2d`. `particle_displacer` and the functions used by the examples are fixed and tested.
