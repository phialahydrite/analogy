#!/usr/bin/env python3
"""
Example: subsurface slip, exhumation & burial from PIV + particle data
======================================================================

This is the script-level workflow from the original ``subsurface_slip.py``
rewritten on top of the ``analogy`` package. For each experiment it:

  1. Loads PIV files, wedge-surface HDF5 store and artificial-particle CSV.
  2. Computes depth / burial / exhumation per particle.
  3. Bins strain, vorticity, burial and exhumation into [frame, y, x] grids.
  4. Converts strain - 1/2 vorticity to instantaneous and cumulative slip,
     scaled by the foreland plate convergence rate.
  5. Samples slip at fixed depths below the evolving wedge surface and plots
     it as slip-vs-convergence maps.
  6. Writes transparent, surface-masked PNGs for MATLAB's ``warp``
     (pseudo-3D views of model evolution).
  7. Plots max instantaneous / cumulative slip per experiment, then compares
     all experiments.

Usage
-----
    python subsurface_slip_example.py --demo        # synthetic data, runs anywhere
    python subsurface_slip_example.py               # your experiments (edit EXPERIMENTS)

Input conventions (same as the original scripts)
------------------------------------------------
* PIV text files: 3 header lines, then columns x, y, u, v (PIVlab export),
  in uncropped image pixel coordinates.
* Surface store: HDF5 with one table per frame, key ``wedgetop_%05i``,
  columns x, y in uncropped image coordinates (y up from the image base).
* Particle CSV: columns frame, particle, x, y, exy, vort, cumexy, cumvort,
  with x, y in ROI coordinates (pixels, y up from the ROI base).
"""
import argparse
import gc
import glob
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")                      # write files only, no GUI
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import analogy as an


# --------------------------------------------------------------------------
# 1. Experiment configuration
# --------------------------------------------------------------------------
@dataclass
class Experiment:
    """Everything that varied between the 12 hard-coded ``if fi == N`` blocks."""
    prefix: str                       # output filename prefix / label
    surfs: str                        # HDF5 store of wedge surfaces
    points_csv: str                   # artificial particle table
    piv_glob: str                     # glob for PIV text files
    # ROI in uncropped image pixels (origin at image top-left, as in the
    # original: x_min..x_max, y_min..y_max)
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    img_h: int                        # full image height [px]
    scale: float                      # pixels per cm
    pixelmm_scale: float              # mm of plate motion per frame-step
    # foreland box used to measure the (steady) plate velocity, in ROI px:
    #   (x_lo, x_hi, y_lo, y_hi)
    foreland_box: Tuple[int, int, int, int] = (4400, 4500, 50, 150)
    frame_start: int = 2              # PIV frame step ("step" in the original)
    depths_cm: Sequence[float] = (0.25, 0.3, 0.75, 1.0, 1.5)
    png_every: int = 200              # transparent PNG every N frames
    image_glob: Optional[str] = None  # optional photos for the overlay figure
    overlay_every: int = 2            # overlay figure every N frames
    ins_lim: float = 1.0              # +/- colour limit, instantaneous slip maps [cm]
    cum_lim: float = 15.0             # +/- colour limit, cumulative slip maps [cm]


@dataclass
class SlipFields:
    """Gridded results, all shaped [frame, y, x]."""
    ins_slip: np.ndarray
    cum_slip: np.ndarray
    burial: np.ndarray
    exhumation: np.ndarray
    cc_corrected: np.ndarray          # plate convergence per frame, [n,1]
    roi_w: float
    roi_h: float
    spacing: float
    corrected_spacing: float


# --------------------------------------------------------------------------
# 2. Core computation (the "main body" of the original script)
# --------------------------------------------------------------------------
def compute_fields(exp: Experiment) -> SlipFields:
    pfiles = sorted(glob.glob(exp.piv_glob))
    if not pfiles:
        raise FileNotFoundError(f"No PIV files match {exp.piv_glob!r}")

    # PIV window spacing: largest positive step between neighbouring x values
    first = pd.read_csv(pfiles[0], skiprows=3, usecols=[0, 1, 2, 3],
                        names=["x", "y", "u", "v"])
    spacing = first.x.diff().max()
    # the original divides by spacing/8 so slip is per 8-px window
    corrected_spacing = spacing / 8

    roi_w, roi_h, *_ = an.roi_geom(pfiles[0])

    # depth, burial & exhumation for every particle at every frame
    pts = an.cumulative_depth_stats(
        an.depth_calc(exp.img_h,
                      pd.read_csv(exp.points_csv, dtype=np.float32),
                      1, exp.surfs,
                      preflipped=True, silent=True,
                      from_larger=True, skip_edge=True,
                      x_min=exp.x_min, y_max=exp.y_max))
    n_frames = int(pts.frame.max())

    # gridded histograms: strain/vorticity layers and burial/exhumation layers
    EXY, VORT, CUMEXY, CUMVORT, part_counts, _ = an.displace_hist(
        pts, roi_w, roi_h, n_frames, square_size=spacing)
    BUR, EXH, _, _ = an.exhum_burial_hist(
        pts, roi_w, roi_h, n_frames, square_size=spacing)

    # steady plate velocity from non-deforming foreland material
    xlo, xhi, ylo, yhi = exp.foreland_box
    _, fit = an.plate_velocity_calc(xlo, xhi, ylo, yhi,
                                    exp.x_min, exp.y_max,
                                    range(len(pfiles)), pfiles,
                                    step=exp.frame_start)
    n = part_counts.shape[0]
    cc = (fit[0] * np.arange(n) + fit[1]).reshape(n, 1)
    cc3 = cc[:, np.newaxis]                       # [n,1,1] for broadcasting

    # Slip from strain minus half the vorticity (Gilman 1961; Fung 1980),
    # normalised by particle counts and scaled by convergence per frame.
    step = exp.frame_start
    ins = step * ((EXY - 0.5 * VORT) / part_counts) * (cc3 / exp.scale)
    cum = step * ((CUMEXY - 0.5 * CUMVORT) / part_counts) * (cc3 / exp.scale)

    # fill gaps (bins without particles, frames not in the PIV set)
    ins = an.fill(ins) / corrected_spacing
    cum = an.fill(cum) / corrected_spacing
    BUR = an.fill(np.ma.filled(BUR, np.nan))
    EXH = an.fill(np.ma.filled(EXH, np.nan))

    return SlipFields(ins, cum, BUR, EXH, cc, roi_w, roi_h,
                      spacing, corrected_spacing)


def load_surface(exp: Experiment, frame: int) -> pd.DataFrame:
    """Wedge surface for [frame], shifted into ROI coordinates."""
    surf = pd.read_hdf(exp.surfs, "wedgetop_%05i" % frame)
    surf.x = surf.x - exp.x_min
    surf.y = surf.y - (exp.img_h - exp.y_max)
    return surf


def extract_subsurface(exp: Experiment, f: SlipFields, depth_cm: float):
    """
    Sample slip at [depth_cm] below the wedge surface for every frame, and
    return (instantaneous, cumulative) grids shaped [frame, x].
    """
    n_keys = len(pd.HDFStore(exp.surfs, "r").keys())
    ins_rows, cum_rows = [], []
    for i in range(f.ins_slip.shape[0]):
        if i >= n_keys:
            break
        surf = load_surface(exp, i)
        x = surf.x / f.spacing
        y = (surf.y - depth_cm * exp.scale) / f.spacing
        coords = np.vstack((y, x))
        ins_rows.append(np.vstack((x, np.full(len(x), i),
                        ndi.map_coordinates(f.ins_slip[i], coords))).T)
        cum_rows.append(np.vstack((x, np.full(len(x), i),
                        ndi.map_coordinates(f.cum_slip[i], coords))).T)
    ins_e, cum_e = np.vstack(ins_rows), np.vstack(cum_rows)
    # columns are (x, frame, value) -> grid with frame on rows, x on columns
    return (an.gridize(ins_e[:, 1], ins_e[:, 0], ins_e[:, 2]),
            an.gridize(cum_e[:, 1], cum_e[:, 0], cum_e[:, 2]))


# --------------------------------------------------------------------------
# 3. Plotting helpers
# --------------------------------------------------------------------------
def haxby_cmap():
    """GMT haxby from CPT-City if pycpt is installed, else a fallback."""
    try:
        import pycpt
        return pycpt.load.cmap_from_cptcity_url("gmt/GMT_haxby.cpt")
    except Exception:
        return plt.cm.turbo


def _slip_map(grid, exp, title, cbar_label, cmap, vmin, vmax, out, divide=1.0):
    """Width vs. cumulative-convergence map of subsurface slip."""
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlabel("Width [cm]")
    ax.set_ylabel("Cumulative Convergence [cm]")
    ax.set_title(title)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(np.flipud(ndi.median_filter(grid, 4)) / divide,
                   aspect="auto", norm=norm, cmap=cmap,
                   # x2: PIV only analyses every other window
                   extent=(0, grid.shape[1] / exp.scale, 0,
                           2 * grid.shape[0] * exp.pixelmm_scale / exp.scale))
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax).set_label(cbar_label)
    ax.invert_yaxis()
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_subsurface_maps(exp, f, cmap_cum):
    for d in exp.depths_cm:
        ins_grid, cum_grid = extract_subsurface(exp, f, d)
        mm = int(round(d * 10))
        _slip_map(ins_grid, exp,
                  "Instantaneous Slip at %.2f cm Below Wedge Surface" % d,
                  "Instantaneous Slip [cm]", plt.cm.seismic,
                  -exp.ins_lim, exp.ins_lim,
                  "%s_subsurface_insslip_%03imm.png" % (exp.prefix, mm))
        _slip_map(cum_grid, exp,
                  "Cumulative Slip at %.2f cm Below Wedge Surface" % d,
                  "Cumulative Slip [cm]", cmap_cum,
                  -exp.cum_lim, exp.cum_lim,
                  "%s_subsurface_cumslip_%03imm.png" % (exp.prefix, mm),
                  divide=10)       # mm -> cm


def save_transparent_layer(field_2d, surf, exp, f, name, frame, cmap,
                           vmin, vmax, centered=False, divide=1.0):
    """
    Surface-masked field on a transparent, frame-filling canvas, ready for
    MATLAB's ``warp``. The wedge surface is drawn as a thick black line.
    """
    masked = an.mask_from_surface(surf, field_2d, f.roi_w, f.roi_h)[:, 2:]
    h, w = masked.shape
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w / 20, h / 20)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    kw = dict(cmap=cmap, extent=(0, f.roi_w / exp.scale, 0, f.roi_h / exp.scale))
    if centered:
        kw["norm"] = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        kw.update(vmin=vmin, vmax=vmax)
    ax.imshow(np.flipud(masked) / divide, **kw)
    ax.set_xlim((0, f.roi_w / exp.scale))
    ax.set_ylim((0, f.roi_h / exp.scale))
    ax.plot(surf.x / exp.scale, surf.y / exp.scale, "k-", linewidth=8)
    fig.savefig("%s_%strans_%03iby%03ipx_f%05i.png"
                % (exp.prefix, name, w, h, frame), dpi=200)
    plt.close(fig)
    return masked


def export_transparent_pngs(exp, f, cmap_cum):
    n_keys = len(pd.HDFStore(exp.surfs, "r").keys())
    for fr in range(f.cum_slip.shape[0])[2::2][::max(exp.png_every // 2, 1)]:
        if fr >= n_keys:
            continue
        surf = load_surface(exp, fr)
        S = lambda a: an.fill(a[fr])        # noqa: E731  (fill, then mask)
        save_transparent_layer(S(f.cum_slip), surf, exp, f, "cumslip", fr,
                               cmap_cum, -10, 10, centered=True)
        save_transparent_layer(S(f.ins_slip), surf, exp, f, "insslip", fr,
                               plt.cm.seismic, -0.05, 0.1, centered=True)
        ex = save_transparent_layer(S(f.exhumation), surf, exp, f,
                                    "exhumation", fr, plt.cm.turbo, 0, 10,
                                    divide=exp.scale)
        bu = save_transparent_layer(S(f.burial), surf, exp, f, "burial", fr,
                                    plt.cm.turbo_r, -10, 0, divide=exp.scale)
        # net burial + exhumation (original saves this layer too)
        h, w = (bu + ex).shape
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w / 20, h / 20)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(np.flipud(bu + ex) / exp.scale, cmap=plt.cm.turbo,
                  extent=(0, f.roi_w / exp.scale, 0, f.roi_h / exp.scale),
                  vmin=-10, vmax=10)
        ax.set_xlim((0, f.roi_w / exp.scale))
        ax.set_ylim((0, f.roi_h / exp.scale))
        ax.plot(surf.x / exp.scale, surf.y / exp.scale, "k-", linewidth=8)
        fig.savefig("%s_burialexhumtrans_%03iby%03ipx_f%05i.png"
                    % (exp.prefix, w, h, fr), dpi=200)
        plt.close(fig)
        gc.collect()


def _decorate_slip_axis(ax, ylim, major, minor, xlim=(0, 125)):
    ax.set_xlabel("Cumulative Convergence [cm]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_minor_locator(MultipleLocator(minor))
    ax.grid(True)
    ax.grid(visible=True, which="minor", color="#999999",
            linestyle="-", alpha=0.2)


def plot_max_slip(exp, f):
    """Max instantaneous and cumulative slip per frame; returns both series."""
    x = np.arange(f.ins_slip.shape[0]) * exp.pixelmm_scale / exp.scale

    max_ins = np.nanmax(np.nanmax(f.ins_slip, axis=2), axis=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    _decorate_slip_axis(ax, (0, 0.26), 0.05, 0.01)
    ax.set_ylabel("Max. Instantaneous Displacement [cm]")
    ax.plot(x, max_ins, "k+", label="Max. Instantaneous Slip")
    ax.plot(-np.squeeze(f.cc_corrected) / 2 / exp.scale, "r-",
            label="Plate Convergence Rate [cm/frame]")
    ax.legend()
    fig.savefig("%s_max_instant_slipcm.png" % exp.prefix,
                bbox_inches="tight", dpi=200)
    plt.close(fig)

    max_cum = np.nanmax(np.nanmax(f.cum_slip, axis=2), axis=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    _decorate_slip_axis(ax, (0, 15), 1, 0.5)
    ax.set_ylabel("Cumulative Maximum Slip [cm]")
    ax.plot(x, max_cum, "k+", label="Max. Cumulative Slip")
    ax.legend(loc="lower right")
    fig.savefig("%s_max_cumul_slipcm.png" % exp.prefix,
                bbox_inches="tight", dpi=200)
    plt.close(fig)
    return max_ins, max_cum


def plot_exhumation_overlays(exp, f, out_dir="watermark"):
    """Photo + exhumation overlay, for presentation figures (needs `pims`)."""
    if not exp.image_glob:
        return
    import pims
    images = pims.ImageSequence(exp.image_glob)
    os.makedirs(out_dir, exist_ok=True)
    n_keys = len(pd.HDFStore(exp.surfs, "r").keys()) - 4
    # the original crops with the same ROI it uses for everything else
    crop = (slice(exp.y_min, exp.y_max), slice(exp.x_min, exp.x_max))
    for n in range(f.exhumation.shape[0])[::exp.overlay_every]:
        if n >= n_keys:
            break
        surf = load_surface(exp, n)
        mex = an.mask_from_surface(surf, f.exhumation[n], f.roi_w, f.roi_h)
        fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True, sharey=True)
        ext = (0, f.roi_w / exp.scale, 0, (exp.y_max - exp.y_min) / exp.scale)
        ax[0].imshow(images[n][crop], extent=ext)
        ax[1].imshow(images[n][crop], extent=ext)
        im = ax[1].imshow(np.flipud(mex) / exp.scale,
                          extent=(0, f.roi_w / exp.scale, 0, f.roi_h / exp.scale),
                          cmap=haxby_cmap(), vmin=0, vmax=10)
        for a in ax:
            a.plot(surf.x / exp.scale, surf.y / exp.scale, "k-", linewidth=3)
            a.set_ylabel("Height [cm]")
        make_axes_locatable(ax[0]).append_axes(
            "right", size="2%", pad=0.05).set_visible(False)
        cax = make_axes_locatable(ax[1]).append_axes("right", size="2%", pad=0.05)
        plt.colorbar(im, cax=cax).set_label("Exhumation [cm]")
        ax[1].set_xlabel("Width [cm]")
        ax[0].set_ylim(0, 15)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "%s_exhum_%05i.png" % (exp.prefix, n)),
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
        gc.collect()


def plot_comparison(labels, series, ylabel, ylim, legend_loc, stem, exp_ref):
    """All experiments on one axis (the closing 'ch3_*' plots)."""
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(labels)))
    fig, ax = plt.subplots(figsize=(15, 7))
    for lab, y, c in zip(labels, series, colors):
        ax.plot(np.arange(len(y)) * exp_ref.pixelmm_scale / exp_ref.scale,
                y, label=lab, color=c)
    ax.set_xlabel("Cumulative Convergence [cm]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 125)
    ax.set_ylim(*ylim)
    ax.legend(ncol=2, loc=legend_loc)
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (stem, ext), bbox_inches="tight", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Driver
# --------------------------------------------------------------------------
def run(experiments: List[Experiment]):
    cmap_cum = haxby_cmap()
    labels, max_ins_all, max_cum_all = [], [], []
    for exp in experiments:
        fields = compute_fields(exp)
        plot_subsurface_maps(exp, fields, cmap_cum)
        export_transparent_pngs(exp, fields, cmap_cum)
        max_ins, max_cum = plot_max_slip(exp, fields)
        plot_exhumation_overlays(exp, fields)
        labels.append(exp.prefix)
        max_ins_all.append(max_ins)
        max_cum_all.append(max_cum)
        print("%s complete" % exp.prefix)
    plot_comparison(labels, max_ins_all, "Maximum Instantaneous Slip [cm]",
                    (0, 1), "upper right", "ch3_maxinstslip", experiments[0])
    plot_comparison(labels, max_cum_all, "Maximum Slip [cm]",
                    (0, 15), "lower right", "ch3_maxcumtslip", experiments[0])


# Edit this list for real data: one Experiment per model run.
EXPERIMENTS = [
    Experiment(
        prefix="pat_onesidefric_15deg_7ero",
        surfs="pat_onesidefric_15deg_7ero/pat_onesidefric_15deg_7ero_surfnocv.h5",
        points_csv="pat_onesidefric_15deg_7ero_artificial_pts_temp.csv",
        piv_glob="pat_onesidefric_15deg_7ero/piv/*.txt",
        x_min=90, x_max=90 + 4800, y_min=400, y_max=1205,
        img_h=1500,                    # images.frame_shape[0] in the original
        scale=63.0, pixelmm_scale=3.15,
        frame_start=2,
        image_glob="pat_onesidefric_15deg_7ero/*.jpg",
    ),
]


# --------------------------------------------------------------------------
# 5. Synthetic demo data (so the example runs without your experiments)
# --------------------------------------------------------------------------
def make_demo_experiment(root: str) -> Experiment:
    """A tiny fake model: a growing wedge, 30 frames, constant plate motion."""
    rng = np.random.default_rng(0)
    n_frames, roi_w, roi_h, sp = 30, 400, 120, 8
    x_min, y_max, img_h = 20, 200, 260
    surfs = os.path.join(root, "demo_surfs.h5")
    for t in range(n_frames):
        x = np.arange(0, roi_w + 1, dtype=float)
        height = 40 + 0.1 * x + 0.5 * t
        pd.DataFrame({"x": x + x_min,
                      "y": height + (img_h - y_max)}).to_hdf(
            surfs, "wedgetop_%05i" % t, mode="a")

    # PIV: steady plate translation (u = -2 px/frame) plus a little noise
    os.makedirs(os.path.join(root, "piv"), exist_ok=True)
    gx, gy = np.meshgrid(np.arange(0, roi_w + 1, sp) + x_min,
                         y_max - np.arange(0, roi_h + 1, sp))
    for t in range(n_frames):
        u = -2 + 0.05 * rng.standard_normal(gx.shape)
        v = 0.05 * rng.standard_normal(gx.shape)
        with open(os.path.join(root, "piv", "PIV_%04i.txt" % t), "w") as fh:
            fh.write("PIVlab demo\nframe %i\nx [px], y [px], u [px/frame], v [px/frame]\n" % t)
            # x varies fastest so diff().max() recovers the window spacing
            for yy, xx, uu, vv in zip(gy.ravel(), gx.ravel(), u.ravel(), v.ravel()):
                fh.write("%g,%g,%g,%g\n" % (xx, yy, uu, vv))

    # artificial particles: slowly buried, small noise strain + one shear band
    rows = []
    for p in range(500):
        px = rng.uniform(0, roi_w)
        depth_frac = rng.uniform(0.2, 0.9)
        cexy = cvort = 0.0
        for t in range(n_frames):
            h = 40 + 0.1 * px + 0.5 * t
            e, w = 0.01 * rng.standard_normal(2)
            if 150 < px < 180 and t > 5:      # a localized shear band (a "fault")
                e += 0.15
            cexy, cvort = cexy + e, cvort + w
            rows.append((t, p, px, depth_frac * h - 0.2 * t, e, w, cexy, cvort))
    pts = os.path.join(root, "demo_pts.csv")
    pd.DataFrame(rows, columns=["frame", "particle", "x", "y", "exy", "vort",
                                "cumexy", "cumvort"]).to_csv(pts, index=False)

    return Experiment(
        prefix=os.path.join(root, "demo"), surfs=surfs, points_csv=pts,
        piv_glob=os.path.join(root, "piv", "*.txt"),
        x_min=x_min, x_max=x_min + roi_w, y_min=y_max - roi_h, y_max=y_max,
        img_h=img_h, scale=10.0, pixelmm_scale=3.15,
        foreland_box=(330, 390, 20, 90), frame_start=2,
        depths_cm=(0.5, 1.0), png_every=20,
        ins_lim=0.1, cum_lim=0.2)      # demo slip is small; tighten colour limits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demo", action="store_true",
                    help="run on synthetic data in a temp folder")
    args = ap.parse_args()
    if args.demo:
        root = tempfile.mkdtemp(prefix="analogy_demo_")
        print("Demo output in", root)
        run([make_demo_experiment(root)])
    else:
        run(EXPERIMENTS)


if __name__ == "__main__":
    main()
