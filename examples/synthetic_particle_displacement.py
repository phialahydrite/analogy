#!/usr/bin/env python3
"""
Example: synthetic particle displacement through a known velocity field
=======================================================================

``analogy.particle_displacer`` advects synthetic markers through a sequence of
PIVlab velocity files (nearest-grid-point lookup, forward step per PIV pair).
Real PIV has no ground truth, so this example builds one:

  1. Define an analytic, incompressible velocity field: a plate moving left at
     speed U that is slowed by a fraction ``s`` across a localized shortening
     zone (a smooth "backthrust" band at x0, width w). Mass conservation
     forces upward motion inside the band, so material thickens there:

         u(x)    = -U * (1 - s * (1 - tanh((x - x0)/w)) / 2)
         v(x, y) = -y * du/dx = y * U*s/(2w) * sech^2((x - x0)/w)      (y up)

  2. Write it to disk as PIVlab-style text files (3 header lines, x,y,u,v).
  3. Seed a grid of markers and displace them with ``an.particle_displacer``.
  4. Integrate the same markers with RK4 on the continuous field and compare.

Usage
-----
    python synthetic_particle_displacement.py [--out DIR] [--files 100]

Writes ``synthetic_displacement_paths.png`` and ``..._error.png`` to DIR.
"""
import argparse
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analogy as an

# ---- model & grid parameters (pixels; one "frame" = one PIV image pair) ----
ROI_W, ROI_H = 800, 300         # region of interest
SPACING = 8                     # PIV window spacing
X_MIN = 100                     # ROI offset inside the full image (PIV coords)
U = 2.0                         # plate speed [px per PIV pair]
S = 0.3                         # fraction of speed lost across the zone
X0, WIDTH = 300.0, 40.0         # zone centre and half-width [px]
PIXELMM_SCALE = 3.15            # as in the experiments; used for strain scaling


# --------------------------------------------------------------------------
# Analytic field (y is up from the base of the ROI)
# --------------------------------------------------------------------------
def velocity(x, y):
    t = (x - X0) / WIDTH
    u = -U * (1.0 - S * 0.5 * (1.0 - np.tanh(t)))
    v = y * (U * S / (2.0 * WIDTH)) / np.cosh(t) ** 2
    return u, v


def advect_rk4(x, y, n_pairs, substeps=20):
    """Reference solution: integrate dx/dt = (u, v) per PIV pair."""
    x, y = np.array(x, float), np.array(y, float)
    h = 1.0 / substeps
    path = [np.column_stack((x, y))]
    for _ in range(n_pairs):
        for _ in range(substeps):
            k1 = velocity(x, y)
            k2 = velocity(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1])
            k3 = velocity(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1])
            k4 = velocity(x + h * k3[0], y + h * k3[1])
            x = x + h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            y = y + h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        path.append(np.column_stack((x, y)))
    return np.array(path)                        # [n_pairs+1, n_particles, 2]


# --------------------------------------------------------------------------
# Write the field as PIVlab-style files
# --------------------------------------------------------------------------
def write_piv_files(folder, n_files):
    """
    PIVlab layout expected by analogy: line 2 holds the A/B image names (words
    5 and 8), data starts on line 4 as x,y,u,v in image coordinates (y DOWN, so
    v is negated). Rows are x-major, as the strain/vorticity step assumes.
    """
    os.makedirs(folder, exist_ok=True)
    xs = np.arange(0, ROI_W + 1, SPACING, dtype=float)
    ys_up = np.arange(0, ROI_H + 1, SPACING, dtype=float)
    gx, gy = np.meshgrid(xs, ys_up, indexing="ij")       # x outer, y inner
    u, v_up = velocity(gx, gy)
    files = []
    for k in range(n_files):
        a, b = 2 * k, 2 * k + 2
        path = os.path.join(folder, "PIV_%04i.txt" % k)
        with open(path, "w") as fh:
            fh.write("PIVlab synthetic velocity field\n")
            fh.write("PIVlab pair of images: img_%04i.jpg compared with: img_%04i.jpg\n" % (a, b))
            fh.write("x [px], y [px], u [px/pair], v [px/pair]\n")
            pd.DataFrame({"x": gx.ravel() + X_MIN,
                          "y": (ROI_H - gy).ravel(),      # image y, downward
                          "u": u.ravel(),
                          "v": (-v_up).ravel()}).to_csv(fh, header=False, index=False)
        files.append(path)
    return files


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output folder (default: temp dir)")
    ap.add_argument("--files", type=int, default=100, help="number of PIV files")
    args = ap.parse_args()
    out = args.out or tempfile.mkdtemp(prefix="analogy_synth_")
    os.makedirs(out, exist_ok=True)

    files = write_piv_files(os.path.join(out, "piv"), args.files)
    framenumbers = pd.DataFrame([an.PIV_framenumbers(f) for f in files],
                                columns=["filenumber", "A_frame", "B_frame"])

    # seed markers in the foreland, in ROI coordinates (y up)
    px, py = np.meshgrid(np.arange(450, 701, 25), np.arange(20, 181, 20))
    particles = pd.DataFrame({"frame": 0, "particle": np.arange(px.size),
                              "x": px.ravel().astype(float),
                              "y": py.ravel().astype(float)})

    tracks = an.particle_displacer(
        files, particles, framenumbers, surface_hdf5=None,
        begin_file=0, end_file=len(files),
        radius=3 * SPACING, mask_piv=False,
        pixelmm_scale=PIXELMM_SCALE, x_min=X_MIN)

    # reference: continuous-field RK4, sampled at the same frames
    ref = advect_rk4(px.ravel(), py.ravel(), len(files))
    frames = np.sort(tracks.frame.unique())               # 0, 2, 4, ... image no.
    err = []
    for i, fr in enumerate(frames):
        g = tracks[tracks.frame == fr].sort_values("particle")
        d = np.hypot(g.x.values - ref[i][:, 0], g.y.values - ref[i][:, 1])
        err.append((fr, np.sqrt(np.mean(d ** 2)), d.max()))
    err = np.array(err)
    print("markers: %i, PIV files: %i, output rows: %i"
          % (px.size, len(files), len(tracks)))
    print("final-frame error vs. analytic [px]: RMS %.2f, max %.2f"
          % (err[-1, 1], err[-1, 2]))

    # ---- figure 1: paths over the velocity field -------------------------
    gx, gy = np.meshgrid(np.linspace(0, ROI_W, 200), np.linspace(0, ROI_H, 80))
    gu, gv = velocity(gx, gy)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.pcolormesh(gx, gy, np.hypot(gu, gv), cmap="Greys", shading="auto")
    plt.colorbar(im, ax=ax, label="Speed [px/pair]", pad=0.01)
    final = tracks[tracks.frame == frames[-1]].sort_values("particle")
    for p in range(px.size):
        t = tracks[tracks.particle == p].sort_values("frame")
        ax.plot(t.x, t.y, "-", color="tab:blue", lw=0.8)
        ax.plot(ref[:, p, 0], ref[:, p, 1], "--", color="tab:orange", lw=0.8)
    ax.scatter(px, py, s=10, c="k", zorder=3, label="start")
    ax.scatter(final.x, final.y, s=12, c="tab:red", zorder=3, label="end (analogy)")
    ax.plot([], [], "-", color="tab:blue", label="particle_displacer")
    ax.plot([], [], "--", color="tab:orange", label="RK4 analytic")
    ax.axvspan(X0 - 2 * WIDTH, X0 + 2 * WIDTH, color="tab:red", alpha=0.06)
    ax.set_xlim(150, 750)
    ax.set_ylim(0, ROI_H)
    ax.set_aspect("equal")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("height [px]")
    ax.set_title("Synthetic markers crossing a shortening zone (shaded)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.savefig(os.path.join(out, "synthetic_displacement_paths.png"),
                bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ---- figure 2: error growth ------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(err[:, 0], err[:, 1], label="RMS")
    ax.plot(err[:, 0], err[:, 2], label="max")
    ax.set_xlabel("Image number")
    ax.set_ylabel("Position error vs. analytic [px]")
    ax.set_title("Nearest-grid-point displacement error")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out, "synthetic_displacement_error.png"),
                bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main()
