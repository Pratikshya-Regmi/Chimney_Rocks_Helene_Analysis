#!/usr/bin/env python3
"""
make_figures.py — publication figures for the Natural Hazards manuscript.

Produces two vector PDFs (plus 600 dpi PNG previews):

  fig_lod_vs_canopy.pdf   Level of detection vs canopy height, both sensor
                          pairs. Log y-axis, because the two pairs differ by
                          ~1.5 orders of magnitude and a linear axis would
                          flatten the lidar-lidar series to a line at zero.

  fig_detection_area.pdf  Detectable-change area under each thresholding
                          scenario, both pairs.

Springer / Natural Hazards figure requirements this targets:
  - vector PDF (preferred) with fonts embedded as Type 42
  - single column 84 mm, double column 174 mm
  - sans-serif, no text below ~7 pt at final print size
  - colour-blind-safe palette, and distinguishable in greyscale via marker
    and line style, not colour alone

Usage:
    python3 make_figures.py                       # reads ./results/tables/*.csv
    python3 make_figures.py --data-dir some/dir
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------------------- style
MM = 1 / 25.4
SINGLE_COL = 84 * MM
DOUBLE_COL = 174 * MM

# Colour-blind-safe (Okabe-Ito). Lidar = blue, SfM = vermillion.
C_LIDAR = "#0072B2"
C_SFM = "#D55E00"
C_GREY = "#4D4D4D"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,       # embed TrueType, not Type 3 — Springer requires
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

BIN_LABELS = ["0–2", "2–5", "5–10", "10–20", "≥20"]


def save(fig, stem, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png", dpi=600)
    plt.close(fig)
    print(f"  wrote {outdir/stem}.pdf  and  .png")


# ============================================================ FIGURE 1
def fig_lod_vs_canopy(df, outdir):
    """LoD vs canopy height for both pairs.

    Log y-axis is not a stylistic choice: lidar-lidar sits at 0.24-0.32 m and
    lidar-SfM at 6-22 m. On a linear axis the lidar series collapses onto the
    x-axis and the reader loses the fact that it *declines* slightly. The log
    axis shows both the ~70x separation and the within-series trends.
    """
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.78))

    x = np.arange(len(BIN_LABELS))

    for pair, colour, marker, ls, label in [
        ("lidar-lidar", C_LIDAR, "o", "-", "Lidar–lidar (2020–2024)"),
        ("lidar-sfm", C_SFM, "s", "--", "Lidar–SfM (2020–2024)"),
    ]:
        sub = df[df["pair"] == pair].sort_values("bin_lo")
        ax.plot(x, sub["lod95"], color=colour, marker=marker, linestyle=ls,
                markersize=3.5, markeredgewidth=0, label=label, zorder=3)

    # annotate the two endpoints that carry the argument
    ll = df[df["pair"] == "lidar-lidar"].sort_values("bin_lo")["lod95"].values
    sf = df[df["pair"] == "lidar-sfm"].sort_values("bin_lo")["lod95"].values

    ax.annotate(f"{sf[0]:.1f} m", (x[0], sf[0]), textcoords="offset points",
                xytext=(4, -11), color=C_SFM, fontsize=6.5)
    ax.annotate(f"{sf[-1]:.1f} m", (x[-1], sf[-1]), textcoords="offset points",
                xytext=(-2, 6), color=C_SFM, fontsize=6.5, ha="right")
    ax.annotate(f"{ll[0]:.2f} m", (x[0], ll[0]), textcoords="offset points",
                xytext=(4, 5), color=C_LIDAR, fontsize=6.5)
    ax.annotate(f"{ll[-1]:.2f} m", (x[-1], ll[-1]), textcoords="offset points",
                xytext=(-2, 6), color=C_LIDAR, fontsize=6.5, ha="right")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Canopy height (m)")
    ax.set_ylabel("Level of detection, LoD$_{95}$ (m)")
    ax.set_ylim(0.15, 40)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"{v:.1f}"))
    ax.grid(axis="y", which="major", color="0.9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    # legend sits in the empty band between the two series
    ax.legend(loc="center", bbox_to_anchor=(0.52, 0.46))

    save(fig, "fig_lod_vs_canopy", outdir)


# ============================================================ FIGURE 2
def fig_detection_area(df, outdir):
    """Detectable-change area by thresholding scenario.

    Stacked erosion/deposition so the reader sees both the total and the
    balance between them -- the lidar-lidar (a)->(b) change is as much about
    the erosion/deposition ratio flipping as about the total falling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.34),
                             sharey=True)

    panels = [
        ("lidar-lidar", "Lidar–lidar",
         ["Uniform LoD\n(uncorrected)", "Uniform LoD\n(co-registered)",
          "Canopy-aware LoD\n(co-registered)"]),
        ("lidar-sfm", "Lidar–SfM",
         ["Uniform LoD\n(uncorrected)", "Uniform LoD\n(bias-corrected)",
          "Canopy-aware LoD\n(bias-corrected)"]),
    ]

    for ax, (pair, title, xlabels) in zip(axes, panels):
        sub = df[df["pair"] == pair].sort_values("scenario")
        x = np.arange(len(sub))
        ero = sub["pct_erosion"].values
        dep = sub["pct_deposition"].values

        ax.bar(x, ero, width=0.62, color=C_SFM, label="Erosion",
               edgecolor="white", linewidth=0.4, zorder=3)
        ax.bar(x, dep, width=0.62, bottom=ero, color=C_LIDAR,
               label="Deposition", edgecolor="white", linewidth=0.4, zorder=3)

        for xi, (e, d) in enumerate(zip(ero, dep)):
            ax.text(xi, e + d + 1.8, f"{e + d:.1f}%", ha="center",
                    fontsize=6.5, color=C_GREY)

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=6.5)
        ax.set_title(title, pad=4)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="0.9", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Basin area above\nlevel of detection (%)")
    axes[1].legend(loc="upper right", ncol=1)

    save(fig, "fig_detection_area", outdir)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="results/tables")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    ddir, odir = Path(args.data_dir), Path(args.out_dir)
    lod = pd.read_csv(ddir / "lod_vs_canopy.csv")
    det = pd.read_csv(ddir / "detection_summary.csv")

    print("Generating figures...")
    fig_lod_vs_canopy(lod, odir)
    fig_detection_area(det, odir)

    # numbers for the captions -- printed so they are never retyped by hand
    ll = lod[lod.pair == "lidar-lidar"].sort_values("bin_lo")["lod95"].values
    sf = lod[lod.pair == "lidar-sfm"].sort_values("bin_lo")["lod95"].values
    print("\nCaption values:")
    print(f"  lidar-lidar LoD95: {ll[0]:.2f} m (open) -> {ll[-1]:.2f} m (>=20 m), "
          f"ratio {ll[-1]/ll[0]:.2f}x")
    print(f"  lidar-SfM   LoD95: {sf[0]:.2f} m (open) -> {sf[-1]:.2f} m (>=20 m), "
          f"ratio {sf[-1]/sf[0]:.2f}x")
    for _, r in det.iterrows():
        print(f"  {r['pair']:<12s} {r['scenario']}: {r['pct_total']:.1f}% total "
              f"({r['pct_erosion']:.1f}% ero / {r['pct_deposition']:.1f}% dep)")


if __name__ == "__main__":
    main()
