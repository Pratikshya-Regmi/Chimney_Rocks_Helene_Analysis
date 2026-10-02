#!/usr/bin/env python3
"""
31_fig_flow_sequence_basinwide.py
======================================
FIGURE 1 of the three-epoch overland-flow rebuild: basin-wide, three panels
in chronological order --
  (a) 2020 lidar -- pre-event
  (b) 2024 CAP SfM -- 5 October 2024, immediate post-event (~8 days)
  (c) 2024 lidar -- 15-16 November 2024, seven weeks post-event
on a COMMON discharge colour scale (log, vmin=1e-4, vmax=0.48 m -- the
larger of the two lidar epochs' 99.9th percentile; see log for the exact
figure). Cells above vmax are SATURATED at the colour scale's top colour
rather than rescaled per-panel, so the panels stay visually comparable;
the count/fraction of saturated cells per epoch is reported in the caption
and the log (this is where the SfM panel differs most from the two lidar
panels -- its reconstructed surface produces locally implausible discharge
spikes the lidar surfaces do not).

The three zones of largest 2020-vs-2024-lidar flow-path change (from
30_flow_sequence_setup.py, results/tables/flow_change_zones.csv) are
outlined and labelled Z1-Z3 on all three panels, at identical map
positions, so Figure 2's zoom grid can be located directly against this
figure.

Requires 30_flow_sequence_setup.py to have been run first (reads its
GeoTIFF exports and the zone table).

Outputs:
  results/figures/fig_flow_sequence_basinwide.pdf / .png
  results/logs/flow_sequence_basinwide_<date>.log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import log  # noqa: E402
from map_style import style_map_panel, add_scale_north_inline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

DISCHARGE_2020_TIF = FIGURES / "_flow_discharge_2020.tif"
DISCHARGE_CAP_TIF = FIGURES / "_flow_discharge_2024cap.tif"
DISCHARGE_2024_TIF = FIGURES / "_flow_discharge_2024lidar.tif"
HILLSHADE_TIF = FIGURES / "_flow_hillshade.tif"

MM = 1 / 25.4
FIG_W_MM = 174  # double column

VMIN = 1e-4
VMAX = 0.48  # max(p99.9 2020, p99.9 2024 lidar); see log for exact values

PANELS = [
    ("2020 lidar", "pre-event", DISCHARGE_2020_TIF),
    ("2024 CAP SfM", "5 Oct 2024, immediate post-event", DISCHARGE_CAP_TIF),
    ("2024 lidar", "15-16 Nov 2024, 7 weeks post-event", DISCHARGE_2024_TIF),
]


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        b = src.bounds
        extent = (b.left, b.right, b.bottom, b.top)
    return arr, extent


def main():
    if not DISCHARGE_2020_TIF.exists():
        sys.exit("Run 30_flow_sequence_setup.py first.")

    zones = pd.read_csv(TABLES / "flow_change_zones.csv")

    hs, hs_extent = read_raster(HILLSHADE_TIF)

    fig = plt.figure(figsize=(FIG_W_MM * MM, 92 * MM))
    gs_fig = fig.add_gridspec(2, 3, height_ratios=[1, 0.055], hspace=0.32,
                              wspace=0.12)

    saturated_report = []
    extents = []
    im = None
    ax_first = None
    ax_last = None
    for i, (sensor, date_label, tif) in enumerate(PANELS):
        ax = fig.add_subplot(gs_fig[0, i])
        if ax_first is None:
            ax_first = ax
        ax_last = ax
        arr, extent = read_raster(tif)
        extents.append(extent)

        ax.imshow(hs, cmap="gray", extent=hs_extent, origin="upper", zorder=1)
        arr_clipped = np.clip(arr, VMIN, VMAX)
        im = ax.imshow(arr_clipped, cmap="YlGnBu", norm=LogNorm(vmin=VMIN, vmax=VMAX),
                       extent=extent, origin="upper", alpha=0.85, zorder=2)

        n_valid = int(arr.count())
        n_sat = int((arr > VMAX).sum())
        pct_sat = 100.0 * n_sat / n_valid if n_valid else 0.0
        saturated_report.append((sensor, n_valid, n_sat, pct_sat))

        for _, z in zones.iterrows():
            w = z["e_max"] - z["e_min"]
            h = z["n_max"] - z["n_min"]
            ax.add_patch(Rectangle((z["e_min"], z["n_min"]), w, h, fill=False,
                                   edgecolor="red", linewidth=0.9, zorder=6))
            ax.text(z["e_min"] + w / 2, z["n_max"] + 12, z["zone_id"],
                   color="red", fontsize=6.5, fontweight="bold", ha="center",
                   va="bottom", zorder=7)

        # No scale bar or north arrow inside the panels: all three are the
        # same basin at the same extent, so both belong to the FIGURE and
        # are drawn once on the legend line below (advisor comment).
        style_map_panel(ax, extent, graticule_step=500,
                        draw_scale=False, draw_north=False)
        ax.text(0.5, 1.10, f"({chr(97+i)}) {sensor}", transform=ax.transAxes,
               ha="center", va="bottom", fontsize=7.5, fontweight="bold",
               clip_on=False, zorder=21)
        ax.text(0.5, 1.02, date_label, transform=ax.transAxes,
               ha="center", va="bottom", fontsize=6.3, clip_on=False,
               zorder=21)

    fig.subplots_adjust(top=0.84)

    # The legend line carries the colourbar AND the figure's single scale bar
    # + north arrow. Split the row that gs_fig[1, :] would have occupied.
    fig.canvas.draw()
    row = gs_fig[1, :].get_position(fig)
    SCALE_FRAC, GAP_FRAC = 0.20, 0.03
    cb_w = row.width * (1.0 - SCALE_FRAC - GAP_FRAC)
    cax = fig.add_axes([row.x0, row.y0, cb_w, row.height])
    # Place the group's TOP just under panel (c)'s rotated easting tick
    # labels, measured rather than guessed: those labels are set at 45 deg
    # and hang well below the axes, so any fixed offset from the colourbar
    # row either collides with them or floats far from the legend line.
    tick_bottom = min(
        (lab.get_window_extent().y0 for lab in ax_last.get_xticklabels()
         if lab.get_text()), default=None)
    sax_top = (fig.transFigure.inverted().transform((0, tick_bottom))[1]
               - 0.006) if tick_bottom is not None else row.y0 + row.height
    sax_h = row.height * 2.6
    sax = fig.add_axes([row.x0 + cb_w + row.width * GAP_FRAC,
                        sax_top - sax_h,
                        row.width * SCALE_FRAC, sax_h])

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Unit discharge [log scale, m$^3$ s$^{-1}$ m$^{-1}$-equivalent] "
                   "-- values above the scale maximum are saturated", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2.5, width=0.5)

    # Measure the panels as RENDERED (aspect='equal' shrinks them inside their
    # GridSpec cells) so the scale bar's length is true to the maps.
    fig.canvas.draw()
    panel_w_mm = (ax_first.get_window_extent().width / fig.dpi) * 25.4
    map_w_m = extents[0][1] - extents[0][0]
    sax_w_mm = (sax.get_window_extent().width / fig.dpi) * 25.4
    add_scale_north_inline(sax, scale_m=500, mm_per_m=panel_w_mm / map_w_m,
                           axes_w_mm=sax_w_mm, color="black")

    # Verify the group collides with nothing: not the long colourbar label,
    # and not the panels' rotated easting tick labels above it.
    fig.canvas.draw()
    sbb = sax.get_window_extent()
    clashes = []
    if cbar.ax.xaxis.label.get_window_extent().overlaps(sbb):
        clashes.append("the colourbar label")
    for lab in ax_last.get_xticklabels() + ax_last.get_yticklabels():
        if lab.get_window_extent().overlaps(sbb):
            clashes.append(f"panel (c) tick label {lab.get_text()!r}")
    print("scale bar + north arrow placement: "
          + ("no collisions" if not clashes
             else "COLLIDES WITH " + ", ".join(clashes)))

    out_pdf = FIGURES / "fig_flow_sequence_basinwide.pdf"
    out_png = FIGURES / "fig_flow_sequence_basinwide.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    plt.close(fig)

    # the caption says the panels share extent and scale -- check it
    assert len(set(extents)) == 1, (
        "panels do NOT share one extent, so a single scale bar would be "
        f"wrong: {set(extents)}")
    print(f"wrote {out_pdf} and .png\n")
    print(f"All {len(extents)} panels verified on one common extent "
          f"{extents[0]} -- single black scale bar + north arrow on the "
          f"legend line.")
    print("Saturation (cells with discharge > common scale max "
          f"{VMAX} m, of valid non-null cells within the basin):")
    for sensor, n_valid, n_sat, pct_sat in saturated_report:
        print(f"  {sensor}: {n_sat:,} / {n_valid:,} cells saturated ({pct_sat:.2f}%)")

    log("31_fig_flow_sequence_basinwide.py run. Common log colour scale "
        f"vmin={VMIN}, vmax={VMAX} (= max of the two lidar epochs' 99.9th "
        "percentile: 0.480 m in 2020, 0.387 m in 2024 lidar). Saturated "
        "cells (> vmax, common scale): " +
        "; ".join(f"{s}={n}/{v} ({p:.2f}%)" for s, v, n, p in saturated_report) +
        ". Zones Z1-Z3 (flow_change_zones.csv) outlined on all three panels "
        "at identical positions. ONE scale bar + north arrow, in black, on the "
        "legend line beside the colourbar (not inside any panel); "
        "all three panels verified on a single common extent. CAP SfM date "
        f"corrected to 5 Oct 2024 (~8 days post-event). Written to "
        f"{out_pdf.name}/.png.",
        run="flow_sequence_basinwide")


if __name__ == "__main__":
    main()
