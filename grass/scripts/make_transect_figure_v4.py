#!/usr/bin/env python3
"""
make_transect_figure_v4.py -- three-transect version, two-column layout:
the locator map fills the left column, three profile pairs (A-A', B-B',
C-C') stack in the right column. GRASS documentation-style map framing
throughout (scripts/map_style.py).

Transect set (as of this version):
  A-A'  cross-section, main scar
  B-B'  longitudinal profile, main scar (follows the r.drain flow path)
  C-C'  longitudinal profile, western corridor -- a TRACED SINUOUS PATH
        along the strongest continuous DoD erosion signal (not a straight
        line between endpoints; distance_m in its CSV is cumulative path
        length, ~622 m, not straight-line displacement)

The old C-C' channel cross-section (perpendicular, at the channel) has been
dropped from this figure and the map. Its CSV (transect3_channel.csv) is
kept in results/tables/ for reference but nothing here reads it. The old
straight-line western-corridor transect (formerly labelled D-D') has been
replaced by the traced path above and renamed into the C-C' slot, so the
figure runs A-A', B-B', C-C' with no gap and no supplementary figure.

Usage (needs the project's conda env):
    /home/pregmi3/mambaforge/envs/helene_dec1/bin/python3 \\
        agent/scripts/make_transect_figure_v4.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_style import style_map_panel  # noqa: E402
from make_transect_figure_v2 import (  # noqa: E402
    export_map_rasters, load_tables, draw_profile_pair, LOD, FIGURES,
)
from make_transect_figure_v3 import draw_locator_v3  # noqa: E402

MM = 1 / 25.4
FIG_W_MM = 174
FIG_H_MM = 205  # within the ~210 mm cap, small margin

# Map column width chosen so the map's true aspect ratio (buffered cat34,
# ~948 x 1509 m) fills essentially the whole figure height once the
# colourbar strip + basin label margin are accounted for.
MAP_COL_W_MM = 96
PROFILE_COL_W_MM = FIG_W_MM - MAP_COL_W_MM - 6  # 6 mm wspace allowance


def main():
    print("Exporting GRASS map-panel rasters...")
    reg = export_map_rasters()
    print(f"  region: n={reg['n']} s={reg['s']} e={reg['e']} w={reg['w']} "
          f"res={reg['nsres']}m rows={reg['rows']} cols={reg['cols']}")

    t1, t2, t3, geom, boundary = load_tables()

    fig = plt.figure(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))
    outer = fig.add_gridspec(
        1, 2, width_ratios=[MAP_COL_W_MM, PROFILE_COL_W_MM], wspace=0.30)

    ax_map = fig.add_subplot(outer[0, 0])
    im, extent = draw_locator_v3(ax_map, geom, boundary)
    avoid_bbox = (boundary["easting"].min(), boundary["easting"].max(),
                 boundary["northing"].min(), boundary["northing"].max())
    style_map_panel(ax_map, extent, mappable=im,
                    cbar_label="Elevation change", cbar_units="m",
                    basin_label=None,  # no watershed text label on the map
                    avoid_bbox=avoid_bbox)

    right = outer[0, 1].subgridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.45)
    pairs = [right[i, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
            for i in range(3)]
    axes = []
    for p in pairs:
        ax_e = fig.add_subplot(p[0, 0])
        ax_d = fig.add_subplot(p[1, 0], sharex=ax_e)
        axes.append((ax_e, ax_d))

    # title="left"-anchored text is not wrapped/clipped to the axes -- it
    # was measured overflowing past the figure's own right edge at ~45
    # characters ("C-C' longitudinal profile, western corridor"). Keeping
    # every title close to B-B's length (27 chars, confirmed to fit) avoids
    # that; "longitudinal profile" is still explained in the caption text.
    titles = ["A–A′  cross-section", "B–B′  longitudinal profile",
             "C–C′  western corridor"]
    dfs = [t1, t2, t3]
    stats = []
    for (ax_e, ax_d), df, title in zip(axes, dfs, titles):
        stats.append(draw_profile_pair(ax_e, ax_d, df, title))
    # A-A's legend was landing on top of the plotted elevation curves in
    # the lower-right corner of its own panel; A-A' is a cross-section so
    # its curves occupy the full width fairly evenly -- "lower right" had
    # nowhere clear to go. "upper left" for a cross-section (curves peak
    # toward the middle/right of a scar cross-section) leaves the corner
    # clear; verified against the rendered PNG below.
    axes[0][0].legend(loc="upper left", ncol=2, handlelength=1.4, columnspacing=1.0)

    out_pdf = FIGURES / "fig_transects_v4.pdf"
    out_png = FIGURES / "fig_transects_v4.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    print(f"\n  wrote {out_pdf} and .png  ({FIG_W_MM} x {FIG_H_MM} mm, "
          f"map column {MAP_COL_W_MM} mm, profile column {PROFILE_COL_W_MM:.0f} mm, "
          f"3 of 3 profile pairs -- A-A', B-B', C-C', no supplementary figure)")

    print("\nCaption values:")
    labels = ["A-A' (cross-section)", "B-B' (longitudinal, main scar)",
             "C-C' (longitudinal, western corridor -- traced sinuous path, "
             "distance_m is path length, not straight-line displacement)"]
    for label, s in zip(labels, stats):
        print(f"  {label}: length={s['length_m']:.1f} m, relief={s['relief_m']:.1f} m, "
              f"max lowering={s['max_lowering_m']:+.2f} m, "
              f"max raising={s['max_raising_m']:+.2f} m, "
              f"{s['pct_above_lod']:.1f}% of {s['n']} samples exceed the "
              f"{LOD} m LoD")


if __name__ == "__main__":
    main()
