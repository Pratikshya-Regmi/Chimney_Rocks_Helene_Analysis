#!/usr/bin/env python3
"""
32_fig_flow_sequence_zoom.py
=================================
FIGURE 2 of the three-epoch overland-flow rebuild: zoom grid, one row per
zone of largest 2020-vs-2024-lidar flow-path change (Z1-Z3, from
30_flow_sequence_setup.py / results/tables/flow_change_zones.csv), three
columns for the three epochs in the same chronological order as Figure 1
(2020 lidar / 2024 CAP SfM / 2024 lidar), at each zone's own padded bounding
box -- small enough (~85-150 m across) that individual channel threads are
legible.

Same common log discharge colour scale as Figure 1 (vmin=1e-4, vmax=0.48 m)
so panel-to-panel colour is directly comparable across both figures.

SCALE BARS: one per ROW, not one per panel, and NOT one for the figure. The
three zones are different physical sizes (Z1 116x93 m, Z2 115x148 m, Z3
108x85 m), so with a fixed column width the rendered scale differs by row --
about 0.47, 0.37 and 0.50 mm per map metre respectively, a spread of ~34%. A
single figure-wide scale bar would therefore be wrong for two rows out of
three. Within a row all three panels ARE the same zone at the same extent,
so the per-panel repeats there carried no information and are removed; the
row's scale bar sits on its rightmost panel. The north arrow is the same for
every panel, so it is drawn once, on the top-right panel.

Requires 30_flow_sequence_setup.py and 31_fig_flow_sequence_basinwide.py's
GeoTIFF exports.

Outputs:
  results/figures/fig_flow_sequence_zoom.pdf / .png
  results/logs/flow_sequence_zoom_<date>.log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import log  # noqa: E402
from map_style import style_map_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

DISCHARGE_2020_TIF = FIGURES / "_flow_discharge_2020.tif"
DISCHARGE_CAP_TIF = FIGURES / "_flow_discharge_2024cap.tif"
DISCHARGE_2024_TIF = FIGURES / "_flow_discharge_2024lidar.tif"
HILLSHADE_TIF = FIGURES / "_flow_hillshade.tif"

MM = 1 / 25.4
FIG_W_MM = 174

VMIN = 1e-4
VMAX = 0.48

COLUMNS = [
    ("2020 lidar (pre-event)", DISCHARGE_2020_TIF),
    ("2024 CAP SfM (5 Oct, +8 d)", DISCHARGE_CAP_TIF),
    ("2024 lidar (15-16 Nov, +7 wk)", DISCHARGE_2024_TIF),
]


def read_window(path, e_min, e_max, n_min, n_max):
    with rasterio.open(path) as src:
        window = rasterio.windows.from_bounds(e_min, n_min, e_max, n_max,
                                              transform=src.transform)
        arr = src.read(1, window=window, masked=True)
        win_transform = src.window_transform(window)
        b = rasterio.windows.bounds(window, src.transform)
    return arr, (b[0], b[2], b[1], b[3])


def main():
    if not DISCHARGE_2020_TIF.exists():
        sys.exit("Run 30_flow_sequence_setup.py first.")
    zones = pd.read_csv(TABLES / "flow_change_zones.csv")

    n_zones = len(zones)
    fig = plt.figure(figsize=(FIG_W_MM * MM, (n_zones * 62 + 14) * MM))
    gs_fig = fig.add_gridspec(n_zones + 1, 3,
                              height_ratios=[1] * n_zones + [0.06],
                              hspace=0.38, wspace=0.10)

    im = None
    for row, (_, z) in enumerate(zones.iterrows()):
        e_min, e_max = z["e_min"], z["e_max"]
        n_min, n_max = z["n_min"], z["n_max"]
        hs, hs_extent = read_window(HILLSHADE_TIF, e_min, e_max, n_min, n_max)

        for col, (col_label, tif) in enumerate(COLUMNS):
            ax = fig.add_subplot(gs_fig[row, col])
            arr, extent = read_window(tif, e_min, e_max, n_min, n_max)

            ax.imshow(hs, cmap="gray", extent=hs_extent, origin="upper", zorder=1)
            arr_clipped = np.clip(arr, VMIN, VMAX)
            im = ax.imshow(arr_clipped, cmap="YlGnBu",
                           norm=LogNorm(vmin=VMIN, vmax=VMAX), extent=extent,
                           origin="upper", alpha=0.88, zorder=2)

            # scale bar once per ROW (rows are at different scales, see the
            # module docstring); north arrow once for the whole figure
            style_map_panel(ax, extent, graticule_step=50, scale_color="white",
                            draw_scale=(col == len(COLUMNS) - 1),
                            draw_north=(row == 0 and col == len(COLUMNS) - 1))

            if row == 0:
                ax.text(0.5, 1.15, col_label, transform=ax.transAxes,
                       ha="center", va="bottom", fontsize=6.8,
                       fontweight="bold", clip_on=False, zorder=21)
            if col == 0:
                ax.text(-0.38, 0.5, z["zone_id"], transform=ax.transAxes,
                       ha="center", va="center", fontsize=9, fontweight="bold",
                       color="red", clip_on=False, zorder=21, rotation=90)

    fig.subplots_adjust(top=0.90)
    cax = fig.add_subplot(gs_fig[n_zones, :])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Unit discharge [log scale, m$^3$ s$^{-1}$ m$^{-1}$-equivalent] "
                   "-- values above the scale maximum are saturated", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2.5, width=0.5)

    out_pdf = FIGURES / "fig_flow_sequence_zoom.pdf"
    out_png = FIGURES / "fig_flow_sequence_zoom.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf} and .png")

    zone_desc = "; ".join(
        f"{z['zone_id']}=E[{z['e_min']:.0f},{z['e_max']:.0f}]"
        f"N[{z['n_min']:.0f},{z['n_max']:.0f}]" for _, z in zones.iterrows())
    log("32_fig_flow_sequence_zoom.py run. Zoom grid, "
        f"{n_zones} rows (zones) x 3 columns (epochs), same log colour "
        f"scale as Figure 1 (vmin={VMIN}, vmax={VMAX}). Zone extents: "
        f"{zone_desc}. Scale bar once per ROW (rows are at different scales: "
        "Z1/Z2/Z3 render at ~0.47/0.37/0.50 mm per map metre, so a single "
        "figure-wide bar would be wrong for two rows); north arrow once, "
        "top-right panel. CAP SfM date corrected to 5 Oct 2024 (+8 d). "
        f"Written to {out_pdf.name}/.png.",
        run="flow_sequence_zoom")


if __name__ == "__main__":
    main()
