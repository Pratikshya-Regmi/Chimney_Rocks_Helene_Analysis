#!/usr/bin/env python3
"""
make_stable_mask_figure.py -- replaces the manuscript's old Figure 4 (nine
digitized stable points on paved surfaces). The revised Methods derives
detection limits from a distributed stable-terrain mask, not a handful of
points, so this figure shows that mask instead: watershed (left) and Lake
Lure (right), each as the stable-terrain cells (spectral stability, >30 m
from the channel, slope <45 deg, water excluded) over its CAP/aerial
orthophoto, GRASS documentation-style framing (map_style.py).

Reuses already-exported GeoTIFFs where they exist (cat34_cap_ortho_2024.tif,
cat34_hillshade_2020.tif from make_transect_figure_v2.py;
_lure_ortho_2024.tif, _lure_stable_mask.tif from
05_lure_stable_mask_validation.py) and exports only what's missing (the
watershed's own stable_mask, not previously exported as a standalone
GeoTIFF).

Usage (needs the project's conda env):
    /home/pregmi3/mambaforge/envs/helene_dec1/bin/python3 \\
        agent/scripts/make_stable_mask_figure.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402
from map_style import style_map_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
DIAGNOSTICS = ROOT / "results" / "diagnostics"

WSHED_ORTHO_TIF = FIGURES / "cat34_cap_ortho_2024.tif"
WSHED_HILLSHADE_TIF = FIGURES / "cat34_hillshade_2020.tif"
WSHED_MASK_TIF = FIGURES / "cat34_stable_mask.tif"
WSHED_BOUNDARY_CSV = TABLES / "cat34_boundary.csv"

LURE_ORTHO_TIF = DIAGNOSTICS / "_lure_ortho_2024.tif"
LURE_MASK_TIF = DIAGNOSTICS / "_lure_stable_mask.tif"

MM = 1 / 25.4
FIG_W_MM = 174
FIG_H_MM = 100

MASK_COLOR = (0.0, 0.55, 0.35)  # teal-green, reads clearly over both an
                                 # orthophoto and a hillshade/ortho blend
MASK_ALPHA = 0.55


def export_watershed_mask():
    gs, _ = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")
    cat34_bbox = dict(gs.parse_command("g.region", flags="g"))
    buf = 200
    gs.run_command("g.region",
                   n=float(cat34_bbox["n"]) + buf, s=float(cat34_bbox["s"]) - buf,
                   e=float(cat34_bbox["e"]) + buf, w=float(cat34_bbox["w"]) - buf,
                   res=1, flags="a")
    n = gs.parse_command("r.univar", map="stable_mask", flags="g").get("n")
    gs.run_command("r.out.gdal", input="stable_mask", output=str(WSHED_MASK_TIF),
                   format="GTiff", type="Byte", createopt="COMPRESS=LZW",
                   nodata=0, overwrite=True, quiet=True)
    log(f"make_stable_mask_figure: exported watershed stable_mask "
        f"({WSHED_MASK_TIF.name}), n={n} cells in the buffered region.",
        run="stable_mask_figure")
    print(f"  watershed stable_mask: n={n} cells (buffered region)")


def read_raster(path, band=None):
    with rasterio.open(path) as src:
        if band is not None:
            arr = src.read(band, masked=True)
        else:
            arr = src.read(masked=True)
        bounds = src.bounds
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return arr, extent


def draw_panel(ax, ortho_tif, mask_tif, mask_band, boundary_xy, label):
    ortho, extent = read_raster(ortho_tif)
    rgb = np.transpose(ortho[:3], (1, 2, 0)).astype(float) / 255.0
    ax.imshow(rgb, extent=extent, origin="upper", zorder=1)

    mask, mask_extent = read_raster(mask_tif, band=mask_band)
    overlay = np.zeros((*mask.shape, 4))
    on = (~mask.mask) & (mask.data > 0) if hasattr(mask, "mask") else mask > 0
    overlay[..., 0] = MASK_COLOR[0]
    overlay[..., 1] = MASK_COLOR[1]
    overlay[..., 2] = MASK_COLOR[2]
    overlay[..., 3] = np.where(on, MASK_ALPHA, 0.0)
    ax.imshow(overlay, extent=mask_extent, origin="upper", zorder=3)

    if boundary_xy is not None:
        ax.plot(boundary_xy[0], boundary_xy[1], color="black", linewidth=0.9,
               zorder=4)

    ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top",
           fontsize=8, fontweight="bold", color="white", zorder=10,
           bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55,
                     edgecolor="none"))
    return extent


def main():
    if not WSHED_MASK_TIF.exists():
        export_watershed_mask()
    else:
        print(f"  reusing existing {WSHED_MASK_TIF.name}")

    import pandas as pd
    boundary = pd.read_csv(WSHED_BOUNDARY_CSV)
    boundary_xy = (boundary["easting"], boundary["northing"])

    fig, (ax_w, ax_l) = plt.subplots(1, 2, figsize=(FIG_W_MM * MM, FIG_H_MM * MM))

    ext_w = draw_panel(ax_w, WSHED_ORTHO_TIF, WSHED_MASK_TIF, 1, boundary_xy,
                       "(a) Watershed")
    style_map_panel(ax_w, ext_w, basin_label=None)

    ext_l = draw_panel(ax_l, LURE_ORTHO_TIF, LURE_MASK_TIF, 1, None,
                       "(b) Lake Lure")
    style_map_panel(ax_l, ext_l, basin_label=None)

    fig.subplots_adjust(wspace=0.32)

    out_pdf = FIGURES / "fig_stable_mask.pdf"
    out_png = FIGURES / "fig_stable_mask.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out_pdf} and .png")


if __name__ == "__main__":
    main()
