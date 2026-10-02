#!/usr/bin/env python3
"""
make_transect_figure_v3.py -- same content as make_transect_figure_v2.py's
locator map + elevation-profile figure, restyled to match GRASS's own
documentation-figure conventions via the shared scripts/map_style.py:

  - coordinate graticule (easting/northing) along all four map edges
  - north arrow + scale bar, bottom-right, inside the map frame
  - a horizontal colourbar strip BELOW the map, "Elevation change [m]",
    4-5 ticks
  - thin border box, opaque white background

This does NOT touch fig_transects.pdf/.png (v2's output) -- it saves under
new names (fig_transects_gdal_style.pdf/.png) specifically so both versions
exist side by side for comparison. All GRASS export and data-loading logic
is reused from make_transect_figure_v2.py rather than duplicated; only the
map panel's drawing/framing differs.

Usage (needs the project's conda env -- matplotlib/pandas/rasterio are not
in the base GRASS Python environment):
    /home/pregmi3/mambaforge/envs/helene_dec1/bin/python3 \\
        agent/scripts/make_transect_figure_v3.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_style import style_map_panel, draw_transects  # noqa: E402
from make_transect_figure_v2 import (  # noqa: E402
    export_map_rasters, load_tables, read_raster, draw_profile_pair,
    ORTHO_TIF, HILLSHADE_TIF, DOD_TIF, ORTHO_ALPHA, SCRIM_COLOR,
    SCRIM_ALPHA, BASIN_LABEL, C_EROSION, C_DEPOSITION, LOD,
    DOUBLE_COL, FIGURES,
)


# ==========================================================================
def draw_locator_v3(ax, geom, boundary, extra_specs=None):
    """Same map content as v2's draw_locator (hillshade, blended orthophoto,
    thresholded DoD, scrim outside the boundary, transects), but returns the
    DoD's imshow mappable so the caller can attach a colourbar, and leaves
    all framing (graticule / north arrow / scale bar / border) to
    map_style.style_map_panel().

    `extra_specs`: additional (transect_name, start_label, end_label)
    tuples to draw alongside A/B/C -- e.g. v4 passes D-D' here so its label
    gets resolved against A/B/C's labels for collisions too, instead of
    being placed afterwards with no visibility into where they landed.
    """
    hill, hill_ext = read_raster(HILLSHADE_TIF)
    ortho, ortho_ext = read_raster(ORTHO_TIF)
    dod, dod_ext = read_raster(DOD_TIF)
    assert hill_ext == ortho_ext == dod_ext, "map rasters are not co-registered"

    ax.imshow(hill[0], cmap="gray", vmin=0, vmax=255, extent=hill_ext,
             origin="upper", zorder=1)

    ortho_img = np.transpose(ortho, (1, 2, 0)).astype(float) / 255.0
    ax.imshow(ortho_img, extent=ortho_ext, origin="upper", alpha=ORTHO_ALPHA,
             zorder=2)

    # DoD as a real colormap + norm (not a hand-built RGBA array) so it can
    # feed a colourbar -- erosion / white(masked-through-alpha) / deposition,
    # symmetric about zero. Values beyond +/-3 m are rare and clipped with
    # the colourbar's "extend" arrows rather than blowing out the scale.
    cmap = LinearSegmentedColormap.from_list(
        "erosion_deposition", [C_EROSION, "#f7f7f7", C_DEPOSITION])
    cmap.set_bad(color=(0, 0, 0, 0))  # masked (below LoD) -> fully transparent
    norm = Normalize(vmin=-3, vmax=3)
    im = ax.imshow(dod[0], extent=dod_ext, origin="upper", cmap=cmap,
                  norm=norm, zorder=3, alpha=0.9)

    # scrim over everything OUTSIDE the basin boundary (see v2 for the
    # winding-order reasoning)
    b = boundary.sort_values("vertex_order")
    ring = np.column_stack([b["easting"].values, b["northing"].values])
    signed_area = np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
    outer = np.array([[hill_ext[0], hill_ext[2]], [hill_ext[1], hill_ext[2]],
                     [hill_ext[1], hill_ext[3]], [hill_ext[0], hill_ext[3]],
                     [hill_ext[0], hill_ext[2]]])
    inner = ring if signed_area < 0 else ring[::-1]
    verts = np.concatenate([outer, inner])
    codes = ([MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
            + [MplPath.MOVETO] + [MplPath.LINETO] * (len(inner) - 2)
            + [MplPath.CLOSEPOLY])
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=SCRIM_COLOR,
                          alpha=SCRIM_ALPHA, edgecolor="none", zorder=4))
    ax.plot(b["easting"], b["northing"], color="black", linewidth=1.1, zorder=5)

    specs = [("transect1_crosssection", "A", "A'"),
            ("transect2_longitudinal", "B", "B'"),
            ("transect3_westcorridor", "C", "C'")]
    if extra_specs:
        specs += extra_specs
    draw_transects(ax, geom, specs)

    # No corridor inset here (unlike v2) -- redundant with the separate
    # location map the manuscript already has.

    return im, hill_ext


# ==========================================================================
def main():
    print("Exporting GRASS map-panel rasters...")
    reg = export_map_rasters()
    print(f"  region: n={reg['n']} s={reg['s']} e={reg['e']} w={reg['w']} "
          f"res={reg['nsres']}m rows={reg['rows']} cols={reg['cols']}")

    t1, t2, t3, geom, boundary = load_tables()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 1.65))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.35], wspace=0.30)

    ax_map = fig.add_subplot(outer[0, 0])
    im, extent = draw_locator_v3(ax_map, geom, boundary)
    avoid_bbox = (boundary["easting"].min(), boundary["easting"].max(),
                 boundary["northing"].min(), boundary["northing"].max())
    # style_map_panel creates the colourbar axes itself (make_axes_locatable)
    # snug against the map's actual aspect-shrunk box -- see map_style.py.
    style_map_panel(ax_map, extent, mappable=im,
                    cbar_label="Elevation change", cbar_units="m",
                    basin_label=BASIN_LABEL, avoid_bbox=avoid_bbox)

    right = outer[0, 1].subgridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.45)
    pair1 = right[0, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
    pair2 = right[1, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
    pair3 = right[2, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
    ax_e1 = fig.add_subplot(pair1[0, 0])
    ax_d1 = fig.add_subplot(pair1[1, 0], sharex=ax_e1)
    ax_e2 = fig.add_subplot(pair2[0, 0])
    ax_d2 = fig.add_subplot(pair2[1, 0], sharex=ax_e2)
    ax_e3 = fig.add_subplot(pair3[0, 0])
    ax_d3 = fig.add_subplot(pair3[1, 0], sharex=ax_e3)

    stats1 = draw_profile_pair(ax_e1, ax_d1, t1, "A–A′  cross-section")
    stats2 = draw_profile_pair(ax_e2, ax_d2, t2, "B–B′  longitudinal profile")
    stats3 = draw_profile_pair(ax_e3, ax_d3, t3, "C–C′  longitudinal profile (western corridor)")
    ax_e1.legend(loc="lower right", ncol=2, handlelength=1.4, columnspacing=1.0)

    out_pdf = FIGURES / "fig_transects_gdal_style.pdf"
    out_png = FIGURES / "fig_transects_gdal_style.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    print(f"\n  wrote {out_pdf} and .png  (fig_transects.pdf/.png from v2 left untouched)")

    print("\nCaption values:")
    for label, s in [("A-A' (cross-section)", stats1),
                     ("B-B' (longitudinal)", stats2),
                     ("C-C' (western corridor, traced sinuous path)", stats3)]:
        print(f"  {label}: length={s['length_m']:.1f} m, relief={s['relief_m']:.1f} m, "
              f"max lowering={s['max_lowering_m']:+.2f} m, "
              f"max raising={s['max_raising_m']:+.2f} m, "
              f"{s['pct_above_lod']:.1f}% of {s['n']} samples exceed the "
              f"{LOD} m LoD")


if __name__ == "__main__":
    main()
