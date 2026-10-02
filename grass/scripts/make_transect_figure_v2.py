#!/usr/bin/env python3
"""
make_transect_figure_v2.py -- locator map + elevation-profile figure for the
two geomorphic transects in basin cat34 (Regmi et al., Hickory Nut Gorge).

Produces a single 3-panel figure:

  Left   locator map -- extends ~200 m (MAP_BUFFER_M) past the cat34 boundary
         on every side for context, hillshade of dtm_2020_filled with the
         2024 CAP orthophoto blended over it (ORTHO_ALPHA), the co-registered
         lidar-lidar DoD (coreg_dh_corrected) overlaid where |change| exceeds
         the 0.32 m LoD (red = erosion, blue = deposition), a scrim dimming
         everything OUTSIDE the cat34 boundary so the analysed basin reads as
         foregrounded, the boundary itself as a clear line, a "Basin cat34"
         label, both transects labelled A-A' and B-B', a north arrow, a scale
         bar, and a small inset showing the basin's position within the
         wider study corridor (hull_mask extent).

  Right  two stacked profile pairs (A-A' = cross-section, B-B' = longitudinal
         flow path). Each pair is an upper/lower panel:
           upper: 2020 and 2024 surfaces, area between them shaded red where
                  ground was lost and blue where gained
           lower: elevation difference (2024-2020), with a grey band at
                  +/-0.32 m marking the level of detection

WHY THE PAIRED UPPER/LOWER DESIGN: relief along these transects (27 m and
111 m) is one to two orders of magnitude larger than the change signal
(1-3 m). Plotting the 2020/2024 surfaces and the difference on one axis would
flatten the difference to a line at zero. This is not a style choice.

This script does two things that must both succeed:
  1. GRASS raster export -- opens a GRASS session (via grass_env.py), sets
     the region to cat34 at 1 m resolution, and exports/re-exports the three
     GeoTIFFs the map panel needs to results/figures/, all on the identical
     grid so they register exactly.
  2. Figure assembly -- reads those GeoTIFFs (rasterio) and the transect/
     geometry CSVs (results/tables/) and builds the publication figure
     (matplotlib).

Re-run this script whenever the underlying GRASS data changes (a new
coreg_dh_corrected, a new transect, etc.) -- nothing here is cached.

Specifications targeted (Natural Hazards / Springer):
  - vector PDF (Type 42 embedded fonts) + 600 dpi PNG preview
  - double column, 174 mm wide
  - sans-serif; body 8 pt, tick labels 7 pt, nothing below 6 pt
  - Okabe-Ito colour-blind-safe palette: erosion #D55E00, deposition #0072B2
  - no top/right spines

Usage (needs the project's conda env -- matplotlib/pandas/rasterio are not
in the base GRASS Python environment):
    /home/pregmi3/mambaforge/envs/helene_dec1/bin/python3 \\
        agent/scripts/make_transect_figure_v2.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402
from map_style import draw_transects  # noqa: E402

# -------------------------------------------------------------- fixed paths
# Anchored to this file's location, not the caller's cwd -- this project has
# already been bitten once by relative "results/..." paths resolving
# differently depending on where a script was launched from.
ROOT = Path(__file__).resolve().parents[2]        # .../helene-agent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

ORTHO_TIF = FIGURES / "cat34_cap_ortho_2024.tif"
HILLSHADE_TIF = FIGURES / "cat34_hillshade_2020.tif"
DOD_TIF = FIGURES / "cat34_dod_lidar_lidar_thresholded.tif"
INSET_ORTHO_TIF = FIGURES / "corridor_ortho_2020_inset.tif"
INSET_HULLMASK_TIF = FIGURES / "corridor_hullmask_inset.tif"

LOD = 0.32  # m, manuscript's uniform lidar-lidar level of detection

# --------------------------------------------------------- tunable knobs
# Everything a reader/editor is likely to want to tweak without touching the
# plotting code lives here.
MAP_BUFFER_M = 200        # context shown beyond the cat34 boundary, all sides
ORTHO_ALPHA = 0.55         # orthophoto opacity over the hillshade (0=hidden, 1=opaque).
                           # Matches fig_transects_gdal_style.png (v3): at
                           # 0.55 the hillshade still dominates the blend,
                           # giving the basin interior a darker, more
                           # textured tone that the DoD colours pop against.
SCRIM_COLOR = "white"      # tints everything OUTSIDE the basin boundary.
                           # v3's contrast runs the OPPOSITE way from an
                           # earlier round's assumption here: the exterior is
                           # the LIGHTER of the two (flattened, whitened by
                           # this scrim), and the basin interior is the
                           # DARKER, more detailed one (plain hillshade+ortho
                           # blend at ORTHO_ALPHA=0.55, no scrim on it at
                           # all) -- that's what gives the clean visual
                           # separation the reference figure has. A dark
                           # exterior scrim was tried and reads as a
                           # different (also legitimate) style, but not this
                           # one.
SCRIM_ALPHA = 0.55         # matches fig_transects_gdal_style.png.
BASIN_LABEL = None  # no text label on the map -- the caption identifies the study unit
SHOW_INSET = True          # small "where is this basin" inset, wider corridor
INSET_RES_M = 5            # coarser resolution keeps the inset GeoTIFF small/fast
INSET_BOX = (0.02, 0.02, 0.40, 0.40)  # (x, y, w, h) in axes fraction, lower-left

# ------------------------------------------------------------------ style
MM = 1 / 25.4
DOUBLE_COL = 174 * MM

C_EROSION = "#D55E00"      # Okabe-Ito vermillion
C_DEPOSITION = "#0072B2"   # Okabe-Ito blue
C_2020 = "#4D4D4D"
C_2024 = "#000000"
C_LOD_BAND = "0.85"

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
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.pad_inches": 0.02,
})


# ==========================================================================
# STEP 1 -- GRASS raster export
# ==========================================================================
def export_map_rasters():
    """(Re)export the GeoTIFFs the locator map needs, all on the same
    buffered-cat34 grid so they register exactly. Returns the region dict."""
    gs, _ = grass_session(project="DEM_generation", mapset="for_codem")
    mapsets = gs.read_command("g.mapsets", flags="l").split()
    gs.run_command("g.mapsets", mapset=",".join(mapsets), operation="set")

    # cat34's own bbox first, then widen it by MAP_BUFFER_M on every side so
    # the map shows surrounding terrain instead of cropping at the boundary.
    gs.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1,
                   flags="a")
    cat34_bbox = dict(gs.parse_command("g.region", flags="g"))
    gs.run_command("g.region",
                   n=float(cat34_bbox["n"]) + MAP_BUFFER_M,
                   s=float(cat34_bbox["s"]) - MAP_BUFFER_M,
                   e=float(cat34_bbox["e"]) + MAP_BUFFER_M,
                   w=float(cat34_bbox["w"]) - MAP_BUFFER_M,
                   res=1, flags="a")
    reg = dict(gs.parse_command("g.region", flags="g"))

    # sanity check: which pipeline is coreg_dh_corrected currently? (this
    # script does NOT enforce a specific value -- it's meant to reflect
    # whatever is currently on disk -- but prints it so a human can tell)
    gs.mapcalc("_chk = if(!isnull(stable_mask), coreg_dh_corrected, null())",
              overwrite=True, quiet=True)
    med = gs.parse_command("r.univar", map="_chk", flags="ge", quiet=True).get("median")
    gs.run_command("g.remove", type="raster", name="_chk", flags="f", quiet=True)
    print(f"  coreg_dh_corrected stable-terrain median right now: {float(med):+.4f} m "
          f"(lidar-lidar pipeline validated value was +0.0127 m -- if this "
          f"doesn't match, coreg_dh_corrected reflects a different run)")

    # 1. 2024 CAP orthophoto, RGB, clipped to the region above
    # i.group APPENDS to an existing group rather than replacing its
    # members -- remove first so re-running this script doesn't silently
    # accumulate stale (and wrongly-typed) members from a prior run.
    gs.run_command("g.remove", type="group", name="cap_ortho_grp", flags="f",
                   quiet=True)
    gs.run_command("i.group", group="cap_ortho_grp",
                   input="A0069A_data_upload_flightA0069_ortho_full.1@PERMANENT,"
                         "A0069A_data_upload_flightA0069_ortho_full.2@PERMANENT,"
                         "A0069A_data_upload_flightA0069_ortho_full.3@PERMANENT",
                   quiet=True)
    gs.run_command("r.out.gdal", input="cap_ortho_grp", output=str(ORTHO_TIF),
                   format="GTiff", type="Byte", createopt="COMPRESS=LZW",
                   overwrite=True, quiet=True)

    # 2. hillshade of dtm_2020_filled -- r.relief can produce rare edge-of-DEM
    # values outside 0-255 (seen: min -155); clamp before export.
    gs.run_command("r.relief", input="dtm_2020_filled@for_codem",
                   output="hillshade_2020", overwrite=True, quiet=True)
    gs.mapcalc("hillshade_2020_scaled = if(hillshade_2020 < 0, 0, "
              "if(hillshade_2020 > 255, 255, hillshade_2020))",
              overwrite=True, quiet=True)
    gs.run_command("r.out.gdal", input="hillshade_2020_scaled",
                   output=str(HILLSHADE_TIF), format="GTiff", type="Byte",
                   createopt="COMPRESS=LZW", overwrite=True, quiet=True)

    # 3. co-registered lidar-lidar DoD, masked to |dh| > LoD AND to the true
    # basin polygon (cat34_mask). coreg_dh_corrected's own native computational
    # region is a plain rectangle (confirmed via r.what: valid from
    # y=192580 northward at every x from 313200-313650) that does not follow
    # cat34's jagged real boundary -- without the cat34_mask term, DoD colour
    # renders in that rectangle's corners beyond the true polygon, and shows
    # through the scrim there (the scrim dims, at partial alpha, it does not
    # erase) as a hard straight edge cutting across the map near N192,580.
    gs.mapcalc(f"dod_locator = if(!isnull(cat34_mask) && "
              f"abs(coreg_dh_corrected) > {LOD}, coreg_dh_corrected, null())",
              overwrite=True, quiet=True)
    gs.run_command("r.out.gdal", input="dod_locator", output=str(DOD_TIF),
                   format="GTiff", type="Float64", createopt="COMPRESS=LZW",
                   nodata=-9999, overwrite=True, quiet=True)

    # 4. inset background -- the wider study corridor (hull_mask extent), so
    # the reader can see where cat34 sits within it. Coarser resolution: this
    # is a small thumbnail, not a measurement surface.
    if SHOW_INSET:
        gs.run_command("g.region", raster="hull_mask@DTM_DSM", res=INSET_RES_M,
                       flags="a")
        # Ortho_NAIP_2020_rgb@DTM_DSM is an r.composite PACKED single band
        # (levels=32, values up to 32767) -- not true RGB. Use the original
        # separate bands it was built from instead (already ~0-255 range,
        # but stored as DCELL -- round to Byte before export).
        for b in ("1", "2", "3"):
            gs.mapcalc(f"naip_2020_b{b} = round(min(max(Ortho_NAIP_2020.{b}@DTM_DSM, 0), 255))",
                      overwrite=True, quiet=True)
        gs.run_command("g.remove", type="group", name="naip_2020_grp", flags="f",
                       quiet=True)
        gs.run_command("i.group", group="naip_2020_grp",
                       input="naip_2020_b1,naip_2020_b2,naip_2020_b3",
                       quiet=True)
        gs.run_command("r.out.gdal", input="naip_2020_grp",
                       output=str(INSET_ORTHO_TIF), format="GTiff", type="Byte",
                       createopt="COMPRESS=LZW", overwrite=True, quiet=True)
        gs.run_command("r.out.gdal", input="hull_mask@DTM_DSM",
                       output=str(INSET_HULLMASK_TIF), format="GTiff",
                       type="Byte", createopt="COMPRESS=LZW", nodata=0,
                       overwrite=True, quiet=True)
        # restore the buffered cat34 region -- not strictly needed again in
        # this function, but leaves the mapset's region in the state the
        # main exports above assume, for anyone extending this script.
        gs.run_command("g.region", n=reg["n"], s=reg["s"], e=reg["e"],
                       w=reg["w"], nsres=reg["nsres"], ewres=reg["ewres"])

    log(f"make_transect_figure_v2: exported map-panel GeoTIFFs. Region: "
        f"n={reg['n']} s={reg['s']} e={reg['e']} w={reg['w']} "
        f"nsres={reg['nsres']} ewres={reg['ewres']} rows={reg['rows']} "
        f"cols={reg['cols']}. coreg_dh_corrected stable-median={float(med):+.4f} m.",
        run="mappanel")
    return reg


# ==========================================================================
# STEP 2 -- data loading
# ==========================================================================
def load_tables():
    """A-A' (cross-section), B-B' (longitudinal, main scar), C-C' (longitudinal,
    western corridor -- a traced sinuous path, not a straight line; see
    transect3_westcorridor.csv's header note). The old C-C' channel
    cross-section (transect3_channel.csv) is dropped from the figure/map per
    task instructions but its CSV is kept on disk for reference."""
    t1 = pd.read_csv(TABLES / "transect1_crosssection.csv")
    t2 = pd.read_csv(TABLES / "transect2_longitudinal.csv")
    t3 = pd.read_csv(TABLES / "transect3_westcorridor.csv", comment="#")
    geom = pd.read_csv(TABLES / "transect_geometry.csv")
    boundary = pd.read_csv(TABLES / "cat34_boundary.csv")
    return t1, t2, t3, geom, boundary


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(masked=True)
        bounds = src.bounds  # left, bottom, right, top
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return arr, extent


# ==========================================================================
# STEP 3 -- locator map panel
# ==========================================================================
def draw_locator(ax, geom, boundary):
    hill, hill_ext = read_raster(HILLSHADE_TIF)
    ortho, ortho_ext = read_raster(ORTHO_TIF)
    dod, dod_ext = read_raster(DOD_TIF)
    assert hill_ext == ortho_ext == dod_ext, "map rasters are not co-registered"

    ax.imshow(hill[0], cmap="gray", vmin=0, vmax=255, extent=hill_ext,
             origin="upper", zorder=1)

    ortho_img = np.transpose(ortho, (1, 2, 0)).astype(float) / 255.0
    ax.imshow(ortho_img, extent=ortho_ext, origin="upper", alpha=ORTHO_ALPHA,
             zorder=2)

    # DoD overlay: erosion / deposition, transparent where masked (below LoD)
    dz = dod[0]
    rgba = np.zeros((*dz.shape, 4))
    ero = (~dz.mask) & (dz.data < 0)
    dep = (~dz.mask) & (dz.data > 0)
    rgba[ero] = (*matplotlib.colors.to_rgb(C_EROSION), 0.85)
    rgba[dep] = (*matplotlib.colors.to_rgb(C_DEPOSITION), 0.85)
    ax.imshow(rgba, extent=dod_ext, origin="upper", zorder=3)

    # scrim over everything OUTSIDE the basin boundary, so the analysed area
    # reads as foregrounded while the surrounding context stays visible.
    # Built as a single compound path: the full map rectangle wound one way,
    # the basin ring wound the OPPOSITE way, so matplotlib's nonzero winding
    # fill rule punches a hole for the basin interior automatically.
    b = boundary.sort_values("vertex_order")
    ring = np.column_stack([b["easting"].values, b["northing"].values])
    # shoelace signed area: positive = CCW
    signed_area = np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
    outer = np.array([[hill_ext[0], hill_ext[2]], [hill_ext[1], hill_ext[2]],
                     [hill_ext[1], hill_ext[3]], [hill_ext[0], hill_ext[3]],
                     [hill_ext[0], hill_ext[2]]])  # CCW rectangle
    inner = ring if signed_area < 0 else ring[::-1]  # force opposite winding
    verts = np.concatenate([outer, inner])
    codes = ([MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
            + [MplPath.MOVETO] + [MplPath.LINETO] * (len(inner) - 2)
            + [MplPath.CLOSEPOLY])
    scrim_path = MplPath(verts, codes)
    ax.add_patch(PathPatch(scrim_path, facecolor=SCRIM_COLOR, alpha=SCRIM_ALPHA,
                           edgecolor="none", zorder=4))

    # basin boundary, drawn clearly on top of the scrim edge
    ax.plot(b["easting"], b["northing"], color="black", linewidth=1.1, zorder=5)

    # transects, labelled A-A' / B-B' / C-C' -- drawn together (not one at
    # a time) so labels from different transects placed close together by
    # design can be separated from EACH OTHER, not just from their own
    # line. See map_style.draw_transects for why that distinction matters.
    specs = [("transect1_crosssection", "A", "A'"),
            ("transect2_longitudinal", "B", "B'"),
            ("transect3_channel", "C", "C'")]
    draw_transects(ax, geom, specs)

    ax.set_xlim(hill_ext[0], hill_ext[1])
    ax.set_ylim(hill_ext[2], hill_ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    # north arrow
    x0f, y0f = 0.90, 0.90
    ax.annotate("", xy=(x0f, y0f + 0.07), xytext=(x0f, y0f - 0.02),
               xycoords="axes fraction",
               arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.2))
    ax.text(x0f, y0f + 0.09, "N", transform=ax.transAxes, ha="center",
           va="bottom", fontsize=7, fontweight="bold")

    # scale bar (200 m)
    bar_m = 200
    x_e0, x_e1 = hill_ext[0], hill_ext[1]
    margin = 0.05 * (x_e1 - x_e0)
    xbar0 = x_e0 + margin
    xbar1 = xbar0 + bar_m
    ybar = hill_ext[2] + 0.04 * (hill_ext[3] - hill_ext[2])
    ax.plot([xbar0, xbar1], [ybar, ybar], color="black", linewidth=2.2,
           solid_capstyle="butt", zorder=9,
           path_effects=[pe.withStroke(linewidth=3.4, foreground="white")])
    ax.text((xbar0 + xbar1) / 2, ybar + 0.015 * (hill_ext[3] - hill_ext[2]),
           f"{bar_m} m", ha="center", va="bottom", fontsize=6.5, zorder=9)

    # No basin-name text label on the map (removed per analyst request --
    # the caption identifies the study unit).

    if SHOW_INSET:
        draw_inset(ax)


def draw_inset(parent_ax):
    """Small "where is this basin" inset: the wider study corridor
    (hull_mask extent), with the cat34 boundary/location marked."""
    ortho, ortho_ext = read_raster(INSET_ORTHO_TIF)
    mask, mask_ext = read_raster(INSET_HULLMASK_TIF)
    assert ortho_ext == mask_ext

    iax = parent_ax.inset_axes(INSET_BOX)
    ortho_img = np.transpose(ortho, (1, 2, 0)).astype(float) / 255.0
    iax.imshow(ortho_img, extent=ortho_ext, origin="upper")

    # hull_mask outline -- contour the binary mask rather than vectorise it,
    # cheap and exact at the resolution the inset is rendered at
    ny, nx = mask[0].shape
    xs = np.linspace(ortho_ext[0], ortho_ext[1], nx)
    ys = np.linspace(ortho_ext[3], ortho_ext[2], ny)  # top to bottom
    iax.contour(xs, ys, mask[0].filled(0), levels=[0.5], colors="yellow",
               linewidths=0.8)

    # cat34's own footprint, highlighted
    cat34_bbox_e = (313173, 313721)
    cat34_bbox_n = (192579, 193688)
    iax.add_patch(plt.Rectangle(
        (cat34_bbox_e[0], cat34_bbox_n[0]),
        cat34_bbox_e[1] - cat34_bbox_e[0], cat34_bbox_n[1] - cat34_bbox_n[0],
        facecolor="none", edgecolor="red", linewidth=1.0))

    iax.set_xlim(ortho_ext[0], ortho_ext[1])
    iax.set_ylim(ortho_ext[2], ortho_ext[3])
    iax.set_aspect("equal")
    iax.set_xticks([])
    iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("black")


# ==========================================================================
# STEP 4 -- profile pair (upper elevation + lower dh)
# ==========================================================================
def draw_profile_pair(ax_elev, ax_dh, df, title):
    x = df["distance_m"].values
    e20 = df["elev_2020"].values
    e24 = df["elev_2024"].values
    dz = e24 - e20

    # --- upper: elevation surfaces, shaded difference ---
    ax_elev.fill_between(x, e20, e24, where=(e24 < e20), interpolate=True,
                         color=C_EROSION, alpha=0.55, linewidth=0)
    ax_elev.fill_between(x, e20, e24, where=(e24 >= e20), interpolate=True,
                         color=C_DEPOSITION, alpha=0.55, linewidth=0)
    ax_elev.plot(x, e20, color=C_2020, linewidth=1.0, linestyle="--", label="2020")
    ax_elev.plot(x, e24, color=C_2024, linewidth=1.0, label="2024")
    ax_elev.set_ylabel("Elevation (m)")
    ax_elev.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax_elev.tick_params(labelbottom=False)
    ax_elev.margins(x=0.01)

    # --- lower: elevation difference with LoD band ---
    ax_dh.axhspan(-LOD, LOD, color=C_LOD_BAND, zorder=0, linewidth=0)
    ax_dh.axhline(0, color="black", linewidth=0.5, zorder=1)
    ax_dh.plot(x, dz, color="black", linewidth=0.7, zorder=2)
    ax_dh.fill_between(x, dz, LOD, where=(dz > LOD), interpolate=True,
                       color=C_DEPOSITION, alpha=0.85, linewidth=0, zorder=1)
    ax_dh.fill_between(x, dz, -LOD, where=(dz < -LOD), interpolate=True,
                       color=C_EROSION, alpha=0.85, linewidth=0, zorder=1)
    ax_dh.set_ylabel(r"$\Delta z$ (m)")
    ax_dh.set_xlabel("Distance along transect (m)")
    ax_dh.margins(x=0.01)

    return dict(
        length_m=float(x[-1] - x[0]),
        relief_m=float(max(e20.max(), e24.max()) - min(e20.min(), e24.min())),
        max_lowering_m=float(dz.min()),
        max_raising_m=float(dz.max()),
        pct_above_lod=float(100.0 * np.mean(np.abs(dz) > LOD)),
        n=int(len(x)),
    )


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print("Exporting GRASS map-panel rasters...")
    reg = export_map_rasters()
    print(f"  region: n={reg['n']} s={reg['s']} e={reg['e']} w={reg['w']} "
          f"res={reg['nsres']}m rows={reg['rows']} cols={reg['cols']}")

    t1, t2, t3, geom, boundary = load_tables()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 1.55))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.35], wspace=0.28)

    ax_map = fig.add_subplot(outer[0, 0])
    draw_locator(ax_map, geom, boundary)

    # three separate sub-gridspecs (not one long grid) so the gap BETWEEN
    # transect pairs can be generous while the gap WITHIN each pair
    # (elevation panel directly over its own dh panel) stays tight -- a
    # single uniform hspace collided adjacent pairs' titles/tick labels.
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
    stats3 = draw_profile_pair(ax_e3, ax_d3, t3, "C–C′  channel cross-section")

    ax_e1.legend(loc="lower right", ncol=2, handlelength=1.4, columnspacing=1.0)

    fig.savefig(FIGURES / "fig_transects.pdf")
    fig.savefig(FIGURES / "fig_transects.png", dpi=600)
    plt.close(fig)
    print(f"\n  wrote {FIGURES/'fig_transects.pdf'} and .png")

    print("\nCaption values:")
    for label, s in [("A-A' (cross-section)", stats1),
                     ("B-B' (longitudinal)", stats2),
                     ("C-C' (channel cross-section)", stats3)]:
        print(f"  {label}: length={s['length_m']:.1f} m, relief={s['relief_m']:.1f} m, "
              f"max lowering={s['max_lowering_m']:+.2f} m, "
              f"max raising={s['max_raising_m']:+.2f} m, "
              f"{s['pct_above_lod']:.1f}% of {s['n']} samples exceed the "
              f"{LOD} m LoD")


if __name__ == "__main__":
    main()
