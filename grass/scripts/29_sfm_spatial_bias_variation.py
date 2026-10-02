#!/usr/bin/env python3
"""
29_sfm_spatial_bias_variation.py
====================================
Quantifies the spatial variation in required SfM vertical correction
between the two analysis areas, for the SINGLE CAP SfM raster
(cap_sfm_regis@for_codem) that spans both continuously (N195153-190878,
E318661-310250 -- confirmed to cover both the watershed and Lake Lure).

Two things, per the analyst's request:

1. Precise summary statistics of the residual (cap_sfm_regis minus the
   LOCAL lidar reference each area actually uses -- dtm_2020_filled for
   the watershed, lidar_2017_DTM_lure for Lake Lure), computed separately
   over each area's own stable mask: n, median, NMAD, IQR (p25/p75).
   Supersedes/refines the two ad hoc single-run numbers already on record
   in RESULTS_FOR_PAPER.md (+1.788 m over a stale 46,188-cell watershed
   mask; +12.987 m over the current 678,811-cell Lake Lure mask) -- this
   run uses the CURRENT, correct watershed stable_mask (68,248 cells, not
   the stale 46,188) for the watershed side.

2. Because cap_sfm_regis and dtm_2017 (the one lidar surface that also
   spans both areas continuously) overlap the whole corridor, a continuous
   spatial map of (cap_sfm_regis - dtm_2017) across the corridor, so the
   residual's spatial trend can be read directly rather than inferred from
   two point estimates. This uses ONE common reference (2017) for both
   areas purely to make the map internally consistent; it is NOT a
   replacement for the two area-specific numbers in (1), which correctly
   use each area's own designated pre-event baseline.

Lake Lure's stable_mask_lure and lidar_2017_DTM_lure are reprojected
(r.proj, same CRS EPSG:3358 -- a resample, not a reprojection) into
DEM_generation/for_codem so both areas can be computed and mapped in one
region.

Outputs:
  results/tables/sfm_vertical_bias_by_area.csv
  results/figures/fig_sfm_bias_corridor_map.png / .pdf
  results/logs/sfm_spatial_bias_variation_<date>.log
"""

import csv
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
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

CORRIDOR_TIF = FIGURES / "_sfm_bias_corridor.tif"
HILLSHADE_TIF = FIGURES / "_sfm_bias_hillshade.tif"


def nmad_and_iqr(gs, rmap, zone=None):
    """NMAD = 1.4826 * median(|x - median(x)|), plus p25/p75 (IQR) and n.
    r.univar doesn't give NMAD directly -- two-pass computation."""
    kwargs = {"map": rmap, "flags": "ge", "quiet": True}
    if zone:
        kwargs["zone"] = zone
    u = gs.parse_command("r.univar", **kwargs)
    n = int(u["n"])
    median = float(u["median"])

    gs.mapcalc(f"_nmad_absdev = abs({rmap} - {median})", overwrite=True, quiet=True)
    kwargs2 = {"map": "_nmad_absdev", "flags": "ge", "quiet": True}
    if zone:
        kwargs2["zone"] = zone
    u2 = gs.parse_command("r.univar", **kwargs2)
    mad = float(u2["median"])
    nmad = 1.4826 * mad

    p = {}
    for pct in (25, 75):
        kwargs3 = {"map": rmap, "flags": "ge", "quiet": True, "percentile": pct}
        if zone:
            kwargs3["zone"] = zone
        u3 = gs.parse_command("r.univar", **kwargs3)
        key = [k for k in u3 if k.startswith("percentile_")][0]
        p[pct] = float(u3[key])

    gs.run_command("g.remove", type="raster", name="_nmad_absdev", flags="f", quiet=True)
    return {"n": n, "median": median, "nmad": nmad, "p25": p[25], "p75": p[75],
           "iqr": p[75] - p[25], "mean": float(u["mean"]), "sd": float(u["stddev"])}


def main():
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")

    # ---- bring Lake Lure's own reference and mask into this project ----
    # region must cover WHERE THE OUTPUT LANDS before r.proj runs (it clips
    # to the current region) -- cat34's own region does not overlap Lake
    # Lure at all, so use the full cap_sfm_regis/dtm_2017 corridor extent
    # (confirmed to span both areas) for the import step, then narrow the
    # region per area for the actual statistics below.
    gs.run_command("g.region", raster="cap_sfm_regis", res=1, flags="a")
    gs.run_command("r.proj", project="DEM_generation_lure", mapset="PERMANENT",
                   input="lidar_2017_DTM_lure", output="lure_dtm2017_in_wshed_prj",
                   resolution=1, method="bilinear", overwrite=True, quiet=True)
    gs.run_command("r.proj", project="DEM_generation_lure", mapset="PERMANENT",
                   input="stable_mask_lure", output="stable_mask_lure_in_wshed_prj",
                   resolution=1, method="nearest", overwrite=True, quiet=True)
    gs.run_command("v.proj", project="DEM_generation_lure", mapset="PERMANENT",
                   input="boundary", output="lure_boundary_in_wshed_prj",
                   overwrite=True, quiet=True)

    # ============================================================
    # PART 1: area-specific summary stats, each area's own reference
    # ============================================================
    rows = []

    # -- watershed: cap_sfm_regis - dtm_2020_filled, over the CURRENT
    #    stable_mask (68,248 cells), watershed region
    gs.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")
    gs.mapcalc("_res_wshed = cap_sfm_regis - dtm_2020_filled", overwrite=True, quiet=True)
    n_mask = gs.parse_command("r.univar", map="stable_mask", flags="g").get("n")
    stats_w = nmad_and_iqr(gs, "_res_wshed", zone="stable_mask")
    stats_w["area"] = "watershed"
    stats_w["reference"] = "dtm_2020_filled (2020 lidar)"
    stats_w["stable_mask_n_check"] = n_mask
    rows.append(stats_w)
    print(f"watershed (n={stats_w['n']}, stable_mask has {n_mask} cells): "
          f"median={stats_w['median']:.3f} m, NMAD={stats_w['nmad']:.3f} m, "
          f"IQR=[{stats_w['p25']:.3f}, {stats_w['p75']:.3f}] m "
          f"(width {stats_w['iqr']:.3f} m)")

    # -- Lake Lure: cap_sfm_regis - lidar_2017_DTM_lure(reprojected), over
    #    stable_mask_lure(reprojected), Lake Lure region (reprojected boundary)
    gs.run_command("g.region", vector="lure_boundary_in_wshed_prj", res=1, flags="a")
    gs.mapcalc("_res_lure = cap_sfm_regis - lure_dtm2017_in_wshed_prj",
              overwrite=True, quiet=True)
    n_mask_l = gs.parse_command("r.univar", map="stable_mask_lure_in_wshed_prj",
                                flags="g").get("n")
    stats_l = nmad_and_iqr(gs, "_res_lure", zone="stable_mask_lure_in_wshed_prj")
    stats_l["area"] = "lake_lure"
    stats_l["reference"] = "lidar_2017_DTM_lure (2017 lidar, reprojected)"
    stats_l["stable_mask_n_check"] = n_mask_l
    rows.append(stats_l)
    print(f"Lake Lure (n={stats_l['n']}, stable mask has {n_mask_l} cells): "
          f"median={stats_l['median']:.3f} m, NMAD={stats_l['nmad']:.3f} m, "
          f"IQR=[{stats_l['p25']:.3f}, {stats_l['p75']:.3f}] m "
          f"(width {stats_l['iqr']:.3f} m)")

    out_csv = TABLES / "sfm_vertical_bias_by_area.csv"
    fieldnames = ["area", "reference", "n", "stable_mask_n_check", "median", "mean",
                 "sd", "nmad", "p25", "p75", "iqr"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    diff_median = stats_l["median"] - stats_w["median"]
    print(f"\nSpatial variation in required correction: {diff_median:+.3f} m "
          f"median difference (Lake Lure minus watershed), same SfM raster, "
          f"each area's own lidar reference.")

    # ============================================================
    # PART 2: continuous corridor-wide spatial map, ONE common reference
    # (dtm_2017, the only lidar surface that also spans both areas)
    # ============================================================
    gs.run_command("g.region", raster="cap_sfm_regis", res=2, flags="a")
    gs.mapcalc("_res_corridor = cap_sfm_regis - dtm_2017", overwrite=True, quiet=True)
    corridor_stats = gs.parse_command("r.univar", map="_res_corridor", flags="ge", quiet=True)
    print(f"\nCorridor-wide (cap_sfm_regis - dtm_2017, common reference, "
          f"no mask, res=2m): n={corridor_stats['n']}, "
          f"median={float(corridor_stats['median']):.2f} m, "
          f"p05={float(corridor_stats.get('first_quartile', 0)):.2f} m")

    gs.run_command("r.relief", input="dtm_2017", output="_res_hillshade_raw",
                   overwrite=True, quiet=True)
    gs.mapcalc("_res_hillshade = if(_res_hillshade_raw < 0, 0, "
              "if(_res_hillshade_raw > 255, 255, _res_hillshade_raw))",
              overwrite=True, quiet=True)
    gs.run_command("r.out.gdal", input="_res_hillshade", output=str(HILLSHADE_TIF),
                   format="GTiff", type="Byte", createopt="COMPRESS=LZW",
                   overwrite=True, quiet=True)
    gs.run_command("r.out.gdal", input="_res_corridor", output=str(CORRIDOR_TIF),
                   format="GTiff", type="Float64", createopt="COMPRESS=LZW",
                   nodata=-9999, overwrite=True, quiet=True)

    import pandas as pd
    boundary_w = pd.read_csv(TABLES / "cat34_boundary.csv")

    with rasterio.open(HILLSHADE_TIF) as src:
        hs = src.read(1, masked=True)
        bounds = src.bounds
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    with rasterio.open(CORRIDOR_TIF) as src:
        res = src.read(1, masked=True)

    fig, ax = plt.subplots(figsize=(174 / 25.4, 110 / 25.4))
    ax.imshow(hs, cmap="gray", extent=extent, origin="upper", zorder=1)
    vmax = 20
    im = ax.imshow(res, cmap="RdYlBu_r", vmin=-vmax, vmax=vmax, extent=extent,
                   origin="upper", alpha=0.75, zorder=2)
    ax.plot(boundary_w["easting"], boundary_w["northing"], color="black",
           linewidth=1.0, zorder=4, label="Watershed boundary")

    ax.text(boundary_w["easting"].mean(), boundary_w["northing"].max() + 60,
           "Watershed", ha="center", fontsize=8, fontweight="bold")
    # Lake Lure boundary polyline export was unreliable across GRASS
    # versions (v.out.ascii type=boundary/format=point); label its
    # approximate centroid directly instead (from boundary@PERMANENT's own
    # known extent, DEM_generation_lure project: roughly
    # E315700-318100, N190900-192200).
    ax.text(316900, 192000, "Lake Lure", ha="center", fontsize=8,
           fontweight="bold")
    style_map_panel(ax, extent, cax=None, mappable=im,
                    cbar_label="CAP SfM $-$ 2017 lidar", cbar_units="m")
    ax.set_title("SfM vertical residual across the corridor "
                 "(common 2017 lidar reference, no mask)", fontsize=9)
    out_pdf = FIGURES / "fig_sfm_bias_corridor_map.pdf"
    out_png = FIGURES / "fig_sfm_bias_corridor_map.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_pdf} and .png")

    gs.run_command("g.remove", type="raster",
                   name="_res_wshed,_res_lure,_res_corridor,_res_hillshade,"
                        "_res_hillshade_raw",
                   flags="f", quiet=True)

    log("29_sfm_spatial_bias_variation.py run. cap_sfm_regis (spans both "
        "areas continuously) vs. each area's own lidar reference, over each "
        "area's own current stable mask: watershed (dtm_2020_filled, "
        f"stable_mask n={stats_w['n']}): median={stats_w['median']:+.3f}m, "
        f"NMAD={stats_w['nmad']:.3f}m, IQR=[{stats_w['p25']:.3f}, "
        f"{stats_w['p75']:.3f}]m. Lake Lure (lidar_2017_DTM_lure reprojected, "
        f"stable_mask_lure n={stats_l['n']}): median={stats_l['median']:+.3f}m, "
        f"NMAD={stats_l['nmad']:.3f}m, IQR=[{stats_l['p25']:.3f}, "
        f"{stats_l['p75']:.3f}]m. Spatial variation (median difference, Lake "
        f"Lure minus watershed) = {diff_median:+.3f}m. Superscedes the two "
        "ad hoc single numbers previously on record (+1.788m over a stale "
        "46,188-cell watershed mask; +12.987m over the Lake Lure mask) -- "
        "this run uses the current, correct watershed stable_mask (68,248 "
        "cells). Corridor-wide common-reference (dtm_2017) map saved to "
        "fig_sfm_bias_corridor_map.png/.pdf. Written to "
        f"{out_csv.name}.", run="sfm_spatial_bias_variation")


if __name__ == "__main__":
    main()
