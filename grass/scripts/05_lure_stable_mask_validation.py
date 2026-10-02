#!/usr/bin/env python3
"""
05_lure_stable_mask_validation.py
==================================
Validates stable_mask_lure (built by 02_build_stable_mask_lure.py) before
trusting the LoD measured from it. Three checks, per the user's explicit
request:

  1a. Shoreline-buffer sensitivity: how many stable cells fall within 20 m of
      lure_water_mask's edge, and does the canopy-binned LoD move if they are
      excluded?
  2b. Slope and canopy-height (land-cover proxy) distribution of the Lake
      Lure stable mask, compared with the watershed's stable_mask.
  3c. Visual check: stable mask overlaid on the 2024 orthophoto, at a scale
      that lets a human eye catch docks / roads under construction / Morse
      Park works / dredged areas that have no business being called "stable."

Run inside DEM_generation_lure/PERMANENT, region = boundary@PERMANENT res=1,
after 02_build_stable_mask_lure.py has produced stable_mask_lure and
lure_water_mask.

Outputs (all real, all kept):
  raster  lure_water_dist          (r.grow.distance from lure_water_mask)
  raster  stable_mask_lure_far     (stable_mask_lure with the <20m shoreline
                                    buffer removed -- kept as a documented
                                    raster, not a throwaway)
  results/tables/lure_shoreline_buffer_sensitivity.csv
  results/tables/lure_mask_slope_canopy_comparison.csv
  results/diagnostics/lure_stable_mask_overlay.png
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
DIAGNOSTICS = Path(__file__).resolve().parents[2] / "results" / "diagnostics"
TABLES.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS.mkdir(parents=True, exist_ok=True)

LOD_BINS = [0.0, 2.0, 5.0, 10.0, 20.0, 1e6]
LOD_CONFIDENCE = 1.96
MIN_BIN_N = 500
SHORELINE_BUFFER_M = 20.0


def nmad(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def canopy_binned_lod(gs, dh_map, canopy_map, mask_map, label):
    """Reproduce stage3's canopy-binned LoD, restricted to `mask_map`."""
    import grass.script.array as garray

    dh = np.ma.masked_invalid(np.array(garray.array(dh_map, dtype=np.float64)))
    canopy = np.ma.masked_invalid(np.array(garray.array(canopy_map, dtype=np.float64)))
    mask = np.ma.masked_invalid(np.array(garray.array(mask_map, dtype=np.float64)))
    valid = (~dh.mask) & (~canopy.mask) & (~mask.mask) & (mask == 1)
    dh_v = np.asarray(dh[valid])
    cov_v = np.asarray(canopy[valid])

    rows = []
    print(f"\n  {label}")
    print(f"  {'canopy (m)':>14s} | {'n':>9s} | {'NMAD (m)':>9s} | {'LoD95 (m)':>9s}")
    for i in range(len(LOD_BINS) - 1):
        lo, hi = LOD_BINS[i], LOD_BINS[i + 1]
        sel = (cov_v >= lo) & (cov_v < hi)
        vals = dh_v[sel]
        vals = vals[np.abs(vals) < 20.0]
        n = vals.size
        if n < MIN_BIN_N:
            print(f"  [{lo:6.1f},{hi:6.1f}) | {n:9,d} | {'--':>9s} | {'merged':>9s}")
            rows.append({"label": label, "lo": lo, "hi": hi, "n": n,
                        "nmad": None, "lod95": None})
            continue
        bin_nmad = nmad(vals)
        lod = LOD_CONFIDENCE * bin_nmad
        print(f"  [{lo:6.1f},{hi:6.1f}) | {n:9,d} | {bin_nmad:9.3f} | {lod:9.3f}")
        rows.append({"label": label, "lo": lo, "hi": hi, "n": n,
                    "nmad": bin_nmad, "lod95": lod})
    return rows


def main():
    gs, gj = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    gs.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")

    # ------------------------------------------------------------------
    print("=== 1a. Shoreline-buffer sensitivity ===")
    gs.run_command("r.grow.distance", input="lure_water_mask",
                   distance="lure_water_dist", overwrite=True, quiet=True)

    n_stable = int(gs.parse_command("r.univar", map="stable_mask_lure",
                                    flags="g").get("n", 0))

    gs.mapcalc("_stable_near_shore = if(!isnull(stable_mask_lure) && "
              f"lure_water_dist < {SHORELINE_BUFFER_M}, 1, null())",
              overwrite=True, quiet=True)
    n_near = int(gs.parse_command("r.univar", map="_stable_near_shore",
                                  flags="g").get("n", 0))

    gs.mapcalc("stable_mask_lure_far = if(!isnull(stable_mask_lure) && "
              f"lure_water_dist >= {SHORELINE_BUFFER_M}, 1, null())",
              overwrite=True, quiet=True)
    n_far = int(gs.parse_command("r.univar", map="stable_mask_lure_far",
                                 flags="g").get("n", 0))

    print(f"  stable_mask_lure total: {n_stable:,}")
    print(f"  within {SHORELINE_BUFFER_M:.0f} m of the water mask's edge: "
          f"{n_near:,} ({100*n_near/n_stable:.1f}%)")
    print(f"  stable_mask_lure_far (>= {SHORELINE_BUFFER_M:.0f} m from water): "
          f"{n_far:,} ({100*n_far/n_stable:.1f}%)")

    # canopy raster already exists as sml_canopy_chk (built in 02_..._lure.py's
    # step 6); recompute defensively in case that mapset state changed
    gs.mapcalc("sml_canopy_chk = lidar_2017_DSM_lure@PERMANENT - "
              "lidar_2017_DTM_lure@PERMANENT", overwrite=True, quiet=True)

    rows_all = canopy_binned_lod(gs, "coreg_lure_lidar_dh_corrected",
                                 "sml_canopy_chk", "stable_mask_lure",
                                 "ALL stable cells (includes <20m shoreline)")
    rows_far = canopy_binned_lod(gs, "coreg_lure_lidar_dh_corrected",
                                 "sml_canopy_chk", "stable_mask_lure_far",
                                 f">= {SHORELINE_BUFFER_M:.0f} m from shoreline only")

    import csv
    out_csv = TABLES / "lure_shoreline_buffer_sensitivity.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "lo", "hi", "n", "nmad", "lod95"])
        w.writeheader()
        for r in rows_all + rows_far:
            w.writerow(r)
    print(f"\n  wrote {out_csv}")

    # ------------------------------------------------------------------
    print("\n=== 1b. Slope / canopy distribution vs. the watershed's mask ===")
    import grass.script.array as garray

    slope_lure = np.ma.masked_invalid(np.array(garray.array("sml_slope", dtype=np.float64)))
    mask_lure = np.ma.masked_invalid(np.array(garray.array("stable_mask_lure", dtype=np.float64)))
    canopy_lure = np.ma.masked_invalid(np.array(garray.array("sml_canopy_chk", dtype=np.float64)))
    sel_lure = (~slope_lure.mask) & (~mask_lure.mask) & (mask_lure == 1) & (~canopy_lure.mask)
    slope_lure_v = np.asarray(slope_lure[sel_lure])
    canopy_lure_v = np.asarray(canopy_lure[sel_lure])

    def pct_below(x, thresh):
        return 100.0 * float(np.mean(x < thresh))

    lure_stats = {
        "reach": "Lake Lure",
        "n": int(sel_lure.sum()),
        "slope_mean_deg": float(np.mean(slope_lure_v)),
        "slope_median_deg": float(np.median(slope_lure_v)),
        "pct_slope_lt_5deg": pct_below(slope_lure_v, 5.0),
        "pct_slope_lt_10deg": pct_below(slope_lure_v, 10.0),
        "canopy_mean_m": float(np.mean(canopy_lure_v)),
        "canopy_median_m": float(np.median(canopy_lure_v)),
        "pct_open_canopy_lt_2m": pct_below(canopy_lure_v, 2.0),
        "pct_forested_canopy_ge_10m": 100.0 - pct_below(canopy_lure_v, 10.0),
    }

    # watershed comparison: switch project via a fresh session
    gsw, gjw = grass_session(project="DEM_generation", mapset="for_codem")
    gsw.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")
    gsw.mapcalc("sm_canopy_chk = lidar_2020_DSM@DTM_DSM - dtm_2020_filled@for_codem",
               overwrite=True, quiet=True)
    slope_w = np.ma.masked_invalid(np.array(garray.array("sm_slope@for_codem", dtype=np.float64)))
    mask_w = np.ma.masked_invalid(np.array(garray.array("stable_mask@for_codem", dtype=np.float64)))
    canopy_w = np.ma.masked_invalid(np.array(garray.array("sm_canopy_chk", dtype=np.float64)))
    sel_w = (~slope_w.mask) & (~mask_w.mask) & (mask_w == 1) & (~canopy_w.mask)
    slope_w_v = np.asarray(slope_w[sel_w])
    canopy_w_v = np.asarray(canopy_w[sel_w])

    wshed_stats = {
        "reach": "Watershed (cat34)",
        "n": int(sel_w.sum()),
        "slope_mean_deg": float(np.mean(slope_w_v)),
        "slope_median_deg": float(np.median(slope_w_v)),
        "pct_slope_lt_5deg": pct_below(slope_w_v, 5.0),
        "pct_slope_lt_10deg": pct_below(slope_w_v, 10.0),
        "canopy_mean_m": float(np.mean(canopy_w_v)),
        "canopy_median_m": float(np.median(canopy_w_v)),
        "pct_open_canopy_lt_2m": pct_below(canopy_w_v, 2.0),
        "pct_forested_canopy_ge_10m": 100.0 - pct_below(canopy_w_v, 10.0),
    }

    for s in (lure_stats, wshed_stats):
        print(f"\n  {s['reach']} (n={s['n']:,}):")
        print(f"    slope: mean={s['slope_mean_deg']:.1f} deg, median="
              f"{s['slope_median_deg']:.1f} deg, {s['pct_slope_lt_5deg']:.1f}% < 5 deg, "
              f"{s['pct_slope_lt_10deg']:.1f}% < 10 deg")
        print(f"    canopy: mean={s['canopy_mean_m']:.1f} m, median="
              f"{s['canopy_median_m']:.1f} m, {s['pct_open_canopy_lt_2m']:.1f}% open "
              f"(<2m), {s['pct_forested_canopy_ge_10m']:.1f}% forested (>=10m)")

    import csv
    out_csv2 = TABLES / "lure_mask_slope_canopy_comparison.csv"
    with open(out_csv2, "w", newline="") as f:
        fieldnames = list(lure_stats.keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(lure_stats)
        w.writerow(wshed_stats)
    print(f"\n  wrote {out_csv2}")

    # ------------------------------------------------------------------
    print("\n=== 1c. Visual overlay: stable mask on the 2024 orthophoto ===")
    # back to the Lake Lure session/region for the export
    gs, gj = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    gs.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")

    gs.run_command("g.remove", type="group", name="lure_ortho_grp", flags="f",
                   quiet=True, stderr_=None) if False else None
    gs.run_command("i.group", group="lure_ortho_grp",
                   input="Ortho_aerial_2024.1,Ortho_aerial_2024.2,Ortho_aerial_2024.3",
                   quiet=True)
    gs.run_command("r.out.gdal", input="lure_ortho_grp",
                   output=str(DIAGNOSTICS / "_lure_ortho_2024.tif"),
                   format="GTiff", type="Byte", createopt="COMPRESS=LZW",
                   overwrite=True, quiet=True)
    gs.run_command("r.out.gdal", input="stable_mask_lure",
                   output=str(DIAGNOSTICS / "_lure_stable_mask.tif"),
                   format="GTiff", type="Byte", nodata=0, overwrite=True, quiet=True)
    gs.run_command("r.out.gdal", input="lure_water_mask",
                   output=str(DIAGNOSTICS / "_lure_water_mask_export.tif"),
                   format="GTiff", type="Byte", nodata=0, overwrite=True, quiet=True)

    log(f"05_lure_stable_mask_validation.py run. Shoreline buffer "
        f"({SHORELINE_BUFFER_M:.0f} m): {n_near:,} of {n_stable:,} stable "
        f"cells ({100*n_near/n_stable:.1f}%) fall within it; "
        f"stable_mask_lure_far (n={n_far:,}) kept as a raster. Slope/canopy "
        f"comparison vs watershed written to {out_csv2.name}. Shoreline-"
        f"buffer LoD sensitivity written to {out_csv.name}.",
        run="lure_mask_validation")

    print(f"\n  exported {DIAGNOSTICS}/_lure_ortho_2024.tif, "
          f"_lure_stable_mask.tif, _lure_water_mask_export.tif for the "
          f"overlay figure (built separately with matplotlib -- see "
          f"make_lure_mask_overlay_figure.py)")


if __name__ == "__main__":
    main()
