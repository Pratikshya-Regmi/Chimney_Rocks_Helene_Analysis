#!/usr/bin/env python3
"""
10_consolidated_lod_table.py
==============================
One consolidated table for the manuscript: for each area (watershed, Lake
Lure), the stable-terrain n, NMAD, and LoD95 per canopy bin, LIDAR-LIDAR
PAIR ONLY. Intended to replace the manuscript's current Table 3 (derived
from 9 -- actually 7, see RESULTS_FOR_PAPER.md -- road points).

Recomputes both areas fresh from their own stable_mask + bias-corrected dh
+ canopy rasters (not transcribed from earlier printed output), so this is
independently traceable to a single logged run rather than copied from two
different sessions' logs.

  Watershed: stable_mask@for_codem, coreg_dh_corrected@for_codem,
             coreg_canopy@for_codem   (DEM_generation/for_codem)
  Lake Lure: stable_mask_lure, coreg_lure_lidar_dh_corrected,
             sml_canopy_chk           (DEM_generation_lure/PERMANENT)

Output: results/tables/consolidated_lod_table.csv (printed too)
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

LOD_BINS = [0.0, 2.0, 5.0, 10.0, 20.0, 1e6]
LOD_CONFIDENCE = 1.96
MIN_BIN_N = 500
DH_CLIP = 20.0


def nmad(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def canopy_bins(dh_map, canopy_map, mask_map, area_label):
    import grass.script.array as garray

    dh = np.ma.masked_invalid(np.array(garray.array(dh_map, dtype=np.float64)))
    canopy = np.ma.masked_invalid(np.array(garray.array(canopy_map, dtype=np.float64)))
    mask = np.ma.masked_invalid(np.array(garray.array(mask_map, dtype=np.float64)))
    valid = (~dh.mask) & (~canopy.mask) & (~mask.mask) & (mask == 1)
    dh_v = np.asarray(dh[valid])
    cov_v = np.asarray(canopy[valid])

    rows = []
    for i in range(len(LOD_BINS) - 1):
        lo, hi = LOD_BINS[i], LOD_BINS[i + 1]
        sel = (cov_v >= lo) & (cov_v < hi)
        vals = dh_v[sel]
        vals = vals[np.abs(vals) < DH_CLIP]
        n = vals.size
        bin_label = f"[{lo:.0f}, {hi:.0f})" if hi < 1e5 else f"[{lo:.0f}, inf)"
        if n < MIN_BIN_N:
            rows.append({"area": area_label, "canopy_bin_m": bin_label,
                        "n": n, "nmad_m": None, "lod95_m": None,
                        "merged_insufficient_n": True})
            continue
        bn = nmad(vals)
        lod = LOD_CONFIDENCE * bn
        rows.append({"area": area_label, "canopy_bin_m": bin_label,
                    "n": n, "nmad_m": round(bn, 3), "lod95_m": round(lod, 3),
                    "merged_insufficient_n": False})
    return rows


def main():
    all_rows = []

    gs_w, _ = grass_session(project="DEM_generation", mapset="for_codem")
    gs_w.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")
    all_rows += canopy_bins("coreg_dh_corrected", "coreg_canopy", "stable_mask",
                            "Watershed (cat34)")

    gs_l, _ = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    gs_l.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")
    all_rows += canopy_bins("coreg_lure_lidar_dh_corrected", "sml_canopy_chk",
                            "stable_mask_lure", "Lake Lure")

    print(f"\n{'area':>18s} | {'canopy bin (m)':>14s} | {'n':>9s} | "
          f"{'NMAD (m)':>9s} | {'LoD95 (m)':>9s}")
    print("-" * 70)
    for r in all_rows:
        nmad_s = f"{r['nmad_m']:.3f}" if r["nmad_m"] is not None else "--"
        lod_s = f"{r['lod95_m']:.3f}" if r["lod95_m"] is not None else "merged"
        print(f"{r['area']:>18s} | {r['canopy_bin_m']:>14s} | {r['n']:9,d} | "
              f"{nmad_s:>9s} | {lod_s:>9s}")

    out_csv = TABLES / "consolidated_lod_table.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    log("10_consolidated_lod_table.py run. Lidar-lidar pair only, both "
        "areas, canopy-binned stable-terrain n/NMAD/LoD95, recomputed fresh "
        "from stable_mask(+_lure)/coreg(_lure_lidar)_dh_corrected/canopy "
        f"rasters. {len(all_rows)} rows written to {out_csv.name}. Intended "
        "to replace the manuscript's Table 3 (currently derived from 7 "
        "road points, not 9 as stated -- see RESULTS_FOR_PAPER.md).",
        run="consolidated_lod_table")


if __name__ == "__main__":
    main()
