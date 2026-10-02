#!/usr/bin/env python3
"""
06_lure_known_stable_ground_test.py
====================================
Independent test of the measured LoD: sample the 2017->2024 lidar DoD
(coreg_lure_lidar_dh_corrected, bias-corrected but NOT LoD-thresholded) over
surfaces that certainly did not change -- building roofs, paved parking,
road surfaces away from the shoreline -- and report the distribution.

WHY THIS IS INDEPENDENT OF stable_mask_lure: that mask was built from
DELTA-NDVI stability (spectral change between epochs) + slope + channel
distance + water exclusion. It was NOT built to specifically target
impervious surfaces, and impervious surfaces are a genuinely different
population (hard, flat, unvegetated) than "any spectrally-stable ground."

HISTORY -- WHY THIS SCRIPT CLASSIFIES ONLY WITHIN TWO FIXED BOXES, NOT THE
WHOLE REACH: the first version classified impervious candidates reach-wide
from single-date NDVI < 0.15 + slope < 5 deg + canopy < 1.5 m (no location
restriction). That produced 8,776 cells, but sampling dh_corrected over them
gave a heavy-tailed, non-credible distribution (SD 1.17 m, only 78.5% within
+/-0.34 m). Visual inspection (results/diagnostics/
lure_impervious_candidate_overlay.png) showed why: the two largest r.clump
components (3,506 and 1,859 cells, 61% of the sample) sit on a large bare/
exposed-sediment area near the reach centre -- genuinely disturbed ground,
not "certainly did not change" ground. Filtering out large clumps (keeping
only <=150-cell components, results/diagnostics/
lure_impervious_small_overlay.png) reduced but did not eliminate this (still
55.2% within +/-0.34 m, p05 -5.24 m) -- some small clump fragments were still
inside the disturbed zone. Both attempts are recorded here and in
RESULTS_FOR_PAPER.md for transparency but WITHDRAWN, not used for any
conclusion -- an automated spectral/slope/canopy classifier cannot tell
"bare because it's pavement" from "bare because it's a dredged/exposed
lakebed" on this imagery.

The fix: restrict classification to areas a human has actually looked at
and confirmed are roofs/pavement/road with no disturbed ground nearby
(results/diagnostics/_candidate_check.png). VERIFIED_BOXES below are those
two areas. This is the ONLY result this script reports a conclusion from.

Outputs (kept, not disposable):
  raster  lure_impervious_verified   (the classification actually used)
  results/tables/lure_known_stable_ground_dh_distribution.csv  (full sample,
    the 1,201-cell VERIFIED result -- NOT either withdrawn attempt)
  results/tables/lure_known_stable_ground_summary.csv
  results/diagnostics/lure_impervious_verified_overlay.png
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
DIAGNOSTICS = Path(__file__).resolve().parents[2] / "results" / "diagnostics"
TABLES.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS.mkdir(parents=True, exist_ok=True)

NDVI_MAX = 0.15
SLOPE_MAX_DEG = 6.0
CANOPY_MAX_M = 2.0
DH_CLIP = 20.0

# Visually verified against the 2024 orthophoto (results/diagnostics/
# _candidate_check.png) to contain unambiguous roofs/pavement/road, with no
# disturbed/bare/dredged ground nearby.
VERIFIED_BOXES = [
    (315955, 316025, 191965, 192015),   # roofs + driveway
    (316650, 316950, 191150, 191230),   # parking lot + building + road
]


def main():
    gs, gj = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    gs.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")

    print("=== Building the VERIFIED impervious-surface classification ===")
    box_expr = " || ".join(
        f"(x() >= {x0} && x() <= {x1} && y() >= {y0} && y() <= {y1})"
        for x0, x1, y0, y1 in VERIFIED_BOXES)
    gs.mapcalc(
        "lure_impervious_verified = if("
        f"  ({box_expr})"
        f"  && ndvi_2024_lure < {NDVI_MAX}"
        f"  && sml_slope < {SLOPE_MAX_DEG}"
        f"  && sml_canopy_chk < {CANOPY_MAX_M}"
        "  && isnull(lure_water_mask),"
        "  1, null())", overwrite=True, quiet=True)

    n = int(gs.parse_command("r.univar", map="lure_impervious_verified",
                             flags="g").get("n", 0))
    print(f"  lure_impervious_verified: {n:,} cells within {len(VERIFIED_BOXES)} "
          f"visually-verified boxes (NDVI<{NDVI_MAX}, slope<{SLOPE_MAX_DEG} deg, "
          f"canopy<{CANOPY_MAX_M} m)")
    if n < 200:
        print("  !! Very few candidate cells -- loosen thresholds or verify "
              "additional boxes before trusting this test.")

    # ------------------------------------------------------------------
    print("\n=== Sampling coreg_lure_lidar_dh_corrected over these cells ===")
    import grass.script.array as garray

    dh = np.ma.masked_invalid(np.array(
        garray.array("coreg_lure_lidar_dh_corrected", dtype=np.float64)))
    cand = np.ma.masked_invalid(np.array(
        garray.array("lure_impervious_verified", dtype=np.float64)))
    sel = (~dh.mask) & (~cand.mask) & (cand == 1)
    vals = np.asarray(dh[sel])
    vals = vals[np.abs(vals) < DH_CLIP]

    def pct_within(x, thresh):
        return 100.0 * float(np.mean(np.abs(x) <= thresh))

    nmad = float(1.4826 * np.median(np.abs(vals - np.median(vals))))
    stats = {
        "n": int(vals.size),
        "mean_m": float(np.mean(vals)),
        "median_m": float(np.median(vals)),
        "sd_m": float(np.std(vals)),
        "nmad_m": nmad,
        "p05_m": float(np.percentile(vals, 5)),
        "p25_m": float(np.percentile(vals, 25)),
        "p75_m": float(np.percentile(vals, 75)),
        "p95_m": float(np.percentile(vals, 95)),
        "pct_within_0.28m": pct_within(vals, 0.28),
        "pct_within_0.30m": pct_within(vals, 0.30),
        "pct_within_0.34m": pct_within(vals, 0.34),
        "pct_within_0.50m": pct_within(vals, 0.50),
        "pct_within_0.94m": pct_within(vals, 0.94),
    }
    print(f"  n={stats['n']:,}  mean={stats['mean_m']:+.3f}  "
          f"median={stats['median_m']:+.3f}  SD={stats['sd_m']:.3f}  "
          f"NMAD={stats['nmad_m']:.3f}")
    print(f"  p05={stats['p05_m']:+.3f}  p25={stats['p25_m']:+.3f}  "
          f"p75={stats['p75_m']:+.3f}  p95={stats['p95_m']:+.3f}")
    print(f"  within +/-0.28 m: {stats['pct_within_0.28m']:.1f}%   "
          f"within +/-0.34 m: {stats['pct_within_0.34m']:.1f}%   "
          f"within +/-0.50 m: {stats['pct_within_0.50m']:.1f}%   "
          f"within +/-0.94 m: {stats['pct_within_0.94m']:.1f}%")

    out_csv = TABLES / "lure_known_stable_ground_dh_distribution.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dh_m"])
        for v in vals:
            w.writerow([f"{v:.6f}"])
    print(f"\n  wrote {out_csv} ({len(vals):,} rows, full VERIFIED sample)")

    out_summary = TABLES / "lure_known_stable_ground_summary.csv"
    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats.keys()))
        w.writeheader()
        w.writerow(stats)
    print(f"  wrote {out_summary}")

    # ------------------------------------------------------------------
    gs.run_command("r.out.gdal", input="lure_impervious_verified",
                   output=str(DIAGNOSTICS / "_lure_impervious_verified.tif"),
                   format="GTiff", type="Byte", nodata=0, overwrite=True, quiet=True)

    log(f"06_lure_known_stable_ground_test.py run (verified-boxes version). "
        f"lure_impervious_verified: {n:,} cells within "
        f"{len(VERIFIED_BOXES)} visually-verified boxes. dh_corrected "
        f"sampled: n={stats['n']:,} median={stats['median_m']:+.3f}m "
        f"SD={stats['sd_m']:.3f}m NMAD={stats['nmad_m']:.3f}m, "
        f"{stats['pct_within_0.34m']:.1f}% within +/-0.34m, "
        f"{stats['pct_within_0.94m']:.1f}% within +/-0.94m. Two earlier "
        f"reach-wide automated attempts (8,776 and 911 cells) were "
        f"contaminated by a bare/dredged-sediment area and withdrawn -- see "
        f"script docstring and RESULTS_FOR_PAPER.md. "
        f"Full sample -> {out_csv.name}, summary -> {out_summary.name}.",
        run="lure_known_stable_ground")


if __name__ == "__main__":
    main()
