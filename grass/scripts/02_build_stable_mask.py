#!/usr/bin/env python3
"""
build_stable_mask.py
====================
Construct a stable-terrain mask for DEM co-registration and Level of
Detection estimation.

WRITTEN FOR: Regmi et al., post-Helene geomorphic change, Hickory Nut Gorge.
Companion to coreg_variable_lod.py (which consumes the mask this produces).

!!  NOT TESTED AGAINST YOUR DATA  !!
Written without access to GRASS or your rasters. Run it, inspect the output
mask visually over the orthophoto BEFORE using it, and adjust thresholds.

--------------------------------------------------------------------------
THE ONE RULE THAT MATTERS
--------------------------------------------------------------------------
NEVER define stable terrain using the DEM difference you are about to
evaluate. If you select cells because their elevation difference is small,
then measure the error from those same cells, you have guaranteed a small
error estimate. That is circular and a reviewer will catch it.

Every criterion below is INDEPENDENT of the DoD:
  - NDVI stability      (spectral, from Sentinel-2 -- you already have it)
  - distance from channel (topographic, from the PRE-event DEM only)
  - slope               (topographic, from the PRE-event DEM only)
  - water exclusion     (your digitised full-pool polygon)
  - disturbance exclusion (your mapped scar / debris polygons)

The mask is deliberately conservative. It is better to keep 15% of the
basin that is genuinely stable than 60% that is arguably stable.

--------------------------------------------------------------------------
WHY THIS FIXES THE PAPER'S WEAKEST POINT
--------------------------------------------------------------------------
Nine points on asphalt cannot characterise the error of an interpolated
bare-earth surface under closed canopy. A mask spanning open ground AND
forest, across the full elevation and canopy range, lets you estimate the
error where it actually matters -- and lets you show that it varies, which
is the paper's methodological contribution.

--------------------------------------------------------------------------
INPUTS you must supply (edit CONFIG)
--------------------------------------------------------------------------
  PRE_DEM        pre-event bare-earth DTM (2020 lidar for the watershed,
                 2017 for Lake Lure)
  NDVI_DIFF      your delta-NDVI raster (post - pre)
  WATER_VECT     digitised full-pool shoreline polygon (or None)
  DISTURB_VECT   mapped landslide scar / debris / channel-disturbance
                 polygons (or None -- but you should digitise them; the
                 scar is obvious in the CAP orthophoto)
  ROAD_POINTS    your existing 9 stable points (optional; used only to
                 CHECK that the new mask agrees with them, not to build it)
"""

import os
import sys

import numpy as np

import grass.script as gs
import grass.script.array as garray

CONFIG = {
    # --- inputs ---
    "PRE_DEM": "dtm_2020_filled@for_codem",
    "NDVI_DIFF": "ndvi_diff",           # post - pre; negative = vegetation loss
                                         # imported via r.proj from
                                         # ndvi_change@test_data_extent (helene_chimney_11_25)
    "WATER_VECT": None,                 # e.g. "lake_fullpool"
    "DISTURB_VECT": None,               # e.g. "disturbance_polygons"
    "ROAD_POINTS": "more_stable_pts@for_codem",

    # --- output ---
    "OUT_MASK": "stable_mask",

    # --- thresholds (TUNE THESE, then look at the result) ---
    # NDVI: vegetation that did not change spectrally almost certainly did
    # not have its ground surface reworked. Tight threshold = safer.
    "NDVI_ABS_MAX": 0.10,

    # Channels concentrate scour and fill. Buffer them out generously.
    # Derived from the PRE-event DEM, so this is not circular.
    "CHANNEL_THRESH_CELLS": 2000,       # flow-accumulation cells defining a channel
    "CHANNEL_BUFFER_M": 30.0,           # exclusion buffer either side

    # Very steep ground is both failure-prone and noise-prone.
    "SLOPE_MAX_DEG": 45.0,

    # Extra buffer around mapped disturbance polygons
    "DISTURB_BUFFER_M": 25.0,

    # Sanity floor: warn if the mask covers less than this fraction of the region
    "MIN_COVERAGE_FRACTION": 0.03,
}


def msg(t):
    print(f"\n=== {t}", flush=True)


def main():
    if not os.environ.get("GISRC"):
        sys.exit("Not inside a GRASS session. Start GRASS first, e.g.\n"
                 "  grass /path/to/project/mapset --exec python3 "
                 "build_stable_mask.py")

    cfg = CONFIG
    print(gs.read_command("g.region", flags="p"))

    # ------------------------------------------------------------------
    msg("1. Terrain criteria from the PRE-event DEM")
    gs.run_command("r.slope.aspect", elevation=cfg["PRE_DEM"],
                   slope="sm_slope", overwrite=True, quiet=True)

    gs.run_command("r.watershed", elevation=cfg["PRE_DEM"],
                   accumulation="sm_accum", flags="a",
                   overwrite=True, quiet=True)

    # channel network from the pre-event surface only
    gs.mapcalc(f"sm_channel = if(abs(sm_accum) > {cfg['CHANNEL_THRESH_CELLS']}, "
               f"1, null())", overwrite=True, quiet=True)

    gs.run_command("r.grow.distance", input="sm_channel",
                   distance="sm_chan_dist", overwrite=True, quiet=True)
    print(f"  channel network extracted (>{cfg['CHANNEL_THRESH_CELLS']} cells), "
          f"buffer {cfg['CHANNEL_BUFFER_M']} m")

    # ------------------------------------------------------------------
    msg("2. Spectral criterion from NDVI change")
    # |delta NDVI| small  ->  canopy/ground cover unchanged
    gs.mapcalc(f"sm_ndvi_ok = if(abs({cfg['NDVI_DIFF']}) < "
               f"{cfg['NDVI_ABS_MAX']}, 1, 0)", overwrite=True, quiet=True)
    print(f"  |dNDVI| < {cfg['NDVI_ABS_MAX']}")

    # ------------------------------------------------------------------
    msg("3. Exclusion zones")
    exclusions = []

    if cfg["WATER_VECT"]:
        gs.run_command("v.to.rast", input=cfg["WATER_VECT"], output="sm_water",
                       use="val", value=1, overwrite=True, quiet=True)
        exclusions.append("sm_water")
        print("  water excluded")
    else:
        print("  !! No water polygon supplied. Open water MUST be excluded --")
        print("     lidar elevations over water are meaningless and will")
        print("     contaminate the error estimate. Digitise it.")

    if cfg["DISTURB_VECT"]:
        gs.run_command("v.buffer", input=cfg["DISTURB_VECT"],
                       output="sm_disturb_buf",
                       distance=cfg["DISTURB_BUFFER_M"],
                       overwrite=True, quiet=True)
        gs.run_command("v.to.rast", input="sm_disturb_buf",
                       output="sm_disturb", use="val", value=1,
                       overwrite=True, quiet=True)
        exclusions.append("sm_disturb")
        print(f"  disturbance polygons excluded (+{cfg['DISTURB_BUFFER_M']} m buffer)")
    else:
        print("  !! No disturbance polygons supplied. The landslide scar and")
        print("     debris paths are clearly visible in the CAP orthophoto --")
        print("     digitise them. Leaving real change inside the 'stable'")
        print("     mask inflates your error estimate and raises the LoD,")
        print("     which would understate genuine change.")

    excl_expr = " || ".join(f"(!isnull({e}) && {e} == 1)" for e in exclusions)
    if not excl_expr:
        excl_expr = "0"

    # ------------------------------------------------------------------
    msg("4. Combine")
    expr = (
        f"{cfg['OUT_MASK']} = if("
        f"  sm_ndvi_ok == 1"
        f"  && sm_slope < {cfg['SLOPE_MAX_DEG']}"
        f"  && sm_chan_dist > {cfg['CHANNEL_BUFFER_M']}"
        f"  && !({excl_expr})"
        f"  && !isnull({cfg['PRE_DEM']}),"
        f"  1, null())"
    )
    gs.mapcalc(expr, overwrite=True, quiet=True)

    # ------------------------------------------------------------------
    msg("5. Coverage check")
    total = int(gs.parse_command("r.univar", map=cfg["PRE_DEM"],
                                 flags="g", quiet=True)["n"])
    kept = int(gs.parse_command("r.univar", map=cfg["OUT_MASK"],
                                flags="g", quiet=True)["n"])
    frac = kept / total if total else 0
    print(f"  stable cells: {kept:,} of {total:,}  ({100*frac:.1f}% of region)")

    if frac < cfg["MIN_COVERAGE_FRACTION"]:
        print("  !! Very small mask. Loosen NDVI_ABS_MAX or CHANNEL_BUFFER_M,")
        print("     or check that NDVI_DIFF is aligned to the same region.")
    if frac > 0.75:
        print("  !! Very large mask -- probably too permissive. You are")
        print("     including disturbed ground, which will inflate the LoD.")

    # ------------------------------------------------------------------
    msg("6. Does the mask span the canopy range? (this is the whole point)")
    print("  If you have a DSM, check that stable cells exist in BOTH open")
    print("  and forested bins. If the mask is all open ground, you have")
    print("  rebuilt the 9-road-points problem at larger scale and the")
    print("  variable LoD will have nothing to vary over.")
    print("    r.mapcalc \"canopy = lidar_2020_dsm - lidar_2020_dtm\"")
    print("    r.univar map=canopy zones=stable_mask")

    # ------------------------------------------------------------------
    if cfg["ROAD_POINTS"]:
        msg("7. Cross-check against your original 9 points")
        print("  Sampling the new mask at the original stable points --")
        print("  they should mostly fall inside it. If they do not,")
        print("  something is misaligned.")
        gs.run_command("v.what.rast", map=cfg["ROAD_POINTS"],
                       raster=cfg["OUT_MASK"], column="in_mask", quiet=True)
        print("  (inspect the in_mask column: v.db.select "
              f"map={cfg['ROAD_POINTS']})")

    msg("NEXT: open stable_mask over the CAP orthophoto and LOOK AT IT")
    print("  Reject it if it includes: the scar, active channel, the lake,")
    print("  building roofs, or the Morse Park area. Iterate the thresholds")
    print("  until you would defend every included cell to a reviewer.")
    print("\n  Then run: python3 coreg_variable_lod.py --stage 1")


if __name__ == "__main__":
    main()
