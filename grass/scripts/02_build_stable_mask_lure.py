#!/usr/bin/env python3
"""
02_build_stable_mask_lure.py
=============================
Lake Lure variant of 02_build_stable_mask.py. Same logic, different CONFIG
and one extra step: water exclusion.

WHY A SEPARATE FILE RATHER THAN EDITING CONFIG IN PLACE: 02_build_stable_mask.py
is the watershed's own script (already run, logged, and referenced from
RESULTS_FOR_PAPER.md as WATERSHED's stable mask). Overwriting its CONFIG in
place would make that prior run unreproducible from the script as it now
reads. Keeping a per-reach copy matches how coreg_variable_lod.py already
gets copied per-pair.

--------------------------------------------------------------------------
WATER EXCLUSION -- HOW THIS DIFFERS FROM THE WATERSHED
--------------------------------------------------------------------------
No digitised full-pool polygon exists for this reach, and true NDWI needs
Sentinel-2 B3 (green) + B8 (NIR); only TCI (true-colour quicklook) composites
and precomputed NDVI were found in helene_chimney_11_25's `imagery` /
`test_data_extent` mapsets -- not the raw bands NDWI needs. A plain NDVI < 0
threshold on the single-date NDVI rasters was tried and rejected: the
histogram is a smooth continuum from -0.84 to +1.0 with no bimodal gap, i.e.
no clean water/land separation at Sentinel-2's ~8.65 m resolution.

Instead, water is extracted directly from the 2017 lidar DTM, at the
resolution the rest of this analysis actually runs at (1 m), the same way a
reservoir surface is classically identified in lidar: it is the one large,
essentially flat, contiguous surface at the bottom of the local relief.
r.stats on lidar_2017_DTM_lure@PERMANENT inside the reach boundary showed one
dominant, sharply-bounded elevation cluster at 297-301 m (cell counts 2-4x
any other 20 cm bin in the whole 297-402 m range) -- thresholding <= 301 m
and keeping only the largest r.clump component isolates a single connected
region (441,400 of 445,758 candidate cells, 99%) that, rendered over the
hillshade, is unmistakably Lake Lure's shoreline in this reach (see
/tmp/lure_water_check.png from the interactive check -- reproduce with
r.mapcalc + r.clump if you want to regenerate it). This is independent of
the DoD being evaluated (built from the 2017 DTM alone), so it is not
circular, but it IS a proxy for "water" rather than a surveyed shoreline --
note that in the manuscript if this matters at the margin.

--------------------------------------------------------------------------
Usage: run inside an active GRASS session (see grass_env.grass_session),
DEM_generation_lure project, PERMANENT mapset, region already set to
boundary@PERMANENT res=1.
"""

import os
import sys

import numpy as np

import grass.script as gs
import grass.script.array as garray

CONFIG = {
    "PRE_DEM": "lidar_2017_DTM_lure@PERMANENT",
    "NDVI_DIFF": "ndvi_diff_lure",       # imported via r.proj from
                                         # ndvi_change@test_data_extent
                                         # (helene_chimney_11_25), bilinear, res=1
    "WATER_RASTER": "lure_water_mask",   # pre-built raster (1=water), see docstring --
                                         # NOT a vector like the watershed CONFIG's
                                         # WATER_VECT; built ahead of this script
    "DISTURB_VECT": None,                # no mapped disturbance polygons for this
                                         # reach -- shoreline/inflow, not a scar
    "ROAD_POINTS": "reg_points@PERMANENT",

    "OUT_MASK": "stable_mask_lure",

    "NDVI_ABS_MAX": 0.10,
    "CHANNEL_THRESH_CELLS": 2000,
    "CHANNEL_BUFFER_M": 30.0,
    "SLOPE_MAX_DEG": 45.0,
    "DISTURB_BUFFER_M": 25.0,
    "MIN_COVERAGE_FRACTION": 0.03,
}


def msg(t):
    print(f"\n=== {t}", flush=True)


def main():
    if not os.environ.get("GISRC"):
        sys.exit("Not inside a GRASS session.")

    cfg = CONFIG
    print(gs.read_command("g.region", flags="p"))

    msg("1. Terrain criteria from the PRE-event DEM (2017 lidar)")
    gs.run_command("r.slope.aspect", elevation=cfg["PRE_DEM"],
                   slope="sml_slope", overwrite=True, quiet=True)
    gs.run_command("r.watershed", elevation=cfg["PRE_DEM"],
                   accumulation="sml_accum", flags="a",
                   overwrite=True, quiet=True)
    gs.mapcalc(f"sml_channel = if(abs(sml_accum) > {cfg['CHANNEL_THRESH_CELLS']}, "
               f"1, null())", overwrite=True, quiet=True)
    gs.run_command("r.grow.distance", input="sml_channel",
                   distance="sml_chan_dist", overwrite=True, quiet=True)
    print(f"  channel network extracted (>{cfg['CHANNEL_THRESH_CELLS']} cells), "
          f"buffer {cfg['CHANNEL_BUFFER_M']} m")

    msg("2. Spectral criterion from NDVI change")
    gs.mapcalc(f"sml_ndvi_ok = if(abs({cfg['NDVI_DIFF']}) < "
               f"{cfg['NDVI_ABS_MAX']}, 1, 0)", overwrite=True, quiet=True)
    print(f"  |dNDVI| < {cfg['NDVI_ABS_MAX']}")

    msg("3. Exclusion zones")
    exclusions = [cfg["WATER_RASTER"]]
    print(f"  water excluded ({cfg['WATER_RASTER']}, lidar-flatness-derived -- see docstring)")

    if cfg["DISTURB_VECT"]:
        gs.run_command("v.buffer", input=cfg["DISTURB_VECT"],
                       output="sml_disturb_buf", distance=cfg["DISTURB_BUFFER_M"],
                       overwrite=True, quiet=True)
        gs.run_command("v.to.rast", input="sml_disturb_buf", output="sml_disturb",
                       use="val", value=1, overwrite=True, quiet=True)
        exclusions.append("sml_disturb")
        print(f"  disturbance polygons excluded (+{cfg['DISTURB_BUFFER_M']} m buffer)")
    else:
        print("  no disturbance polygons for this reach (shoreline/inflow, not a scar)")

    # BUG (found and fixed here, present in 02_build_stable_mask.py's ORIGINAL
    # form too but never exercised there -- the watershed run had both
    # exclusions=None, so excl_expr fell back to the literal "0" branch and
    # never actually evaluated this expression): `e == 1` is NULL, not 0/False,
    # wherever e itself is null (GRASS mapcalc comparison operators propagate
    # NULL rather than treating null-vs-number as False). `!isnull(e) && e==1`
    # therefore evaluates to NULL (not 0) on every non-excluded cell, and
    # `!(...)` of a NULL condition is also NULL -- so the final if() saw a
    # NULL condition EVERYWHERE and returned null() = its own else-branch,
    # zeroing out the whole mask regardless of any other criterion. Confirmed
    # by testing in isolation: the raw expression gave n=0 cells. Fixed by
    # coercing each exclusion raster to a real 0/1 (never null) first.
    excl_expr = " || ".join(f"(if(isnull({e}),0,{e}) == 1)" for e in exclusions)

    msg("4. Combine")
    expr = (
        f"{cfg['OUT_MASK']} = if("
        f"  sml_ndvi_ok == 1"
        f"  && sml_slope < {cfg['SLOPE_MAX_DEG']}"
        f"  && sml_chan_dist > {cfg['CHANNEL_BUFFER_M']}"
        f"  && !({excl_expr})"
        f"  && !isnull({cfg['PRE_DEM']}),"
        f"  1, null())"
    )
    gs.mapcalc(expr, overwrite=True, quiet=True)

    msg("5. Coverage check")
    total = int(gs.parse_command("r.univar", map=cfg["PRE_DEM"],
                                 flags="g", quiet=True)["n"])
    kept = int(gs.parse_command("r.univar", map=cfg["OUT_MASK"],
                                flags="g", quiet=True)["n"])
    frac = kept / total if total else 0
    print(f"  stable cells: {kept:,} of {total:,}  ({100*frac:.1f}% of region)")
    if frac < cfg["MIN_COVERAGE_FRACTION"]:
        print("  !! Very small mask.")
    if frac > 0.75:
        print("  !! Very large mask -- probably too permissive.")

    msg("6. Canopy-range check (uses 2017 DSM)")
    gs.mapcalc("sml_canopy_chk = lidar_2017_DSM_lure@PERMANENT - "
              f"{cfg['PRE_DEM']}", overwrite=True, quiet=True)
    print(gs.read_command("r.univar", map="sml_canopy_chk", zones=cfg["OUT_MASK"],
                          flags="g"))

    if cfg["ROAD_POINTS"]:
        msg("7. Cross-check against reg_points")
        gs.run_command("v.what.rast", map=cfg["ROAD_POINTS"],
                       raster=cfg["OUT_MASK"], column="in_mask_lure", quiet=True)
        sel = gs.read_command("v.db.select", map=cfg["ROAD_POINTS"],
                              columns="cat,in_mask_lure")
        print(sel)

    msg("done")


if __name__ == "__main__":
    main()
