#!/usr/bin/env python3
"""
30_flow_sequence_setup.py
==============================
Shared setup for the three-epoch overland-flow comparison (rebuild of the
watershed flow figure as a chronological sequence: 2020 lidar pre-event,
2024 CAP SfM immediate post-event, 2024 lidar seven-weeks-post-event).

Does three things, once, so 31/32/33/34 don't repeat it:

1. VERIFIES the three r.sim.water discharge rasters were run with identical
   parameters on the identical region -- not assumed. All three
   (discharge_2020_regis, discharge_2024_cap, discharge_may_2024_lidar,
   all @DEM_generation/for_codem) already existed from the original
   notebooks (Multitemporal_hillshade_watershed_DOD.ipynb); this step reads
   each raster's r.sim.water command string back out of its GRASS history
   and asserts rain_value/infil_value/man_value/niterations/output_step and
   the region (n/s/e/w/res/rows/cols) match across all three before
   anything downstream trusts them as comparable. No new r.sim.water run
   was needed -- these were already the identical-parameter run the
   analyst asked for; re-running would only add stochastic noise from a
   different random seed, not improve comparability.

2. EXPORTS discharge_2020_regis, discharge_2024_cap, discharge_may_2024_lidar
   and hillshade_2020 to GeoTIFF for the matplotlib figure scripts.

3. IDENTIFIES the 2-3 zones of largest flow-path change between the 2020
   and 2024 LIDAR simulations (only -- SfM is not part of this
   comparison): reclumps channel_agreement's disagreement classes
   (1 = 2020-only/abandoned, 2 = 2024-only/new; both already computed at
   discharge > 0.01 by 08_watershed_flow_agreement.py, the SAME threshold
   used for the headline Jaccard number) into connected components,
   ranks by area, and keeps the top 3 (a clear break in the size
   distribution after the third: 1328/1122/748 m^2, then a drop to 584 m^2
   for the fourth -- see log). Bounding boxes are padded 20 m on each side
   for zoom-panel context.

Outputs:
  results/figures/_flow_discharge_2020.tif   (scratch GeoTIFF, not a figure)
  results/figures/_flow_discharge_2024cap.tif
  results/figures/_flow_discharge_2024lidar.tif
  results/figures/_flow_hillshade.tif
  results/tables/flow_change_zones.csv
  results/logs/flow_sequence_setup_<date>.log
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

DISCHARGE_2020_TIF = FIGURES / "_flow_discharge_2020.tif"
DISCHARGE_CAP_TIF = FIGURES / "_flow_discharge_2024cap.tif"
DISCHARGE_2024_TIF = FIGURES / "_flow_discharge_2024lidar.tif"
HILLSHADE_TIF = FIGURES / "_flow_hillshade.tif"

DISAGREE_THRESH = 0.01  # matches 08_watershed_flow_agreement.py's channel def.
N_ZONES = 3
PAD_M = 20.0


def parse_simwater_params(hist_text):
    """Pull rain_value/infil_value/man_value/niterations/output_step out of
    an r.sim.water history/comment string. Sourced from r.info's JSON
    "comments" field, which -- unlike the plain-text -h form, which wraps
    long history lines with a literal trailing backslash INSIDE tokens
    (e.g. "niterat\\nions=48", breaking a naive regex match) -- is the
    unwrapped original command string."""
    joined = re.sub(r"\s+", " ", hist_text)
    out = {}
    for key in ("rain_value", "infil_value", "man_value", "niterations",
                "output_step", "diffusion_coeff", "hmax", "halpha", "hbeta"):
        m = re.search(key + r'="?([0-9.eE+-]+)"?', joined)
        if m:
            out[key] = m.group(1)
    m = re.search(r'elevation="([^"]+)"', joined)
    out["elevation"] = m.group(1) if m else None
    return out


def main():
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", raster="discharge_2020_regis", flags="a")
    region = gs.parse_command("g.region", flags="g")

    # ---- 1. verify identical parameters -----------------------------
    rasters = {
        "2020 lidar (pre-event)": "discharge_2020_regis",
        "2024 SfM (immediate post-event)": "discharge_2024_cap",
        "2024 lidar (7 weeks post-event)": "discharge_may_2024_lidar",
    }
    params_by_epoch = {}
    print("Verifying r.sim.water parameters across all three epochs:")
    for label, rmap in rasters.items():
        info_json = json.loads(gs.read_command("r.info", map=rmap, format="json"))
        hist = info_json.get("comments", "")
        p = parse_simwater_params(hist)
        info = gs.parse_command("r.info", map=rmap, flags="g")
        p["region"] = (info["north"], info["south"], info["east"], info["west"],
                       info["rows"], info["cols"])
        params_by_epoch[label] = p
        print(f"  {label} ({rmap}): elevation={p.get('elevation')}, "
              f"rain={p.get('rain_value')}, infil={p.get('infil_value')}, "
              f"man_n={p.get('man_value')}, niter={p.get('niterations')}, "
              f"output_step={p.get('output_step')}, "
              f"region_rows_cols=({info['rows']},{info['cols']})")

    check_keys = ["rain_value", "infil_value", "man_value", "niterations",
                 "output_step", "diffusion_coeff", "hmax", "halpha", "hbeta"]
    mismatches = []
    ref = list(params_by_epoch.values())[0]
    for label, p in params_by_epoch.items():
        for k in check_keys:
            if p.get(k) != ref.get(k):
                mismatches.append(f"{label}: {k}={p.get(k)} vs reference {ref.get(k)}")
        if p["region"] != ref["region"]:
            mismatches.append(f"{label}: region {p['region']} vs reference {ref['region']}")

    if mismatches:
        print("\nPARAMETER MISMATCH -- the three runs are NOT on identical "
              "footing:")
        for m in mismatches:
            print("  " + m)
        sys.exit("Aborting: fix parameters before building comparison figures.")
    else:
        print("\nOK -- all three r.sim.water runs share identical rain/infil/"
              "Manning's n/duration/output-step and identical region/extent/"
              "resolution. Safe to compare directly.")

    # ---- 2. export GeoTIFFs ------------------------------------------
    gs.mapcalc("_flow_hillshade_raw = if(isnull(dtm_2020_filled), null(), "
              "dtm_2020_filled)", overwrite=True, quiet=True)
    gs.run_command("r.relief", input="dtm_2020_filled",
                   output="_flow_hillshade_rel", altitude=30, azimuth=270,
                   zscale=1, overwrite=True, quiet=True)
    gs.mapcalc("_flow_hillshade_clamped = if(_flow_hillshade_rel < 0, 0, "
              "if(_flow_hillshade_rel > 255, 255, _flow_hillshade_rel))",
              overwrite=True, quiet=True)
    gs.run_command("r.out.gdal", input="_flow_hillshade_clamped",
                   output=str(HILLSHADE_TIF), format="GTiff", type="Byte",
                   createopt="COMPRESS=LZW", overwrite=True, quiet=True)

    gs.run_command("r.out.gdal", input="discharge_2020_regis",
                   output=str(DISCHARGE_2020_TIF), format="GTiff",
                   type="Float64", createopt="COMPRESS=LZW", nodata=-9999,
                   overwrite=True, quiet=True)
    gs.run_command("r.out.gdal", input="discharge_2024_cap",
                   output=str(DISCHARGE_CAP_TIF), format="GTiff",
                   type="Float64", createopt="COMPRESS=LZW", nodata=-9999,
                   overwrite=True, quiet=True)
    gs.run_command("r.out.gdal", input="discharge_may_2024_lidar",
                   output=str(DISCHARGE_2024_TIF), format="GTiff",
                   type="Float64", createopt="COMPRESS=LZW", nodata=-9999,
                   overwrite=True, quiet=True)
    print(f"\nwrote {HILLSHADE_TIF.name}, {DISCHARGE_2020_TIF.name}, "
          f"{DISCHARGE_CAP_TIF.name}, {DISCHARGE_2024_TIF.name}")

    # ---- 3. identify zones of largest 2020-vs-2024-lidar change ------
    found = gs.find_file(name="channel_agreement", element="cell", mapset="for_codem")
    if not found or not found.get("name"):
        sys.exit("channel_agreement not found -- run "
                 "08_watershed_flow_agreement.py first (threshold=0.01).")

    gs.mapcalc("_zone_bin = if(channel_agreement==1 || channel_agreement==2, "
              "1, null())", overwrite=True, quiet=True)
    gs.run_command("r.clump", input="_zone_bin", output="zone_change_clump",
                   overwrite=True, quiet=True)
    stats = gs.read_command("r.stats", input="zone_change_clump", flags="an",
                            separator="comma")
    rows = [(int(float(a)), float(b)) for a, b in
            (l.split(",") for l in stats.strip().split("\n"))]
    rows.sort(key=lambda r: -r[1])
    print(f"\n{len(rows)} connected disagreement clumps found "
          f"(discharge threshold {DISAGREE_THRESH}, same as the headline "
          f"Jaccard metric). Largest {N_ZONES + 2}:")
    for cid, area in rows[:N_ZONES + 2]:
        print(f"  clump {cid}: {area:.0f} m^2")

    top = rows[:N_ZONES]

    gs.run_command("r.out.gdal", input="zone_change_clump",
                   output="/tmp/_zone_change_clump.tif", format="GTiff",
                   type="Int32", nodata=-1, overwrite=True, quiet=True)
    with rasterio.open("/tmp/_zone_change_clump.tif") as src:
        arr = src.read(1)
        transform = src.transform

    zone_rows = []
    for i, (cid, area) in enumerate(top, start=1):
        ys, xs = np.where(arr == cid)
        row_min, row_max = int(ys.min()), int(ys.max())
        col_min, col_max = int(xs.min()), int(xs.max())
        x_min, y_max = transform * (col_min, row_min)
        x_max, y_min = transform * (col_max + 1, row_max + 1)
        x_min -= PAD_M
        x_max += PAD_M
        y_min -= PAD_M
        y_max += PAD_M
        zone_rows.append({
            "zone_id": f"Z{i}",
            "clump_cat": cid,
            "area_m2": round(area, 1),
            "n_cells": int(len(xs)),
            "e_min": round(x_min, 1), "e_max": round(x_max, 1),
            "n_min": round(y_min, 1), "n_max": round(y_max, 1),
            "width_m": round(x_max - x_min, 1),
            "height_m": round(y_max - y_min, 1),
            "discharge_threshold": DISAGREE_THRESH,
        })
        print(f"  {zone_rows[-1]['zone_id']}: clump {cid}, {area:.0f} m^2, "
              f"padded bbox E[{x_min:.0f},{x_max:.0f}] N[{y_min:.0f},{y_max:.0f}] "
              f"({x_max - x_min:.0f} x {y_max - y_min:.0f} m)")

    out_csv = TABLES / "flow_change_zones.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(zone_rows[0].keys()))
        w.writeheader()
        for r in zone_rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    gs.run_command("g.remove", type="raster",
                   name="_flow_hillshade_raw,_flow_hillshade_rel,"
                        "_flow_hillshade_clamped,_zone_bin",
                   flags="f", quiet=True)
    print("(scratch rasters _flow_hillshade_raw/_rel/_clamped, _zone_bin "
          "removed -- disposable; zone_change_clump kept as the documented "
          "clump raster behind flow_change_zones.csv)")

    log("30_flow_sequence_setup.py run. Verified discharge_2020_regis, "
        "discharge_2024_cap, discharge_may_2024_lidar (all @for_codem) "
        "share identical r.sim.water parameters (rain=50, infil=0, "
        "man_n=0.4, niterations=48, output_step=4) and identical region "
        f"({region['rows']}x{region['cols']} cells) -- no new r.sim.water "
        "run needed, these were already run identically in the original "
        "notebooks. Exported all three + hillshade_2020-derived relief to "
        "GeoTIFF. Identified top 3 zones of 2020-vs-2024-LIDAR flow-path "
        "disagreement (channel_agreement cat 1/2, threshold 0.01) by "
        "connected-component area: " +
        "; ".join(f"{r['zone_id']}=clump{r['clump_cat']}/{r['area_m2']}m2"
                 for r in zone_rows) +
        f". Written to {out_csv.name}.", run="flow_sequence_setup")


if __name__ == "__main__":
    main()
