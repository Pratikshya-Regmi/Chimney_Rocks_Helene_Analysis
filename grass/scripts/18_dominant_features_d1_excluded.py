#!/usr/bin/env python3
"""
18_dominant_features_d1_excluded.py
=======================================
Supersedes the watershed half of 13_dominant_features_both_sites.py now
that D1 (deposition rank 1, +27,343 m3, 49.7% of gross deposition) is
resolved as an artifact. Identical method (uniform 0.32 m LoD, r.clump,
r.volume, ranked by |volume|) with D1's footprint (d1_feature_mask)
excluded from the deposition mask before clumping.

Lake Lure's table (dominant_features_lure.csv) is UNCHANGED -- D1 is a
watershed-only feature and does not touch Lake Lure's mask or DoD -- so it
is not recomputed here, only re-cited.

Output: results/tables/dominant_features_watershed_d1_excluded.csv
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

LOD = 0.32


def rank_features(gs, dh, exclude_mask=None):
    excl_clause = f"isnull({exclude_mask}) && " if exclude_mask else ""
    gs.mapcalc(f"_d18_tmp = if(abs({dh}) > {LOD}, {dh}, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_d18_ero = if(_d18_tmp < 0, _d18_tmp, null())",
              overwrite=True, quiet=True)
    gs.mapcalc(f"_d18_dep = if({excl_clause}_d18_tmp > 0, _d18_tmp, null())",
              overwrite=True, quiet=True)

    results = {}
    for kind, dhmap in (("erosion", "_d18_ero"), ("deposition", "_d18_dep")):
        gs.mapcalc(f"_d18_bin = if(!isnull({dhmap}), 1, null())",
                  overwrite=True, quiet=True)
        gs.run_command("r.clump", input="_d18_bin", output="_d18_clump",
                       overwrite=True, quiet=True)
        raw = gs.read_command("r.volume", input=dhmap, clump="_d18_clump",
                              format="json")
        rows = json.loads(raw)
        rows = sorted(rows, key=lambda r: -abs(r["volume"]))
        results[kind] = rows
    return results["erosion"], results["deposition"]


def write_csv(path, ero_rows, dep_rows, total_ero_vol, total_dep_vol):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "rank", "area_m2", "mean_depth_m", "volume_m3",
                   "pct_of_gross", "easting", "northing", "d1_excluded"])
        for kind, rows, total in (("erosion", ero_rows, total_ero_vol),
                                  ("deposition", dep_rows, total_dep_vol)):
            for i, r in enumerate(rows, start=1):
                pct = 100.0 * abs(r["volume"]) / abs(total) if total else 0
                w.writerow([kind, i, r["cells"], f"{r['average']:.3f}",
                           f"{r['volume']:.2f}", f"{pct:.2f}",
                           f"{r['easting']:.1f}", f"{r['northing']:.1f}",
                           "yes"])


def main():
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")

    found = gs.find_file(name="d1_feature_mask", element="cell", mapset="for_codem")
    if not found or not found.get("name"):
        sys.exit("d1_feature_mask not found -- run 16_resolve_d1_deposition.py first.")

    ero, dep = rank_features(gs, "coreg_dh_corrected", exclude_mask="d1_feature_mask")
    tot_ero = sum(r["volume"] for r in ero)
    tot_dep = sum(r["volume"] for r in dep)

    print(f"\n=== WATERSHED (cat34), D1 excluded: top 8 erosion "
          f"(of {len(ero)} total) ===")
    print(f"{'rank':>4s} {'area_m2':>9s} {'depth_m':>8s} {'vol_m3':>10s} "
          f"{'%gross':>7s} {'easting':>11s} {'northing':>11s}")
    for i, r in enumerate(ero[:8], start=1):
        pct = 100.0 * abs(r["volume"]) / abs(tot_ero) if tot_ero else 0
        print(f"{i:4d} {r['cells']:9,d} {r['average']:8.2f} {r['volume']:10,.1f} "
              f"{pct:7.1f} {r['easting']:11.1f} {r['northing']:11.1f}")

    print(f"\n=== WATERSHED (cat34), D1 excluded: top 8 deposition "
          f"(of {len(dep)} total) ===")
    print(f"{'rank':>4s} {'area_m2':>9s} {'depth_m':>8s} {'vol_m3':>10s} "
          f"{'%gross':>7s} {'easting':>11s} {'northing':>11s}")
    for i, r in enumerate(dep[:8], start=1):
        pct = 100.0 * abs(r["volume"]) / abs(tot_dep) if tot_dep else 0
        print(f"{i:4d} {r['cells']:9,d} {r['average']:8.2f} {r['volume']:10,.1f} "
              f"{pct:7.1f} {r['easting']:11.1f} {r['northing']:11.1f}")

    top8_ero_pct = 100 * sum(abs(r["volume"]) for r in ero[:8]) / abs(tot_ero)
    top8_dep_pct = 100 * sum(abs(r["volume"]) for r in dep[:8]) / abs(tot_dep)
    print(f"\ntop 8 erosion = {top8_ero_pct:.1f}% of gross erosion")
    print(f"top 8 deposition = {top8_dep_pct:.1f}% of gross deposition "
          f"(was 70.6% including D1 -- D1 alone was 49.7% of gross, so "
          f"removing it changes both the ranking and the top-8 share)")

    out_csv = TABLES / "dominant_features_watershed_d1_excluded.csv"
    write_csv(out_csv, ero, dep, tot_ero, tot_dep)
    print(f"\nwrote {out_csv} ({len(ero)} erosion + {len(dep)} deposition rows)")

    gs.run_command("g.remove", type="raster",
                   name="_d18_tmp,_d18_ero,_d18_dep,_d18_bin,_d18_clump",
                   flags="f", quiet=True)

    log("18_dominant_features_d1_excluded.py run. Supersedes the watershed "
        "half of 13_dominant_features_both_sites.py -- D1 excluded from "
        "the deposition mask before clumping. LoD=0.32m. "
        f"Watershed (D1 excluded): {len(ero)} erosion features "
        f"(top1={ero[0]['volume']:.0f}m3), {len(dep)} deposition features "
        f"(top1={dep[0]['volume']:.0f}m3 @ {dep[0]['easting']:.0f}E "
        f"{dep[0]['northing']:.0f}N, {100*abs(dep[0]['volume'])/abs(tot_dep):.1f}% "
        f"of D1-excluded gross deposition). top8 erosion={top8_ero_pct:.1f}% "
        f"of gross, top8 deposition={top8_dep_pct:.1f}% of gross. "
        "Lake Lure's dominant_features_lure.csv is unaffected by D1 and not "
        f"recomputed here. Written to {out_csv.name}.",
        run="dominant_features_d1_excluded")


if __name__ == "__main__":
    main()
