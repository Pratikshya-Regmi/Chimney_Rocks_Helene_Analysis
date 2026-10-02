#!/usr/bin/env python3
"""
13_dominant_features_both_sites.py
====================================
Dominant individual erosion/deposition features at both sites, identical
method: threshold the lidar-lidar DoD at the uniform 0.32 m LoD, split into
erosion/deposition, r.clump (4-connectivity), r.volume per clump (area,
mean depth, volume, centroid). Ranked by |volume| descending. Rebuilds the
manuscript's Table 4 (Lake Lure feature table) from the corrected DoD, and
produces the same for the watershed (which the manuscript does not
currently have).

Orthophoto identification (scar / channel / inflow deposit / etc.) is done
separately, by hand, against each top feature's centroid -- see
RESULTS_FOR_PAPER.md for the write-up; this script only produces the
ranked numeric table plus a centroid-marked overlay figure to do that
identification from.

Outputs:
  results/tables/dominant_features_watershed.csv   (full ranked clump list)
  results/tables/dominant_features_lure.csv         (full ranked clump list)
  results/figures/dominant_features_watershed_map.png
  results/figures/dominant_features_lure_map.png
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
FIGURES = Path(__file__).resolve().parents[2] / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

LOD = 0.32


def rank_features(gs, dh, water_mask=None):
    """Returns (erosion_rows, deposition_rows), each a list of dicts sorted
    by |volume| descending, from r.volume run on 0.32 m-thresholded clumps."""
    water_clause = f"isnull({water_mask}) && " if water_mask else ""
    gs.mapcalc(f"_dom_tmp = if({water_clause}abs({dh}) > {LOD}, {dh}, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_dom_ero = if(_dom_tmp < 0, _dom_tmp, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_dom_dep = if(_dom_tmp > 0, _dom_tmp, null())",
              overwrite=True, quiet=True)

    results = {}
    for kind, dhmap in (("erosion", "_dom_ero"), ("deposition", "_dom_dep")):
        gs.mapcalc(f"_dom_bin = if(!isnull({dhmap}), 1, null())",
                  overwrite=True, quiet=True)
        gs.run_command("r.clump", input="_dom_bin", output="_dom_clump",
                       overwrite=True, quiet=True)
        rows = gs.parse_command("r.volume", input=dhmap, clump="_dom_clump",
                                format="json")
        # parse_command with format=json on r.volume returns the raw dict
        # structure already (a list under some modules, but r.volume's json
        # is a bare top-level list) -- handle both possibilities defensively
        import json as _json
        if isinstance(rows, str):
            rows = _json.loads(rows)
        elif hasattr(rows, "get") and "volume" not in rows and len(rows) and \
                not isinstance(list(rows.values())[0], (int, float, str)):
            # unexpected dict-of-something; fall back to raw command
            raw = gs.read_command("r.volume", input=dhmap, clump="_dom_clump",
                                  format="json")
            rows = _json.loads(raw)
        rows = sorted(rows, key=lambda r: -abs(r["volume"]))
        results[kind] = rows
    return results["erosion"], results["deposition"]


def write_csv(path, ero_rows, dep_rows, total_ero_vol, total_dep_vol):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "rank", "area_m2", "mean_depth_m", "volume_m3",
                   "pct_of_gross", "easting", "northing"])
        for kind, rows, total in (("erosion", ero_rows, total_ero_vol),
                                  ("deposition", dep_rows, total_dep_vol)):
            for i, r in enumerate(rows, start=1):
                pct = 100.0 * abs(r["volume"]) / abs(total) if total else 0
                w.writerow([kind, i, r["cells"], f"{r['average']:.3f}",
                           f"{r['volume']:.2f}", f"{pct:.2f}",
                           f"{r['easting']:.1f}", f"{r['northing']:.1f}"])


def print_top(label, ero_rows, dep_rows, total_ero_vol, total_dep_vol, n=8):
    print(f"\n=== {label}: top {n} erosion features (of {len(ero_rows)} total) ===")
    print(f"{'rank':>4s} {'area_m2':>9s} {'depth_m':>8s} {'vol_m3':>10s} "
          f"{'%gross':>7s} {'easting':>11s} {'northing':>11s}")
    for i, r in enumerate(ero_rows[:n], start=1):
        pct = 100.0 * abs(r["volume"]) / abs(total_ero_vol) if total_ero_vol else 0
        print(f"{i:4d} {r['cells']:9,d} {r['average']:8.2f} {r['volume']:10,.1f} "
              f"{pct:7.1f} {r['easting']:11.1f} {r['northing']:11.1f}")
    cum = sum(abs(r["volume"]) for r in ero_rows[:n])
    print(f"  top {n} = {100*cum/abs(total_ero_vol):.1f}% of gross erosion "
          f"({cum:,.0f} of {abs(total_ero_vol):,.0f} m3)")

    print(f"\n=== {label}: top {n} deposition features (of {len(dep_rows)} total) ===")
    print(f"{'rank':>4s} {'area_m2':>9s} {'depth_m':>8s} {'vol_m3':>10s} "
          f"{'%gross':>7s} {'easting':>11s} {'northing':>11s}")
    for i, r in enumerate(dep_rows[:n], start=1):
        pct = 100.0 * abs(r["volume"]) / abs(total_dep_vol) if total_dep_vol else 0
        print(f"{i:4d} {r['cells']:9,d} {r['average']:8.2f} {r['volume']:10,.1f} "
              f"{pct:7.1f} {r['easting']:11.1f} {r['northing']:11.1f}")
    cum = sum(abs(r["volume"]) for r in dep_rows[:n])
    print(f"  top {n} = {100*cum/abs(total_dep_vol):.1f}% of gross deposition "
          f"({cum:,.0f} of {abs(total_dep_vol):,.0f} m3)")


def main():
    # -------------------------------------------------------- watershed
    gs_w, _ = grass_session(project="DEM_generation", mapset="for_codem")
    gs_w.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")
    ero_w, dep_w = rank_features(gs_w, "coreg_dh_corrected")
    tot_ero_w = sum(r["volume"] for r in ero_w)
    tot_dep_w = sum(r["volume"] for r in dep_w)
    print_top("WATERSHED (cat34)", ero_w, dep_w, tot_ero_w, tot_dep_w)
    csv_w = TABLES / "dominant_features_watershed.csv"
    write_csv(csv_w, ero_w, dep_w, tot_ero_w, tot_dep_w)
    print(f"\nwrote {csv_w} ({len(ero_w)} erosion + {len(dep_w)} deposition rows)")

    # -------------------------------------------------------- Lake Lure
    gs_l, _ = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    gs_l.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")
    ero_l, dep_l = rank_features(gs_l, "coreg_lure_lidar_dh_corrected",
                                 water_mask="lure_water_mask")
    tot_ero_l = sum(r["volume"] for r in ero_l)
    tot_dep_l = sum(r["volume"] for r in dep_l)
    print_top("LAKE LURE", ero_l, dep_l, tot_ero_l, tot_dep_l)
    csv_l = TABLES / "dominant_features_lure.csv"
    write_csv(csv_l, ero_l, dep_l, tot_ero_l, tot_dep_l)
    print(f"\nwrote {csv_l} ({len(ero_l)} erosion + {len(dep_l)} deposition rows)")

    for gs in (gs_w, gs_l):
        gs.run_command("g.remove", type="raster",
                       name="_dom_tmp,_dom_ero,_dom_dep,_dom_bin,_dom_clump",
                       flags="f", quiet=True)

    log("13_dominant_features_both_sites.py run. LoD=0.32m both sites. "
        f"Watershed: {len(ero_w)} erosion features (top1={ero_w[0]['volume']:.0f}m3 "
        f"@ {ero_w[0]['easting']:.0f}E {ero_w[0]['northing']:.0f}N), "
        f"{len(dep_w)} deposition features (top1={dep_w[0]['volume']:.0f}m3 "
        f"@ {dep_w[0]['easting']:.0f}E {dep_w[0]['northing']:.0f}N). "
        f"Lake Lure: {len(ero_l)} erosion features (top1={ero_l[0]['volume']:.0f}m3 "
        f"@ {ero_l[0]['easting']:.0f}E {ero_l[0]['northing']:.0f}N), "
        f"{len(dep_l)} deposition features (top1={dep_l[0]['volume']:.0f}m3 "
        f"@ {dep_l[0]['easting']:.0f}E {dep_l[0]['northing']:.0f}N). "
        f"Written to {csv_w.name}, {csv_l.name}.",
        run="dominant_features")


if __name__ == "__main__":
    main()
