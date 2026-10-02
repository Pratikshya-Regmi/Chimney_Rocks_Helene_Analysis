#!/usr/bin/env python3
"""
11_lure_lod_min_feature_area.py
=================================
Lake Lure, uniform 0.32 m LoD, Pair 1 (lidar 2017->2024): recomputes gross
erosion, gross deposition, net volume, feature count and mean feature size
with a MINIMUM FEATURE AREA applied on top of the plain |dh| > 0.32 m
threshold, at 50 m^2 and 100 m^2 -- matching the r.clump minimum-area
criterion the manuscript's Methods already specifies -- alongside the
existing no-minimum row from 07_lure_lod_sensitivity.py for comparison.

Method: threshold -> binarize -> r.clump (4-connectivity) -> r.stats for
per-clump cell counts -> keep only clumps with count >= MIN_AREA_M2 (1 m
resolution, so cell count == area in m^2) -> recompute volume/area/feature
count over the survivors only. Water excluded throughout (same as
07_lure_lod_sensitivity.py's stage4-equivalent fix).

Output: results/tables/lure_lod_min_feature_area.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

LOD = 0.32
MIN_AREAS = [0, 50, 100]  # m^2; 0 = no minimum (matches the existing row)


def build_clump_sizes(gs, binmap, sizemap_name):
    """r.clump `binmap` (a 0/1-style presence raster), then paint each
    cell with ITS OWN clump's total cell count via r.stats.zonal-style
    reclass, so a single mapcalc comparison (>= MIN_AREA_M2) can filter by
    feature size."""
    gs.run_command("r.clump", input=binmap, output="_lure_mfa_clump",
                   overwrite=True, quiet=True)
    stats = gs.read_command("r.stats", input="_lure_mfa_clump", flags="cn")
    sizes = {}
    for line in stats.strip().split("\n"):
        cat, cnt = line.split()
        sizes[int(cat)] = int(cnt)
    rules_path = "/tmp/_lure_mfa_reclass_rules.txt"
    with open(rules_path, "w") as f:
        for cat, cnt in sizes.items():
            f.write(f"{cat} = {cnt}\n")
    gs.run_command("r.reclass", input="_lure_mfa_clump", output=sizemap_name,
                   rules=rules_path, overwrite=True, quiet=True)
    return sizes


def compute_one(gs, label, dh, water, min_area):
    gs.mapcalc(f"_lure_mfa_tmp = if(isnull({water}) && abs({dh}) > {LOD}, "
              f"{dh}, null())", overwrite=True, quiet=True)
    gs.mapcalc("_lure_mfa_ero = if(_lure_mfa_tmp < 0, _lure_mfa_tmp, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_lure_mfa_dep = if(_lure_mfa_tmp > 0, _lure_mfa_tmp, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_lure_mfa_ero_bin = if(!isnull(_lure_mfa_ero), 1, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_lure_mfa_dep_bin = if(!isnull(_lure_mfa_dep), 1, null())",
              overwrite=True, quiet=True)

    row = {"min_area_m2": min_area, "label": label}
    for kind, dhmap, binmap in (("erosion", "_lure_mfa_ero", "_lure_mfa_ero_bin"),
                               ("deposition", "_lure_mfa_dep", "_lure_mfa_dep_bin")):
        sizes = build_clump_sizes(gs, binmap, "_lure_mfa_sizes")
        if min_area > 0:
            gs.mapcalc(f"_lure_mfa_filtered = if(_lure_mfa_sizes >= {min_area}, "
                      f"{dhmap}, null())", overwrite=True, quiet=True)
        else:
            gs.run_command("g.copy", raster=f"{dhmap},_lure_mfa_filtered",
                           overwrite=True, quiet=True)

        uni = gs.parse_command("r.univar", map="_lure_mfa_filtered", flags="ge",
                               quiet=True)
        n = int(uni.get("n", 0))
        vol = float(uni.get("sum", 0))
        n_features_surviving = sum(1 for c in sizes.values() if c >= min_area)

        row[f"{kind}_volume_m3"] = vol
        row[f"{kind}_area_m2"] = n
        row[f"n_{kind}_features"] = n_features_surviving
        row[f"mean_{kind}_feature_size_m2"] = (n / n_features_surviving
                                               if n_features_surviving else 0)

    row["net_volume_m3"] = row["erosion_volume_m3"] + row["deposition_volume_m3"]
    return row


def main():
    gs, gj = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    gs.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")

    dh = "coreg_lure_lidar_dh_corrected"
    water = "lure_water_mask"

    rows = []
    for min_area in MIN_AREAS:
        label = (f"uniform {LOD:.2f} m, no minimum feature area" if min_area == 0
                 else f"uniform {LOD:.2f} m, min feature area >= {min_area} m2")
        r = compute_one(gs, label, dh, water, min_area)
        rows.append(r)
        print(f"\n  {label}")
        print(f"    erosion    : {r['erosion_volume_m3']:+,.0f} m3, "
              f"area {r['erosion_area_m2']:,} m2, "
              f"{r['n_erosion_features']:,} features, "
              f"mean {r['mean_erosion_feature_size_m2']:.1f} m2/feature")
        print(f"    deposition : {r['deposition_volume_m3']:+,.0f} m3, "
              f"area {r['deposition_area_m2']:,} m2, "
              f"{r['n_deposition_features']:,} features, "
              f"mean {r['mean_deposition_feature_size_m2']:.1f} m2/feature")
        print(f"    net        : {r['net_volume_m3']:+,.0f} m3")

    out_csv = TABLES / "lure_lod_min_feature_area.csv"
    fieldnames = ["min_area_m2", "label", "erosion_volume_m3", "erosion_area_m2",
                 "n_erosion_features", "mean_erosion_feature_size_m2",
                 "deposition_volume_m3", "deposition_area_m2",
                 "n_deposition_features", "mean_deposition_feature_size_m2",
                 "net_volume_m3"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    scratch = ["_lure_mfa_tmp", "_lure_mfa_ero", "_lure_mfa_dep",
              "_lure_mfa_ero_bin", "_lure_mfa_dep_bin", "_lure_mfa_clump",
              "_lure_mfa_sizes", "_lure_mfa_filtered"]
    gs.run_command("g.remove", type="raster", name=",".join(scratch),
                   flags="f", quiet=True)
    print(f"({len(scratch)} scratch rasters removed -- disposable "
          "intermediates re-created fresh per (min_area x erosion/"
          "deposition) loop iteration, not results)")

    log("11_lure_lod_min_feature_area.py run. LoD=0.32m, min feature area "
        f"in {MIN_AREAS} m2 (0=no minimum, matches "
        "07_lure_lod_sensitivity.py's existing row). Results: " +
        "; ".join(f"min{r['min_area_m2']}m2: net={r['net_volume_m3']:+,.0f}m3 "
                 f"(ero {r['n_erosion_features']}feat/{r['mean_erosion_feature_size_m2']:.0f}m2avg, "
                 f"dep {r['n_deposition_features']}feat/{r['mean_deposition_feature_size_m2']:.0f}m2avg)"
                 for r in rows) +
        f". Written to {out_csv.name}.", run="lure_lod_min_feature_area")


if __name__ == "__main__":
    main()
