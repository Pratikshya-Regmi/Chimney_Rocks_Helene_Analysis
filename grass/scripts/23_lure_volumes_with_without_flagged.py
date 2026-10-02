#!/usr/bin/env python3
"""
23_lure_volumes_with_without_flagged.py
===========================================
Lake Lure gross erosion/deposition/net at uniform 0.32 m LoD, water-
excluded, 0/50/100 m^2 minimum feature area -- WITH all top-8 features
(the published/existing numbers, from 11_lure_lod_min_feature_area.py) vs.
WITHOUT the five features 22_resolve_lure_flagged_features.py investigated
and this session judged artifacts: E5, E6, E7, E8 (erosion) and D7
(deposition). Method identical to 11_lure_lod_min_feature_area.py /
17_watershed_lod_min_feature_area_d1_excluded.py.

This is a COMPUTED COMPARISON ONLY -- per instruction, not applied to any
manuscript-facing table (lure_lod_min_feature_area.csv,
sediment_budget_comparison_d1_excluded.csv, etc.) without confirmation.

Output: results/tables/lure_volumes_with_without_flagged.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

LOD = 0.32
MIN_AREAS = [0, 50, 100]
FLAGGED_MASKS = ["lure_E5_mask", "lure_E6_mask", "lure_E7_mask", "lure_E8_mask",
                "lure_D7_mask"]


def build_clump_sizes(gs, binmap, sizemap_name):
    gs.run_command("r.clump", input=binmap, output="_lv_clump",
                   overwrite=True, quiet=True)
    stats = gs.read_command("r.stats", input="_lv_clump", flags="cn")
    sizes = {}
    for line in stats.strip().split("\n"):
        cat, cnt = line.split()
        sizes[int(cat)] = int(cnt)
    rules_path = "/tmp/_lv_reclass_rules.txt"
    with open(rules_path, "w") as f:
        for cat, cnt in sizes.items():
            f.write(f"{cat} = {cnt}\n")
    gs.run_command("r.reclass", input="_lv_clump", output=sizemap_name,
                   rules=rules_path, overwrite=True, quiet=True)
    return sizes


def compute_one(gs, label, exclude_flagged, min_area):
    excl = ""
    if exclude_flagged:
        clauses = " && ".join(f"isnull({m})" for m in FLAGGED_MASKS)
        excl = clauses + " && "
    gs.mapcalc(f"_lv_tmp = if(isnull(lure_water_mask) && {excl}"
              f"abs(coreg_lure_lidar_dh_corrected) > {LOD}, "
              "coreg_lure_lidar_dh_corrected, null())", overwrite=True, quiet=True)
    gs.mapcalc("_lv_ero = if(_lv_tmp < 0, _lv_tmp, null())", overwrite=True, quiet=True)
    gs.mapcalc("_lv_dep = if(_lv_tmp > 0, _lv_tmp, null())", overwrite=True, quiet=True)
    gs.mapcalc("_lv_ero_bin = if(!isnull(_lv_ero), 1, null())", overwrite=True, quiet=True)
    gs.mapcalc("_lv_dep_bin = if(!isnull(_lv_dep), 1, null())", overwrite=True, quiet=True)

    row = {"min_area_m2": min_area, "label": label,
          "flagged_excluded": "yes" if exclude_flagged else "no"}
    for kind, dhmap, binmap in (("erosion", "_lv_ero", "_lv_ero_bin"),
                               ("deposition", "_lv_dep", "_lv_dep_bin")):
        sizes = build_clump_sizes(gs, binmap, "_lv_sizes")
        if min_area > 0:
            gs.mapcalc(f"_lv_filtered = if(_lv_sizes >= {min_area}, {dhmap}, null())",
                      overwrite=True, quiet=True)
        else:
            gs.run_command("g.copy", raster=f"{dhmap},_lv_filtered",
                           overwrite=True, quiet=True)
        uni = gs.parse_command("r.univar", map="_lv_filtered", flags="ge", quiet=True)
        n = int(uni.get("n", 0))
        vol = float(uni.get("sum", 0))
        n_feat = sum(1 for c in sizes.values() if c >= min_area)
        row[f"{kind}_volume_m3"] = vol
        row[f"{kind}_area_m2"] = n
        row[f"n_{kind}_features"] = n_feat
    row["net_volume_m3"] = row["erosion_volume_m3"] + row["deposition_volume_m3"]
    return row


def main():
    gs, gj = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    gs.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")

    for m in FLAGGED_MASKS:
        found = gs.find_file(name=m, element="cell", mapset="PERMANENT")
        if not found or not found.get("name"):
            sys.exit(f"{m} not found -- run 22_resolve_lure_flagged_features.py first.")

    rows = []
    for min_area in MIN_AREAS:
        for exclude in (False, True):
            label = (f"uniform {LOD:.2f} m, min feature area >= {min_area} m2, "
                     f"{'flagged features excluded' if exclude else 'as published (with flagged)'}")
            r = compute_one(gs, label, exclude, min_area)
            rows.append(r)
            print(f"\n  {label}")
            print(f"    erosion    : {r['erosion_volume_m3']:+,.0f} m3, "
                  f"{r['n_erosion_features']:,} features")
            print(f"    deposition : {r['deposition_volume_m3']:+,.0f} m3, "
                  f"{r['n_deposition_features']:,} features")
            print(f"    net        : {r['net_volume_m3']:+,.0f} m3")

    out_csv = TABLES / "lure_volumes_with_without_flagged.csv"
    fieldnames = ["min_area_m2", "label", "flagged_excluded", "erosion_volume_m3",
                 "erosion_area_m2", "n_erosion_features", "deposition_volume_m3",
                 "deposition_area_m2", "n_deposition_features", "net_volume_m3"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    # cross-check vs. the published table (min_area=0 no-exclude row)
    published = list(csv.DictReader(open(TABLES / "lure_lod_min_feature_area.csv")))
    for pr in published:
        match = next(r for r in rows if r["min_area_m2"] == int(pr["min_area_m2"])
                    and not r["flagged_excluded"] == "yes")
        d = float(pr["erosion_volume_m3"]) - match["erosion_volume_m3"]
        print(f"  cross-check min_area={pr['min_area_m2']}: erosion delta vs. "
              f"published = {d:.2f} m3 (expect 0.0)")

    gs.run_command("g.remove", type="raster",
                   name="_lv_tmp,_lv_ero,_lv_dep,_lv_ero_bin,_lv_dep_bin,"
                        "_lv_clump,_lv_sizes,_lv_filtered", flags="f", quiet=True)

    log("23_lure_volumes_with_without_flagged.py run. Lake Lure gross/net "
        "volumes, uniform 0.32m LoD, water-excluded, 0/50/100 m2 min "
        "feature area, WITH vs. WITHOUT E5/E6/E7/E8/D7 (this session's "
        "judged artifacts). Results: " +
        "; ".join(f"min{r['min_area_m2']}m2 {'excl' if r['flagged_excluded']=='yes' else 'incl'}: "
                 f"net={r['net_volume_m3']:+,.0f}m3" for r in rows) +
        f". Cross-checked vs. lure_lod_min_feature_area.csv (erosion delta = 0.0 "
        "at every min-area level for the not-excluded rows). NOT applied to any "
        "manuscript table -- computed and reported only, per instruction. "
        f"Written to {out_csv.name}.", run="lure_volumes_with_without_flagged")


if __name__ == "__main__":
    main()
