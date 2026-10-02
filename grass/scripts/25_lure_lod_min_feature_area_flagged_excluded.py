#!/usr/bin/env python3
"""
25_lure_lod_min_feature_area_flagged_excluded.py
====================================================
Supersedes 11_lure_lod_min_feature_area.py now that E5, E6, E7, E8
(erosion) and D7 (deposition) are resolved as artifacts
(22_resolve_lure_flagged_features.py, RESULTS_FOR_PAPER.md "Check 1").
Identical method (uniform 0.32 m LoD, water-excluded, r.clump
4-connectivity, minimum feature area at 0/50/100 m^2) with the five
features' exact footprints (lure_E5_mask ... lure_D7_mask, persisted by
script 22) removed from their respective erosion/deposition mask BEFORE
clumping, at every minimum-area level -- same pattern as
17_watershed_lod_min_feature_area_d1_excluded.py.

flagged_excluded="yes" stated explicitly in every row.

Output: results/tables/lure_lod_min_feature_area_flagged_excluded.csv
(canonical, manuscript-facing -- supersedes lure_lod_min_feature_area.csv,
which is kept on disk, marked superseded).
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
FLAGGED_EROSION_MASKS = ["lure_E5_mask", "lure_E6_mask", "lure_E7_mask", "lure_E8_mask"]
FLAGGED_DEPOSITION_MASKS = ["lure_D7_mask"]


def build_clump_sizes(gs, binmap, sizemap_name):
    gs.run_command("r.clump", input=binmap, output="_lfx_clump",
                   overwrite=True, quiet=True)
    stats = gs.read_command("r.stats", input="_lfx_clump", flags="cn")
    sizes = {}
    for line in stats.strip().split("\n"):
        cat, cnt = line.split()
        sizes[int(cat)] = int(cnt)
    rules_path = "/tmp/_lfx_reclass_rules.txt"
    with open(rules_path, "w") as f:
        for cat, cnt in sizes.items():
            f.write(f"{cat} = {cnt}\n")
    gs.run_command("r.reclass", input="_lfx_clump", output=sizemap_name,
                   rules=rules_path, overwrite=True, quiet=True)
    return sizes


def compute_one(gs, label, dh, water, min_area):
    ero_excl = " && ".join(f"isnull({m})" for m in FLAGGED_EROSION_MASKS)
    dep_excl = " && ".join(f"isnull({m})" for m in FLAGGED_DEPOSITION_MASKS)
    gs.mapcalc(f"_lfx_tmp = if(isnull({water}) && abs({dh}) > {LOD}, "
              f"{dh}, null())", overwrite=True, quiet=True)
    gs.mapcalc(f"_lfx_ero = if(_lfx_tmp < 0 && {ero_excl}, _lfx_tmp, null())",
              overwrite=True, quiet=True)
    gs.mapcalc(f"_lfx_dep = if(_lfx_tmp > 0 && {dep_excl}, _lfx_tmp, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_lfx_ero_bin = if(!isnull(_lfx_ero), 1, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_lfx_dep_bin = if(!isnull(_lfx_dep), 1, null())",
              overwrite=True, quiet=True)

    row = {"min_area_m2": min_area, "label": label, "flagged_excluded": "yes"}
    for kind, dhmap, binmap in (("erosion", "_lfx_ero", "_lfx_ero_bin"),
                               ("deposition", "_lfx_dep", "_lfx_dep_bin")):
        sizes = build_clump_sizes(gs, binmap, "_lfx_sizes")
        if min_area > 0:
            gs.mapcalc(f"_lfx_filtered = if(_lfx_sizes >= {min_area}, "
                      f"{dhmap}, null())", overwrite=True, quiet=True)
        else:
            gs.run_command("g.copy", raster=f"{dhmap},_lfx_filtered",
                           overwrite=True, quiet=True)

        uni = gs.parse_command("r.univar", map="_lfx_filtered", flags="ge", quiet=True)
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

    for m in FLAGGED_EROSION_MASKS + FLAGGED_DEPOSITION_MASKS:
        found = gs.find_file(name=m, element="cell", mapset="PERMANENT")
        if not found or not found.get("name"):
            sys.exit(f"{m} not found -- run 22_resolve_lure_flagged_features.py first.")

    dh = "coreg_lure_lidar_dh_corrected"
    water = "lure_water_mask"

    rows = []
    for min_area in MIN_AREAS:
        label = (f"uniform {LOD:.2f} m, no minimum feature area, E5/E6/E7/E8/D7 excluded"
                 if min_area == 0 else
                 f"uniform {LOD:.2f} m, min feature area >= {min_area} m2, E5/E6/E7/E8/D7 excluded")
        r = compute_one(gs, label, dh, water, min_area)
        rows.append(r)
        print(f"\n  {label}")
        print(f"    erosion    : {r['erosion_volume_m3']:+,.0f} m3, "
              f"{r['n_erosion_features']:,} features, "
              f"mean {r['mean_erosion_feature_size_m2']:.1f} m2/feature")
        print(f"    deposition : {r['deposition_volume_m3']:+,.0f} m3, "
              f"{r['n_deposition_features']:,} features, "
              f"mean {r['mean_deposition_feature_size_m2']:.1f} m2/feature")
        print(f"    net        : {r['net_volume_m3']:+,.0f} m3")

    out_csv = TABLES / "lure_lod_min_feature_area_flagged_excluded.csv"
    fieldnames = ["min_area_m2", "label", "flagged_excluded", "erosion_volume_m3",
                 "erosion_area_m2", "n_erosion_features", "mean_erosion_feature_size_m2",
                 "deposition_volume_m3", "deposition_area_m2", "n_deposition_features",
                 "mean_deposition_feature_size_m2", "net_volume_m3"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    published = list(csv.DictReader(open(TABLES / "lure_lod_min_feature_area.csv")))
    for pub, new in zip(published, rows):
        d_ero_vol = float(pub["erosion_volume_m3"]) - new["erosion_volume_m3"]
        d_dep_vol = float(pub["deposition_volume_m3"]) - new["deposition_volume_m3"]
        print(f"  cross-check min_area={new['min_area_m2']}: erosion delta="
              f"{d_ero_vol:.1f} m3, deposition delta={d_dep_vol:.1f} m3 vs. published")

    scratch = ["_lfx_tmp", "_lfx_ero", "_lfx_dep", "_lfx_ero_bin", "_lfx_dep_bin",
              "_lfx_clump", "_lfx_sizes", "_lfx_filtered"]
    gs.run_command("g.remove", type="raster", name=",".join(scratch), flags="f", quiet=True)

    log("25_lure_lod_min_feature_area_flagged_excluded.py run. Supersedes "
        "11_lure_lod_min_feature_area.py -- E5/E6/E7/E8 (erosion) and D7 "
        "(deposition) excluded from their respective masks before "
        f"clumping, at every min-area level. Results: " +
        "; ".join(f"min{r['min_area_m2']}m2: net={r['net_volume_m3']:+,.0f}m3 "
                 f"(ero {r['n_erosion_features']}feat, dep {r['n_deposition_features']}feat)"
                 for r in rows) +
        f". Written to {out_csv.name}.",
        run="lure_lod_min_feature_area_flagged_excluded")


if __name__ == "__main__":
    main()
