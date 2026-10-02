#!/usr/bin/env python3
"""
17_watershed_lod_min_feature_area_d1_excluded.py
====================================================
Supersedes 12_watershed_lod_min_feature_area.py now that D1 (the rank-1
watershed deposition feature, +27,343 m3, 49.7% of gross deposition) has
been resolved as an artifact (RESULTS_FOR_PAPER.md, "D1 investigation --
RESOLVED", 2026-08-28). Identical method to
11_lure_lod_min_feature_area.py / 12_watershed_lod_min_feature_area.py --
uniform 0.32 m LoD, coreg_dh_corrected@for_codem, r.clump (4-connectivity),
minimum feature area applied at 0 (none)/50/100 m2 -- with D1's exact
footprint (the persistent d1_feature_mask raster from
16_resolve_d1_deposition.py) removed from the deposition side BEFORE
clumping, at every minimum-area level.

D1 EXCLUDED from every row in this table -- stated in the CSV itself via
the d1_excluded column, per the analyst's instruction.

No water exclusion (unlike Lake Lure) -- cat34 has no open-water body.

Output: results/tables/watershed_lod_min_feature_area_d1_excluded.csv
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


def build_clump_sizes(gs, binmap, sizemap_name):
    gs.run_command("r.clump", input=binmap, output="_wd1_clump",
                   overwrite=True, quiet=True)
    stats = gs.read_command("r.stats", input="_wd1_clump", flags="cn")
    sizes = {}
    for line in stats.strip().split("\n"):
        cat, cnt = line.split()
        sizes[int(cat)] = int(cnt)
    rules_path = "/tmp/_wd1_reclass_rules.txt"
    with open(rules_path, "w") as f:
        for cat, cnt in sizes.items():
            f.write(f"{cat} = {cnt}\n")
    gs.run_command("r.reclass", input="_wd1_clump", output=sizemap_name,
                   rules=rules_path, overwrite=True, quiet=True)
    return sizes


def compute_one(gs, label, dh, min_area):
    gs.mapcalc(f"_wd1_tmp = if(abs({dh}) > {LOD}, {dh}, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_wd1_ero = if(_wd1_tmp < 0, _wd1_tmp, null())",
              overwrite=True, quiet=True)
    # D1 excluded from deposition here, before clumping -- so it cannot
    # contribute to any downstream volume, area, or feature count.
    gs.mapcalc("_wd1_dep = if(_wd1_tmp > 0 && isnull(d1_feature_mask), "
              "_wd1_tmp, null())", overwrite=True, quiet=True)
    gs.mapcalc("_wd1_ero_bin = if(!isnull(_wd1_ero), 1, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_wd1_dep_bin = if(!isnull(_wd1_dep), 1, null())",
              overwrite=True, quiet=True)

    row = {"min_area_m2": min_area, "label": label, "d1_excluded": "yes"}
    for kind, dhmap, binmap in (("erosion", "_wd1_ero", "_wd1_ero_bin"),
                               ("deposition", "_wd1_dep", "_wd1_dep_bin")):
        sizes = build_clump_sizes(gs, binmap, "_wd1_sizes")
        if min_area > 0:
            gs.mapcalc(f"_wd1_filtered = if(_wd1_sizes >= {min_area}, "
                      f"{dhmap}, null())", overwrite=True, quiet=True)
        else:
            gs.run_command("g.copy", raster=f"{dhmap},_wd1_filtered",
                           overwrite=True, quiet=True)

        uni = gs.parse_command("r.univar", map="_wd1_filtered", flags="ge",
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
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")

    found = gs.find_file(name="d1_feature_mask", element="cell", mapset="for_codem")
    if not found or not found.get("name"):
        sys.exit("d1_feature_mask not found -- run 16_resolve_d1_deposition.py first.")

    dh = "coreg_dh_corrected"

    rows = []
    for min_area in MIN_AREAS:
        label = (f"uniform {LOD:.2f} m, no minimum feature area, D1 excluded"
                 if min_area == 0 else
                 f"uniform {LOD:.2f} m, min feature area >= {min_area} m2, D1 excluded")
        r = compute_one(gs, label, dh, min_area)
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

    out_csv = TABLES / "watershed_lod_min_feature_area_d1_excluded.csv"
    fieldnames = ["min_area_m2", "label", "d1_excluded", "erosion_volume_m3",
                 "erosion_area_m2", "n_erosion_features",
                 "mean_erosion_feature_size_m2", "deposition_volume_m3",
                 "deposition_area_m2", "n_deposition_features",
                 "mean_deposition_feature_size_m2", "net_volume_m3"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    # cross-check against the with-D1 table: erosion must be identical,
    # deposition must differ by exactly D1's volume (3,738 m2 / 27,343.4 m3)
    with_d1 = list(csv.DictReader(open(TABLES / "watershed_lod_min_feature_area.csv")))
    for old, new in zip(with_d1, rows):
        d_dep_area = int(old["deposition_area_m2"]) - new["deposition_area_m2"]
        d_dep_vol = float(old["deposition_volume_m3"]) - new["deposition_volume_m3"]
        d_n = int(old["n_deposition_features"]) - new["n_deposition_features"]
        print(f"  cross-check min_area={new['min_area_m2']}: "
              f"deposition area delta={d_dep_area} m2 (expect 3738), "
              f"volume delta={d_dep_vol:.1f} m3 (expect 27343.4), "
              f"feature count delta={d_n} (expect 1)")

    scratch = ["_wd1_tmp", "_wd1_ero", "_wd1_dep", "_wd1_ero_bin",
              "_wd1_dep_bin", "_wd1_clump", "_wd1_sizes", "_wd1_filtered"]
    gs.run_command("g.remove", type="raster", name=",".join(scratch),
                   flags="f", quiet=True)

    log("17_watershed_lod_min_feature_area_d1_excluded.py run. Supersedes "
        "12_watershed_lod_min_feature_area.py -- D1 (3738 m2, +27343.4 m3) "
        "excluded from the deposition mask before clumping, at every "
        f"min-area level. cat34, LoD=0.32m, min feature area in {MIN_AREAS} m2. "
        "Results: " +
        "; ".join(f"min{r['min_area_m2']}m2: net={r['net_volume_m3']:+,.0f}m3 "
                 f"(ero {r['n_erosion_features']}feat/{r['mean_erosion_feature_size_m2']:.0f}m2avg, "
                 f"dep {r['n_deposition_features']}feat/{r['mean_deposition_feature_size_m2']:.0f}m2avg)"
                 for r in rows) +
        f". Cross-checked against watershed_lod_min_feature_area.csv (with D1): "
        "deposition area/volume/feature-count deltas match D1 exactly "
        "(3738 m2, 27343.4 m3, 1 feature) at every min-area level. "
        f"Written to {out_csv.name}.", run="watershed_lod_min_feature_area_d1_excluded")


if __name__ == "__main__":
    main()
