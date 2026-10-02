#!/usr/bin/env python3
"""
12_watershed_lod_min_feature_area.py
======================================
Watershed (cat34), uniform 0.32 m LoD, lidar-lidar pair (coreg_dh_corrected
@for_codem, the fully co-registered + bias-corrected DoD): gross erosion,
gross deposition, net volume, feature count and mean feature size, with a
MINIMUM FEATURE AREA applied on top of the plain |dh| > 0.32 m threshold, at
50 m^2 and 100 m^2 -- identical method to
11_lure_lod_min_feature_area.py, so the two sites are directly comparable.

No water exclusion here (unlike Lake Lure) -- cat34 is a hillslope
tributary watershed with no open-water body.

Method: threshold -> binarize -> r.clump (4-connectivity) -> r.stats for
per-clump cell counts -> keep only clumps with count >= MIN_AREA_M2 (1 m
resolution, so cell count == area in m^2) -> recompute volume/area/feature
count over the survivors only.

Output: results/tables/watershed_lod_min_feature_area.csv
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
    gs.run_command("r.clump", input=binmap, output="_wshed_mfa_clump",
                   overwrite=True, quiet=True)
    stats = gs.read_command("r.stats", input="_wshed_mfa_clump", flags="cn")
    sizes = {}
    for line in stats.strip().split("\n"):
        cat, cnt = line.split()
        sizes[int(cat)] = int(cnt)
    rules_path = "/tmp/_wshed_mfa_reclass_rules.txt"
    with open(rules_path, "w") as f:
        for cat, cnt in sizes.items():
            f.write(f"{cat} = {cnt}\n")
    gs.run_command("r.reclass", input="_wshed_mfa_clump", output=sizemap_name,
                   rules=rules_path, overwrite=True, quiet=True)
    return sizes


def compute_one(gs, label, dh, min_area):
    gs.mapcalc(f"_wshed_mfa_tmp = if(abs({dh}) > {LOD}, {dh}, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_wshed_mfa_ero = if(_wshed_mfa_tmp < 0, _wshed_mfa_tmp, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_wshed_mfa_dep = if(_wshed_mfa_tmp > 0, _wshed_mfa_tmp, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_wshed_mfa_ero_bin = if(!isnull(_wshed_mfa_ero), 1, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_wshed_mfa_dep_bin = if(!isnull(_wshed_mfa_dep), 1, null())",
              overwrite=True, quiet=True)

    row = {"min_area_m2": min_area, "label": label}
    for kind, dhmap, binmap in (("erosion", "_wshed_mfa_ero", "_wshed_mfa_ero_bin"),
                               ("deposition", "_wshed_mfa_dep", "_wshed_mfa_dep_bin")):
        sizes = build_clump_sizes(gs, binmap, "_wshed_mfa_sizes")
        if min_area > 0:
            gs.mapcalc(f"_wshed_mfa_filtered = if(_wshed_mfa_sizes >= {min_area}, "
                      f"{dhmap}, null())", overwrite=True, quiet=True)
        else:
            gs.run_command("g.copy", raster=f"{dhmap},_wshed_mfa_filtered",
                           overwrite=True, quiet=True)

        uni = gs.parse_command("r.univar", map="_wshed_mfa_filtered", flags="ge",
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

    dh = "coreg_dh_corrected"

    rows = []
    for min_area in MIN_AREAS:
        label = (f"uniform {LOD:.2f} m, no minimum feature area" if min_area == 0
                 else f"uniform {LOD:.2f} m, min feature area >= {min_area} m2")
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

    out_csv = TABLES / "watershed_lod_min_feature_area.csv"
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

    scratch = ["_wshed_mfa_tmp", "_wshed_mfa_ero", "_wshed_mfa_dep",
              "_wshed_mfa_ero_bin", "_wshed_mfa_dep_bin", "_wshed_mfa_clump",
              "_wshed_mfa_sizes", "_wshed_mfa_filtered"]
    gs.run_command("g.remove", type="raster", name=",".join(scratch),
                   flags="f", quiet=True)
    print(f"({len(scratch)} scratch rasters removed -- disposable, mirrors "
          "11_lure_lod_min_feature_area.py's cleanup)")

    log("12_watershed_lod_min_feature_area.py run. cat34, LoD=0.32m, "
        f"min feature area in {MIN_AREAS} m2. Results: " +
        "; ".join(f"min{r['min_area_m2']}m2: net={r['net_volume_m3']:+,.0f}m3 "
                 f"(ero {r['n_erosion_features']}feat/{r['mean_erosion_feature_size_m2']:.0f}m2avg, "
                 f"dep {r['n_deposition_features']}feat/{r['mean_deposition_feature_size_m2']:.0f}m2avg)"
                 for r in rows) +
        f". Written to {out_csv.name}.", run="watershed_lod_min_feature_area")


if __name__ == "__main__":
    main()
