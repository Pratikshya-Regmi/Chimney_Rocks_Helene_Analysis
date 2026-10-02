#!/usr/bin/env python3
"""
07_lure_lod_sensitivity.py
===========================
Volume sensitivity to the choice of Level of Detection, for the Lake Lure
lidar 2017->2024 pair. Computes gross erosion, gross deposition, and net
volume at four thresholds: 0.32 m (watershed's own uniform LoD, for cross-
reach context), 0.50 m, 0.94 m (the manuscript's published Lake Lure
uniform LoD), and the canopy-binned variable LoD already measured in
03_coreg_variable_lod_lure_lidar.py stage 3 (coreg_lure_lidar_lod). Also
reports the NUMBER and AREA of qualifying features at each threshold
(erosion and deposition clumped separately via r.clump, 4-connectivity,
diagonal not counted so two cells touching only at a corner are separate
features) -- volume alone doesn't say whether a threshold is finding a few
large coherent patches or many small scattered ones.

Water (lure_water_mask) is excluded from every threshold's statistics, same
fix as 03_coreg_variable_lod_lure_lidar.py's stage4.

Output: results/tables/lure_lod_sensitivity.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

FIXED_THRESHOLDS = [0.32, 0.50, 0.94]


def main():
    gs, gj = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    gs.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")

    dh = "coreg_lure_lidar_dh_corrected"
    water = "lure_water_mask"

    land_n = int(gs.parse_command("r.univar", map="lidar_2017_DTM_lure@PERMANENT",
                                  flags="g").get("n", 0))
    water_n = int(gs.parse_command("r.univar", map=water, flags="g").get("n", 0))
    land_n -= water_n
    print(f"land cells (total minus water): {land_n:,}")

    rows = []

    def n_clumps(rastmap):
        """Number of distinct connected features (r.clump, 4-connectivity;
        no diagonal flag, so corner-touching cells count as separate
        features).

        BUG CAUGHT AND FIXED HERE: r.clump groups cells by CATEGORY
        (identical value), not by "both non-null." Run directly on the
        continuous dh_corrected values, essentially every cell has a
        distinct floating-point value, so nothing ever merges -- the first
        version of this function returned one "clump" per cell (n_features
        == area for every single row, confirmed via r.stats -c: e.g. 0.32 m
        erosion, 155,347 cells, ALL 155,347 categories present with count 1
        each). Fixed by binarizing to a constant (1) first, so "same
        category" correctly means "same qualifying region," not "same dh
        value."
        """
        gs.mapcalc(f"_lure_lod_bin = if(!isnull({rastmap}), 1, null())",
                  overwrite=True, quiet=True)
        gs.run_command("r.clump", input="_lure_lod_bin", output="_lure_lod_clump",
                       overwrite=True, quiet=True)
        info = gs.parse_command("r.info", map="_lure_lod_clump", flags="r")
        return int(float(info.get("max", 0)))

    def compute_one(label, thresh_expr, extra_note=""):
        gs.mapcalc(f"_lure_lod_tmp = if(isnull({water}) && ({thresh_expr}), "
                  f"{dh}, null())", overwrite=True, quiet=True)
        gs.mapcalc("_lure_lod_ero = if(_lure_lod_tmp < 0, _lure_lod_tmp, null())",
                  overwrite=True, quiet=True)
        gs.mapcalc("_lure_lod_dep = if(_lure_lod_tmp > 0, _lure_lod_tmp, null())",
                  overwrite=True, quiet=True)
        ero = gs.parse_command("r.univar", map="_lure_lod_ero", flags="ge", quiet=True)
        dep = gs.parse_command("r.univar", map="_lure_lod_dep", flags="ge", quiet=True)
        n_ero = int(ero.get("n", 0))
        n_dep = int(dep.get("n", 0))
        ero_vol = float(ero.get("sum", 0))
        dep_vol = float(dep.get("sum", 0))
        n_ero_features = n_clumps("_lure_lod_ero") if n_ero else 0
        n_dep_features = n_clumps("_lure_lod_dep") if n_dep else 0
        row = {
            "label": label,
            "erosion_pct_of_land": 100.0 * n_ero / land_n if land_n else 0,
            "deposition_pct_of_land": 100.0 * n_dep / land_n if land_n else 0,
            "erosion_median_m": float(ero.get("median", "nan")),
            "deposition_median_m": float(dep.get("median", "nan")),
            "erosion_volume_m3": ero_vol,
            "deposition_volume_m3": dep_vol,
            "net_volume_m3": ero_vol + dep_vol,
            "erosion_area_m2": n_ero,           # 1 m resolution -> cells == m2
            "deposition_area_m2": n_dep,
            "n_erosion_features": n_ero_features,
            "n_deposition_features": n_dep_features,
            "note": extra_note,
        }
        rows.append(row)
        print(f"\n  {label}")
        print(f"    erosion    : {row['erosion_pct_of_land']:5.1f}% of land, "
              f"median {row['erosion_median_m']:+.2f} m, {ero_vol:+,.0f} m3, "
              f"area {n_ero:,} m2, {n_ero_features:,} features")
        print(f"    deposition : {row['deposition_pct_of_land']:5.1f}% of land, "
              f"median {row['deposition_median_m']:+.2f} m, {dep_vol:+,.0f} m3, "
              f"area {n_dep:,} m2, {n_dep_features:,} features")
        print(f"    net        : {row['net_volume_m3']:+,.0f} m3")

    for t in FIXED_THRESHOLDS:
        note = "manuscript's published Lake Lure LoD" if t == 0.94 else \
               ("watershed's uniform LoD, shown for cross-reach context" if t == 0.32 else "")
        compute_one(f"uniform {t:.2f} m", f"abs({dh}) > {t}", note)

    compute_one("canopy-binned variable LoD (0.28-0.34 m)",
               f"abs({dh}) > coreg_lure_lidar_lod",
               "measured in 03_coreg_variable_lod_lure_lidar.py stage 3; "
               "independently supported by known-stable-ground test "
               "(06_lure_known_stable_ground_test.py): verified roof/"
               "pavement cells show NMAD=0.034m, 99.9% within 0.34m")

    out_csv = TABLES / "lure_lod_sensitivity.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    gs.run_command("g.remove", type="raster",
                   name="_lure_lod_tmp,_lure_lod_ero,_lure_lod_dep,_lure_lod_clump,_lure_lod_bin",
                   flags="f", quiet=True)
    print("(scratch rasters _lure_lod_tmp/_ero/_dep/_clump removed -- disposable "
          "intermediates re-created fresh each loop iteration, not results)")

    log("07_lure_lod_sensitivity.py run. Volume sensitivity across "
        f"thresholds 0.32/0.50/0.94 m (uniform) and the canopy-binned "
        f"variable LoD, water-excluded (land_n={land_n:,}). "
        f"Net volumes: " +
        "; ".join(f"{r['label']}={r['net_volume_m3']:+,.0f} m3" for r in rows) +
        f". Written to {out_csv.name}.", run="lure_lod_sensitivity")


if __name__ == "__main__":
    main()
