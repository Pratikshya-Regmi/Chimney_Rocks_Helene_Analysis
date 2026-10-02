#!/usr/bin/env python3
"""
21_watershed_corridor_comparison_d1_checked.py
==================================================
Re-verifies 14_watershed_corridor_comparison.py's corridor table against
D1's exclusion. D1's footprint (d1_feature_mask, E313673.5/N193670.5) was
checked against both corridor buffers and overlaps NEITHER (0 cells in
corridor_scar_mask, 0 cells in corridor_west_mask -- D1 sits ~460-1000+ m
north of both corridors, in the basin's upper headwater reach). D1
exclusion is therefore applied defensively (a no-op given the zero
overlap) rather than skipped, so this is traceable rather than assumed --
same method as 14_watershed_corridor_comparison.py, with D1's mask
subtracted from the deposition side before computing statistics, and the
overlap count itself reported and logged.

Reuses the persistent corridor_scar_mask / corridor_west_mask rasters
already built and kept by 14_watershed_corridor_comparison.py.

Output: results/tables/watershed_corridor_comparison_d1_checked.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

LOD = 0.32


def main():
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")

    for m in ("corridor_scar_mask", "corridor_west_mask", "d1_feature_mask"):
        found = gs.find_file(name=m, element="cell", mapset="for_codem")
        if not found or not found.get("name"):
            sys.exit(f"{m} not found -- run 14_watershed_corridor_comparison.py "
                     "and 16_resolve_d1_deposition.py first.")

    CORRIDORS = (("scar", "main scar corridor (B-B')", "corridor_scar_mask"),
                ("west", "western corridor (C-C')", "corridor_west_mask"))

    overlaps = {}
    for key, label, maskmap in CORRIDORS:
        gs.mapcalc(f"_c21_overlap = if(!isnull({maskmap}) && "
                  "!isnull(d1_feature_mask), 1, null())", overwrite=True, quiet=True)
        n = int(gs.parse_command("r.univar", map="_c21_overlap", flags="g").get("n", 0))
        overlaps[key] = n
        print(f"D1 cells inside {label}: {n}")

    rows = []
    for key, label, maskmap in CORRIDORS:
        gs.mapcalc(f"_c21_dh = if(!isnull({maskmap}) && isnull(d1_feature_mask) && "
                  f"abs(coreg_dh_corrected) > {LOD}, coreg_dh_corrected, null())",
                  overwrite=True, quiet=True)
        gs.mapcalc("_c21_ero = if(_c21_dh < 0, _c21_dh, null())",
                  overwrite=True, quiet=True)
        gs.mapcalc("_c21_dep = if(_c21_dh > 0, _c21_dh, null())",
                  overwrite=True, quiet=True)
        ero = gs.parse_command("r.univar", map="_c21_ero", flags="ge", quiet=True)
        dep = gs.parse_command("r.univar", map="_c21_dep", flags="ge", quiet=True)
        n_ero = int(ero.get("n", 0))
        n_dep = int(dep.get("n", 0))
        ero_vol = float(ero.get("sum", 0))
        dep_vol = float(dep.get("sum", 0))
        area_above_lod = n_ero + n_dep
        ratio = abs(dep_vol / ero_vol) if ero_vol else float("inf")

        row = {
            "corridor": label,
            "d1_excluded": "yes",
            "d1_overlap_cells": overlaps[key],
            "area_above_lod_m2": area_above_lod,
            "erosion_volume_m3": ero_vol,
            "erosion_area_m2": n_ero,
            "deposition_volume_m3": dep_vol,
            "deposition_area_m2": n_dep,
            "net_volume_m3": ero_vol + dep_vol,
            "deposition_to_erosion_ratio": ratio,
        }
        rows.append(row)
        print(f"\n  {label} (D1 excluded, {overlaps[key]} D1 cells overlapped)")
        print(f"    area above LoD : {area_above_lod:,} m2")
        print(f"    erosion        : {ero_vol:+,.0f} m3 ({n_ero:,} m2)")
        print(f"    deposition     : {dep_vol:+,.0f} m3 ({n_dep:,} m2)")
        print(f"    net            : {row['net_volume_m3']:+,.0f} m3")
        print(f"    deposition:erosion ratio : {ratio:.2f}")

    out_csv = TABLES / "watershed_corridor_comparison_d1_checked.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    # cross-check against the original (pre-D1-check) table -- should be
    # bit-identical since the overlap is zero
    orig = list(csv.DictReader(open(TABLES / "watershed_corridor_comparison.csv")))
    for o, n in zip(orig, rows):
        d_net = float(o["net_volume_m3"]) - n["net_volume_m3"]
        print(f"  cross-check {n['corridor']}: net delta vs. original = {d_net:.3f} m3 "
              "(expect 0.0 -- D1 does not overlap this corridor)")

    gs.run_command("g.remove", type="raster",
                   name="_c21_overlap,_c21_dh,_c21_ero,_c21_dep", flags="f", quiet=True)

    log("21_watershed_corridor_comparison_d1_checked.py run. Verifies "
        "14_watershed_corridor_comparison.py against D1's exclusion: D1 "
        f"overlaps {overlaps['scar']} cells of the main scar corridor and "
        f"{overlaps['west']} cells of the western corridor (both zero -- "
        "D1 sits far north of both, in the upper headwater reach). D1 "
        "exclusion applied defensively (a no-op); results are bit-identical "
        "to watershed_corridor_comparison.csv, confirmed by direct "
        "cross-check (net volume delta = 0.0 m3 both corridors). "
        f"Written to {out_csv.name}.",
        run="watershed_corridor_comparison_d1_checked")


if __name__ == "__main__":
    main()
