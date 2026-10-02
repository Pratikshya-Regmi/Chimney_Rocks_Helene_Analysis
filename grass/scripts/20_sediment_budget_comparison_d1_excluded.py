#!/usr/bin/env python3
"""
20_sediment_budget_comparison_d1_excluded.py
================================================
Supersedes 15_sediment_budget_comparison.py now that D1 is resolved as an
artifact (not a sensitivity row -- the basis). Consolidates
watershed_lod_min_feature_area_d1_excluded.csv (17_...py) against
lure_lod_min_feature_area.csv (11_...py, unaffected by D1) -- no new GRASS
computation.

HARD CAVEAT (state this every time these numbers are quoted): the
watershed (cat34) is ONE tributary among many draining to Lake Lure. This
is a comparison of MAGNITUDES, not a mass-balance and not a closed budget --
do not describe the watershed as "explaining" or "accounting for" any
fraction of the reservoir's deposition.

Output: results/tables/sediment_budget_comparison_d1_excluded.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"


def load(path):
    return list(csv.DictReader(open(path)))


def main():
    wshed = {r["min_area_m2"]: r for r in
             load(TABLES / "watershed_lod_min_feature_area_d1_excluded.csv")}
    lure = {r["min_area_m2"]: r for r in load(TABLES / "lure_lod_min_feature_area.csv")}

    rows = []
    for min_area in ["0", "50", "100"]:
        w = wshed[min_area]
        l = lure[min_area]
        w_net = float(w["net_volume_m3"])
        w_gross_ero = float(w["erosion_volume_m3"])
        l_net = float(l["net_volume_m3"])
        l_gross_dep = float(l["deposition_volume_m3"])

        row = {
            "min_feature_area_m2": min_area,
            "d1_excluded": "yes",
            "watershed_net_volume_m3": w_net,
            "watershed_gross_erosion_m3": w_gross_ero,
            "lure_net_volume_m3": l_net,
            "lure_gross_deposition_m3": l_gross_dep,
            "ratio_lure_net_to_watershed_net": abs(l_net / w_net) if w_net else None,
            "ratio_lure_gross_dep_to_watershed_gross_ero": abs(l_gross_dep / w_gross_ero) if w_gross_ero else None,
        }
        rows.append(row)
        print(f"\nmin feature area >= {min_area} m2 (D1 excluded from watershed side):")
        print(f"  watershed: net {w_net:+,.0f} m3, gross erosion {w_gross_ero:+,.0f} m3")
        print(f"  Lake Lure: net {l_net:+,.0f} m3, gross deposition {l_gross_dep:+,.0f} m3")
        print(f"  Lake Lure net / watershed net (magnitude): "
              f"{row['ratio_lure_net_to_watershed_net']:.2f}x")
        print(f"  Lake Lure gross deposition / watershed gross erosion (magnitude): "
              f"{row['ratio_lure_gross_dep_to_watershed_gross_ero']:.2f}x")

    out_csv = TABLES / "sediment_budget_comparison_d1_excluded.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")
    print("\nCAVEAT (restate whenever these numbers are used): cat34 is one "
          "tributary among many draining to Lake Lure. This is a magnitude "
          "comparison only, not a closed sediment budget.")

    log("20_sediment_budget_comparison_d1_excluded.py run. Supersedes "
        "15_sediment_budget_comparison.py -- D1-excluded watershed table is "
        "now the primary basis, not a sensitivity row. Consolidated from "
        "watershed_lod_min_feature_area_d1_excluded.csv, "
        "lure_lod_min_feature_area.csv. Lake Lure net / watershed net ratio "
        f"{rows[0]['ratio_lure_net_to_watershed_net']:.2f}-"
        f"{max(r['ratio_lure_net_to_watershed_net'] for r in rows):.2f}x "
        "across min-area thresholds (was 2.8-2.9x with D1 included); Lake "
        "Lure gross deposition / watershed gross erosion ratio "
        f"{min(r['ratio_lure_gross_dep_to_watershed_gross_ero'] for r in rows):.2f}-"
        f"{max(r['ratio_lure_gross_dep_to_watershed_gross_ero'] for r in rows):.2f}x "
        "(unchanged by D1, since D1 was a deposition feature and this ratio "
        "uses watershed gross EROSION). Explicit caveat: magnitude "
        "comparison only, cat34 is one of many tributaries, not a closed "
        f"budget. Written to {out_csv.name}.",
        run="sediment_budget_comparison_d1_excluded")


if __name__ == "__main__":
    main()
