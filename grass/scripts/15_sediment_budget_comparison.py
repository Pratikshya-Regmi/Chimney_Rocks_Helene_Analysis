#!/usr/bin/env python3
"""
15_sediment_budget_comparison.py
==================================
Assembles the watershed-vs-Lake-Lure magnitude comparison from the already-
computed, already-saved tables (watershed_lod_min_feature_area.csv,
lure_lod_min_feature_area.csv, dominant_features_watershed.csv) -- no new
GRASS computation, just consolidation with the ratios worked out, so the
comparison itself is traceable to a script rather than typed by hand.

HARD CAVEAT (state this every time these numbers are quoted): the
watershed (cat34) is ONE tributary among many draining to Lake Lure. This
is a comparison of MAGNITUDES, not a mass-balance and not a closed budget --
do not describe the watershed as "explaining" or "accounting for" any
fraction of the reservoir's deposition.

Also reports the same watershed net figures with the likely-artifact
dominant deposition feature (D1, +27,343 m3, flagged in
13_dominant_features_both_sites.py's write-up, NOT removed from the
official watershed_lod_min_feature_area.csv rows) excluded, as a sensitivity
row -- for awareness, not as a replacement result.

Output: results/tables/sediment_budget_comparison.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"

D1_VOLUME_M3 = 27343.4  # dominant_features_watershed.csv, deposition rank 1
                        # -- flagged as a likely canopy/interpolation
                        # artifact, see RESULTS_FOR_PAPER.md


def load(path):
    return list(csv.DictReader(open(path)))


def main():
    wshed = {r["min_area_m2"]: r for r in load(TABLES / "watershed_lod_min_feature_area.csv")}
    lure = {r["min_area_m2"]: r for r in load(TABLES / "lure_lod_min_feature_area.csv")}

    rows = []
    for min_area in ["0", "50", "100"]:
        w = wshed[min_area]
        l = lure[min_area]
        w_net = float(w["net_volume_m3"])
        w_gross_ero = float(w["erosion_volume_m3"])
        l_net = float(l["net_volume_m3"])
        l_gross_dep = float(l["deposition_volume_m3"])
        w_net_d1_excl = w_net - D1_VOLUME_M3

        row = {
            "min_feature_area_m2": min_area,
            "watershed_net_volume_m3": w_net,
            "watershed_gross_erosion_m3": w_gross_ero,
            "lure_net_volume_m3": l_net,
            "lure_gross_deposition_m3": l_gross_dep,
            "ratio_lure_net_to_watershed_net": abs(l_net / w_net) if w_net else None,
            "ratio_lure_gross_dep_to_watershed_gross_ero": abs(l_gross_dep / w_gross_ero) if w_gross_ero else None,
            "watershed_net_volume_m3_D1_excluded_sensitivity": w_net_d1_excl,
        }
        rows.append(row)
        print(f"\nmin feature area >= {min_area} m2:")
        print(f"  watershed: net {w_net:+,.0f} m3, gross erosion {w_gross_ero:+,.0f} m3 "
              f"(net excl. likely-artifact D1: {w_net_d1_excl:+,.0f} m3)")
        print(f"  Lake Lure: net {l_net:+,.0f} m3, gross deposition {l_gross_dep:+,.0f} m3")
        print(f"  Lake Lure net / watershed net (magnitude): "
              f"{row['ratio_lure_net_to_watershed_net']:.2f}x")
        print(f"  Lake Lure gross deposition / watershed gross erosion (magnitude): "
              f"{row['ratio_lure_gross_dep_to_watershed_gross_ero']:.2f}x")

    out_csv = TABLES / "sediment_budget_comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")
    print("\nCAVEAT (restate whenever these numbers are used): cat34 is one "
          "tributary among many draining to Lake Lure. This is a magnitude "
          "comparison only, not a closed sediment budget.")

    log("15_sediment_budget_comparison.py run. Consolidated from "
        "watershed_lod_min_feature_area.csv, lure_lod_min_feature_area.csv. "
        "Lake Lure net / watershed net ratio ~2.8-2.9x across min-area "
        "thresholds; Lake Lure gross deposition / watershed gross erosion "
        "ratio ~2.1-2.3x. D1-excluded watershed net sensitivity row included "
        f"(D1={D1_VOLUME_M3:+,.0f} m3 flagged as likely artifact). Explicit "
        "caveat: magnitude comparison only, cat34 is one of many "
        "tributaries, not a closed budget. Written to sediment_budget_"
        "comparison.csv.", run="sediment_budget_comparison")


if __name__ == "__main__":
    main()
