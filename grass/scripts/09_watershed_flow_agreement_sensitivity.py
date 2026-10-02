#!/usr/bin/env python3
"""
09_watershed_flow_agreement_sensitivity.py
============================================
Sensitivity of the 2020-vs-2024 drainage-network agreement metric
(08_watershed_flow_agreement.py) to the discharge threshold used to define
a "channel" cell. Recomputes the Jaccard index and persist/new percentages
at DISCHARGE_THRESHOLDS = [0.005, 0.01, 0.02, 0.05].

Output: results/tables/watershed_flowpath_agreement_sensitivity.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

DISCHARGE_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]


def one_threshold(gs, thresh):
    gs.mapcalc(f"_ch2020 = if(discharge_2020_regis > {thresh}, 1, null())",
              overwrite=True, quiet=True)
    gs.mapcalc(f"_ch2024 = if(discharge_may_2024_lidar > {thresh}, 1, null())",
              overwrite=True, quiet=True)
    n2020 = int(gs.parse_command("r.univar", map="_ch2020", flags="g").get("n", 0))
    n2024 = int(gs.parse_command("r.univar", map="_ch2024", flags="g").get("n", 0))

    gs.mapcalc(
        "_ch_agree = if(!isnull(_ch2020) && !isnull(_ch2024), 3, "
        "if(!isnull(_ch2020) && isnull(_ch2024), 1, "
        "if(isnull(_ch2020) && !isnull(_ch2024), 2, 0)))",
        overwrite=True, quiet=True)
    stats = gs.read_command("r.stats", input="_ch_agree", flags="cn")
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for line in stats.strip().split("\n"):
        cat, cnt = line.split()
        counts[int(cat)] = int(cnt)

    n_2020_only, n_2024_only, n_both = counts[1], counts[2], counts[3]
    n_union = n_2020_only + n_2024_only + n_both
    n_2020_total = n_2020_only + n_both
    n_2024_total = n_2024_only + n_both

    jaccard = n_both / n_union if n_union else float("nan")
    pct_2020_retained = 100.0 * n_both / n_2020_total if n_2020_total else float("nan")
    pct_2024_new = 100.0 * n_2024_only / n_2024_total if n_2024_total else float("nan")
    pct_union_common = 100.0 * n_both / n_union if n_union else float("nan")

    return {
        "discharge_threshold": thresh,
        "n_channel_2020": n2020,
        "n_channel_2024": n2024,
        "n_common": n_both,
        "n_2020_only": n_2020_only,
        "n_2024_only": n_2024_only,
        "n_union": n_union,
        "jaccard_index": jaccard,
        "pct_union_common": pct_union_common,
        "pct_2020_retained_in_2024": pct_2020_retained,
        "pct_2024_that_is_new": pct_2024_new,
    }


def main():
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", raster="discharge_2020_regis@for_codem", flags="a")

    rows = []
    print(f"{'threshold':>10s} {'n2020':>9s} {'n2024':>9s} {'jaccard':>8s} "
          f"{'%common':>8s} {'%2020->2024':>12s} {'%2024 new':>10s}")
    for t in DISCHARGE_THRESHOLDS:
        r = one_threshold(gs, t)
        rows.append(r)
        print(f"{t:10.3f} {r['n_channel_2020']:9,d} {r['n_channel_2024']:9,d} "
              f"{r['jaccard_index']:8.3f} {r['pct_union_common']:8.1f} "
              f"{r['pct_2020_retained_in_2024']:12.1f} {r['pct_2024_that_is_new']:10.1f}")

    jaccards = [r["jaccard_index"] for r in rows]
    jmin, jmax = min(jaccards), max(jaccards)
    spread = jmax - jmin
    print(f"\nJaccard range across thresholds {DISCHARGE_THRESHOLDS}: "
          f"{jmin:.3f} - {jmax:.3f} (spread {spread:.3f})")
    verdict = ("STABLE -- the agreement metric does not depend meaningfully "
              "on the exact threshold chosen" if spread < 0.05 else
              "SENSITIVE -- report the range, not a single number")
    print(f"Verdict: {verdict}")

    out_csv = TABLES / "watershed_flowpath_agreement_sensitivity.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    gs.run_command("g.remove", type="raster", name="_ch2020,_ch2024,_ch_agree",
                   flags="f", quiet=True)
    print("(scratch rasters _ch2020/_ch2024/_ch_agree removed -- disposable, "
          "recreated fresh per threshold in the loop above; the single-"
          "threshold 0.01 case's kept/documented equivalents are "
          "channel_2020_regis/channel_2024_lidar/channel_agreement from "
          "08_watershed_flow_agreement.py)")

    log("09_watershed_flow_agreement_sensitivity.py run. Thresholds "
        f"{DISCHARGE_THRESHOLDS}. Jaccard range {jmin:.3f}-{jmax:.3f} "
        f"(spread {spread:.3f}). Verdict: {verdict}. Per-threshold detail: " +
        "; ".join(f"{r['discharge_threshold']}=J{r['jaccard_index']:.3f}"
                 f"/{r['pct_union_common']:.1f}%common"
                 for r in rows) +
        f". Written to {out_csv.name}.", run="watershed_flow_agreement_sensitivity")


if __name__ == "__main__":
    main()
