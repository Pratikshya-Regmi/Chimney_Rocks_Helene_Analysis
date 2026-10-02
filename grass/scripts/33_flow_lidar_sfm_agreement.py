#!/usr/bin/env python3
"""
33_flow_lidar_sfm_agreement.py
===================================
Extends the drainage-network agreement metric (08_watershed_flow_agreement.py
/ 09_watershed_flow_agreement_sensitivity.py, so far computed only for
2020-lidar-vs-2024-lidar) to the 2024-lidar-vs-2024-SfM pair, at the SAME
four discharge thresholds (0.005, 0.01, 0.02, 0.05), so the two pairs can be
read side by side.

COMPARABILITY CHECK (done before trusting the numbers, not assumed):
  - same footing? discharge_2024_cap and discharge_may_2024_lidar
    (30_flow_sequence_setup.py already verified this for the 2020 raster
    too) share IDENTICAL r.sim.water parameters (rain=50, infil=0, man_n=0.4,
    niterations=48, output_step=4) and IDENTICAL region/extent/resolution
    (1508x947 cells, same corner coordinates). Nothing here needed
    re-scaling or reprojecting -- the two rasters are on the same footing
    and a shared fixed threshold is as valid for this pair as it is for the
    lidar-lidar pair.

Output: results/tables/flowpath_agreement_lidar_vs_sfm.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

DISCHARGE_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]


def one_threshold(gs, thresh):
    gs.mapcalc(f"_sfmch_lidar = if(discharge_may_2024_lidar > {thresh}, 1, null())",
              overwrite=True, quiet=True)
    gs.mapcalc(f"_sfmch_sfm = if(discharge_2024_cap > {thresh}, 1, null())",
              overwrite=True, quiet=True)
    n_lidar = int(gs.parse_command("r.univar", map="_sfmch_lidar", flags="g").get("n", 0))
    n_sfm = int(gs.parse_command("r.univar", map="_sfmch_sfm", flags="g").get("n", 0))

    gs.mapcalc(
        "_sfmch_agree = if(!isnull(_sfmch_lidar) && !isnull(_sfmch_sfm), 3, "
        "if(!isnull(_sfmch_lidar) && isnull(_sfmch_sfm), 1, "
        "if(isnull(_sfmch_lidar) && !isnull(_sfmch_sfm), 2, 0)))",
        overwrite=True, quiet=True)
    stats = gs.read_command("r.stats", input="_sfmch_agree", flags="cn")
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for line in stats.strip().split("\n"):
        cat, cnt = line.split()
        counts[int(cat)] = int(cnt)

    n_lidar_only, n_sfm_only, n_both = counts[1], counts[2], counts[3]
    n_union = n_lidar_only + n_sfm_only + n_both
    n_lidar_total = n_lidar_only + n_both
    n_sfm_total = n_sfm_only + n_both

    jaccard = n_both / n_union if n_union else float("nan")
    pct_lidar_shared = 100.0 * n_both / n_lidar_total if n_lidar_total else float("nan")
    pct_sfm_shared = 100.0 * n_both / n_sfm_total if n_sfm_total else float("nan")
    pct_union_common = 100.0 * n_both / n_union if n_union else float("nan")

    return {
        "discharge_threshold": thresh,
        "n_channel_2024lidar": n_lidar,
        "n_channel_2024sfm": n_sfm,
        "n_common": n_both,
        "n_lidar_only": n_lidar_only,
        "n_sfm_only": n_sfm_only,
        "n_union": n_union,
        "jaccard_index": jaccard,
        "pct_union_common": pct_union_common,
        "pct_lidar_channel_shared_with_sfm": pct_lidar_shared,
        "pct_sfm_channel_shared_with_lidar": pct_sfm_shared,
    }


def main():
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", raster="discharge_may_2024_lidar", flags="a")

    # -- comparability check --------------------------------------------
    info_lidar = gs.parse_command("r.info", map="discharge_may_2024_lidar", flags="g")
    info_sfm = gs.parse_command("r.info", map="discharge_2024_cap", flags="g")
    region_keys = ["north", "south", "east", "west", "nsres", "ewres", "rows", "cols"]
    mismatches = [k for k in region_keys if info_lidar[k] != info_sfm[k]]
    if mismatches:
        print("COMPARABILITY WARNING -- extent/resolution differ between "
              f"discharge_may_2024_lidar and discharge_2024_cap on: {mismatches}. "
              "A shared fixed threshold would NOT be a fair comparison; "
              "stopping rather than forcing one.")
        sys.exit(1)
    print("Comparability check passed: discharge_may_2024_lidar and "
          "discharge_2024_cap share identical region/extent/resolution "
          f"({info_lidar['rows']}x{info_lidar['cols']} cells, {info_lidar['nsres']} m). "
          "Both were run with identical r.sim.water rain/infil/Manning's n/"
          "duration/output-step (verified in 30_flow_sequence_setup.py). "
          "A shared fixed discharge threshold is valid for this pair.\n")

    rows = []
    print(f"{'threshold':>10s} {'n_lidar':>9s} {'n_sfm':>9s} {'jaccard':>8s} "
          f"{'%common':>8s} {'%lidar_shared':>14s} {'%sfm_shared':>12s}")
    for t in DISCHARGE_THRESHOLDS:
        r = one_threshold(gs, t)
        rows.append(r)
        print(f"{t:10.3f} {r['n_channel_2024lidar']:9,d} {r['n_channel_2024sfm']:9,d} "
              f"{r['jaccard_index']:8.3f} {r['pct_union_common']:8.1f} "
              f"{r['pct_lidar_channel_shared_with_sfm']:14.1f} "
              f"{r['pct_sfm_channel_shared_with_lidar']:12.1f}")

    jaccards = [r["jaccard_index"] for r in rows]
    common_pcts = [r["pct_union_common"] for r in rows]
    jmin, jmax = min(jaccards), max(jaccards)
    cmin, cmax = min(common_pcts), max(common_pcts)
    print(f"\n2024 lidar vs. 2024 SfM, across thresholds {DISCHARGE_THRESHOLDS}:")
    print(f"  Jaccard range: {jmin:.3f} - {jmax:.3f}")
    print(f"  Shared-cell (%% of union) range: {cmin:.1f}% - {cmax:.1f}%")

    # -- side-by-side against the lidar-lidar pair already on record ----
    lidar_lidar_csv = TABLES / "watershed_flowpath_agreement_sensitivity.csv"
    if lidar_lidar_csv.exists():
        import csv as csv_mod
        with open(lidar_lidar_csv) as f:
            ll_rows = list(csv_mod.DictReader(f))
        ll_j = [float(r["jaccard_index"]) for r in ll_rows]
        ll_c = [float(r["pct_union_common"]) for r in ll_rows]
        print(f"\nFor comparison, 2020-lidar-vs-2024-lidar (already on record, "
              f"same 4 thresholds): Jaccard {min(ll_j):.3f}-{max(ll_j):.3f}, "
              f"%% common {min(ll_c):.1f}%-{max(ll_c):.1f}%.")
        print(f"Lidar-vs-SfM agreement is {'LOWER' if jmax < min(ll_j) else 'higher/overlapping'} "
              "than lidar-vs-lidar agreement across the same threshold range.")

    out_csv = TABLES / "flowpath_agreement_lidar_vs_sfm.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    gs.run_command("g.remove", type="raster",
                   name="_sfmch_lidar,_sfmch_sfm,_sfmch_agree", flags="f", quiet=True)

    log("33_flow_lidar_sfm_agreement.py run. Comparability check passed "
        "(identical region/resolution/r.sim.water parameters between "
        "discharge_may_2024_lidar and discharge_2024_cap). Thresholds "
        f"{DISCHARGE_THRESHOLDS}: Jaccard {jmin:.3f}-{jmax:.3f}, %union "
        f"common {cmin:.1f}%-{cmax:.1f}%. Per-threshold: " +
        "; ".join(f"{r['discharge_threshold']}=J{r['jaccard_index']:.3f}"
                 f"/{r['pct_union_common']:.1f}%common" for r in rows) +
        f". Written to {out_csv.name}.", run="flow_lidar_sfm_agreement")


if __name__ == "__main__":
    main()
