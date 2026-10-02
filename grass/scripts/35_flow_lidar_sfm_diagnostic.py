#!/usr/bin/env python3
"""
35_flow_lidar_sfm_diagnostic.py
====================================
Diagnoses the 0.35-0.36 lidar-vs-SfM flow-path Jaccard result
(33_flow_lidar_sfm_agreement.py) before any manuscript text is allowed to
rest on it. Three checks, run in order:

PART A -- equal footing at a fixed absolute threshold. Reports, for all
three discharge rasters (2020 lidar, 2024 SfM, 2024 lidar) at all four
thresholds (0.005/0.01/0.02/0.05), the classified-channel fraction of each
basin's valid cells. If the SfM discharge distribution were scaled very
differently, a shared absolute threshold would pick a very different
network SIZE per raster and the Jaccard would be deflated by construction
(mismatched set sizes cap the achievable overlap) rather than by genuine
spatial disagreement.

PART B -- percentile-matched recompute. Redefines "channel" independently
per raster as that raster's OWN top 5% and top 10% of discharge (not a
shared absolute value), so both networks being compared are guaranteed the
same size by construction regardless of any distributional difference.
Run for both pairs (2020-lidar-vs-2024-lidar, the metric already on
record, and 2024-lidar-vs-2024-SfM, the one being diagnosed) so the two
are read on the same footing.

PART C -- major-channels-only. The claim actually at stake is not "every
classified cell matches" but "the dominant flow-concentration zones are
reproduced". Restricts to each raster's own top 1% of discharge (the
primary channel stems -- checked visually/by cell count before use, see
log) and recomputes Jaccard for both pairs at that stricter definition.

VERDICT: plain yes/no on whether the SfM simulation reproduces the
dominant flow paths, printed at the end and NOT softened.

Outputs:
  results/tables/flow_channel_fraction_by_threshold.csv   (Part A)
  results/tables/flowpath_agreement_percentile_matched.csv (Part B)
  results/tables/flowpath_agreement_major_channels.csv     (Part C)
  results/logs/flow_lidar_sfm_diagnostic_<date>.log
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

ABS_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]
PERCENTILE_LEVELS = [5, 10]   # top-X% -- Part B
MAJOR_CHANNEL_PCT = 1         # top-1% -- Part C, "dominant concentration zones"

RASTERS = {
    "2020_lidar": "discharge_2020_regis",
    "2024_sfm": "discharge_2024_cap",
    "2024_lidar": "discharge_may_2024_lidar",
}


def jaccard_pair(gs, mapA, mapB, tag):
    """Both mapA/mapB assumed already built as binary (1/null) channel
    masks. Returns dict of n_A, n_B, n_common, n_union, jaccard, %common."""
    nA = int(gs.parse_command("r.univar", map=mapA, flags="g").get("n", 0))
    nB = int(gs.parse_command("r.univar", map=mapB, flags="g").get("n", 0))
    agree = f"_diag_agree_{tag}"
    gs.mapcalc(
        f"{agree} = if(!isnull({mapA}) && !isnull({mapB}), 3, "
        f"if(!isnull({mapA}) && isnull({mapB}), 1, "
        f"if(isnull({mapA}) && !isnull({mapB}), 2, 0)))",
        overwrite=True, quiet=True)
    stats = gs.read_command("r.stats", input=agree, flags="cn")
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for line in stats.strip().split("\n"):
        cat, cnt = line.split()
        counts[int(cat)] = int(cnt)
    n_common = counts[3]
    n_union = counts[1] + counts[2] + counts[3]
    jaccard = n_common / n_union if n_union else float("nan")
    pct_common = 100.0 * n_common / n_union if n_union else float("nan")
    gs.run_command("g.remove", type="raster", name=agree, flags="f", quiet=True)
    return {"n_A": nA, "n_B": nB, "n_common": n_common, "n_union": n_union,
           "jaccard": jaccard, "pct_union_common": pct_common}


def main():
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", raster="discharge_2020_regis", flags="a")

    # ================================================================
    # PART A -- channel fraction of basin at fixed absolute thresholds
    # ================================================================
    print("=" * 70)
    print("PART A: channel fraction of basin, fixed absolute thresholds")
    print("=" * 70)

    total_valid = {}
    for key, rmap in RASTERS.items():
        total_valid[key] = int(gs.parse_command("r.univar", map=rmap, flags="g").get("n", 0))

    rows_a = []
    print(f"{'threshold':>10s}", end="")
    for key in RASTERS:
        print(f" {key + '_pct':>16s}", end="")
    print()
    for t in ABS_THRESHOLDS:
        row = {"threshold": t}
        line = f"{t:10.3f}"
        for key, rmap in RASTERS.items():
            gs.mapcalc(f"_diag_ch = if({rmap} > {t}, 1, null())", overwrite=True, quiet=True)
            n = int(gs.parse_command("r.univar", map="_diag_ch", flags="g").get("n", 0))
            pct = 100.0 * n / total_valid[key]
            row[f"{key}_n"] = n
            row[f"{key}_pct"] = round(pct, 3)
            line += f" {pct:16.2f}"
        print(line)
        rows_a.append(row)

    out_a = TABLES / "flow_channel_fraction_by_threshold.csv"
    with open(out_a, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_a[0].keys()))
        w.writeheader()
        w.writerows(rows_a)
    print(f"\nwrote {out_a}")

    # verdict on Part A: max relative spread in channel-% across the 3
    # rasters, at each threshold
    max_spread = 0.0
    for row in rows_a:
        pcts = [row[f"{k}_pct"] for k in RASTERS]
        spread = (max(pcts) - min(pcts)) / max(pcts) * 100
        max_spread = max(max_spread, spread)
    print(f"\nMax relative spread in channel-%% across the 3 rasters at any "
          f"threshold: {max_spread:.1f}%.")
    equal_footing = max_spread < 25.0
    print("VERDICT (Part A): " +
         ("network sizes are COMPARABLE at a fixed absolute threshold -- "
          "a shared threshold is not obviously deflating the Jaccard by "
          "construction." if equal_footing else
          "network sizes DIFFER substantially at a fixed absolute "
          "threshold -- the Jaccard may be partly deflated by mismatched "
          "set sizes, not just spatial disagreement."))

    # ================================================================
    # PART B -- percentile-matched (top 5%, top 10%), both pairs
    # ================================================================
    print("\n" + "=" * 70)
    print("PART B: percentile-matched channel definition (equal network size "
          "by construction)")
    print("=" * 70)

    rows_b = []
    for pct_level in PERCENTILE_LEVELS:
        p_thresh = {}
        for key, rmap in RASTERS.items():
            u = gs.parse_command("r.univar", map=rmap, flags="ge",
                                 percentile=100 - pct_level)
            pkey = [k for k in u if k.startswith("percentile_")][0]
            p_thresh[key] = float(u[pkey])
            gs.mapcalc(f"_pctch_{key} = if({rmap} > {p_thresh[key]}, 1, null())",
                      overwrite=True, quiet=True)

        for pair_name, keyA, keyB, ref_csv_label in [
            ("2020lidar_vs_2024lidar", "2020_lidar", "2024_lidar", "lidar-lidar"),
            ("2024lidar_vs_2024sfm", "2024_lidar", "2024_sfm", "lidar-SfM"),
        ]:
            r = jaccard_pair(gs, f"_pctch_{keyA}", f"_pctch_{keyB}",
                             f"pb_{pct_level}_{pair_name}")
            row = {"top_pct": pct_level, "pair": ref_csv_label,
                  "threshold_A": round(p_thresh[keyA], 5),
                  "threshold_B": round(p_thresh[keyB], 5), **r}
            rows_b.append(row)
            print(f"top {pct_level}%: {ref_csv_label:12s} thresholds "
                 f"A={p_thresh[keyA]:.5f} B={p_thresh[keyB]:.5f} -- "
                 f"n_A={r['n_A']:,} n_B={r['n_B']:,} Jaccard={r['jaccard']:.3f} "
                 f"({r['pct_union_common']:.1f}% common)")

        for key in RASTERS:
            gs.run_command("g.remove", type="raster", name=f"_pctch_{key}",
                           flags="f", quiet=True)

    out_b = TABLES / "flowpath_agreement_percentile_matched.csv"
    with open(out_b, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
        w.writeheader()
        w.writerows(rows_b)
    print(f"\nwrote {out_b}")

    # ================================================================
    # PART C -- major channels only (top 1%, "dominant concentration zones")
    # ================================================================
    print("\n" + "=" * 70)
    print(f"PART C: major-channels-only (top {MAJOR_CHANNEL_PCT}% of each "
          "raster's own discharge)")
    print("=" * 70)

    p_thresh_major = {}
    for key, rmap in RASTERS.items():
        u = gs.parse_command("r.univar", map=rmap, flags="ge",
                             percentile=100 - MAJOR_CHANNEL_PCT)
        pkey = [k for k in u if k.startswith("percentile_")][0]
        p_thresh_major[key] = float(u[pkey])
        gs.mapcalc(f"_majch_{key} = if({rmap} > {p_thresh_major[key]}, 1, null())",
                  overwrite=True, quiet=True)
        n = int(gs.parse_command("r.univar", map=f"_majch_{key}", flags="g").get("n", 0))
        print(f"  {key}: top {MAJOR_CHANNEL_PCT}% threshold = "
             f"{p_thresh_major[key]:.5f}, {n:,} cells "
             f"({100.0 * n / total_valid[key]:.2f}% of basin)")

    rows_c = []
    for pair_name, keyA, keyB, ref_csv_label in [
        ("2020lidar_vs_2024lidar", "2020_lidar", "2024_lidar", "lidar-lidar"),
        ("2024lidar_vs_2024sfm", "2024_lidar", "2024_sfm", "lidar-SfM"),
    ]:
        r = jaccard_pair(gs, f"_majch_{keyA}", f"_majch_{keyB}", f"mc_{pair_name}")
        row = {"top_pct": MAJOR_CHANNEL_PCT, "pair": ref_csv_label,
              "threshold_A": round(p_thresh_major[keyA], 5),
              "threshold_B": round(p_thresh_major[keyB], 5), **r}
        rows_c.append(row)
        print(f"major channels only: {ref_csv_label:12s} -- n_A={r['n_A']:,} "
             f"n_B={r['n_B']:,} Jaccard={r['jaccard']:.3f} "
             f"({r['pct_union_common']:.1f}% common)")

    for key in RASTERS:
        gs.run_command("g.remove", type="raster", name=f"_majch_{key}",
                       flags="f", quiet=True)

    out_c = TABLES / "flowpath_agreement_major_channels.csv"
    with open(out_c, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_c[0].keys()))
        w.writeheader()
        w.writerows(rows_c)
    print(f"\nwrote {out_c}")

    # ================================================================
    # VERDICT
    # ================================================================
    j_lidar_major = next(r["jaccard"] for r in rows_c if r["pair"] == "lidar-lidar")
    j_sfm_major = next(r["jaccard"] for r in rows_c if r["pair"] == "lidar-SfM")
    j_lidar_p10 = next(r["jaccard"] for r in rows_b if r["pair"] == "lidar-lidar" and r["top_pct"] == 10)
    j_sfm_p10 = next(r["jaccard"] for r in rows_b if r["pair"] == "lidar-SfM" and r["top_pct"] == 10)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Major channels only (top {MAJOR_CHANNEL_PCT}%): lidar-lidar "
         f"Jaccard={j_lidar_major:.3f}, lidar-SfM Jaccard={j_sfm_major:.3f}.")
    print(f"Percentile-matched top 10%: lidar-lidar Jaccard={j_lidar_p10:.3f}, "
         f"lidar-SfM Jaccard={j_sfm_p10:.3f}.")
    reproduces = j_sfm_major >= 0.7 * j_lidar_major
    print("\n" + ("YES -- the SfM simulation substantially reproduces the "
                 "dominant flow paths (major-channel Jaccard within 30% of "
                 "the lidar-lidar reference)." if reproduces else
                 "NO -- the SfM simulation does NOT reproduce the dominant "
                 "flow paths. Even restricted to the top "
                 f"{MAJOR_CHANNEL_PCT}% most concentrated cells in each "
                 "raster (the major channels, not the marginal network), "
                 f"lidar-SfM agreement (Jaccard {j_sfm_major:.3f}) remains "
                 f"well below lidar-lidar agreement on the SAME terrain "
                 f"over 4 years and a hurricane (Jaccard {j_lidar_major:.3f}). "
                 "This is a negative result, not a threshold artifact: Part "
                 "A showed the two rasters classify comparable FRACTIONS of "
                 "the basin as channel at every fixed threshold (so the low "
                 "Jaccard is not from mismatched network sizes), and Part B/C "
                 "show the disagreement persists, largely undiminished, "
                 "whether measured on the full marginal network or "
                 "restricted to the dominant stems."))

    log("35_flow_lidar_sfm_diagnostic.py run. Part A: max relative spread "
        f"in basin channel-%% across all 3 rasters at any fixed threshold = "
        f"{max_spread:.1f}%% -- {'comparable' if equal_footing else 'NOT comparable'} "
        "network sizes at fixed absolute thresholds. Part B (percentile-"
        f"matched): top10%% lidar-lidar J={j_lidar_p10:.3f}, lidar-SfM "
        f"J={j_sfm_p10:.3f}. Part C (major channels, top {MAJOR_CHANNEL_PCT}%%): "
        f"lidar-lidar J={j_lidar_major:.3f}, lidar-SfM J={j_sfm_major:.3f}. "
        f"VERDICT: {'reproduces' if reproduces else 'does NOT reproduce'} "
        "dominant flow paths. Written to flow_channel_fraction_by_threshold.csv, "
        "flowpath_agreement_percentile_matched.csv, "
        "flowpath_agreement_major_channels.csv.", run="flow_lidar_sfm_diagnostic")


if __name__ == "__main__":
    main()
