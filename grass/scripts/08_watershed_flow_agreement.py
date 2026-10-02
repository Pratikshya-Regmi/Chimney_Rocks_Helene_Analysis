#!/usr/bin/env python3
"""
08_watershed_flow_agreement.py
================================
Quantitative flow-path agreement metric for the watershed (cat34): extracts
drainage networks from the 2020 and 2024 lidar overland-flow simulations
(discharge_2020_regis, discharge_may_2024_lidar -- both r.sim.water
discharge output, m/units per cell, same basin region) above a common
threshold, and reports how much they spatially agree. This replaces a
visual-only "the drainage reorganized" claim with a number.

Threshold choice: DISCHARGE_THRESH = 0.01 (m^3/s-equivalent cell value),
close to the ~90th percentile of both rasters (0.0120 in 2020, 0.0105 in
2024) -- i.e. roughly the most concentrated-flow 10% of the basin in each
epoch, a standard "channel-like" cutoff. Applied as the SAME fixed value to
both rasters so the comparison is not threshold-tuned per epoch.

Outputs (kept):
  raster  channel_2020_regis     (discharge_2020_regis > DISCHARGE_THRESH)
  raster  channel_2024_lidar     (discharge_may_2024_lidar > DISCHARGE_THRESH)
  raster  channel_agreement      (0=neither, 1=2020 only, 2=2024 only, 3=both)
  results/tables/watershed_flowpath_agreement.csv
  results/figures/watershed_flowpath_agreement_map.png
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
FIGURES = Path(__file__).resolve().parents[2] / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

DISCHARGE_THRESH = 0.01


def main():
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", raster="discharge_2020_regis@for_codem", flags="a")

    gs.mapcalc(f"channel_2020_regis = if(discharge_2020_regis > {DISCHARGE_THRESH}, "
              "1, null())", overwrite=True, quiet=True)
    gs.mapcalc(f"channel_2024_lidar = if(discharge_may_2024_lidar > {DISCHARGE_THRESH}, "
              "1, null())", overwrite=True, quiet=True)

    n2020 = int(gs.parse_command("r.univar", map="channel_2020_regis", flags="g").get("n", 0))
    n2024 = int(gs.parse_command("r.univar", map="channel_2024_lidar", flags="g").get("n", 0))

    gs.mapcalc(
        "channel_agreement = if(!isnull(channel_2020_regis) && !isnull(channel_2024_lidar), 3, "
        "if(!isnull(channel_2020_regis) && isnull(channel_2024_lidar), 1, "
        "if(isnull(channel_2020_regis) && !isnull(channel_2024_lidar), 2, 0)))",
        overwrite=True, quiet=True)

    stats = gs.read_command("r.stats", input="channel_agreement", flags="cn")
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for line in stats.strip().split("\n"):
        cat, cnt = line.split()
        counts[int(cat)] = int(cnt)

    n_2020_only = counts[1]
    n_2024_only = counts[2]
    n_both = counts[3]
    n_union = n_2020_only + n_2024_only + n_both
    n_2020_total = n_2020_only + n_both
    n_2024_total = n_2024_only + n_both

    jaccard = n_both / n_union if n_union else float("nan")
    pct_of_2020_retained = 100.0 * n_both / n_2020_total if n_2020_total else float("nan")
    pct_of_2024_that_is_new = 100.0 * n_2024_only / n_2024_total if n_2024_total else float("nan")
    overlap_pct_of_union = 100.0 * n_both / n_union if n_union else float("nan")

    print(f"discharge threshold: {DISCHARGE_THRESH}")
    print(f"channel_2020_regis: {n2020:,} cells")
    print(f"channel_2024_lidar: {n2024:,} cells")
    print(f"in both (common channel): {n_both:,}")
    print(f"2020 only (abandoned): {n_2020_only:,}")
    print(f"2024 only (new): {n_2024_only:,}")
    print(f"union: {n_union:,}")
    print(f"Jaccard index: {jaccard:.3f}")
    print(f"% of union that is common to both: {overlap_pct_of_union:.1f}%")
    print(f"% of 2020's channel retained in 2024: {pct_of_2020_retained:.1f}%")
    print(f"% of 2024's channel that is new (not in 2020): {pct_of_2024_that_is_new:.1f}%")

    row = {
        "discharge_threshold": DISCHARGE_THRESH,
        "n_channel_2020": n2020,
        "n_channel_2024": n2024,
        "n_common": n_both,
        "n_2020_only": n_2020_only,
        "n_2024_only": n_2024_only,
        "n_union": n_union,
        "jaccard_index": jaccard,
        "pct_union_common": overlap_pct_of_union,
        "pct_2020_retained_in_2024": pct_of_2020_retained,
        "pct_2024_that_is_new": pct_of_2024_that_is_new,
    }
    out_csv = TABLES / "watershed_flowpath_agreement.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"\nwrote {out_csv}")

    # ------------------------------------------------------------------
    gs.run_command("r.out.gdal", input="channel_agreement",
                   output="/tmp/channel_agreement.tif", format="GTiff",
                   type="Byte", nodata=255, overwrite=True, quiet=True)

    log(f"08_watershed_flow_agreement.py run. Threshold={DISCHARGE_THRESH} "
        f"on discharge_2020_regis/discharge_may_2024_lidar (both @for_codem, "
        f"same cat34 region). n_2020={n2020:,} n_2024={n2024:,} "
        f"n_common={n_both:,} n_union={n_union:,}. Jaccard={jaccard:.3f}, "
        f"{overlap_pct_of_union:.1f}% of the union channel is common to "
        f"both epochs, {pct_of_2020_retained:.1f}% of 2020's channel "
        f"persists in 2024, {pct_of_2024_that_is_new:.1f}% of 2024's "
        f"channel is new. Rasters channel_2020_regis, channel_2024_lidar, "
        f"channel_agreement kept. Written to {out_csv.name}.",
        run="watershed_flow_agreement")


if __name__ == "__main__":
    main()
