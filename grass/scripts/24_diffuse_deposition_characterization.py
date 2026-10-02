#!/usr/bin/env python3
"""
24_diffuse_deposition_characterization.py
=============================================
Tests whether the watershed's diffuse deposition (the 56% of gross
deposition NOT in the top-8 features, D1 excluded) is geomorphically
plausible or looks like dispersed artifact contamination.

Characterises, per feature (area, canopy height, distance to the
pipeline's own established channel-distance raster -- sm_chan_dist for the
watershed, built by 02_build_stable_mask.py with CHANNEL_BUFFER_M=30,
accumulation-threshold channel; sml_chan_dist for Lake Lure, the identical
method from 02_build_stable_mask_lure.py -- a genuine apples-to-apples
metric, not a re-improvised one) for three populations:

  1. Watershed deposition, D1 excluded, ranks 9+ ("the diffuse 56%")
  2. Watershed erosion, all ranks (comparison population -- concentrated,
     72.8% in top 8)
  3. Lake Lure deposition, all ranks (comparison population -- also fairly
     concentrated, 55.8% in top 8, and independently validated as mostly
     real in RESULTS_FOR_PAPER.md's LoD validation section)

Then reports a size-distribution / canopy-height / channel-distance
comparison across all three, and quantifies watershed deposition volume
under a stricter reading: excluding diffuse (rank 9+) features whose
centroid sits BOTH under closed canopy (>= CLOSED_CANOPY_M) AND far from
the established channel (>= CHANNEL_BUFFER_M -- the same 30 m buffer
already used project-wide to define "stable"/unlikely-to-have-changed
terrain, not a new threshold invented for this test).

COMPUTED AND REPORTED ONLY -- not applied to any manuscript-facing table
without confirmation, per instruction.

Outputs:
  results/tables/watershed_deposition_remainder_features.csv (564 rows,
    per-feature canopy/chan-dist)
  results/tables/watershed_erosion_features_context.csv (708 rows)
  results/tables/lure_deposition_features_context.csv (4302 rows)
  results/tables/diffuse_deposition_comparison_summary.csv (the
    cross-population comparison table)
  results/tables/watershed_deposition_stricter_reading.csv (volume under
    the closed-canopy + far-from-channel exclusion, at several threshold
    combinations)
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

CLOSED_CANOPY_M = 5.0     # matches the [5,10) bin floor used throughout the
                          # project's canopy-binned LoD work
CHANNEL_BUFFER_M = 30.0   # matches 02_build_stable_mask.py's CHANNEL_BUFFER_M


def load_rows(path):
    return list(csv.DictReader(open(path)))


def sample_points(gs, rows, out_prefix, canopy_map, chan_map):
    """Build a point vector from rows (needs easting/northing/rank), sample
    canopy_map and chan_map at each point, return list of dicts with the
    sampled values merged in, keyed by rank (int)."""
    lines = []
    for r in rows:
        lines.append(f"{r['easting']}|{r['northing']}|{r['rank']}")
    gs.write_command("v.in.ascii", input="-", output=out_prefix, separator="pipe",
                     format="point", overwrite=True, quiet=True,
                     stdin="\n".join(lines))
    gs.run_command("v.db.addcolumn", map=out_prefix,
                   columns="canopy_m double precision, chan_dist_m double precision",
                   quiet=True)
    gs.run_command("v.what.rast", map=out_prefix, raster=canopy_map, column="canopy_m",
                   quiet=True)
    gs.run_command("v.what.rast", map=out_prefix, raster=chan_map, column="chan_dist_m",
                   quiet=True)
    out = gs.read_command("v.db.select", map=out_prefix, separator="comma",
                          columns="cat,canopy_m,chan_dist_m")
    sampled = {}
    lines_out = out.strip().split("\n")[1:]
    for line in lines_out:
        cat, canopy, chan = line.split(",")
        sampled[int(cat)] = {
            "canopy_m": float(canopy) if canopy not in ("", "*") else None,
            "chan_dist_m": float(chan) if chan not in ("", "*") else None,
        }
    gs.run_command("g.remove", type="vector", name=out_prefix, flags="f", quiet=True)
    return sampled


def merge_and_write(rows, sampled, out_path):
    fieldnames = list(rows[0].keys()) + ["canopy_m", "chan_dist_m"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            s = sampled.get(int(r["rank"]), {})
            row = dict(r)
            row["canopy_m"] = s.get("canopy_m", "")
            row["chan_dist_m"] = s.get("chan_dist_m", "")
            w.writerow(row)
    return out_path


def percentiles(values, ps=(10, 25, 50, 75, 90)):
    vs = sorted(values)
    n = len(vs)
    out = {}
    for p in ps:
        idx = min(n - 1, max(0, round(p / 100 * (n - 1))))
        out[p] = vs[idx]
    return out


def summarize(label, merged_rows, volume_key="volume_m3", area_key="area_m2"):
    areas = [float(r[area_key]) for r in merged_rows]
    canopies = [r["canopy_m"] for r in merged_rows if r["canopy_m"] != ""]
    canopies = [float(c) for c in canopies]
    chans = [r["chan_dist_m"] for r in merged_rows if r["chan_dist_m"] != ""]
    chans = [float(c) for c in chans]
    vols = [abs(float(r[volume_key])) for r in merged_rows]
    ap = percentiles(areas)
    return {
        "population": label,
        "n_features": len(merged_rows),
        "total_abs_volume_m3": sum(vols),
        "area_mean_m2": sum(areas) / len(areas),
        "area_p10_m2": ap[10], "area_p25_m2": ap[25], "area_p50_m2": ap[50],
        "area_p75_m2": ap[75], "area_p90_m2": ap[90],
        "canopy_mean_m": sum(canopies) / len(canopies) if canopies else None,
        "canopy_median_m": percentiles(canopies)[50] if canopies else None,
        "pct_closed_canopy_ge5m": (100 * sum(1 for c in canopies if c >= CLOSED_CANOPY_M)
                                   / len(canopies) if canopies else None),
        "chan_dist_mean_m": sum(chans) / len(chans) if chans else None,
        "chan_dist_median_m": percentiles(chans)[50] if chans else None,
        "pct_far_from_channel_ge30m": (100 * sum(1 for c in chans if c >= CHANNEL_BUFFER_M)
                                       / len(chans) if chans else None),
    }


def main():
    # ---------------------------------------------------------- watershed
    gs_w, gj_w = grass_session(project="DEM_generation", mapset="for_codem")
    gs_w.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")

    all_dep = load_rows(TABLES / "dominant_features_watershed_d1_excluded.csv")
    dep_rows = [r for r in all_dep if r["kind"] == "deposition"]
    ero_rows = [r for r in all_dep if r["kind"] == "erosion"]
    dep_remainder = [r for r in dep_rows if int(r["rank"]) > 8]
    print(f"watershed deposition: {len(dep_rows)} total, "
          f"{len(dep_remainder)} beyond rank 8 (the diffuse population)")
    print(f"watershed erosion: {len(ero_rows)} total")

    sampled_dep_rem = sample_points(gs_w, dep_remainder, "wdiff_dep",
                                    "coreg_canopy", "sm_chan_dist")
    merged_dep_rem = []
    for r in dep_remainder:
        s = sampled_dep_rem.get(int(r["rank"]), {})
        merged_dep_rem.append({**r, "canopy_m": s.get("canopy_m", ""),
                              "chan_dist_m": s.get("chan_dist_m", "")})
    merge_and_write(dep_remainder, sampled_dep_rem,
                    TABLES / "watershed_deposition_remainder_features.csv")

    sampled_ero = sample_points(gs_w, ero_rows, "wdiff_ero",
                                "coreg_canopy", "sm_chan_dist")
    merged_ero = []
    for r in ero_rows:
        s = sampled_ero.get(int(r["rank"]), {})
        merged_ero.append({**r, "canopy_m": s.get("canopy_m", ""),
                          "chan_dist_m": s.get("chan_dist_m", "")})
    merge_and_write(ero_rows, sampled_ero,
                    TABLES / "watershed_erosion_features_context.csv")

    # ------------------------------------------------------------ Lake Lure
    gs_l, gj_l = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    gs_l.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")

    lure_dep = load_rows(TABLES / "dominant_features_lure.csv")
    lure_dep_rows = [r for r in lure_dep if r["kind"] == "deposition"]
    print(f"Lake Lure deposition: {len(lure_dep_rows)} total")

    sampled_lure = sample_points(gs_l, lure_dep_rows, "ldiff_dep",
                                 "coreg_lure_lidar_canopy", "sml_chan_dist")
    merged_lure = []
    for r in lure_dep_rows:
        s = sampled_lure.get(int(r["rank"]), {})
        merged_lure.append({**r, "canopy_m": s.get("canopy_m", ""),
                           "chan_dist_m": s.get("chan_dist_m", "")})
    merge_and_write(lure_dep_rows, sampled_lure,
                    TABLES / "lure_deposition_features_context.csv")

    # ------------------------------------------------------------ summary
    summary_rows = [
        summarize("watershed deposition, diffuse (rank 9+, D1 excluded)", merged_dep_rem),
        summarize("watershed erosion, all ranks", merged_ero),
        summarize("Lake Lure deposition, all ranks", merged_lure),
    ]
    out_summary = TABLES / "diffuse_deposition_comparison_summary.csv"
    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\nwrote {out_summary}")
    for r in summary_rows:
        print(f"\n{r['population']}: n={r['n_features']}, "
              f"total|vol|={r['total_abs_volume_m3']:,.0f} m3, "
              f"area p50={r['area_p50_m2']:.0f} m2, "
              f"canopy mean={r['canopy_mean_m']:.1f} m "
              f"({r['pct_closed_canopy_ge5m']:.0f}% >= {CLOSED_CANOPY_M:.0f}m), "
              f"chan_dist mean={r['chan_dist_mean_m']:.0f} m "
              f"({r['pct_far_from_channel_ge30m']:.0f}% >= {CHANNEL_BUFFER_M:.0f}m)")

    # ---------------------------------------------------- stricter reading
    strict_rows = []
    top8_dep_vol = sum(abs(float(r["volume_m3"])) for r in dep_rows if int(r["rank"]) <= 8)
    total_dep_vol = sum(abs(float(r["volume_m3"])) for r in dep_rows)
    for canopy_thresh in (0, CLOSED_CANOPY_M, 10.0):
        for chan_thresh in (0, CHANNEL_BUFFER_M, 50.0):
            if canopy_thresh == 0 and chan_thresh == 0:
                kept_vol = total_dep_vol
                excluded_n = 0
            else:
                kept_vol = top8_dep_vol
                excluded_n = 0
                for r in merged_dep_rem:
                    c = r["canopy_m"]
                    d = r["chan_dist_m"]
                    is_contaminated = (c != "" and float(c) >= canopy_thresh and
                                      d != "" and float(d) >= chan_thresh)
                    if is_contaminated:
                        excluded_n += 1
                    else:
                        kept_vol += abs(float(r["volume_m3"]))
            strict_rows.append({
                "canopy_threshold_m": canopy_thresh, "chan_dist_threshold_m": chan_thresh,
                "deposition_volume_kept_m3": kept_vol,
                "pct_of_original_gross_deposition": 100 * kept_vol / total_dep_vol,
                "n_diffuse_features_excluded": excluded_n,
                "n_diffuse_features_total": len(dep_remainder),
            })
            print(f"  canopy>={canopy_thresh}m & chan_dist>={chan_thresh}m: "
                 f"kept={kept_vol:,.0f} m3 ({100*kept_vol/total_dep_vol:.1f}% of gross), "
                 f"excluded {excluded_n}/{len(dep_remainder)} diffuse features")

    out_strict = TABLES / "watershed_deposition_stricter_reading.csv"
    with open(out_strict, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(strict_rows[0].keys()))
        w.writeheader()
        for r in strict_rows:
            w.writerow(r)
    print(f"\nwrote {out_strict}")

    log("24_diffuse_deposition_characterization.py run. Characterised "
        f"watershed diffuse deposition (rank 9+, D1 excluded, "
        f"{len(dep_remainder)} features), watershed erosion "
        f"({len(ero_rows)} features), Lake Lure deposition "
        f"({len(lure_dep_rows)} features) by area, canopy height "
        "(coreg_canopy / coreg_lure_lidar_canopy), and distance to the "
        "pipeline's own established channel raster (sm_chan_dist / "
        "sml_chan_dist, both from the 02_build_stable_mask*.py scripts, "
        f"CHANNEL_BUFFER_M={CHANNEL_BUFFER_M}). Summary: " +
        "; ".join(f"{r['population']}: canopy_mean={r['canopy_mean_m']:.1f}m, "
                 f"chan_dist_mean={r['chan_dist_mean_m']:.0f}m"
                 for r in summary_rows) +
        f". Stricter-reading sensitivity in {out_strict.name}. NOT applied "
        "to any manuscript table -- computed and reported only, per "
        f"instruction. Written to {out_summary.name}.",
        run="diffuse_deposition_characterization")


if __name__ == "__main__":
    main()
