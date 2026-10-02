#!/usr/bin/env python3
"""
28_manuscript_final_summary.py
==================================
Assembles the single consolidated, post-exclusion summary requested for
the manuscript: both sites, uniform 0.32 m LoD with 50 m^2 minimum feature
area, gross erosion, gross deposition, net, feature count, and the
fraction of THAT (50 m^2-filtered) gross volume carried by the top 8
features -- computed exactly from the already-saved, already-cross-checked
tables (no new GRASS computation, no hand arithmetic):

  watershed_lod_min_feature_area_d1_excluded.csv       (17_...py)
  dominant_features_watershed_d1_excluded.csv          (18_...py)
  lure_lod_min_feature_area_flagged_excluded.csv       (25_...py)
  dominant_features_lure_flagged_excluded.csv          (26_...py)
  watershed_corridor_comparison_d1_checked.csv         (21_...py)
  sediment_budget_comparison_final.csv                 (27_...py)

Output: results/tables/manuscript_final_summary.csv
        results/MANUSCRIPT_FINAL_NUMBERS.md (the single human-readable
        reference page)
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"


def load(path):
    return list(csv.DictReader(open(TABLES / path)))


def main():
    w_vol = {r["min_area_m2"]: r for r in load("watershed_lod_min_feature_area_d1_excluded.csv")}["50"]
    l_vol = {r["min_area_m2"]: r for r in load("lure_lod_min_feature_area_flagged_excluded.csv")}["50"]
    w_feat = load("dominant_features_watershed_d1_excluded.csv")
    l_feat = load("dominant_features_lure_flagged_excluded.csv")
    corridor = load("watershed_corridor_comparison_d1_checked.csv")
    budget = {r["min_feature_area_m2"]: r for r in load("sediment_budget_comparison_final.csv")}["50"]

    def top8_fraction(feat_rows, kind, gross_vol):
        top8 = sum(abs(float(r["volume_m3"])) for r in feat_rows
                  if r["kind"] == kind and int(r["rank"]) <= 8)
        return top8, 100 * top8 / gross_vol if gross_vol else 0

    w_ero_top8, w_ero_pct = top8_fraction(w_feat, "erosion", abs(float(w_vol["erosion_volume_m3"])))
    w_dep_top8, w_dep_pct = top8_fraction(w_feat, "deposition", abs(float(w_vol["deposition_volume_m3"])))
    l_ero_top8, l_ero_pct = top8_fraction(l_feat, "erosion", abs(float(l_vol["erosion_volume_m3"])))
    l_dep_top8, l_dep_pct = top8_fraction(l_feat, "deposition", abs(float(l_vol["deposition_volume_m3"])))

    summary_rows = [
        {
            "site": "Watershed (cat34)",
            "exclusions": "D1",
            "min_lod_m": 0.32, "min_area_m2": 50,
            "gross_erosion_m3": float(w_vol["erosion_volume_m3"]),
            "n_erosion_features": int(w_vol["n_erosion_features"]),
            "top8_erosion_pct_of_gross": round(w_ero_pct, 1),
            "gross_deposition_m3": float(w_vol["deposition_volume_m3"]),
            "n_deposition_features": int(w_vol["n_deposition_features"]),
            "top8_deposition_pct_of_gross": round(w_dep_pct, 1),
            "net_m3": float(w_vol["net_volume_m3"]),
        },
        {
            "site": "Lake Lure",
            "exclusions": "E5,E6,E7,E8,D7",
            "min_lod_m": 0.32, "min_area_m2": 50,
            "gross_erosion_m3": float(l_vol["erosion_volume_m3"]),
            "n_erosion_features": int(l_vol["n_erosion_features"]),
            "top8_erosion_pct_of_gross": round(l_ero_pct, 1),
            "gross_deposition_m3": float(l_vol["deposition_volume_m3"]),
            "n_deposition_features": int(l_vol["n_deposition_features"]),
            "top8_deposition_pct_of_gross": round(l_dep_pct, 1),
            "net_m3": float(l_vol["net_volume_m3"]),
        },
    ]

    out_csv = TABLES / "manuscript_final_summary.csv"
    with open(out_csv, "w", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        for r in summary_rows:
            wcsv.writerow(r)
    print(f"wrote {out_csv}")
    for r in summary_rows:
        print(f"\n{r['site']} (exclusions: {r['exclusions']}):")
        print(f"  erosion:    {r['gross_erosion_m3']:+,.0f} m3, {r['n_erosion_features']} features, "
              f"top8={r['top8_erosion_pct_of_gross']}%")
        print(f"  deposition: {r['gross_deposition_m3']:+,.0f} m3, {r['n_deposition_features']} features, "
              f"top8={r['top8_deposition_pct_of_gross']}%")
        print(f"  net:        {r['net_m3']:+,.0f} m3")

    # ---- write the single human-readable reference page
    md = f"""# Manuscript final numbers

Single reference page — post-exclusion, final values. Do not hand-copy
numbers from any other file; if a number is needed that is not here,
regenerate this page rather than pulling from an intermediate table.

**Exclusions applied:** watershed D1 (canopy/interpolation artifact,
+27,343 m³, resolved 2026-08-28); Lake Lure E5, E6, E7, E8 (erosion,
shared branching-shape artifact) and D7 (deposition, shoreline/waterline
differencing artifact) (resolved 2026-08-28). Full reasoning for every
exclusion: `RESULTS_FOR_PAPER.md`, "D1 investigation — RESOLVED" and
"Check 1: the D1 standard applied to Lake Lure E5, E6, E7, E8, D7".

**Method, both sites:** uniform 0.32 m LoD, `r.clump` 4-connectivity,
minimum feature area ≥ 50 m², coregistered + bias-corrected lidar-lidar
DoD (`coreg_dh_corrected` / `coreg_lure_lidar_dh_corrected`, water
excluded at Lake Lure). Produced by `28_manuscript_final_summary.py`,
consolidating already-cross-checked tables (17, 18, 25, 26, 21, 27) — no
new computation, no hand arithmetic.

## Volumes and features

| site | exclusions | gross erosion (m³) | erosion features | top-8 % of gross | gross deposition (m³) | deposition features | top-8 % of gross | **net (m³)** |
|---|---|---|---|---|---|---|---|---|
| Watershed (cat34) | D1 | {summary_rows[0]['gross_erosion_m3']:+,.0f} | {summary_rows[0]['n_erosion_features']} | {summary_rows[0]['top8_erosion_pct_of_gross']}% | {summary_rows[0]['gross_deposition_m3']:+,.0f} | {summary_rows[0]['n_deposition_features']} | {summary_rows[0]['top8_deposition_pct_of_gross']}% | **{summary_rows[0]['net_m3']:+,.0f}** |
| Lake Lure | E5,E6,E7,E8,D7 | {summary_rows[1]['gross_erosion_m3']:+,.0f} | {summary_rows[1]['n_erosion_features']} | {summary_rows[1]['top8_erosion_pct_of_gross']}% | {summary_rows[1]['gross_deposition_m3']:+,.0f} | {summary_rows[1]['n_deposition_features']} | {summary_rows[1]['top8_deposition_pct_of_gross']}% | **{summary_rows[1]['net_m3']:+,.0f}** |

Full sensitivity across 0/50/100 m²: `watershed_lod_min_feature_area_d1_excluded.csv`,
`lure_lod_min_feature_area_flagged_excluded.csv`.

## The two watershed corridors (D1 does not overlap either — verified, not assumed)

| corridor | area above LoD (m²) | erosion (m³) | deposition (m³) | **net (m³)** | deposition:erosion ratio |
|---|---|---|---|---|---|
"""
    for r in corridor:
        md += (f"| {r['corridor']} | {float(r['area_above_lod_m2']):,.0f} | "
              f"{float(r['erosion_volume_m3']):+,.0f} | {float(r['deposition_volume_m3']):+,.0f} | "
              f"**{float(r['net_volume_m3']):+,.0f}** | {float(r['deposition_to_erosion_ratio']):.2f} |\n")

    md += f"""
Full table: `watershed_corridor_comparison_d1_checked.csv`. The western
corridor retains proportionally more of its own eroded material locally
(higher deposition:erosion ratio); the main scar corridor evacuates a
larger share of what it erodes. Both are net erosional.

## Sediment magnitude comparison (50 m² minimum feature area)

| | value |
|---|---|
| Lake Lure net ÷ watershed net (magnitude) | **{float(budget['ratio_lure_net_to_watershed_net']):.2f}×** |
| Lake Lure gross deposition ÷ watershed gross erosion (magnitude) | **{float(budget['ratio_lure_gross_dep_to_watershed_gross_ero']):.2f}×** |

Full sensitivity across 0/50/100 m²: `sediment_budget_comparison_final.csv`.

**Mandatory caveat, restate wherever this is used: cat34 is one tributary
among many draining to Lake Lure. This is a comparison of magnitudes
only — it is not a mass balance, not a closed sediment budget, and does
not attribute any specific fraction of the reservoir's deposition to this
watershed.**

## Superseded versions (kept on disk for provenance, do not cite)

| superseded file | superseded by |
|---|---|
| `watershed_lod_min_feature_area.csv` (D1 included) | `watershed_lod_min_feature_area_d1_excluded.csv` |
| `dominant_features_watershed.csv` (D1 included) | `dominant_features_watershed_d1_excluded.csv` |
| `lure_lod_min_feature_area.csv` (E5-E8/D7 included) | `lure_lod_min_feature_area_flagged_excluded.csv` |
| `dominant_features_lure.csv` (E5-E8/D7 included) | `dominant_features_lure_flagged_excluded.csv` |
| `dominant_features_lure_map.png` (old ranking) | `dominant_features_lure_map_flagged_excluded.png` |
| `sediment_budget_comparison.csv`, `_d1_excluded.csv` | `sediment_budget_comparison_final.csv` |
| `watershed_volumes_with_without_d1.csv`, `lure_volumes_with_without_flagged.csv` | comparison-only, not superseded — kept as the with/without sensitivity record |
"""
    out_md = ROOT / "results" / "MANUSCRIPT_FINAL_NUMBERS.md"
    out_md.write_text(md)
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
