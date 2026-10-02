#!/usr/bin/env python3
"""
26_dominant_features_lure_flagged_excluded.py
==================================================
Supersedes the Lake Lure half of 13_dominant_features_both_sites.py now
that E5, E6, E7, E8 (erosion) and D7 (deposition) are resolved as
artifacts. Identical method (uniform 0.32 m LoD, water-excluded, r.clump,
r.volume, ranked by |volume|) with the five features' footprints excluded
from their respective mask before clumping -- same pattern as
18_dominant_features_d1_excluded.py.

Also regenerates the labelled orthophoto overview map with the new
ranking (E5-E8/D7 no longer appear; ranks shift up).

Output:
  results/tables/dominant_features_lure_flagged_excluded.csv (canonical,
    manuscript-facing -- supersedes dominant_features_lure.csv, which is
    kept on disk, marked superseded)
  results/figures/dominant_features_lure_map_flagged_excluded.png
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

LOD = 0.32
FLAGGED_EROSION_MASKS = ["lure_E5_mask", "lure_E6_mask", "lure_E7_mask", "lure_E8_mask"]
FLAGGED_DEPOSITION_MASKS = ["lure_D7_mask"]


def rank_features(gs, dh, water_mask, exclude_ero, exclude_dep):
    ero_excl = " && ".join(f"isnull({m})" for m in exclude_ero)
    dep_excl = " && ".join(f"isnull({m})" for m in exclude_dep)
    gs.mapcalc(f"_d26_tmp = if(isnull({water_mask}) && abs({dh}) > {LOD}, "
              f"{dh}, null())", overwrite=True, quiet=True)
    gs.mapcalc(f"_d26_ero = if(_d26_tmp < 0 && {ero_excl}, _d26_tmp, null())",
              overwrite=True, quiet=True)
    gs.mapcalc(f"_d26_dep = if(_d26_tmp > 0 && {dep_excl}, _d26_tmp, null())",
              overwrite=True, quiet=True)

    results = {}
    for kind, dhmap in (("erosion", "_d26_ero"), ("deposition", "_d26_dep")):
        gs.mapcalc(f"_d26_bin = if(!isnull({dhmap}), 1, null())",
                  overwrite=True, quiet=True)
        gs.run_command("r.clump", input="_d26_bin", output="_d26_clump",
                       overwrite=True, quiet=True)
        raw = gs.read_command("r.volume", input=dhmap, clump="_d26_clump",
                              format="json")
        rows = json.loads(raw)
        rows = sorted(rows, key=lambda r: -abs(r["volume"]))
        results[kind] = rows
    return results["erosion"], results["deposition"]


def write_csv(path, ero_rows, dep_rows, total_ero_vol, total_dep_vol):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "rank", "area_m2", "mean_depth_m", "volume_m3",
                   "pct_of_gross", "easting", "northing", "flagged_excluded"])
        for kind, rows, total in (("erosion", ero_rows, total_ero_vol),
                                  ("deposition", dep_rows, total_dep_vol)):
            for i, r in enumerate(rows, start=1):
                pct = 100.0 * abs(r["volume"]) / abs(total) if total else 0
                w.writerow([kind, i, r["cells"], f"{r['average']:.3f}",
                           f"{r['volume']:.2f}", f"{pct:.2f}",
                           f"{r['easting']:.1f}", f"{r['northing']:.1f}", "yes"])


def main():
    gs, gj = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    gs.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")

    for m in FLAGGED_EROSION_MASKS + FLAGGED_DEPOSITION_MASKS:
        found = gs.find_file(name=m, element="cell", mapset="PERMANENT")
        if not found or not found.get("name"):
            sys.exit(f"{m} not found -- run 22_resolve_lure_flagged_features.py first.")

    ero, dep = rank_features(gs, "coreg_lure_lidar_dh_corrected", "lure_water_mask",
                             FLAGGED_EROSION_MASKS, FLAGGED_DEPOSITION_MASKS)
    tot_ero = sum(r["volume"] for r in ero)
    tot_dep = sum(r["volume"] for r in dep)

    print(f"\n=== LAKE LURE, flagged excluded: top 8 erosion (of {len(ero)} total) ===")
    for i, r in enumerate(ero[:8], start=1):
        pct = 100.0 * abs(r["volume"]) / abs(tot_ero) if tot_ero else 0
        print(f"{i:4d} {r['cells']:9,d} {r['average']:8.2f} {r['volume']:10,.1f} {pct:7.1f}")
    print(f"\n=== LAKE LURE, flagged excluded: top 8 deposition (of {len(dep)} total) ===")
    for i, r in enumerate(dep[:8], start=1):
        pct = 100.0 * abs(r["volume"]) / abs(tot_dep) if tot_dep else 0
        print(f"{i:4d} {r['cells']:9,d} {r['average']:8.2f} {r['volume']:10,.1f} {pct:7.1f}")

    top8_ero_pct = 100 * sum(abs(r["volume"]) for r in ero[:8]) / abs(tot_ero)
    top8_dep_pct = 100 * sum(abs(r["volume"]) for r in dep[:8]) / abs(tot_dep)
    print(f"\ntop 8 erosion = {top8_ero_pct:.1f}% of gross erosion")
    print(f"top 8 deposition = {top8_dep_pct:.1f}% of gross deposition")

    out_csv = TABLES / "dominant_features_lure_flagged_excluded.csv"
    write_csv(out_csv, ero, dep, tot_ero, tot_dep)
    print(f"\nwrote {out_csv} ({len(ero)} erosion + {len(dep)} deposition rows)")

    # ---- regenerate the labelled map with the new ranking
    def make_points_vector(rows, outname):
        lines = [f"{r['easting']}|{r['northing']}|{i}" for i, r in enumerate(rows, start=1)]
        gs.write_command("v.in.ascii", input="-", output=outname, separator="pipe",
                         format="point", overwrite=True, quiet=True, stdin="\n".join(lines))

    gs.run_command("i.group", group="_d26_ortho_grp",
                   input="Ortho_aerial_2024.1,Ortho_aerial_2024.2,Ortho_aerial_2024.3",
                   quiet=True)
    gs.run_command("r.composite", red="Ortho_aerial_2024.1", green="Ortho_aerial_2024.2",
                   blue="Ortho_aerial_2024.3", output="_d26_ortho_rgb",
                   overwrite=True, quiet=True)
    reg = gs.parse_command("g.region", flags="g")
    gs.run_command("g.region", n=float(reg["n"]) + 60, s=float(reg["s"]) - 60,
                   e=float(reg["e"]) + 60, w=float(reg["w"]) - 60, res=1, flags="a")

    make_points_vector(ero[:8], "d26_ero_pts")
    make_points_vector(dep[:8], "d26_dep_pts")
    img = gj.Map(width=1400, height=1000, use_region=True)
    img.d_rast(map="_d26_ortho_rgb")
    img.d_vect(map="d26_ero_pts", type="point", color="red", fill_color="red",
              icon="basic/circle", size=16)
    img.d_vect(map="d26_dep_pts", type="point", color="blue", fill_color="blue",
              icon="basic/circle", size=16)
    for i, r in enumerate(ero[:8], start=1):
        img.d_text(text=f"E{i}", at=f"{r['easting']},{r['northing']}", flags="g",
                  color="red", size=3)
    for i, r in enumerate(dep[:8], start=1):
        img.d_text(text=f"D{i}", at=f"{r['easting']},{r['northing']}", flags="g",
                  color="blue", size=3)
    img.d_text(text="Lake Lure, E5/E6/E7/E8/D7 EXCLUDED: top-8 erosion (red, E) / "
              "deposition (blue, D)", at="2,97", color="black", bgcolor="white", size=2.5)
    out_png = FIGURES / "dominant_features_lure_map_flagged_excluded.png"
    img.save(str(out_png))
    print(f"wrote {out_png}")

    gs.run_command("g.remove", type="vector", name="d26_ero_pts,d26_dep_pts",
                   flags="f", quiet=True)
    gs.run_command("g.remove", type="raster",
                   name="_d26_tmp,_d26_ero,_d26_dep,_d26_bin,_d26_clump,_d26_ortho_rgb",
                   flags="f", quiet=True)
    gs.run_command("g.remove", type="group", name="_d26_ortho_grp", flags="f", quiet=True)

    log("26_dominant_features_lure_flagged_excluded.py run. Supersedes the "
        "Lake Lure half of 13_dominant_features_both_sites.py -- E5/E6/E7/E8 "
        "(erosion) and D7 (deposition) excluded before clumping. LoD=0.32m, "
        f"water-excluded. {len(ero)} erosion features (top1={ero[0]['volume']:.0f}m3), "
        f"{len(dep)} deposition features (top1={dep[0]['volume']:.0f}m3). "
        f"top8 erosion={top8_ero_pct:.1f}% of gross, top8 deposition="
        f"{top8_dep_pct:.1f}% of gross. Regenerated "
        "dominant_features_lure_map_flagged_excluded.png. Written to "
        f"{out_csv.name}.", run="dominant_features_lure_flagged_excluded")


if __name__ == "__main__":
    main()
