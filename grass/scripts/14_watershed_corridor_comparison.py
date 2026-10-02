#!/usr/bin/env python3
"""
14_watershed_corridor_comparison.py
=====================================
Treats the main scar corridor (B-B', transect2_longitudinal) and the
western corridor (C-C', transect3_westcorridor) as separate spatial units
and reports, for each: area above the 0.32 m LoD, gross erosion, gross
deposition, net, and the deposition:erosion ratio -- quantifying the
transects' visual contrast (one corridor evacuating material, the other
storing it) with actual zonal statistics.

CORRIDOR DEFINITION (a stated analysis choice, not an authoritative
geomorphic delineation): a buffer of CORRIDOR_HALFWIDTH_M around each
transect's own traced line (from transect_geometry.csv), matching the
scale of the visible scar (A-A' crossed ~31 m of clear erosion signal at
its widest per RESULTS_FOR_PAPER.md / MANUSCRIPT_ASSETS.md). 20 m
half-width (40 m total corridor width) is used for BOTH corridors so the
comparison is not width-biased. Overlap between the two buffers is checked
and reported, not silently resolved.

Outputs:
  vector  corridor_scar_buffer, corridor_west_buffer   (kept, documented)
  raster  corridor_scar_mask, corridor_west_mask        (kept, documented)
  results/tables/watershed_corridor_comparison.csv
"""

import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

LOD = 0.32
CORRIDOR_HALFWIDTH_M = 20.0


def write_line_ascii(df, transect_name, path):
    sub = df[df["transect"] == transect_name].sort_values("vertex_order")
    with open(path, "w") as f:
        for _, row in sub.iterrows():
            f.write(f"{row['easting']},{row['northing']}\n")
    return len(sub)


def main():
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")

    geom = pd.read_csv(TABLES / "transect_geometry.csv")

    n_scar = write_line_ascii(geom, "transect2_longitudinal", "/tmp/_corr_scar.txt")
    n_west = write_line_ascii(geom, "transect3_westcorridor", "/tmp/_corr_west.txt")
    print(f"B-B' (main scar) line: {n_scar} vertices; "
          f"C-C' (western corridor) line: {n_west} vertices")

    gs.run_command("v.in.lines", input="/tmp/_corr_scar.txt", output="corridor_scar_line",
                   separator="comma", overwrite=True, quiet=True)
    gs.run_command("v.in.lines", input="/tmp/_corr_west.txt", output="corridor_west_line",
                   separator="comma", overwrite=True, quiet=True)

    gs.run_command("v.buffer", input="corridor_scar_line", output="corridor_scar_buffer",
                   distance=CORRIDOR_HALFWIDTH_M, overwrite=True, quiet=True)
    gs.run_command("v.buffer", input="corridor_west_line", output="corridor_west_buffer",
                   distance=CORRIDOR_HALFWIDTH_M, overwrite=True, quiet=True)

    gs.run_command("v.to.rast", input="corridor_scar_buffer", output="corridor_scar_mask",
                   use="val", value=1, overwrite=True, quiet=True)
    gs.run_command("v.to.rast", input="corridor_west_buffer", output="corridor_west_mask",
                   use="val", value=1, overwrite=True, quiet=True)

    # overlap check
    gs.mapcalc("_corr_overlap = if(!isnull(corridor_scar_mask) && "
              "!isnull(corridor_west_mask), 1, null())", overwrite=True, quiet=True)
    n_overlap = int(gs.parse_command("r.univar", map="_corr_overlap",
                                     flags="g").get("n", 0))
    print(f"overlap between the two {CORRIDOR_HALFWIDTH_M:.0f} m-halfwidth "
          f"buffers: {n_overlap:,} cells")

    rows = []
    for label, maskmap in (("main scar corridor (B-B')", "corridor_scar_mask"),
                           ("western corridor (C-C')", "corridor_west_mask")):
        gs.mapcalc(f"_corr_dh = if(!isnull({maskmap}) && abs(coreg_dh_corrected) > "
                  f"{LOD}, coreg_dh_corrected, null())", overwrite=True, quiet=True)
        gs.mapcalc("_corr_ero = if(_corr_dh < 0, _corr_dh, null())",
                  overwrite=True, quiet=True)
        gs.mapcalc("_corr_dep = if(_corr_dh > 0, _corr_dh, null())",
                  overwrite=True, quiet=True)
        ero = gs.parse_command("r.univar", map="_corr_ero", flags="ge", quiet=True)
        dep = gs.parse_command("r.univar", map="_corr_dep", flags="ge", quiet=True)
        n_ero = int(ero.get("n", 0))
        n_dep = int(dep.get("n", 0))
        ero_vol = float(ero.get("sum", 0))
        dep_vol = float(dep.get("sum", 0))
        area_above_lod = n_ero + n_dep
        ratio = abs(dep_vol / ero_vol) if ero_vol else float("inf")

        row = {
            "corridor": label,
            "area_above_lod_m2": area_above_lod,
            "erosion_volume_m3": ero_vol,
            "erosion_area_m2": n_ero,
            "deposition_volume_m3": dep_vol,
            "deposition_area_m2": n_dep,
            "net_volume_m3": ero_vol + dep_vol,
            "deposition_to_erosion_ratio": ratio,
        }
        rows.append(row)
        print(f"\n  {label}")
        print(f"    area above LoD : {area_above_lod:,} m2")
        print(f"    erosion        : {ero_vol:+,.0f} m3 ({n_ero:,} m2)")
        print(f"    deposition     : {dep_vol:+,.0f} m3 ({n_dep:,} m2)")
        print(f"    net            : {row['net_volume_m3']:+,.0f} m3")
        print(f"    deposition:erosion ratio : {ratio:.2f}")

    out_csv = TABLES / "watershed_corridor_comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["overlap_cells_m2",
                                                                  "corridor_halfwidth_m"])
        w.writeheader()
        for r in rows:
            r["overlap_cells_m2"] = n_overlap
            r["corridor_halfwidth_m"] = CORRIDOR_HALFWIDTH_M
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    gs.run_command("g.remove", type="raster",
                   name="_corr_overlap,_corr_dh,_corr_ero,_corr_dep",
                   flags="f", quiet=True)

    log("14_watershed_corridor_comparison.py run. Corridors = "
        f"{CORRIDOR_HALFWIDTH_M:.0f} m half-width buffer around each "
        "transect's traced line (B-B' for main scar, C-C' for western "
        f"corridor). Overlap: {n_overlap:,} cells. Results: " +
        "; ".join(f"{r['corridor']}: net={r['net_volume_m3']:+,.0f}m3, "
                 f"dep:ero={r['deposition_to_erosion_ratio']:.2f}"
                 for r in rows) +
        f". Written to {out_csv.name}. Rasters/vectors corridor_scar_line/"
        "buffer/mask and corridor_west_line/buffer/mask kept (documented, "
        "not disposable).", run="watershed_corridor_comparison")


if __name__ == "__main__":
    main()
