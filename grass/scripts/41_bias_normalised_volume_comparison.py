#!/usr/bin/env python3
"""
41_bias_normalised_volume_comparison.py
----------------------------------------
Apples-to-apples watershed volumes: every DoD variant bias-corrected the same
way before the volumes are compared.

WHY THIS SCRIPT EXISTS
    Script 38 rebuilt both epochs on a common exact 1 m grid and got a net
    volume of +31,320 m3 against the published -32,603 m3 -- a sign flip.
    Script 40 showed the production epochs sit on mutually misaligned grids
    and that bilinear rather than nearest-neighbour regridding also flips the
    sign (+37,364 m3).

    But the script-38 rebuilds have a stable-terrain median of +0.199 m,
    while the production DoD sits at +0.013 m. An uncorrected ~0.19 m offset
    inflates deposition and suppresses erosion by itself, so part or all of
    that sign flip may be a missing bias correction rather than the grid.

    The production pipeline's stage 2 removes exactly this: it zeroes the DoD
    over stable terrain. This script applies that same correction to EVERY
    variant -- b = median(dh over stable_mask), dh_norm = dh - b -- and only
    then compares volumes. Any difference that survives is a real difference
    between the variants, not a bias artifact.

    Sign convention (CLAUDE.md rule 6): DoD = z_post - z_pre, positive =
    deposition, negative = erosion.

VARIANTS
    production      coreg_dh_corrected@for_codem (as published)
    bilinear        production surfaces, bilinear onto the common grid
    rebuild t10/20/40   both epochs re-interpolated onto one exact 1 m grid

OUTPUTS
    results/tables/bias_normalised_volume_comparison.csv
    results/logs/bias_norm_<date>.log
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grass_env import grass_session, log  # noqa: E402

RUN = "bias_norm"
PROJECT = "DEM_generation"
WORK_MAPSET = "xval_dod"

STABLE = "stable_mask@for_codem"
LOD = 0.32
BASIN = dict(w=313173.0, s=192579.0, e=313722.0, n=193688.0)

VARIANTS = [
    ("production coreg_dh_corrected (published)", "coreg_dh_corrected@for_codem"),
    ("production surfaces, bilinear regrid",      "dh_dod_raw_bil"),
    ("rebuild on common 1 m grid, tension=10",    "dod_dh_t10"),
    ("rebuild on common 1 m grid, tension=20",    "dod_dh_t20"),
    ("rebuild on common 1 m grid, tension=40",    "dod_dh_t40"),
]


def analyse(g, label, dh):
    g.run_command("g.region", w=BASIN["w"], s=BASIN["s"], e=BASIN["e"],
                  n=BASIN["n"], res=1, flags="a")

    # --- stage-2 style bias correction: zero the stable terrain -----------
    g.mapcalc(f"_st = if(!isnull({STABLE}), {dh}, null())",
              overwrite=True, quiet=True)
    su = g.parse_command("r.univar", map="_st", flags="ge")
    bias = float(su["median"])
    stable_n = int(su["n"])
    stable_nmad_src = float(su["stddev"])

    norm = f"norm_{dh.split('@')[0]}"
    g.mapcalc(f"{norm} = {dh} - {bias}", overwrite=True, quiet=True)

    # --- volumes at the LoD ----------------------------------------------
    g.mapcalc(f"_t = if(abs({norm}) > {LOD}, {norm}, null())",
              overwrite=True, quiet=True)
    g.mapcalc("_e = if(_t < 0, _t, null())", overwrite=True, quiet=True)
    g.mapcalc("_d = if(_t > 0, _t, null())", overwrite=True, quiet=True)
    tot = g.parse_command("r.univar", map=norm, flags="g")
    e = g.parse_command("r.univar", map="_e", flags="g")
    d = g.parse_command("r.univar", map="_d", flags="g")

    n_valid, n_e, n_d = int(tot["n"]), int(e["n"]), int(d["n"])
    ero, dep = float(e["sum"]), float(d["sum"])
    for m in ("_st", "_t", "_e", "_d"):
        g.run_command("g.remove", type="raster", name=m, flags="f", quiet=True)

    row = dict(variant=label, dh_map=dh, normalised_map=norm,
               stable_bias_removed_m=bias, stable_n_cells=stable_n,
               stable_sd_m=stable_nmad_src, n_valid_cells=n_valid,
               erosion_m3=ero, deposition_m3=dep, net_m3=ero + dep,
               erosion_area_m2=n_e, deposition_area_m2=n_d,
               pct_above_lod=100.0 * (n_e + n_d) / n_valid,
               pct_erosion=100.0 * n_e / n_valid,
               pct_deposition=100.0 * n_d / n_valid, lod_m=LOD)
    log(f"{label}: bias removed={bias:+.4f} m (stable n={stable_n}) -> "
        f"erosion={ero:+.0f} deposition={dep:+.0f} net={ero+dep:+.0f} m3 | "
        f"{row['pct_above_lod']:.2f}% of {n_valid} cells > {LOD} m", run=RUN)
    return row


def main():
    tables = Path("results/tables")
    gs, _ = grass_session(project=PROJECT, mapset="PERMANENT")
    import grass.script as g
    g.run_command("g.mapset", mapset=WORK_MAPSET, quiet=True)
    log(f"START mapset={WORK_MAPSET} stable={STABLE} LoD={LOD} basin={BASIN}",
        run=RUN)

    rows = []
    for label, dh in VARIANTS:
        if not g.find_file(name=dh.split("@")[0],
                           element="cell",
                           mapset=dh.split("@")[1] if "@" in dh else ".")["name"]:
            log(f"SKIP {label}: {dh} not found", run=RUN)
            continue
        rows.append(analyse(g, label, dh))

    fields = ["variant", "dh_map", "normalised_map", "stable_bias_removed_m",
              "stable_n_cells", "stable_sd_m", "n_valid_cells", "erosion_m3",
              "deposition_m3", "net_m3", "erosion_area_m2",
              "deposition_area_m2", "pct_above_lod", "pct_erosion",
              "pct_deposition", "lod_m"]
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "bias_normalised_volume_comparison.csv", "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {tables/'bias_normalised_volume_comparison.csv'}", flush=True)

    print("\n=== volumes AFTER identical stable-terrain bias correction ===",
          flush=True)
    print(f"  {'variant':<44}{'bias':>8}{'erosion':>12}{'deposition':>12}"
          f"{'net':>12}{'%>LoD':>8}", flush=True)
    for r in rows:
        print(f"  {r['variant']:<44}{r['stable_bias_removed_m']:>+8.3f}"
              f"{r['erosion_m3']:>12,.0f}{r['deposition_m3']:>12,.0f}"
              f"{r['net_m3']:>12,.0f}{r['pct_above_lod']:>7.2f}%", flush=True)

    reb = [r for r in rows if "rebuild" in r["variant"]]
    if reb:
        for k in ("erosion_m3", "deposition_m3", "net_m3", "pct_above_lod"):
            vals = [r[k] for r in reb]
            base = next(r[k] for r in reb if "tension=20" in r["variant"])
            spread = max(vals) - min(vals)
            print(f"  tension 10-40 spread, {k}: {spread:,.1f} "
                  f"({100*spread/abs(base):.1f}% of tension-20)", flush=True)
            log(f"TENSION SPREAD {k}: {spread:.1f} "
                f"({100*spread/abs(base):.1f}%)", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
