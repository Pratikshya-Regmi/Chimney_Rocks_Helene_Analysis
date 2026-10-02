#!/usr/bin/env python3
"""
38_dod_volume_tension_sensitivity.py
-------------------------------------
Are the reported watershed volumes robust to the RST interpolation tension?

BACKGROUND
    Script 37 showed that changing v.surf.rst tension over 10-40 moves
    15.7-22.8% of cells across the 0.32 m LoD, but with a mean difference of
    ~0.0000 m -- zero-mean and spatially varying. That predicts per-cell area
    statistics should move a lot while volumes, which sum a zero-mean
    perturbation over a large area, should move much less. This script tests
    that prediction on the actual watershed numbers instead of assuming it.

METHOD -- tension is the ONLY thing that varies
    Production chain (verified reproduced, see VALIDATION below):
        coreg_dh_corrected = coreg_target_corrected - dtm_2020_filled
    where coreg_target_corrected is the 2024 surface after the pipeline's
    registration, Nuth & Kaab shift and canopy bias model.

    Production's total correction to the 2024 surface is therefore the field
        TOTAL_CORR = coreg_target_corrected@for_codem - dtm_2024_filled@for_codem
    which is recovered once and reapplied unchanged at every tension. The
    co-registration is thus held EXACTLY at production settings -- only the
    interpolation of the two epochs changes:

        dh_t = (z2024_t + TOTAL_CORR) - z2020_t

    Both epochs are rebuilt from the original point clouds
    (class_filter="2,11", the production filter) at tension 10 / 20 / 40 with
    smooth=1 npmin=300 dmin=2 segmax=40, then r.fillnulls exactly as
    production did (method=rst tension=10 smooth=1 edge=15 npmin=100
    segmax=80).

    Interpolation runs over the basin bbox buffered by BUFFER m so that RST
    edge behaviour -- which is itself tension-dependent -- stays well outside
    the area the volumes are computed on.

    Volumes are computed over the production region: g.region
    vector=basins_90_v_cat34 res=1 -a (the basin BOUNDING BOX, 608,841 cells;
    production applies no polygon mask, and this script matches that).

VALIDATION
    The t=20 rebuild is compared against the published coreg_dh_corrected
    numbers (erosion -87,636 m3, deposition +55,033 m3, net -32,603 m3). If
    the rebuild does not reproduce those, the tension comparison is still
    internally valid but the absolute baseline is not, and the script says so.

OUTPUTS
    results/tables/dod_volume_tension_sensitivity.csv
    results/logs/dod_tension_<date>.log
    rasters kept in mapset xval_dod
"""

import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grass_env import grass_session, log  # noqa: E402

RUN = "dod_tension"
PROJECT = "DEM_generation"
WORK_MAPSET = "xval_dod"

LAS_DIR = Path("/home/pregmi3/CODEM/demo/All_elevation")
SRC = {
    "2020": LAS_DIR / "lidar_2020.laz",
    "2024": LAS_DIR / "NEWAREA.las",
}
CLASS_FILTER = "2,11"

# production interpolation settings; only tension varies
BASE = dict(smooth=1.0, npmin=300, dmin=2, segmax=40)
FILLNULLS = dict(method="rst", tension=10, smooth=1, edge=15, npmin=100,
                 segmax=80, memory=5000)
TENSIONS = [10, 20, 40]
RES = 1.0
NPROCS = int(os.environ.get("DOD_NPROCS", "12"))
LOD = 0.32

# production volume region: bbox of basins_90_v_cat34
BASIN = dict(w=313173.0, s=192579.0, e=313722.0, n=193688.0)
BUFFER = 150.0          # interpolation margin around the basin bbox
AREA = dict(w=BASIN["w"] - BUFFER, s=BASIN["s"] - BUFFER,
            e=BASIN["e"] + BUFFER, n=BASIN["n"] + BUFFER)

REF_PROD = "dtm_2020_filled@for_codem"
TGT_PROD = "dtm_2024_filled@for_codem"
TGT_CORR = "coreg_target_corrected@for_codem"
DH_PROD = "coreg_dh_corrected@for_codem"
STABLE = "stable_mask@for_codem"

PUBLISHED = dict(erosion=-87635.5069568134, deposition=55032.9383437886)


def set_area(g):
    g.run_command("g.region", w=AREA["w"], s=AREA["s"], e=AREA["e"],
                  n=AREA["n"], res=RES, flags="a")


def set_basin(g):
    g.run_command("g.region", w=BASIN["w"], s=BASIN["s"], e=BASIN["e"],
                  n=BASIN["n"], res=RES, flags="a")


def import_points(g, epoch):
    name = f"dodpts_{epoch}"
    if g.find_file(name=name, element="vector", mapset=".")["name"]:
        log(f"reusing {name}", run=RUN)
        return name
    t0 = time.time()
    g.run_command("v.in.pdal", input=str(SRC[epoch]), output=name,
                  class_filter=CLASS_FILTER, flags="w",
                  spatial=f"{AREA['w']},{AREA['s']},{AREA['e']},{AREA['n']}",
                  overwrite=True, quiet=True)
    n = int(g.parse_command("v.info", map=name, flags="t")["points"])
    log(f"import {name}: {n} pts from {SRC[epoch].name} in {time.time()-t0:.0f}s",
        run=RUN)
    return name


def build_surface(g, epoch, tension):
    """v.surf.rst at the given tension, then r.fillnulls, as production did."""
    raw = f"dod_{epoch}_t{tension}_raw"
    filled = f"dod_{epoch}_t{tension}"
    if g.find_file(name=filled, element="cell", mapset=".")["name"]:
        log(f"reusing {filled}", run=RUN)
        return filled
    set_area(g)
    t0 = time.time()
    g.run_command("v.surf.rst", input=f"dodpts_{epoch}", elevation=raw,
                  tension=tension, smooth=BASE["smooth"], npmin=BASE["npmin"],
                  dmin=BASE["dmin"], segmax=BASE["segmax"],
                  nprocs=NPROCS, overwrite=True, quiet=True)
    g.run_command("r.fillnulls", input=raw, output=filled,
                  overwrite=True, quiet=True, **FILLNULLS)
    log(f"{filled}: tension={tension} + r.fillnulls in {time.time()-t0:.0f}s",
        run=RUN)
    return filled


def volumes(g, dh, label, tension):
    """Threshold at the LoD and sum, over the production basin-bbox region."""
    set_basin(g)
    thr, ero, dep = f"_thr_{tension}", f"_ero_{tension}", f"_dep_{tension}"
    g.mapcalc(f"{thr} = if(abs({dh}) > {LOD}, {dh}, null())",
              overwrite=True, quiet=True)
    g.mapcalc(f"{ero} = if({thr} < 0, {thr}, null())", overwrite=True, quiet=True)
    g.mapcalc(f"{dep} = if({thr} > 0, {thr}, null())", overwrite=True, quiet=True)

    tot = g.parse_command("r.univar", map=dh, flags="g")
    e = g.parse_command("r.univar", map=ero, flags="g")
    d = g.parse_command("r.univar", map=dep, flags="g")

    n_valid = int(tot["n"])
    n_e, n_d = int(e["n"]), int(d["n"])
    ero_v, dep_v = float(e["sum"]), float(d["sum"])

    # stable-terrain median: does the tension shift the surface globally?
    st = f"_stable_{tension}"
    g.mapcalc(f"{st} = if(!isnull({STABLE}), {dh}, null())",
              overwrite=True, quiet=True)
    su = g.parse_command("r.univar", map=st, flags="ge")
    stable_med = float(su["median"])
    stable_n = int(su["n"])

    for m in (thr, ero, dep, st):
        g.run_command("g.remove", type="raster", name=m, flags="f", quiet=True)

    row = dict(
        label=label, tension=tension, dh_map=dh, n_valid_cells=n_valid,
        erosion_m3=ero_v, deposition_m3=dep_v, net_m3=ero_v + dep_v,
        erosion_area_m2=n_e, deposition_area_m2=n_d,
        pct_above_lod=100.0 * (n_e + n_d) / n_valid,
        pct_erosion=100.0 * n_e / n_valid, pct_deposition=100.0 * n_d / n_valid,
        mean_dh_m=float(tot["mean"]),
        stable_median_dh_m=stable_med, stable_n_cells=stable_n, lod_m=LOD,
    )
    log(f"{label}: erosion={ero_v:+.0f} m3 deposition={dep_v:+.0f} m3 "
        f"net={row['net_m3']:+.0f} m3 | {row['pct_above_lod']:.2f}% of "
        f"{n_valid} cells > {LOD} m | stable median={stable_med:+.4f} m",
        run=RUN)
    return row


def main():
    tables = Path("results/tables")
    gs, _ = grass_session(project=PROJECT, mapset="PERMANENT")
    import grass.script as g
    try:
        g.run_command("g.mapset", flags="c", mapset=WORK_MAPSET, quiet=True)
    except Exception:                                   # noqa: BLE001
        g.run_command("g.mapset", mapset=WORK_MAPSET, quiet=True)
    log(f"START mapset={WORK_MAPSET} tensions={TENSIONS} base={BASE} "
        f"fillnulls={FILLNULLS} area={AREA} basin={BASIN} LoD={LOD} "
        f"nprocs={NPROCS}", run=RUN)

    rows = []

    # --- 0. published baseline, recomputed from the production DoD ---------
    rows.append(volumes(g, DH_PROD, "production coreg_dh_corrected (published)",
                        "published"))

    # --- 1. production's own total correction to the 2024 surface ---------
    set_area(g)
    if not g.find_file(name="dod_total_corr", element="cell", mapset=".")["name"]:
        g.mapcalc(f"dod_total_corr = {TGT_CORR} - {TGT_PROD}",
                  overwrite=True, quiet=True)
    cu = g.parse_command("r.univar", map="dod_total_corr", flags="ge")
    log(f"dod_total_corr = {TGT_CORR} - {TGT_PROD}: n={cu['n']} "
        f"mean={float(cu['mean']):+.4f} median={float(cu['median']):+.4f} "
        f"SD={float(cu['stddev']):.4f} m (held fixed across tensions)", run=RUN)

    # --- 2. rebuild both epochs at each tension ---------------------------
    for epoch in ("2020", "2024"):
        import_points(g, epoch)
    for tension in TENSIONS:
        for epoch in ("2020", "2024"):
            build_surface(g, epoch, tension)

    # --- 3. DoD and volumes per tension -----------------------------------
    for tension in TENSIONS:
        dh = f"dod_dh_t{tension}"
        set_area(g)
        g.mapcalc(f"{dh} = (dod_2024_t{tension} + dod_total_corr) "
                  f"- dod_2020_t{tension}", overwrite=True, quiet=True)
        rows.append(volumes(g, dh, f"rebuild tension={tension}", tension))

    # --- 4. validation: does the t=20 rebuild reproduce production? -------
    set_basin(g)
    g.mapcalc(f"dod_resid_t20 = dod_dh_t20 - {DH_PROD}", overwrite=True, quiet=True)
    ru = g.parse_command("r.univar", map="dod_resid_t20", flags="ge")
    log(f"VALIDATION dod_dh_t20 - coreg_dh_corrected: n={ru['n']} "
        f"mean={float(ru['mean']):+.5f} median={float(ru['median']):+.5f} "
        f"SD={float(ru['stddev']):.5f} m", run=RUN)

    t20 = next(r for r in rows if r["tension"] == 20)
    pub = next(r for r in rows if r["tension"] == "published")
    log(f"VALIDATION volumes t=20 rebuild vs published: "
        f"erosion {t20['erosion_m3']:+.0f} vs {pub['erosion_m3']:+.0f} "
        f"({100*(t20['erosion_m3']-pub['erosion_m3'])/abs(pub['erosion_m3']):+.1f}%), "
        f"deposition {t20['deposition_m3']:+.0f} vs {pub['deposition_m3']:+.0f} "
        f"({100*(t20['deposition_m3']-pub['deposition_m3'])/abs(pub['deposition_m3']):+.1f}%)",
        run=RUN)

    # --- 5. write table ----------------------------------------------------
    fields = ["label", "tension", "dh_map", "n_valid_cells", "erosion_m3",
              "deposition_m3", "net_m3", "erosion_area_m2", "deposition_area_m2",
              "pct_above_lod", "pct_erosion", "pct_deposition", "mean_dh_m",
              "stable_median_dh_m", "stable_n_cells", "lod_m"]
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "dod_volume_tension_sensitivity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {tables/'dod_volume_tension_sensitivity.csv'}", flush=True)

    # --- 6. report ---------------------------------------------------------
    print("\n=== watershed volumes vs interpolation tension "
          "(cat34 bbox, 0.32 m LoD) ===", flush=True)
    hdr = (f"  {'variant':<38} {'erosion':>12} {'deposition':>12} "
           f"{'net':>12} {'% > LoD':>9}")
    print(hdr, flush=True)
    for r in rows:
        print(f"  {r['label']:<38} {r['erosion_m3']:>12,.0f} "
              f"{r['deposition_m3']:>12,.0f} {r['net_m3']:>12,.0f} "
              f"{r['pct_above_lod']:>8.2f}%", flush=True)

    reb = [r for r in rows if r["tension"] != "published"]
    for key in ("erosion_m3", "deposition_m3", "net_m3", "pct_above_lod"):
        vals = [r[key] for r in reb]
        base = next(r[key] for r in reb if r["tension"] == 20)
        spread = max(vals) - min(vals)
        pct = 100.0 * spread / abs(base) if base else float("nan")
        print(f"  spread across tension 10-40, {key}: {spread:,.1f} "
              f"({pct:.1f}% of the tension-20 value)", flush=True)
        log(f"SPREAD {key}: {spread:.2f} = {pct:.1f}% of tension-20 value",
            run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
