#!/usr/bin/env python3
"""
40_grid_alignment_diagnostic.py
--------------------------------
Why the tension-20 rebuild (script 38) does not reproduce the published
watershed DoD -- and what that implies for the published volumes.

WHAT WAS FOUND
    The production DTMs are not on a 1 m grid, and not on the same grid as
    each other:

        dtm_2020_filled   nsres=0.99989716802504  ewres=1.00002012153101
        dtm_2024_filled   nsres=0.99988170605126  ewres=1.00007707042887

    This is the signature of `g.region vector=<points> res=1` without -a
    (as in raster_creation_for_codem.ipynb): GRASS fits a near-1 m resolution
    to each epoch's own point bounding box. Because the two epochs have
    different extents, they get different resolutions and different grid
    origins. Across the full extent the two grids drift apart by ~0.4-0.5 m,
    a spatially varying sub-cell misregistration, and differencing them
    resamples nearest-neighbour.

TEST A -- is grid alignment really the cause?
    Rebuild the 2020 surface at production tension=20 over the basin area
    but on the PRODUCTION grid (g.region align=dtm_2020_filled), and compare
    against dtm_2020_filled. Script 38's rebuild on an exact 1 m grid
    differed by SD 0.239 m. If the aligned rebuild collapses to near zero,
    the grid is confirmed as the cause and the rebuild method is sound.

TEST B -- how much does the misalignment move the published volumes?
    Within production's own data, build the raw 2024-2020 DoD two ways over
    the production region:
      nn  : straight r.mapcalc difference (nearest-neighbour resampling,
            what the pipeline does)
      bil : each epoch first resampled onto the common grid with
            r.resamp.interp method=bilinear, then differenced
    Difference the two DoDs and recompute erosion / deposition / net /
    % above LoD from each. This isolates the resampling artifact alone --
    same surfaces, same tension, same everything else.

OUTPUTS
    results/tables/grid_alignment_diagnostic.csv
    results/logs/grid_align_<date>.log
    rasters kept in mapset xval_dod
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grass_env import grass_session, log  # noqa: E402

RUN = "grid_align"
PROJECT = "DEM_generation"
WORK_MAPSET = "xval_dod"

REF = "dtm_2020_filled@for_codem"
TGT = "dtm_2024_filled@for_codem"
CORR = "dod_total_corr"          # built by script 38, held fixed
LOD = 0.32
BASIN = dict(w=313173.0, s=192579.0, e=313722.0, n=193688.0)
AREA = dict(w=313023.0, s=192429.0, e=313872.0, n=193838.0)
NPROCS = int(os.environ.get("DOD_NPROCS", "12"))
BASE = dict(tension=20, smooth=1.0, npmin=300, dmin=2, segmax=40)
FILLNULLS = dict(method="rst", tension=10, smooth=1, edge=15, npmin=100,
                 segmax=80, memory=5000)


def stats(g, m, label, percentile=None):
    kw = dict(map=m, flags="ge")
    if percentile:
        kw["percentile"] = percentile
    u = g.parse_command("r.univar", **kw)
    med = float(u["median"])
    ad = "_ad_tmp"
    g.mapcalc(f"{ad} = abs({m} - {med})", overwrite=True, quiet=True)
    mad = float(g.parse_command("r.univar", map=ad, flags="ge")["median"])
    g.run_command("g.remove", type="raster", name=ad, flags="f", quiet=True)
    d = dict(label=label, n=int(u["n"]), mean=float(u["mean"]),
             stddev=float(u["stddev"]), median=med, nmad=1.4826 * mad,
             min=float(u["min"]), max=float(u["max"]))
    log(f"{label}: n={d['n']} mean={d['mean']:+.4f} median={d['median']:+.4f} "
        f"SD={d['stddev']:.4f} NMAD={d['nmad']:.4f} "
        f"range=[{d['min']:+.3f},{d['max']:+.3f}]", run=RUN)
    return d


def volumes(g, dh, label):
    g.run_command("g.region", w=BASIN["w"], s=BASIN["s"], e=BASIN["e"],
                  n=BASIN["n"], res=1, flags="a")
    g.mapcalc(f"_t = if(abs({dh}) > {LOD}, {dh}, null())", overwrite=True, quiet=True)
    g.mapcalc("_e = if(_t < 0, _t, null())", overwrite=True, quiet=True)
    g.mapcalc("_d = if(_t > 0, _t, null())", overwrite=True, quiet=True)
    tot = g.parse_command("r.univar", map=dh, flags="g")
    e = g.parse_command("r.univar", map="_e", flags="g")
    d = g.parse_command("r.univar", map="_d", flags="g")
    n_valid, n_e, n_d = int(tot["n"]), int(e["n"]), int(d["n"])
    ero, dep = float(e["sum"]), float(d["sum"])
    for m in ("_t", "_e", "_d"):
        g.run_command("g.remove", type="raster", name=m, flags="f", quiet=True)
    row = dict(variant=label, dh_map=dh, n_valid_cells=n_valid,
               erosion_m3=ero, deposition_m3=dep, net_m3=ero + dep,
               erosion_area_m2=n_e, deposition_area_m2=n_d,
               pct_above_lod=100.0 * (n_e + n_d) / n_valid, lod_m=LOD)
    log(f"{label}: erosion={ero:+.0f} deposition={dep:+.0f} net={ero+dep:+.0f} m3 "
        f"| {row['pct_above_lod']:.2f}% of {n_valid} cells > {LOD} m", run=RUN)
    return row


def main():
    tables = Path("results/tables")
    gs, _ = grass_session(project=PROJECT, mapset="PERMANENT")
    import grass.script as g
    g.run_command("g.mapset", mapset=WORK_MAPSET, quiet=True)
    log(f"START mapset={WORK_MAPSET} ref={REF} tgt={TGT} LoD={LOD}", run=RUN)

    diag, vols = [], []

    # ---------------- TEST A: rebuild on the production grid --------------
    g.run_command("g.region", w=AREA["w"], s=AREA["s"], e=AREA["e"],
                  n=AREA["n"], align=REF)
    reg = g.parse_command("g.region", flags="g")
    log(f"TEST A region aligned to {REF}: nsres={reg['nsres']} "
        f"ewres={reg['ewres']} rows={reg['rows']} cols={reg['cols']}", run=RUN)

    if not g.find_file(name="dod_2020_t20_aligned", element="cell",
                       mapset=".")["name"]:
        g.run_command("v.surf.rst", input="dodpts_2020",
                      elevation="_dod_2020_t20_aligned_raw",
                      tension=BASE["tension"], smooth=BASE["smooth"],
                      npmin=BASE["npmin"], dmin=BASE["dmin"],
                      segmax=BASE["segmax"], nprocs=NPROCS,
                      overwrite=True, quiet=True)
        g.run_command("r.fillnulls", input="_dod_2020_t20_aligned_raw",
                      output="dod_2020_t20_aligned", overwrite=True,
                      quiet=True, **FILLNULLS)
        g.run_command("g.remove", type="raster",
                      name="_dod_2020_t20_aligned_raw", flags="f", quiet=True)
        log("built dod_2020_t20_aligned on the production grid", run=RUN)

    # compare both rebuilds against the production 2020 surface
    g.run_command("g.region", w=BASIN["w"], s=BASIN["s"], e=BASIN["e"],
                  n=BASIN["n"], align=REF)
    g.mapcalc(f"_resid_aligned = dod_2020_t20_aligned - {REF}",
              overwrite=True, quiet=True)
    diag.append(stats(g, "_resid_aligned",
                      "TEST A: rebuild ON production grid - dtm_2020_filled"))
    g.run_command("g.region", w=BASIN["w"], s=BASIN["s"], e=BASIN["e"],
                  n=BASIN["n"], res=1, flags="a")
    g.mapcalc(f"_resid_1m = dod_2020_t20 - {REF}", overwrite=True, quiet=True)
    diag.append(stats(g, "_resid_1m",
                      "TEST A: rebuild on exact 1 m grid - dtm_2020_filled"))

    # ---------------- TEST B: resampling artifact in production data ------
    g.run_command("g.region", w=BASIN["w"], s=BASIN["s"], e=BASIN["e"],
                  n=BASIN["n"], res=1, flags="a")
    # nearest-neighbour (what the pipeline does)
    g.mapcalc(f"dod_raw_nn = {TGT} - {REF}", overwrite=True, quiet=True)
    # bilinear: resample each epoch onto the common grid first
    for src, dst in ((REF, "_ref_bil"), (TGT, "_tgt_bil")):
        g.run_command("r.resamp.interp", input=src, output=dst,
                      method="bilinear", overwrite=True, quiet=True)
    g.mapcalc("dod_raw_bil = _tgt_bil - _ref_bil", overwrite=True, quiet=True)
    g.mapcalc("dod_resamp_artifact = dod_raw_nn - dod_raw_bil",
              overwrite=True, quiet=True)
    diag.append(stats(g, "dod_resamp_artifact",
                      "TEST B: nearest-neighbour DoD - bilinear DoD"))

    # volumes from each, with production's fixed correction applied
    for base, label in (("dod_raw_nn", "production nearest-neighbour resampling"),
                        ("dod_raw_bil", "bilinear resampling onto common grid")):
        dh = f"dh_{base}"
        g.mapcalc(f"{dh} = {base} + {CORR}", overwrite=True, quiet=True)
        vols.append(volumes(g, dh, label))
    for m in ("_ref_bil", "_tgt_bil", "_resid_aligned", "_resid_1m"):
        g.run_command("g.remove", type="raster", name=m, flags="f", quiet=True)

    # ---------------- write -----------------------------------------------
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "grid_alignment_diagnostic.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "label", "n", "mean_m", "stddev_m", "median_m",
                    "nmad_m", "min_m", "max_m"])
        for d in diag:
            w.writerow(["difference_stats", d["label"], d["n"],
                        f"{d['mean']:.5f}", f"{d['stddev']:.5f}",
                        f"{d['median']:.5f}", f"{d['nmad']:.5f}",
                        f"{d['min']:.4f}", f"{d['max']:.4f}"])
        w.writerow([])
        w.writerow(["section", "variant", "n_valid_cells", "erosion_m3",
                    "deposition_m3", "net_m3", "erosion_area_m2",
                    "deposition_area_m2", "pct_above_lod", "lod_m"])
        for v in vols:
            w.writerow(["volumes", v["variant"], v["n_valid_cells"],
                        f"{v['erosion_m3']:.1f}", f"{v['deposition_m3']:.1f}",
                        f"{v['net_m3']:.1f}", v["erosion_area_m2"],
                        v["deposition_area_m2"],
                        f"{v['pct_above_lod']:.3f}", v["lod_m"]])
    print(f"\nwrote {tables/'grid_alignment_diagnostic.csv'}", flush=True)

    print("\n=== resampling artifact: same surfaces, only the regridding "
          "differs ===", flush=True)
    for v in vols:
        print(f"  {v['variant']:<42} erosion {v['erosion_m3']:>12,.0f}  "
              f"deposition {v['deposition_m3']:>12,.0f}  "
              f"net {v['net_m3']:>12,.0f}  {v['pct_above_lod']:.2f}% > LoD",
              flush=True)
    if len(vols) == 2:
        a, b = vols
        for k in ("erosion_m3", "deposition_m3", "net_m3", "pct_above_lod"):
            d = a[k] - b[k]
            pct = 100.0 * d / abs(b[k]) if b[k] else float("nan")
            print(f"  delta {k}: {d:+,.1f} ({pct:+.1f}% of the bilinear value)",
                  flush=True)
            log(f"DELTA {k}: {d:+.2f} ({pct:+.1f}%)", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
