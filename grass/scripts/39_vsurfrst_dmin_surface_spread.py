#!/usr/bin/env python3
"""
39_vsurfrst_dmin_surface_spread.py
-----------------------------------
Does the v.surf.rst dmin choice move the surface more than tension does?

RATIONALE
    Script 37 tested tension (10/20/40) and smoothing (0.1/1/5) and flagged
    dmin as the untested lever that is plausibly larger: dmin discards points
    closer than dmin metres to an already-accepted point, so it does not
    merely reweight the fit, it decides how much of the point cloud is used
    at all. At production dmin=2 a 250 m window of 2020 lidar drops from
    ~434,000 points to ~11,600 retained -- a 97% reduction. Halving or
    doubling that distance changes the retained cloud by roughly 4x either
    way.

METHOD
    Identical to script 37 so the two are directly comparable: one epoch
    (2020 lidar ground+road, class_filter="2,11"), one area (cat 34 basin
    bbox), tension=20 smooth=1 npmin=300 segmax=40 held at production values,
    dmin varied over 1 / 2 / 4. Surfaces differenced pairwise; statistics
    over an interior region inset 25 m. Spread is reported against the
    manuscript's 0.32 m level of detection.

    Retained-point counts are recorded per dmin, since that is the mechanism.

OUTPUTS
    results/tables/vsurfrst_dmin_surface_spread.csv
    results/logs/vsurfrst_dmin_<date>.log
    rasters kept in mapset xval_dmin: rst_2020_d{1,2,4}, diff_*
"""

import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grass_env import grass_session, log  # noqa: E402

RUN = "vsurfrst_dmin"
PROJECT = "DEM_generation"
WORK_MAPSET = "xval_dmin"
POINTS = "area_lidar_2020@xval"      # same point set script 37 used

BASE = dict(tension=20, smooth=1.0, npmin=300, segmax=40)
DMINS = [1, 2, 4]                    # production = 2
RES = 1.0
NPROCS = int(os.environ.get("DMIN_NPROCS", "12"))

AREA = dict(w=313170.0, s=192570.0, e=313725.0, n=193690.0)
EDGE_INSET = 25.0
LOD = 0.32

PAIRS = [
    ("dmin 1 vs 2", 1, 2),
    ("dmin 4 vs 2", 4, 2),
    ("dmin 4 vs 1", 4, 1),
]


def interpolate(g, dmin):
    name = f"rst_2020_d{dmin}"
    if g.find_file(name=name, element="cell", mapset=".")["name"]:
        log(f"reusing {name}", run=RUN)
        return name
    g.run_command("g.region", w=AREA["w"], s=AREA["s"], e=AREA["e"],
                  n=AREA["n"], res=RES, flags="a")
    t0 = time.time()
    g.run_command("v.surf.rst", input=POINTS, elevation=name,
                  tension=BASE["tension"], smooth=BASE["smooth"],
                  npmin=BASE["npmin"], segmax=BASE["segmax"], dmin=dmin,
                  nprocs=NPROCS, overwrite=True, quiet=True)
    log(f"{name}: dmin={dmin} tension={BASE['tension']} "
        f"smooth={BASE['smooth']} npmin={BASE['npmin']} "
        f"segmax={BASE['segmax']} res={RES} in {time.time()-t0:.0f}s", run=RUN)
    return name


def retained_points(g, dmin):
    """How many points survive the dmin filter (via a cheap CV-mode run).

    v.surf.rst -c writes one residual point per RETAINED input point, so the
    cvdev feature count is the retained-point count. Run on one 250 m window
    to keep it quick; the ratio between dmin values is what matters.
    """
    tag = f"dminprobe_d{dmin}"
    g.run_command("g.region", w=313200, s=193150, e=313450, n=193400,
                  res=RES, flags="a")
    try:
        g.run_command("v.surf.rst", flags="c", input="pts_lidar_2020_w3_upper@xval",
                      cvdev=tag, tension=BASE["tension"], smooth=BASE["smooth"],
                      npmin=BASE["npmin"], segmax=BASE["segmax"], dmin=dmin,
                      nprocs=NPROCS, overwrite=True, quiet=True)
    except Exception as exc:                              # noqa: BLE001
        log(f"retained-point probe failed for dmin={dmin}: {exc}", run=RUN)
        return None
    n = int(g.parse_command("v.info", map=tag, flags="t")["points"])
    g.run_command("g.remove", type="vector", name=tag, flags="f", quiet=True)
    log(f"dmin={dmin}: {n} points retained in the 250 m w3_upper window", run=RUN)
    return n


def diff_stats(g, label, a, b, retained):
    g.run_command("g.region", w=AREA["w"] + EDGE_INSET, s=AREA["s"] + EDGE_INSET,
                  e=AREA["e"] - EDGE_INSET, n=AREA["n"] - EDGE_INSET,
                  res=RES, flags="a")
    dm = f"diff_d{a}_minus_d{b}"
    g.mapcalc(f"{dm} = rst_2020_d{a} - rst_2020_d{b}", overwrite=True, quiet=True)

    u = g.parse_command("r.univar", map=dm, flags="ge", percentile="5,95")
    median = float(u["median"])

    ad = f"_absdev_d{a}_d{b}"
    g.mapcalc(f"{ad} = abs({dm} - {median})", overwrite=True, quiet=True)
    mad = float(g.parse_command("r.univar", map=ad, flags="ge")["median"])
    g.run_command("g.remove", type="raster", name=ad, flags="f", quiet=True)

    ex = f"_exceed_d{a}_d{b}"
    g.mapcalc(f"{ex} = if(abs({dm}) > {LOD}, 1, 0)", overwrite=True, quiet=True)
    frac = float(g.parse_command("r.univar", map=ex, flags="g")["mean"])
    g.run_command("g.remove", type="raster", name=ex, flags="f", quiet=True)

    am = f"_absd_d{a}_d{b}"
    g.mapcalc(f"{am} = abs({dm})", overwrite=True, quiet=True)
    mean_abs = float(g.parse_command("r.univar", map=am, flags="g")["mean"])
    g.run_command("g.remove", type="raster", name=am, flags="f", quiet=True)

    row = dict(
        comparison=label, map_a=f"rst_2020_d{a}", map_b=f"rst_2020_d{b}",
        diff_map=dm, n_cells=int(u["n"]), mean=float(u["mean"]),
        stddev=float(u["stddev"]), median=median, nmad=1.4826 * mad,
        mean_abs=mean_abs, p5=float(u["percentile_5"]),
        p95=float(u["percentile_95"]), min=float(u["min"]), max=float(u["max"]),
        pct_cells_gt_lod=100.0 * frac, lod_m=LOD,
        points_retained_a=retained.get(a), points_retained_b=retained.get(b),
    )
    log(f"{label}: n={row['n_cells']} mean={row['mean']:+.4f} "
        f"SD={row['stddev']:.4f} NMAD={row['nmad']:.4f} "
        f"mean|d|={mean_abs:.4f} p5={row['p5']:+.4f} p95={row['p95']:+.4f} "
        f"range=[{row['min']:+.3f},{row['max']:+.3f}] "
        f"{row['pct_cells_gt_lod']:.3f}% of cells > {LOD} m", run=RUN)
    return row


def main():
    tables = Path("results/tables")
    gs, _ = grass_session(project=PROJECT, mapset="PERMANENT")
    import grass.script as g
    try:
        g.run_command("g.mapset", flags="c", mapset=WORK_MAPSET, quiet=True)
    except Exception:                                     # noqa: BLE001
        g.run_command("g.mapset", mapset=WORK_MAPSET, quiet=True)
    log(f"START mapset={WORK_MAPSET} points={POINTS} base={BASE} dmins={DMINS} "
        f"res={RES} area={AREA} inset={EDGE_INSET}m LoD={LOD} nprocs={NPROCS}",
        run=RUN)

    retained = {}
    for dmin in DMINS:
        retained[dmin] = retained_points(g, dmin)
    for dmin in DMINS:
        interpolate(g, dmin)

    rows = [diff_stats(g, label, a, b, retained) for label, a, b in PAIRS]

    fields = ["comparison", "map_a", "map_b", "diff_map", "n_cells", "mean",
              "stddev", "median", "nmad", "mean_abs", "p5", "p95", "min", "max",
              "pct_cells_gt_lod", "lod_m", "points_retained_a", "points_retained_b"]
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "vsurfrst_dmin_surface_spread.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {tables/'vsurfrst_dmin_surface_spread.csv'} ({len(rows)} rows)",
          flush=True)

    print(f"\n=== dmin surface spread vs the {LOD:.2f} m level of detection ===",
          flush=True)
    print("  retained points in the 250 m w3_upper window: "
          + ", ".join(f"dmin={d}: {retained[d]:,}" for d in DMINS
                      if retained.get(d)), flush=True)
    for r in rows:
        print(f"  {r['comparison']:<14} mean={r['mean']:+.4f} m  "
              f"SD={r['stddev']:.4f}  NMAD={r['nmad']:.4f}  "
              f"mean|d|={r['mean_abs']:.4f}  "
              f"p5..p95=[{r['p5']:+.3f},{r['p95']:+.3f}]  "
              f"{r['pct_cells_gt_lod']:.3f}% > LoD", flush=True)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
