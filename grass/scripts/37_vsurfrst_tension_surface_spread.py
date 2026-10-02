#!/usr/bin/env python3
"""
37_vsurfrst_tension_surface_spread.py
--------------------------------------
Does the v.surf.rst tension choice actually move the surface?

RATIONALE
    Cross-validation (script 36) measures predictive error at the data points.
    The question that matters for this manuscript is different and simpler:
    if the interpolation tension is changed over the plausible range, how far
    does the resulting DEM move -- and is that movement large or small next to
    the 0.32 m level of detection the manuscript applies?

    A cross-validation RMSE cannot answer that on its own. Differencing the
    surfaces can, directly and in the units the manuscript uses.

METHOD
    One epoch (2020 lidar ground+road, the pre-event baseline), one area (the
    cat 34 basin bbox), interpolated at tension 10 / 20 / 40 with everything
    else at production settings (smooth=1 npmin=300 dmin=2 segmax=40, 1 m).
    Smoothing 0.1 / 5 at tension=20 is also run so the second sweep axis is
    covered against the same yardstick.

    Surfaces are differenced pairwise. Statistics are taken over an interior
    region inset EDGE_INSET m from the interpolated extent, because RST edge
    behaviour differs between tensions and would otherwise dominate the tails.

    Points come from area_lidar_2020@xval -- the sample area read straight out
    of lidar_2020.laz with class_filter="2,11", the production class filter.

OUTPUTS
    results/tables/vsurfrst_tension_surface_spread.csv   pairwise difference stats
    results/logs/vsurfrst_tension_<date>.log
    rasters kept in mapset xval_tension: rst_2020_t{10,20,40}_s1, rst_2020_t20_s{0p1,5}
"""

import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grass_env import grass_session, log  # noqa: E402

RUN = "vsurfrst_tension"
PROJECT = "DEM_generation"
WORK_MAPSET = "xval_tension"      # own mapset: script 36 owns xval's region
POINTS = "area_lidar_2020@xval"   # 2020 lidar ground+road over the cat 34 bbox

BASE = dict(smooth=1.0, npmin=300, dmin=2, segmax=40)
RES = 1.0
NPROCS = int(os.environ.get("TENSION_NPROCS", "4"))

AREA = dict(w=313170.0, s=192570.0, e=313725.0, n=193690.0)
EDGE_INSET = 25.0                 # m trimmed off each side before statistics

LOD = 0.32                        # manuscript level of detection, metres

# (map suffix, tension, smooth)
VARIANTS = [
    ("t10_s1",   10, 1.0),
    ("t20_s1",   20, 1.0),        # production
    ("t40_s1",   40, 1.0),
    ("t20_s0p1", 20, 0.1),
    ("t20_s5",   20, 5.0),
]

# (name, map_a, map_b) -- a minus b
PAIRS = [
    ("tension 10 vs 20", "t10_s1", "t20_s1"),
    ("tension 40 vs 20", "t40_s1", "t20_s1"),
    ("tension 40 vs 10", "t40_s1", "t10_s1"),
    ("smooth 0.1 vs 1",  "t20_s0p1", "t20_s1"),
    ("smooth 5 vs 1",    "t20_s5", "t20_s1"),
]


def interpolate(g, suffix, tension, smooth):
    name = f"rst_2020_{suffix}"
    if g.find_file(name=name, element="cell", mapset=".")["name"]:
        log(f"reusing {name}", run=RUN)
        return name
    g.run_command("g.region", w=AREA["w"], s=AREA["s"], e=AREA["e"],
                  n=AREA["n"], res=RES, flags="a")
    t0 = time.time()
    g.run_command("v.surf.rst", input=POINTS, elevation=name,
                  tension=tension, smooth=smooth, npmin=BASE["npmin"],
                  dmin=BASE["dmin"], segmax=BASE["segmax"],
                  nprocs=NPROCS, overwrite=True, quiet=True)
    log(f"{name}: tension={tension} smooth={smooth} npmin={BASE['npmin']} "
        f"dmin={BASE['dmin']} segmax={BASE['segmax']} res={RES} "
        f"in {time.time()-t0:.0f}s", run=RUN)
    return name


def set_interior(g):
    g.run_command("g.region", w=AREA["w"] + EDGE_INSET, s=AREA["s"] + EDGE_INSET,
                  e=AREA["e"] - EDGE_INSET, n=AREA["n"] - EDGE_INSET,
                  res=RES, flags="a")


def diff_stats(g, label, a, b):
    """Difference two surfaces and describe the spread over the interior."""
    set_interior(g)
    dm = f"diff_{a}_minus_{b}"
    g.run_command("r.mapcalc",
                  expression=f"{dm} = rst_2020_{a} - rst_2020_{b}",
                  overwrite=True, quiet=True)

    u = g.parse_command("r.univar", map=dm, flags="ge", percentile="5,95")
    median = float(u["median"])

    # NMAD (rule 4): 1.4826 * median(|d - median(d)|)
    ad = f"absdev_{a}_minus_{b}"
    g.run_command("r.mapcalc", expression=f"{ad} = abs({dm} - {median})",
                  overwrite=True, quiet=True)
    mad = float(g.parse_command("r.univar", map=ad, flags="ge")["median"])
    g.run_command("g.remove", type="raster", name=ad, flags="f", quiet=True)

    # fraction of cells whose difference exceeds the manuscript LoD
    ex = f"exceed_{a}_minus_{b}"
    g.run_command("r.mapcalc",
                  expression=f"{ex} = if(abs({dm}) > {LOD}, 1, 0)",
                  overwrite=True, quiet=True)
    eu = g.parse_command("r.univar", map=ex, flags="g")
    frac_exceed = float(eu["mean"])
    g.run_command("g.remove", type="raster", name=ex, flags="f", quiet=True)

    n = int(u["n"])
    row = dict(
        comparison=label, map_a=f"rst_2020_{a}", map_b=f"rst_2020_{b}",
        diff_map=dm, n_cells=n,
        mean=float(u["mean"]), stddev=float(u["stddev"]),
        median=median, nmad=1.4826 * mad,
        min=float(u["min"]), max=float(u["max"]),
        p5=float(u["percentile_5"]), p95=float(u["percentile_95"]),
        mean_abs=None, pct_cells_gt_lod=100.0 * frac_exceed, lod_m=LOD,
    )
    # mean |difference|
    am = f"absd_{a}_minus_{b}"
    g.run_command("r.mapcalc", expression=f"{am} = abs({dm})",
                  overwrite=True, quiet=True)
    row["mean_abs"] = float(g.parse_command("r.univar", map=am, flags="g")["mean"])
    g.run_command("g.remove", type="raster", name=am, flags="f", quiet=True)

    log(f"{label}: n={n} mean={row['mean']:+.4f} SD={row['stddev']:.4f} "
        f"NMAD={row['nmad']:.4f} mean|d|={row['mean_abs']:.4f} "
        f"p5={row['p5']:+.4f} p95={row['p95']:+.4f} "
        f"range=[{row['min']:+.3f},{row['max']:+.3f}] "
        f"{row['pct_cells_gt_lod']:.3f}% of cells > {LOD} m", run=RUN)
    return row


def main():
    tables = Path("results/tables")
    gs, _ = grass_session(project=PROJECT, mapset="PERMANENT")
    import grass.script as g
    try:
        g.run_command("g.mapset", flags="c", mapset=WORK_MAPSET, quiet=True)
    except Exception:                                  # noqa: BLE001
        g.run_command("g.mapset", mapset=WORK_MAPSET, quiet=True)
    log(f"START mapset={WORK_MAPSET} points={POINTS} base={BASE} res={RES} "
        f"area={AREA} inset={EDGE_INSET}m nprocs={NPROCS} LoD={LOD}", run=RUN)

    for suffix, tension, smooth in VARIANTS:
        interpolate(g, suffix, tension, smooth)

    rows = [diff_stats(g, label, a, b) for label, a, b in PAIRS]

    fields = ["comparison", "map_a", "map_b", "diff_map", "n_cells", "mean",
              "stddev", "median", "nmad", "mean_abs", "p5", "p95", "min",
              "max", "pct_cells_gt_lod", "lod_m"]
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "vsurfrst_tension_surface_spread.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {tables/'vsurfrst_tension_surface_spread.csv'} "
          f"({len(rows)} rows)", flush=True)

    print("\n=== surface spread vs the %.2f m level of detection ===" % LOD,
          flush=True)
    for r in rows:
        print(f"  {r['comparison']:<18} mean={r['mean']:+.4f} m  "
              f"SD={r['stddev']:.4f}  NMAD={r['nmad']:.4f}  "
              f"mean|d|={r['mean_abs']:.4f}  "
              f"p5..p95=[{r['p5']:+.3f},{r['p95']:+.3f}]  "
              f"{r['pct_cells_gt_lod']:.3f}% > LoD", flush=True)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
