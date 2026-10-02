#!/usr/bin/env python3
"""
36_vsurfrst_cross_validation.py
--------------------------------
Leave-one-out cross-validation (v.surf.rst -c) of the RST interpolation used
to build the project's DTMs, plus a sensitivity sweep around the parameters
the manuscript reports.

WHAT IS BEING VALIDATED
    Every production surface was interpolated in
    `raster_creation_for_codem.ipynb` (project DEM_generation, mapset
    for_codem) with

        v.surf.rst tension=20 smooth=1 npmin=300 dmin=2 segmax=40   (res 1 m)

    from three point sets, each imported with
    `v.in.pdal class_filter="2,11"` (ground + road):

        dtm_2020_points      <- lidar_2020.laz       2020 lidar   64,449,900 pts
        dtm_2024_points      <- NEWAREA.las          2024 lidar   59,197,045 pts
        cap_dtm_2024_points  <- sfm_2024.copc.laz    2024 CAP SfM  7,737,384 pts

WHY A SPATIAL SUBSET
    v.surf.rst -c is leave-one-out: one approximation per retained data point.
    The GRASS manual states CV is "usually reasonable for up to several
    thousands of points. For larger data sets, CV should be applied to a
    representative subset of the data."  CV over 59-64 million points is not
    computable.

    The subset here is SPATIAL, not a random thinning.  Fixed sample windows
    are re-read from the original LAS/LAZ with the same class filter, so point
    density, spacing and geometry inside each window are exactly what the
    production interpolation saw.  Random thinning would change the density
    and hence the CV error for reasons unrelated to the parameters tested.

    Windows sit inside the cat 34 watershed analysis area, within the region
    where all three point clouds overlap.  Each dataset is read once over the
    whole sample area; the individual windows are cut from that read, so all
    three datasets and all nine parameter combinations are validated on
    identical ground.

NOTE ON dmin
    dmin=2 discards points closer than 2 m to an already-accepted point, so
    the number of points actually cross-validated is far smaller than the
    number imported.  Both counts are reported.  dmin, npmin and segmax are
    held fixed across the sweep (as requested), so every combination is
    validated on the same retained point set.

OUTPUTS
    results/tables/vsurfrst_cv_baseline.csv         chosen params, per dataset x window
    results/tables/vsurfrst_cv_baseline_pooled.csv  chosen params, per dataset
    results/tables/vsurfrst_cv_parameter_sweep.csv  per dataset x window x (tension, smooth)
    results/tables/vsurfrst_cv_sweep_pooled.csv     per dataset x (tension, smooth)
    results/tables/vsurfrst_cv_sample_windows.csv   window geometry + point counts
    results/logs/vsurfrst_cv_<date>.log
"""

import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grass_env import grass_session, log  # noqa: E402

# ------------------------------------------------------------------ config
RUN = "vsurfrst_cv"
PROJECT = "DEM_generation"
WORK_MAPSET = "xval"           # new mapset -- for_codem is read-only (hard rule 1)

LAS_DIR = Path("/home/pregmi3/CODEM/demo/All_elevation")
SCRATCH = Path(os.environ.get("CV_SCRATCH", "/tmp/vsurfrst_cv"))

DATASETS = {
    # key          source point cloud          production vector      label
    "lidar_2020": (LAS_DIR / "lidar_2020.laz",    "dtm_2020_points",
                   "2020 lidar ground+road"),
    "lidar_2024": (LAS_DIR / "NEWAREA.las",       "dtm_2024_points",
                   "2024 lidar ground+road"),
    "sfm_2024":   (LAS_DIR / "sfm_2024.copc.laz", "cap_dtm_2024_points",
                   "2024 CAP SfM ground+road"),
}
CLASS_FILTER = "2,11"          # ground + road, as in raster_creation_for_codem.ipynb

# Production parameters under test
BASE = dict(tension=20, smooth=1.0, npmin=300, dmin=2, segmax=40)

# Requested sweep: tension 10/20/40 x smoothing 0.1/1/5; npmin/dmin/segmax held.
TENSIONS = [10, 20, 40]
SMOOTHS = [0.1, 1.0, 5.0]

RES = 1.0                      # production grid resolution
NPROCS = int(os.environ.get("CV_NPROCS", "12"))

# Sample area: the cat 34 basin bbox (w=313173.3 e=313720.3 s=192579.2
# n=193687.2), which lies wholly inside the 2020 / 2024 / SfM common overlap.
# One v.in.pdal pass per dataset covers this box; windows are cut from it.
AREA = dict(w=313170.0, s=192570.0, e=313725.0, n=193690.0)

# Four 250 m sample windows spanning the basin, low to high.
WIN = 250.0
WINDOWS = {                    # (west, south) corner
    "w1_lower": (313200.0, 192600.0),
    "w2_mid":   (313450.0, 192880.0),
    "w3_upper": (313200.0, 193150.0),
    "w4_head":  (313450.0, 193400.0),
}


# ------------------------------------------------------------------ helpers
def window_bbox(key):
    w, s = WINDOWS[key]
    return w, s, w + WIN, s + WIN


def import_area(gs, ds_key):
    """Read the whole sample area from one point cloud, once."""
    src, _, _ = DATASETS[ds_key]
    name = f"area_{ds_key}"
    if gs.find_file(name=name, element="vector", mapset=".")["name"]:
        log(f"reusing {name}", run=RUN)
        return name
    if not src.exists():
        sys.exit(f"Source point cloud missing: {src}")
    t0 = time.time()
    gs.run_command(
        "v.in.pdal", input=str(src), output=name,
        class_filter=CLASS_FILTER, flags="w",
        spatial=f"{AREA['w']},{AREA['s']},{AREA['e']},{AREA['n']}",
        overwrite=True, quiet=True,
    )
    npts = int(gs.parse_command("v.info", map=name, flags="t")["points"])
    log(f"import {name}: {npts} pts from {src.name} "
        f"class_filter={CLASS_FILTER} bbox={AREA} in {time.time()-t0:.0f}s",
        run=RUN)
    return name


def cut_windows(gs, ds_key, area_map):
    """Cut the sample windows out of the area point set.

    Done through an x,y,z dump rather than a spatial query so the operation is
    exact and cheap on point-only maps; the dump is deleted afterwards.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    dump = SCRATCH / f"{ds_key}.xyz"
    names = {wk: f"pts_{ds_key}_{wk}" for wk in WINDOWS}
    if all(gs.find_file(name=n, element="vector", mapset=".")["name"]
           for n in names.values()):
        log(f"reusing window vectors for {ds_key}", run=RUN)
        return names

    t0 = time.time()
    with open(dump, "w") as fh:
        subprocess.run(
            # layer=-1 : coordinates only, no attribute lookup (the PDAL
            # import carries no attribute table)
            ["v.out.ascii", f"input={area_map}", "layer=-1", "type=point",
             "format=point", "separator=space", "--q"],
            stdout=fh, check=True,
        )

    handles, counts = {}, {}
    paths = {wk: SCRATCH / f"{ds_key}_{wk}.xyz" for wk in WINDOWS}
    boxes = {wk: window_bbox(wk) for wk in WINDOWS}
    for wk in WINDOWS:
        handles[wk] = open(paths[wk], "w")
        counts[wk] = 0
    with open(dump) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 3:
                continue
            x, y = float(p[0]), float(p[1])
            for wk, (w, s, e, n) in boxes.items():
                if w <= x < e and s <= y < n:
                    handles[wk].write(f"{p[0]} {p[1]} {p[2]}\n")
                    counts[wk] += 1
                    break
    for fhh in handles.values():
        fhh.close()
    dump.unlink(missing_ok=True)

    for wk in WINDOWS:
        gs.run_command("v.in.ascii", input=str(paths[wk]), output=names[wk],
                       format="point", separator="space", x=1, y=2, z=3,
                       flags="zt", overwrite=True, quiet=True)
        paths[wk].unlink(missing_ok=True)
        log(f"window {names[wk]}: {counts[wk]} pts imported", run=RUN)
    log(f"cut windows for {ds_key} in {time.time()-t0:.0f}s", run=RUN)
    return names


def cv_stats(gs, cvdev):
    """Leave-one-out residual statistics from a cvdev vector map (column flt1)."""
    raw = gs.read_command("v.db.select", map=cvdev, columns="flt1",
                          flags="c").split()
    vals = [float(v) for v in raw if v not in ("", "NULL")]
    n = len(vals)
    if n == 0:
        raise RuntimeError(f"{cvdev} produced no residuals")
    mean = sum(vals) / n
    rmse = math.sqrt(sum(v * v for v in vals) / n)
    mae = sum(abs(v) for v in vals) / n
    sv = sorted(vals)
    median = sv[n // 2] if n % 2 else 0.5 * (sv[n // 2 - 1] + sv[n // 2])
    absdev = sorted(abs(v - median) for v in vals)
    mad = absdev[n // 2] if n % 2 else 0.5 * (absdev[n // 2 - 1] + absdev[n // 2])
    p95 = sv[min(n - 1, int(round(0.95 * (n - 1))))]
    return dict(n=n, rmse=rmse, mae=mae, mean=mean, median=median,
                nmad=1.4826 * mad, p95=p95)


def run_cv(gs, pts, ds_key, win_key, tension, smooth):
    """One v.surf.rst -c run."""
    w, s, e, n = window_bbox(win_key)
    gs.run_command("g.region", w=w, s=s, e=e, n=n, res=RES, flags="a")
    tag = (f"cv_{ds_key}_{win_key}_t{tension}"
           f"_s{str(smooth).replace('.', 'p')}")
    t0 = time.time()
    try:
        gs.run_command(
            "v.surf.rst", flags="c", input=pts, cvdev=tag,
            tension=tension, smooth=smooth, npmin=BASE["npmin"],
            dmin=BASE["dmin"], segmax=BASE["segmax"],
            nprocs=NPROCS, overwrite=True, quiet=True,
        )
    except Exception as exc:                             # noqa: BLE001
        log(f"FAILED {tag}: {exc}", run=RUN)
        return None
    secs = time.time() - t0
    st = cv_stats(gs, tag)
    st.update(dataset=ds_key, window=win_key, tension=tension, smooth=smooth,
              npmin=BASE["npmin"], dmin=BASE["dmin"], segmax=BASE["segmax"],
              cvdev_map=tag, seconds=round(secs, 1))
    log(f"{tag}: n={st['n']} RMSE={st['rmse']:.4f} MAE={st['mae']:.4f} "
        f"NMAD={st['nmad']:.4f} ({secs:.0f}s)", run=RUN)
    return st


def pool(rows):
    """Pool per-window residuals: RMSE as sqrt(sum(n_i*rmse_i^2)/sum(n_i))."""
    n = sum(r["n"] for r in rows)
    return (n,
            math.sqrt(sum(r["n"] * r["rmse"] ** 2 for r in rows) / n),
            sum(r["n"] * r["mae"] for r in rows) / n,
            sum(r["n"] * r["mean"] for r in rows) / n)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


# ------------------------------------------------------------------ main
def main():
    tables = Path("results/tables")
    gs, _ = grass_session(project=PROJECT, mapset="PERMANENT")
    import grass.script as g

    try:
        g.run_command("g.mapset", flags="c", mapset=WORK_MAPSET, quiet=True)
    except Exception:                                    # noqa: BLE001
        g.run_command("g.mapset", mapset=WORK_MAPSET, quiet=True)
    log(f"START mapset={WORK_MAPSET} nprocs={NPROCS} res={RES} base={BASE} "
        f"area={AREA} win={WIN}m windows={list(WINDOWS)}", run=RUN)

    # --- one LAS pass per dataset, then cut the windows -------------------
    pts, win_counts = {}, []
    for ds in DATASETS:
        area_map = import_area(g, ds)
        names = cut_windows(g, ds, area_map)
        for wk, nm in names.items():
            pts[(ds, wk)] = nm
            w, s, e, n = window_bbox(wk)
            npts = int(g.parse_command("v.info", map=nm, flags="t")["points"])
            win_counts.append(dict(
                dataset=ds, label=DATASETS[ds][2], window=wk,
                west=w, south=s, east=e, north=n, size_m=WIN,
                area_m2=WIN * WIN, points_imported=npts,
                density_pts_per_m2=round(npts / (WIN * WIN), 4)))
    write_csv(tables / "vsurfrst_cv_sample_windows.csv", win_counts,
              ["dataset", "label", "window", "west", "south", "east", "north",
               "size_m", "area_m2", "points_imported", "density_pts_per_m2"])

    # --- sweep (the production combination is one cell of it) -------------
    rows = []
    for ds in DATASETS:
        for wk in WINDOWS:
            for tension in TENSIONS:
                for smooth in SMOOTHS:
                    st = run_cv(g, pts[(ds, wk)], ds, wk, tension, smooth)
                    if st:
                        rows.append(st)

    fields = ["dataset", "window", "tension", "smooth", "npmin", "dmin",
              "segmax", "n", "rmse", "mae", "mean", "median", "nmad", "p95",
              "seconds", "cvdev_map"]
    write_csv(tables / "vsurfrst_cv_parameter_sweep.csv", rows, fields)
    write_csv(tables / "vsurfrst_cv_baseline.csv",
              [r for r in rows if r["tension"] == BASE["tension"]
               and r["smooth"] == BASE["smooth"]], fields)

    # --- pooled -----------------------------------------------------------
    pooled = []
    for ds in DATASETS:
        for tension in TENSIONS:
            for smooth in SMOOTHS:
                grp = [r for r in rows if r["dataset"] == ds
                       and r["tension"] == tension and r["smooth"] == smooth]
                if not grp:
                    continue
                n, rmse, mae, mean = pool(grp)
                pooled.append(dict(
                    dataset=ds, label=DATASETS[ds][2], tension=tension,
                    smooth=smooth, npmin=BASE["npmin"], dmin=BASE["dmin"],
                    segmax=BASE["segmax"], n_windows=len(grp), n=n,
                    rmse=rmse, mae=mae, mean=mean,
                    is_production=(tension == BASE["tension"]
                                   and smooth == BASE["smooth"])))
    pf = ["dataset", "label", "tension", "smooth", "npmin", "dmin", "segmax",
          "n_windows", "n", "rmse", "mae", "mean", "is_production"]
    write_csv(tables / "vsurfrst_cv_sweep_pooled.csv", pooled, pf)
    write_csv(tables / "vsurfrst_cv_baseline_pooled.csv",
              [p for p in pooled if p["is_production"]], pf)

    # --- report -----------------------------------------------------------
    print("\n=== pooled leave-one-out CV RMSE (m) ===", flush=True)
    for ds in DATASETS:
        grp = [p for p in pooled if p["dataset"] == ds]
        if not grp:
            continue
        best = min(grp, key=lambda p: p["rmse"])
        prod = next((p for p in grp if p["is_production"]), None)
        print(f"\n{ds}  ({DATASETS[ds][2]})", flush=True)
        for p in sorted(grp, key=lambda p: p["rmse"]):
            mark = "   <- production" if p["is_production"] else ""
            star = "  *best*" if p is best else ""
            print(f"  tension={p['tension']:>2}  smooth={p['smooth']:<4} "
                  f"n={p['n']:>7}  RMSE={p['rmse']:.4f}  MAE={p['mae']:.4f}"
                  f"{star}{mark}", flush=True)
        if prod:
            d = 100 * (prod["rmse"] - best["rmse"]) / prod["rmse"]
            log(f"{ds}: production(t=20,s=1) RMSE={prod['rmse']:.4f}; "
                f"best t={best['tension']} s={best['smooth']} "
                f"RMSE={best['rmse']:.4f}; production is {d:.1f}% higher",
                run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
