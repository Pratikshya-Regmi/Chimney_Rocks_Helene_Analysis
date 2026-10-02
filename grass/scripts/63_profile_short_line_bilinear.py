#!/usr/bin/env python3
"""
63_profile_short_line_bilinear.py
=================================
Re-extract the short-line profiles with BILINEAR sampling.

WHY
    r.profile samples nearest-neighbour. On a 1 m grid, an oblique line
    therefore repeats a cell's value for two or three consecutive samples and
    then jumps to the next cell, which renders as a staircase -- an artefact
    of the sampling, not of the terrain. The surfaces are continuous
    interpolations (v.surf.rst), so bilinear sampling is the correct way to
    read them along a line, and the profiles come out smooth without any
    cosmetic filtering of the values.

    Nothing is smoothed, averaged or filtered here. The only change is how
    the continuous surface is read between cell centres.

Exports each surface to a small GeoTIFF around the line, samples with
scipy.ndimage.map_coordinates (order=1, bilinear), and rewrites the CSVs.

Outputs:
  results/tables/profile_short_line_before.csv   (rewritten, bilinear)
  results/tables/profile_short_line_after.csv    (rewritten, bilinear)
  results/tables/profile_short_line_summary.csv  (rewritten)
  results/figures/_pl_surf_<key>.tif             (working extracts)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import grass_session, log  # noqa: E402

RUN = "profile_short_line"
ROOT = HERE.parents[1]
TABLES = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

SURFACES = {
    "2017 lidar": ("dtm_2017", "dtm_2017"),
    "2020 lidar": ("dtm_2020_filled", "lidar_2020_regis"),
    "2024 lidar": ("dtm_2024_filled", "lidar_2024_regis"),
    "2024 SfM": ("cap_dtm_2024_filled", "cap_sfm_regis"),
}
REFERENCE = "2020 lidar"
STEP = 1.0          # CSV spacing, as specified
PAD = 30.0


def export(gs, rast, key):
    out = FIG / f"_pl_surf_{key}.tif"
    gs.run_command("g.region", raster=rast, quiet=True)
    gs.run_command("g.region", n=192029.41 + PAD, s=192015.03 - PAD,
                   e=315432.11 + PAD, w=315287.43 - PAD, align=rast,
                   quiet=True)
    gs.run_command("r.out.gdal", input=rast, output=str(out),
                   format="GTiff", type="Float64", overwrite=True,
                   flags="c", quiet=True)
    return out


def densify(verts, step):
    """Points every `step` metres along the polyline, with their distance."""
    v = np.asarray(verts, dtype=float)
    seg = np.hypot(*np.diff(v, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    d = np.arange(0.0, total + 1e-9, step)
    x = np.interp(d, cum, v[:, 0])
    y = np.interp(d, cum, v[:, 1])
    return d, x, y


def sample_bilinear(tif, x, y):
    with rasterio.open(tif) as src:
        arr = src.read(1, masked=True)
        t = src.transform
    z = np.asarray(arr.filled(np.nan), dtype=float)
    col = (x - t.c) / t.a - 0.5
    row = (y - t.f) / t.e - 0.5
    return ndi.map_coordinates(z, [row, col], order=1, mode="nearest")


def staircase_metric(v):
    """Fraction of consecutive samples that are exactly equal -- the
    signature of nearest-neighbour sampling on a grid."""
    d = np.diff(v)
    return float(np.mean(np.abs(d) < 1e-9))


def main():
    gs, _ = grass_session(project="DEM_generation", mapset="profile_short")
    for ms in gs.read_command("g.mapsets", flags="l").split():
        gs.run_command("g.mapsets", mapset=ms, operation="add")

    pts = gs.read_command("v.out.ascii", input="short_line_3358@profile_short",
                          format="standard", quiet=True)
    verts = []
    for line in pts.splitlines():
        p = line.split()
        if len(p) >= 2:
            try:
                x, y = float(p[0]), float(p[1])
            except ValueError:
                continue
            if 200000 < x < 900000 and 100000 < y < 400000:
                verts.append((x, y))

    d, xs, ys = densify(verts, STEP)
    print(f"line {d.max():.2f} m, {len(d)} samples at {STEP} m")

    before = pd.read_csv(TABLES / "profile_short_line_before.csv")
    after = pd.read_csv(TABLES / "profile_short_line_after.csv")
    old = {"before": before, "after": after}

    out = {}
    print(f"\n{'surface':12s} {'state':7s} "
          f"{'repeat frac NN':>14s} {'repeat frac bilinear':>21s}")
    for label, (rb, ra) in SURFACES.items():
        for state, rast in (("before", rb), ("after", ra)):
            key = f"{state}_{label.replace(' ', '')}"
            tif = export(gs, rast, key)
            v = sample_bilinear(tif, xs, ys)
            out.setdefault(state, {})[label] = v
            nn = staircase_metric(old[state][label].values)
            bl = staircase_metric(v)
            print(f"{label:12s} {state:7s} {nn:14.3f} {bl:21.3f}")

    rows = []
    for state in ("before", "after"):
        df = pd.DataFrame({"distance_m": np.round(d, 3)})
        for label in SURFACES:
            df[label] = out[state][label]
        df.to_csv(TABLES / f"profile_short_line_{state}.csv", index=False,
                  float_format="%.4f")
        print(f"\nrewrote profile_short_line_{state}.csv ({len(df)} rows, "
              f"bilinear)")

    b = pd.read_csv(TABLES / "profile_short_line_before.csv")
    a = pd.read_csv(TABLES / "profile_short_line_after.csv")
    print("\nCAPTION NUMBERS after bilinear resampling")
    print(f"{'surface':12s} {'offset before':>14s} {'offset after':>13s} "
          f"{'correction':>11s}")
    for s in SURFACES:
        ob = float((b[s] - b[REFERENCE]).mean())
        oa = float((a[s] - a[REFERENCE]).mean())
        rows.append(dict(surface=s, raster_before=SURFACES[s][0],
                         raster_after=SURFACES[s][1],
                         before_mean_m=round(float(b[s].mean()), 4),
                         after_mean_m=round(float(a[s].mean()), 4),
                         before_offset_vs_2020_m=round(ob, 4),
                         after_offset_vs_2020_m=round(oa, 4),
                         after_min_m=round(float(a[s].min()), 4),
                         after_max_m=round(float(a[s].max()), 4)))
        print(f"{s:12s} {ob:14.3f} {oa:13.3f} {oa-ob:11.3f}")
    allv = np.concatenate([a[s].values for s in SURFACES])
    print(f"\nprofile length {a['distance_m'].max():.1f} m; elevation range "
          f"after {allv.min():.2f} to {allv.max():.2f} m "
          f"(span {allv.max()-allv.min():.2f} m)")
    pd.DataFrame(rows).to_csv(TABLES / "profile_short_line_summary.csv",
                              index=False)
    log("re-extracted short-line profiles with bilinear sampling", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
