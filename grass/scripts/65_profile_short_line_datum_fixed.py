#!/usr/bin/env python3
"""
65_profile_short_line_datum_fixed.py
====================================
Re-extract the short-line A--B profiles with the DATUM problem removed from
the "before" panel.

WHY THIS EXISTS
    Script 63 sampled the before panel from `dtm_2024_filled@for_codem`, the
    2024 lidar surface as delivered -- NAD83(2011) ELLIPSOIDAL heights. That
    sits ~31 m below every other surface, which is the geoid separation in
    western North Carolina, not a vertical bias. The before panel therefore
    showed the datum offset, not the residual bias the figure is meant to
    illustrate, and needed a broken y-axis to display at all.

    Per Methods 2.3.3 the datum harmonisation (NOAA VDatum, NAVD88/GEOID18)
    is a separate, EARLIER step than co-registration. The profile figure is
    meant to show only what the residual vertical bias correction does, i.e.
    the state after the datum transform.

WHAT THE TWO PANELS NOW SHOW
    before  after datum harmonisation, before co-registration and residual
            vertical bias correction
    after   after co-registration and residual vertical bias correction

SURFACE MAPPING, AND WHY
    The pipeline chain in for_codem, read from each raster's own r.info -h:

        dtm_2024_filled          raw v.surf.rst surface, ELLIPSOIDAL
        lidar_2024_regis         = dtm_2024_filled + 30.976715
                                 -> the DATUM-HARMONISED surface. The
                                    constant is the geoid separation; it is
                                    also the TARGET that 03_coreg_variable_lod
                                    was run against (see CONFIG in that file
                                    and results/logs/coreg_*.log).
        coreg_target_coreg       = <lidar_2024_regis shifted> + (-0.13543027)
                                    Nuth & Kaab solution
                                    dx=+0.3779 dy=-0.7046 dz=-0.1354
        coreg_target_corrected   = coreg_target_coreg - coreg_bias_model
                                    bias_model = 0.00101304*canopy + 0.08691537

    So `lidar_2024_regis` -- which script 63 used as the AFTER surface -- is
    in fact the BEFORE surface under the definition above, and the true after
    surface is `coreg_target_corrected`.

THE ONE RECONSTRUCTION, STATED PLAINLY
    `coreg_target_corrected` was computed only inside the tributary watershed
    (region basins_90_v_cat34). The A--B line lies ~1.7 km outside it and the
    raster has NO data there. It is rebuilt here at the line from the stored
    solution, using the same code path as 03_coreg_variable_lod._apply_shift
    (r.region + r.resamp.interp bilinear, signs as corrected 2026-08-25) and
    the same canopy bias model. Nothing is refitted; the corridor-wide
    solution is applied at the line.

Outputs:
  results/tables/profile_short_line_datumfixed_before.csv
  results/tables/profile_short_line_datumfixed_after.csv
  results/tables/profile_short_line_datumfixed_summary.csv
  results/figures/_pl_dfx_<key>.tif            (working extracts)
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

RUN = "profile_short_line_datumfixed"
ROOT = HERE.parents[1]
TABLES = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

# --- Nuth & Kaab solution, from results/logs/coreg_20260825.log -------------
DX, DY, DZ = 0.3779, -0.7046, -0.13543027
# --- canopy bias model, from r.info -h coreg_bias_model --------------------
BIAS_C1, BIAS_C0 = 0.00101304, 0.08691537
REF_DSM = "lidar_2020_DSM@DTM_DSM"
REF_DTM = "dtm_2020_filled@for_codem"

# 2020 lidar is the co-registration REFERENCE and is used unchanged in BOTH
# panels, so the two panels share one fixed reference and stay comparable.
SURFACES = {
    "2017 lidar": ("dtm_2017@for_codem",          "dtm_2017@for_codem"),
    "2020 lidar": (REF_DTM,                        REF_DTM),
    "2024 lidar": ("lidar_2024_regis@for_codem",  "_pl_2024_corrected"),
    "2024 SfM":   ("cap_dtm_2024_filled@for_codem", "cap_sfm_regis@for_codem"),
}
REFERENCE = "2020 lidar"
STEP = 1.0
PAD = 30.0        # padding for the exported sampling window
WORK_PAD = 80.0   # wider padding for the shift/resample, so bilinear has data

# line bounding box (PROFILE_OPTIONS.md)
LN, LS, LE, LW_ = 192029.41, 192015.03, 315432.11, 315287.43


def set_work_region(gs):
    gs.run_command("g.region", raster=REF_DTM, quiet=True)
    gs.run_command("g.region", n=LN + WORK_PAD, s=LS - WORK_PAD,
                   e=LE + WORK_PAD, w=LW_ - WORK_PAD,
                   align=REF_DTM, quiet=True)


def build_corrected(gs):
    """Rebuild coreg_target_corrected at the line from the stored solution.

    Mirrors 03_coreg_variable_lod._apply_shift exactly: move the map's
    georeferencing by (+dx, +dy), resample back onto the computational grid
    with bilinear interpolation, then add dz.  Then subtract the canopy bias
    model.  Everything is written into the current (profile_short) mapset;
    for_codem is never modified.
    """
    set_work_region(gs)

    # canopy = clamp(DSM - DTM, 0, 60), exactly as stage2 builds it
    gs.mapcalc("_pl_canopy_raw = %s - %s" % (REF_DSM, REF_DTM),
               overwrite=True, quiet=True)
    gs.mapcalc("_pl_canopy = if(_pl_canopy_raw < 0, 0, "
               "if(_pl_canopy_raw > 60, 60, _pl_canopy_raw))",
               overwrite=True, quiet=True)
    gs.mapcalc("_pl_bias_model = (%.8f * _pl_canopy) + (%.8f)"
               % (BIAS_C1, BIAS_C0), overwrite=True, quiet=True)

    # horizontal shift, same code path as _apply_shift
    gs.run_command("g.copy", raster="lidar_2024_regis@for_codem,_pl_tgt_tmp",
                   overwrite=True, quiet=True)
    info = gs.raster_info("lidar_2024_regis@for_codem")
    gs.run_command("r.region", map="_pl_tgt_tmp",
                   n=info["north"] + DY, s=info["south"] + DY,
                   e=info["east"] + DX, w=info["west"] + DX, quiet=True)
    gs.run_command("r.resamp.interp", input="_pl_tgt_tmp",
                   output="_pl_tgt_h", method="bilinear",
                   overwrite=True, quiet=True)
    gs.mapcalc("_pl_2024_coreg = _pl_tgt_h + (%.8f)" % DZ,
               overwrite=True, quiet=True)
    gs.mapcalc("_pl_2024_corrected = _pl_2024_coreg - _pl_bias_model",
               overwrite=True, quiet=True)
    gs.run_command("g.remove", type="raster", name="_pl_tgt_tmp,_pl_tgt_h",
                   flags="f", quiet=True)

    st = gs.parse_command("r.univar", map="_pl_bias_model", flags="g",
                          quiet=True)
    print(f"  bias model over the line window: "
          f"{float(st['min']):.4f} to {float(st['max']):.4f} m "
          f"(mean {float(st['mean']):.4f})")
    return "_pl_2024_corrected"


def export(gs, rast, key):
    out = FIG / f"_pl_dfx_{key}.tif"
    gs.run_command("g.region", raster=REF_DTM, quiet=True)
    gs.run_command("g.region", n=LN + PAD, s=LS - PAD, e=LE + PAD,
                   w=LW_ - PAD, align=REF_DTM, quiet=True)
    gs.run_command("r.out.gdal", input=rast, output=str(out), format="GTiff",
                   type="Float64", overwrite=True, flags="c", quiet=True)
    return out


def densify(verts, step):
    v = np.asarray(verts, dtype=float)
    seg = np.hypot(*np.diff(v, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    d = np.arange(0.0, cum[-1] + 1e-9, step)
    return d, np.interp(d, cum, v[:, 0]), np.interp(d, cum, v[:, 1])


def sample_bilinear(tif, x, y):
    with rasterio.open(tif) as src:
        arr = src.read(1, masked=True)
        t = src.transform
    z = np.asarray(arr.filled(np.nan), dtype=float)
    col = (x - t.c) / t.a - 0.5
    row = (y - t.f) / t.e - 0.5
    return ndi.map_coordinates(z, [row, col], order=1, mode="nearest")


def geoid_check(x, y):
    """Independent check that lidar_2024_regis really is datum-harmonised."""
    try:
        import pyproj
        from pyproj import Transformer
        pyproj.network.set_network_enabled(True)
        t1 = Transformer.from_crs("EPSG:3358", "EPSG:6318", always_xy=True)
        t2 = Transformer.from_crs("EPSG:6319", "EPSG:6349", always_xy=True)
        seps = []
        for xi, yi in zip(x[::40], y[::40]):
            lon, lat = t1.transform(xi, yi)
            _, _, z = t2.transform(lon, lat, 0.0)
            seps.append(z)
        seps = np.asarray(seps)
        if np.allclose(seps, 0.0):
            print("  !! GEOID18 grid unavailable (transform returned 0) "
                  "-- skipping check")
            return None
        print(f"  GEOID18 separation along the line: {seps.min():.4f} to "
              f"{seps.max():.4f} m (add to ellipsoidal height)")
        print(f"  constant baked into lidar_2024_regis:  +30.976715 m")
        print(f"  difference: {seps.mean() - 30.976715:+.4f} m")
        return float(seps.mean())
    except Exception as e:  # pragma: no cover
        print(f"  !! geoid check skipped: {e}")
        return None


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
    print(f"line {d.max():.2f} m, {len(d)} samples at {STEP} m\n")

    print("GEOID18 CHECK")
    geoid_check(xs, ys)

    print("\nREBUILDING coreg_target_corrected AT THE LINE")
    build_corrected(gs)

    out = {}
    for label, (rb, ra) in SURFACES.items():
        for state, rast in (("before", rb), ("after", ra)):
            key = f"{state}_{label.replace(' ', '')}"
            v = sample_bilinear(export(gs, rast, key), xs, ys)
            out.setdefault(state, {})[label] = v
            if np.isnan(v).any():
                print(f"  !! {label} {state}: {np.isnan(v).sum()} null samples")

    for state in ("before", "after"):
        df = pd.DataFrame({"distance_m": np.round(d, 3)})
        for label in SURFACES:
            df[label] = out[state][label]
        df.to_csv(TABLES / f"profile_short_line_datumfixed_{state}.csv",
                  index=False, float_format="%.4f")

    b = pd.read_csv(TABLES / "profile_short_line_datumfixed_before.csv")
    a = pd.read_csv(TABLES / "profile_short_line_datumfixed_after.csv")

    print("\nMEAN OFFSET FROM THE 2020 LIDAR REFERENCE")
    print(f"{'surface':12s} {'before (m)':>12s} {'after (m)':>11s} "
          f"{'change (m)':>12s}")
    rows = []
    for s in SURFACES:
        ob = float((b[s] - b[REFERENCE]).mean())
        oa = float((a[s] - a[REFERENCE]).mean())
        rows.append(dict(surface=s, raster_before=SURFACES[s][0],
                         raster_after=SURFACES[s][1],
                         before_mean_m=round(float(b[s].mean()), 4),
                         after_mean_m=round(float(a[s].mean()), 4),
                         before_offset_vs_2020_m=round(ob, 4),
                         after_offset_vs_2020_m=round(oa, 4)))
        print(f"{s:12s} {ob:12.4f} {oa:11.4f} {oa - ob:12.4f}")
    pd.DataFrame(rows).to_csv(
        TABLES / "profile_short_line_datumfixed_summary.csv", index=False)

    allv = np.concatenate([b[s].values for s in SURFACES]
                          + [a[s].values for s in SURFACES])
    print(f"\nelevation range across BOTH panels: {allv.min():.2f} to "
          f"{allv.max():.2f} m (span {allv.max() - allv.min():.2f} m)")
    sprd_b = max(abs(float((b[s] - b[REFERENCE]).mean())) for s in SURFACES)
    sprd_a = max(abs(float((a[s] - a[REFERENCE]).mean())) for s in SURFACES)
    print(f"largest |offset|: before {sprd_b:.3f} m, after {sprd_a:.3f} m")
    log("re-extracted short-line profiles with datum-harmonised before panel",
        run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
