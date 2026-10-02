#!/usr/bin/env python3
"""
60_profile_short_line.py
========================
Extract elevation profiles along the new short profile line (short_line.kml)
for all four surfaces, twice: before and after datum harmonisation and
co-registration.

THE LINE
    short_line.kml is WGS84 geographic; it is imported with v.import, which
    reprojects it to the project CRS, EPSG:3358 (NAD83(HARN) / North
    Carolina, metres). Five vertices, four segments.

SURFACE NAMES -- taken from the established mapping in this project
(49_fig_profile_zoom_options.py), NOT guessed:

    epoch          BEFORE (raw, as first built)   AFTER (co-registered)
    2017 lidar     dtm_2017                       dtm_2017
    2020 lidar     dtm_2020_filled                lidar_2020_regis
    2024 lidar     dtm_2024_filled                lidar_2024_regis
    2024 SfM       cap_dtm_2024_filled            cap_sfm_regis

    All in DEM_generation/for_codem. Two things to know:

    (1) 2017 is the SAME raster on both sides. It is the surface the others
        were brought onto, so it has no "before" state of its own. It is
        carried through both panels as the fixed reference.
    (2) dtm_2024_filled is delivered on a different vertical datum -- it sits
        about 31 m below the 2020 surface, the NAVD88 / NAD83-ellipsoid geoid
        separation in western North Carolina. The CSV holds the RAW value, as
        asked; the figures handle the offset explicitly and say so.

Writes the raw sampled values, nothing rescaled, at 1 m spacing.

Outputs:
  results/tables/profile_short_line_before.csv
  results/tables/profile_short_line_after.csv
  results/tables/profile_short_line_summary.csv
  results/logs/profile_short_line_<date>.log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import grass_session, log  # noqa: E402

RUN = "profile_short_line"
ROOT = HERE.parents[1]
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
KML = ROOT / "short_line.kml"

MAPSET = "profile_short"
LINE_VECT = "short_line_3358"

SURFACES = {
    "2017 lidar": ("dtm_2017", "dtm_2017"),
    "2020 lidar": ("dtm_2020_filled", "lidar_2020_regis"),
    "2024 lidar": ("dtm_2024_filled", "lidar_2024_regis"),
    "2024 SfM": ("cap_dtm_2024_filled", "cap_sfm_regis"),
}
STEP = 1.0


def import_line(gs):
    gs.run_command("v.import", input=str(KML), output=LINE_VECT,
                   overwrite=True, quiet=True)
    pts = gs.read_command("v.out.ascii", input=LINE_VECT,
                          format="standard", quiet=True)
    verts = []
    for line in pts.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if 200000 < x < 900000 and 100000 < y < 400000:
                verts.append((x, y))
    # length straight from the reprojected vertices; v.to.db's -p output
    # parsed to zero here, and the vertex sum is the same quantity
    v = np.asarray(verts)
    length = float(np.hypot(*np.diff(v, axis=0).T).sum())
    return length, verts


def profile(gs, raster, coords):
    flat = []
    for x, y in coords:
        flat += [x, y]
    txt = gs.read_command("r.profile", input=raster, coordinates=flat,
                          resolution=STEP, flags="g", null="nan", quiet=True)
    d, v = [], []
    for line in txt.strip().splitlines():
        p = line.split()
        if len(p) >= 4:
            d.append(float(p[2]))
            v.append(float("nan") if p[3] in ("nan", "*") else float(p[3]))
    return np.array(d), np.array(v)


def main():
    gs, _ = grass_session(project="DEM_generation", mapset=MAPSET)
    for ms in gs.read_command("g.mapsets", flags="l").split():
        gs.run_command("g.mapsets", mapset=ms, operation="add")

    length, verts = import_line(gs)
    print("=" * 72)
    print("THE LINE  (short_line.kml -> EPSG:3358, NAD83(HARN) / North "
          "Carolina)")
    print("=" * 72)
    for i, (x, y) in enumerate(verts):
        tag = "  START" if i == 0 else ("  END" if i == len(verts) - 1 else "")
        print(f"  v{i}   E {x:11.2f}   N {y:11.2f}{tag}")
    print(f"  total length along the line : {length:.2f} m")
    print(f"  straight start->end distance: "
          f"{np.hypot(verts[-1][0]-verts[0][0], verts[-1][1]-verts[0][1]):.2f} m")

    # coverage check
    print("\nCOVERAGE CHECK")
    bad = []
    prof = {"before": {}, "after": {}}
    for label, (r_before, r_after) in SURFACES.items():
        for state, rast in (("before", r_before), ("after", r_after)):
            gs.run_command("g.region", raster=rast, quiet=True)
            d, v = profile(gs, rast, verts)
            prof[state][label] = (d, v)
            n_null = int(np.isnan(v).sum())
            flag = "OK" if n_null == 0 else f"{n_null} NULL of {v.size}"
            print(f"  {state:6s} {label:11s} {rast:22s} {v.size:4d} samples  "
                  f"{flag}")
            if n_null:
                bad.append((state, label, rast, n_null))

    if bad:
        print("\nSTOPPING: the line leaves the coverage of:")
        for state, label, rast, n in bad:
            print(f"    {state} {label} ({rast}): {n} null samples")
        log("STOPPED: line outside coverage of " +
            ", ".join(b[2] for b in bad), run=RUN)
        sys.exit(1)
    print("  -> the line lies inside every surface, with no null samples.")

    rows = []
    for state in ("before", "after"):
        d0 = prof[state]["2017 lidar"][0]
        out = pd.DataFrame({"distance_m": np.round(d0, 3)})
        for label in SURFACES:
            d, v = prof[state][label]
            out[label] = np.interp(d0, d, v) if len(d) != len(d0) else v
        path = TABLES / f"profile_short_line_{state}.csv"
        out.to_csv(path, index=False, float_format="%.4f")
        print(f"\nwrote {path.relative_to(ROOT)}  ({len(out)} rows)")

    print("\n" + "=" * 72)
    print("PER-SURFACE SUMMARY  (elevation in m, along the whole line)")
    print("=" * 72)
    print(f"{'surface':12s} {'before mean':>12s} {'after mean':>11s} "
          f"{'shift':>9s} {'after min':>10s} {'after max':>10s}")
    ref_b = prof["before"]["2020 lidar"][1]
    ref_a = prof["after"]["2020 lidar"][1]
    for label in SURFACES:
        b = prof["before"][label][1]
        a = prof["after"][label][1]
        rows.append(dict(surface=label,
                         raster_before=SURFACES[label][0],
                         raster_after=SURFACES[label][1],
                         before_mean_m=round(float(np.mean(b)), 4),
                         after_mean_m=round(float(np.mean(a)), 4),
                         shift_m=round(float(np.mean(a) - np.mean(b)), 4),
                         before_offset_vs_2020_m=round(
                             float(np.mean(b - ref_b)), 4),
                         after_offset_vs_2020_m=round(
                             float(np.mean(a - ref_a)), 4),
                         after_min_m=round(float(np.min(a)), 4),
                         after_max_m=round(float(np.max(a)), 4)))
        print(f"{label:12s} {np.mean(b):12.3f} {np.mean(a):11.3f} "
              f"{np.mean(a)-np.mean(b):9.3f} {np.min(a):10.3f} "
              f"{np.max(a):10.3f}")

    print(f"\n{'surface':12s} {'offset vs 2020 BEFORE':>22s} "
          f"{'offset vs 2020 AFTER':>22s}")
    for r in rows:
        print(f"{r['surface']:12s} {r['before_offset_vs_2020_m']:22.3f} "
              f"{r['after_offset_vs_2020_m']:22.3f}")

    allafter = np.concatenate([prof["after"][k][1] for k in SURFACES])
    print(f"\nprofile length        {prof['after']['2017 lidar'][0].max():.1f} m")
    print(f"elevation range (after, all four surfaces) "
          f"{allafter.min():.2f} to {allafter.max():.2f} m "
          f"(span {allafter.max()-allafter.min():.2f} m)")

    pd.DataFrame(rows).to_csv(TABLES / "profile_short_line_summary.csv",
                              index=False)
    log(f"short line {length:.2f} m, 8 profiles at {STEP} m, no nulls", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
