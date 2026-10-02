#!/usr/bin/env python3
"""
01_ground_density_check.py — is the reported deposition a classification artifact?

THE HYPOTHESIS BEING TESTED
---------------------------
The 2017 and 2020 point clouds carry the data *providers'* ground
classification. The 2024 cloud was classified separately. If the 2024
classification recovers fewer ground returns beneath closed canopy, the
RST-interpolated bare-earth surface there is built from sparser points, sits
systematically higher, and the DoD reports that as DEPOSITION across the
forested part of the basin.

The manuscript reports deposition over 41% of the watershed with a median gain
of 1.69 m, and erosion over a further 38% -- 79% of a forested basin exceeding
a 0.32 m threshold. That is not plausible as geomorphology. This script tests
whether classification is the cause.

WHAT IT MEASURES
----------------
Ground-return density (points per m^2, class 2) for each epoch, binned by
canopy height, plus the mean DoD in the same bins. The diagnostic signature of
a classification artifact is: ground density diverging between epochs as canopy
height increases, AND mean DoD rising in the same bins.

If the two track each other, part of the reported deposition is not sediment.

REQUIREMENTS
------------
Raw point clouds for each epoch (all classes retained), and a canopy-height
raster (DSM - DTM) for the pre-event epoch.

    python3 scripts/01_ground_density_check.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from grass_env import grass_session, log, require_maps  # noqa: E402

CONFIG = {
    "PROJECT": "DEM_generation",
    "MAPSET": "PERMANENT",

    # Raw point clouds -- ALL classes, not pre-filtered
    "CLOUDS": {
        "2020": "/path/to/lidar_2020_all_classes.laz",
        "2024": "/path/to/lidar_2024_all_classes.laz",
    },

    # Existing rasters
    "PRE_DTM": "dtm_2020_filled",
    "PRE_DSM": "dsm_2020_filled",          # generate first if missing
    "DOD": "Change_in_DTM_lidar_may",      # 2024 - 2020
    "EXTENT": "hull_mask",

    "CANOPY_BINS": [0, 2, 5, 10, 20, 60],  # metres
    "OUT_JSON": "results/tables/ground_density_check.json",
}


def main():
    cfg = CONFIG
    gs, _ = grass_session(project=cfg["PROJECT"], mapset=cfg["MAPSET"])
    import grass.script.array as garray

    gs.run_command("g.region", raster=cfg["EXTENT"], res=1)
    require_maps(gs, [cfg["PRE_DTM"], cfg["PRE_DSM"], cfg["DOD"]])

    # ------------------------------------------------------------ canopy
    gs.mapcalc(f"gd_canopy = {cfg['PRE_DSM']} - {cfg['PRE_DTM']}",
               overwrite=True, quiet=True)
    gs.mapcalc("gd_canopy = if(gd_canopy < 0, 0, "
               "if(gd_canopy > 60, 60, gd_canopy))",
               overwrite=True, quiet=True)
    log("built canopy height raster", run="density")

    # ------------------------------------- ground-point count rasters
    # r.in.pdal method=n counts returns per cell. class_filter=2 -> ground only.
    for epoch, path in cfg["CLOUDS"].items():
        if not Path(path).exists():
            sys.exit(f"Point cloud not found for {epoch}: {path}\n"
                     f"Edit CONFIG['CLOUDS'] with real paths. This script needs "
                     f"the UNFILTERED clouds (all classes retained).")
        gs.run_command("r.in.pdal", input=path,
                       output=f"gd_ground_n_{epoch}",
                       method="n", class_filter=2,
                       overwrite=True, quiet=True)
        gs.run_command("r.in.pdal", input=path,
                       output=f"gd_all_n_{epoch}",
                       method="n",
                       overwrite=True, quiet=True)
        log(f"counted returns for {epoch}", run="density")

    # cells with no returns are null; treat as zero density
    for epoch in cfg["CLOUDS"]:
        for kind in ("ground", "all"):
            m = f"gd_{kind}_n_{epoch}"
            gs.mapcalc(f"{m} = if(isnull({m}), 0, {m})",
                       overwrite=True, quiet=True)

    # ------------------------------------------------------------ read
    def arr(name):
        return np.ma.masked_invalid(np.array(garray.array(name, dtype=np.float64)))

    canopy = arr("gd_canopy")
    dod = arr(cfg["DOD"])
    epochs = list(cfg["CLOUDS"].keys())
    ground = {e: arr(f"gd_ground_n_{e}") for e in epochs}
    allret = {e: arr(f"gd_all_n_{e}") for e in epochs}

    # ------------------------------------------------------------ bin
    edges = cfg["CANOPY_BINS"]
    rows = []
    hdr = ("canopy (m)".rjust(12) + " | " + "cells".rjust(9) + " | "
           + " | ".join((e + " grd/m2").rjust(12) for e in epochs)
           + " | " + " | ".join((e + " grd%").rjust(9) for e in epochs)
           + " | " + "mean DoD".rjust(9))
    print("\n" + hdr)
    print("-" * len(hdr))

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (~canopy.mask) & (canopy >= lo) & (canopy < hi)
        n = int(np.count_nonzero(sel))
        if n == 0:
            continue

        rec = {"canopy_lo": lo, "canopy_hi": hi, "cells": n}
        dens, frac = {}, {}
        for e in epochs:
            g = np.asarray(ground[e][sel])
            a = np.asarray(allret[e][sel])
            dens[e] = float(np.mean(g))                       # pts per m^2 (1 m cells)
            tot = float(np.sum(a))
            frac[e] = 100.0 * float(np.sum(g)) / tot if tot else float("nan")
            rec[f"ground_density_{e}"] = dens[e]
            rec[f"ground_fraction_pct_{e}"] = frac[e]

        d = np.asarray(dod[sel & ~dod.mask])
        mean_dod = float(np.mean(d)) if d.size else float("nan")
        rec["mean_dod_m"] = mean_dod
        rows.append(rec)

        print(f"{lo:5.0f}-{hi:<6.0f} | {n:9,} | "
              + " | ".join(f"{dens[e]:12.2f}" for e in epochs)
              + " | " + " | ".join(f"{frac[e]:9.1f}" for e in epochs)
              + f" | {mean_dod:+9.3f}")

    # ------------------------------------------------------------ verdict
    print("\nHOW TO READ THIS")
    print("  Ground density should fall with canopy height for BOTH epochs --")
    print("  that is normal. The diagnostic is whether the two epochs DIVERGE.")
    print("  If 2024 ground density drops faster than 2020 as canopy thickens,")
    print("  AND mean DoD rises across the same bins, then part of the reported")
    print("  deposition is a ground-classification artifact, not sediment.")

    if len(rows) >= 2 and len(epochs) == 2:
        e0, e1 = epochs
        open_bin, closed_bin = rows[0], rows[-1]
        r_open = (open_bin[f"ground_density_{e1}"] /
                  open_bin[f"ground_density_{e0}"]
                  if open_bin[f"ground_density_{e0}"] else float("nan"))
        r_closed = (closed_bin[f"ground_density_{e1}"] /
                    closed_bin[f"ground_density_{e0}"]
                    if closed_bin[f"ground_density_{e0}"] else float("nan"))
        print(f"\n  {e1}/{e0} ground-density ratio, open canopy:   {r_open:.2f}")
        print(f"  {e1}/{e0} ground-density ratio, closed canopy: {r_closed:.2f}")
        print(f"  mean DoD, open canopy:   {open_bin['mean_dod_m']:+.3f} m")
        print(f"  mean DoD, closed canopy: {closed_bin['mean_dod_m']:+.3f} m")
        if np.isfinite(r_open) and np.isfinite(r_closed) and r_closed < 0.7 * r_open:
            print("\n  >> DIVERGENCE DETECTED. The 2024 classification recovers")
            print("     materially less ground under canopy. Treat forested")
            print("     deposition in the watershed DoD as suspect until the")
            print("     classifications are harmonised.")
        else:
            print("\n  >> No strong divergence. Classification is probably not")
            print("     the main driver; the LoD is then the prime suspect.")

    out = Path(cfg["OUT_JSON"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"config": {k: v for k, v in cfg.items()}, "bins": rows},
                  fh, indent=2)
    log(f"wrote {out}", run="density")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
