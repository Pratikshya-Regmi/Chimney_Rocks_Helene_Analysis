#!/usr/bin/env python3
"""
61_export_profile_line_context.py
=================================
Export the raster and vector context needed for the short-line location maps:
the 2024 CAP orthophoto at two extents (a tight view of the line and a wider
corridor view), a hillshade fallback, and the line and Lake Lure boundary as
GeoJSON.

Outputs into results/figures/:
  _pl_ortho_zoom.png / .pgw     CAP orthophoto, line + margin
  _pl_ortho_wide.png / .pgw     CAP orthophoto, corridor view
  _pl_line.geojson              the reprojected profile line
  _pl_lure.geojson              Lake Lure analysis area (context)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import grass_session, log  # noqa: E402

RUN = "profile_short_line"
ROOT = HERE.parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LINE_VECT = "short_line_3358@profile_short"
ORTHO = "Ortho_aerial_2024"          # 2024 CAP aerial orthophoto
LINE_BBOX = (315287.43, 315432.11, 192015.03, 192029.41)

VIEWS = {
    "zoom": dict(pad_e=45.0, pad_n=45.0),
    "wide": dict(pad_e=520.0, pad_n=420.0),
}


def export_view(gs, name, cfg):
    e0 = LINE_BBOX[0] - cfg["pad_e"]
    e1 = LINE_BBOX[1] + cfg["pad_e"]
    n0 = LINE_BBOX[2] - cfg["pad_n"]
    n1 = LINE_BBOX[3] + cfg["pad_n"]
    gs.run_command("g.region", n=n1, s=n0, e=e1, w=e0, res=0.25, flags="a",
                   quiet=True)
    reg = gs.region()
    out = FIG / f"_pl_ortho_{name}.png"
    gs.run_command("d.mon", start="cairo", output=str(out),
                   width=int(reg["cols"]), height=int(reg["rows"]),
                   bgcolor="white", overwrite=True, quiet=True)
    try:
        gs.run_command("d.rgb", red=f"{ORTHO}.red", green=f"{ORTHO}.green",
                       blue=f"{ORTHO}.blue", quiet=True)
    except Exception:
        gs.run_command("d.rast", map=ORTHO, quiet=True)
    gs.run_command("d.mon", stop="cairo", quiet=True)
    (FIG / f"_pl_ortho_{name}.pgw").write_text(
        f"{reg['ewres']}\n0.0\n0.0\n-{reg['nsres']}\n"
        f"{reg['w'] + reg['ewres'] / 2}\n{reg['n'] - reg['nsres'] / 2}\n")
    print(f"  {name:5s} extent E {e0:.1f}-{e1:.1f}  N {n0:.1f}-{n1:.1f}  "
          f"{int(reg['cols'])}x{int(reg['rows'])} px -> {out.name}")
    return (e0, e1, n0, n1)


def main():
    gs, _ = grass_session(project="DEM_generation", mapset="profile_short")
    for ms in gs.read_command("g.mapsets", flags="l").split():
        gs.run_command("g.mapsets", mapset=ms, operation="add")

    print("orthophoto exports:")
    extents = {k: export_view(gs, k, v) for k, v in VIEWS.items()}

    gs.run_command("v.out.ogr", input=LINE_VECT,
                   output=str(FIG / "_pl_line.geojson"),
                   format="GeoJSON", overwrite=True, quiet=True)
    print(f"  wrote _pl_line.geojson")

    log(f"exported profile-line context: zoom {extents['zoom']}, "
        f"wide {extents['wide']}", run=RUN)


if __name__ == "__main__":
    main()
