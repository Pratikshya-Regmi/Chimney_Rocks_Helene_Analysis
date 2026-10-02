#!/usr/bin/env python3
"""
50_export_ndvi_figure_data.py
==============================
Export everything the rebuilt NDVI-change figure needs. The inputs live in
two different GRASS projects, so each is exported to GeoTIFF/GeoJSON in a
common CRS and composited in matplotlib afterwards.

    ndvi_change            helene_chimney_11_25 / test_data_extent  (8.6 m)
    basins_90_v_cat34      DEM_generation / for_codem
    lure_boundary_in_wshed_prj  DEM_generation / for_codem
    NAIP 2024 RGB          DEM_generation / DTM_DSM

Both projects are NAD83(HARN) / North Carolina (their PROJ_INFO differ only
by a null towgs84 line), so coordinates are directly comparable and no
reprojection is applied.

Outputs -> results/figures/_ndvi_*.tif|.png|.geojson
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import grass_session, log  # noqa: E402

RUN = "ndvi_fig_data"
ROOT = HERE.parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# combined extent of the two analysis areas, plus a margin for context
PAD = 250.0
W, E = 313173.0 - PAD, 318040.0 + PAD
S, N = 190948.0 - PAD, 193687.0 + PAD


def main():
    # ---- NDVI change, from the chimney project -------------------------
    gs, _ = grass_session(project="helene_chimney_11_25",
                          mapset="test_data_extent",
                          gisdbase="/home/pregmi3/Desktop/helene_nov_2025")
    import grass.script as g
    ms = g.read_command("g.mapsets", flags="l").split()
    g.run_command("g.mapsets", mapset=",".join(ms), operation="set")
    g.run_command("g.region", w=W, e=E, s=S, n=N, res=8.6, flags="a")
    g.run_command("r.out.gdal", input="ndvi_change", output=str(FIG / "_ndvi_change.tif"),
                  format="GTiff", type="Float32", nodata=-9999,
                  createopt="COMPRESS=LZW", overwrite=True, quiet=True)
    log(f"exported ndvi_change over {W:.0f}-{E:.0f} x {S:.0f}-{N:.0f}", run=RUN)

    # ---- boundaries + imagery, from the DEM project --------------------
    gs, gj = grass_session(project="DEM_generation", mapset="DTM_DSM")
    import grass.script as g
    ms = g.read_command("g.mapsets", flags="l").split()
    g.run_command("g.mapsets", mapset=",".join(ms), operation="set")
    for vec, out in (("basins_90_v_cat34@for_codem", "_ndvi_bnd_watershed.geojson"),
                     ("lure_boundary_in_wshed_prj@for_codem", "_ndvi_bnd_lure.geojson")):
        p = FIG / out
        if p.exists():
            p.unlink()
        g.run_command("v.out.ogr", input=vec, output=str(p), format="GeoJSON",
                      type="area", quiet=True)
        log(f"exported {vec} -> {out}", run=RUN)

    g.run_command("g.region", w=W, e=E, s=S, n=N, res=1.0, flags="a")
    # which 2024 imagery actually covers the whole strip?
    best, best_n = None, -1
    for cand in ("Ortho_aerial_2024_rgb", "Ortho_NAIP_2024_1_rgb",
                 "Ortho_NAIP_2024_2_rgb", "ortho_rgb_2024"):
        if not g.find_file(name=cand, element="cell")["name"]:
            continue
        u = g.parse_command("r.univar", map=cand, flags="g")
        n = int(u["n"]) if u and "n" in u else 0
        log(f"imagery candidate {cand}: {n} cells in strip", run=RUN)
        if n > best_n:
            best, best_n = cand, n
    # patch the two NAIP tiles, then fall back to the best single raster
    g.mapcalc("ndvi_ctx_raw = if(isnull(Ortho_NAIP_2024_1_rgb), "
              "Ortho_NAIP_2024_2_rgb, Ortho_NAIP_2024_1_rgb)",
              overwrite=True, quiet=True)
    nn = int(g.parse_command("r.univar", map="ndvi_ctx_raw", flags="g")["n"])
    if nn < best_n:
        g.mapcalc(f"ndvi_ctx_raw = {best}", overwrite=True, quiet=True)
        src = best
    else:
        src = "NAIP 2024 (tiles 1+2 patched)"
    g.run_command("r.colors", map="ndvi_ctx_raw",
                  raster="Ortho_NAIP_2024_1_rgb", quiet=True)
    total = g.parse_command("g.region", flags="g")
    log(f"context imagery = {src}: {max(nn, best_n)} of {total['cells']} cells",
        run=RUN)
    print(f"context imagery: {src}, {max(nn,best_n)} of {total['cells']} cells "
          f"({100*max(nn,best_n)/int(total['cells']):.1f}%)")

    m = gj.Map(use_region=True, filename=str(FIG / "_ndvi_context.png"), width=2600)
    m.d_rast(map="ndvi_ctx_raw")
    m.show()
    g.run_command("g.remove", type="raster", name="ndvi_ctx_raw", flags="f",
                  quiet=True)
    print(f"extent W{W:.0f} E{E:.0f} S{S:.0f} N{N:.0f}")
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
