#!/usr/bin/env python3
"""
19_dominant_features_inspection.py
======================================
Per-feature inspection of the top-8 erosion + top-8 deposition features at
both sites (watershed: D1-excluded ranking from
18_dominant_features_d1_excluded.py; Lake Lure: unchanged, from
13_dominant_features_both_sites.py -- D1 is watershed-only and does not
touch Lake Lure).

For each of the 32 features: canopy height and slope (mean over a 40x40 m
window centred on the feature's r.volume centroid -- a window, not the
exact clump footprint, since 32 features at full-footprint precision is
disproportionate; D1 got the full-footprint treatment because it was ~50%
of gross deposition, these are not), plus a site-appropriate flow/activity
context (watershed: flow accumulation max + discharge-simulation coverage
in the same window, using the same rasters as 16_resolve_d1_deposition.py;
Lake Lure: distance to lure_water_mask's shoreline, since the plausible
real mechanism there is shoreline/inflow reworking, not hillslope channel
flow -- Lake Lure has no r.sim.water discharge output for the lidar-lidar
pair).

Also regenerates the labelled orthophoto overview map for the watershed
(ranks shifted after D1's exclusion; old D2-D9 are new D1-D8) and for Lake
Lure (unchanged ranks, regenerated here as a real, reusable script instead
of the ad hoc plot the original map came from).

This is a SCREENING pass, not a D1-depth investigation of every feature --
flags are read together with the orthophoto overview map and the
identification already on record in dominant_features_20260827.log; any
feature that still looks ambiguous after this gets an individual zoom
crop (see results/diagnostics/).

Outputs:
  results/tables/dominant_features_inspection_metrics.csv
  results/figures/dominant_features_watershed_map_d1_excluded.png
  results/figures/dominant_features_lure_map.png (regenerated, same ranks)
  results/diagnostics/feature_zoom_<site>_<label>.png (ambiguous cases only)
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
DIAG = ROOT / "results" / "diagnostics"
for d in (TABLES, FIGURES, DIAG):
    d.mkdir(parents=True, exist_ok=True)

WINDOW_HALF_M = 20.0  # 40x40 m window for context sampling


def load_top8(path):
    rows = list(csv.DictReader(open(path)))
    ero = [r for r in rows if r["kind"] == "erosion"][:8]
    dep = [r for r in rows if r["kind"] == "deposition"][:8]
    return ero, dep


def sample_window(gs, e, n, half, rasters):
    gs.run_command("g.region", n=n + half, s=n - half, e=e + half, w=e - half,
                   res=1, flags="a")
    out = {}
    for name, rmap in rasters.items():
        try:
            u = gs.parse_command("r.univar", map=rmap, flags="g", quiet=True)
            out[name] = u
        except Exception:
            out[name] = None
    return out


def make_points_vector(gs, rows, outname):
    lines = []
    for i, r in enumerate(rows, start=1):
        lines.append(f"{r['easting']}|{r['northing']}|{i}")
    txt = "\n".join(lines)
    gs.write_command("v.in.ascii", input="-", output=outname, separator="pipe",
                     format="point", overwrite=True, quiet=True, stdin=txt)


def render_site_map(gs, gj, ero_rows, dep_rows, ortho_map, region_kwargs,
                    title, out_path, pad_m=60):
    gs.run_command("g.region", flags="a", **region_kwargs)
    reg = gs.parse_command("g.region", flags="g")
    gs.run_command("g.region", n=float(reg["n"]) + pad_m, s=float(reg["s"]) - pad_m,
                   e=float(reg["e"]) + pad_m, w=float(reg["w"]) - pad_m,
                   res=1, flags="a")

    make_points_vector(gs, ero_rows, "insp_ero_pts")
    make_points_vector(gs, dep_rows, "insp_dep_pts")

    img = gj.Map(width=1400, height=1000, use_region=True)
    img.d_rast(map=ortho_map)
    img.d_vect(map="insp_ero_pts", type="point", color="red", fill_color="red",
              icon="basic/circle", size=16)
    img.d_vect(map="insp_dep_pts", type="point", color="blue", fill_color="blue",
              icon="basic/circle", size=16)
    for i, r in enumerate(ero_rows, start=1):
        img.d_text(text=f"E{i}", at=f"{r['easting']},{r['northing']}", flags="g",
                  color="red", size=3)
    for i, r in enumerate(dep_rows, start=1):
        img.d_text(text=f"D{i}", at=f"{r['easting']},{r['northing']}", flags="g",
                  color="blue", size=3)
    img.d_text(text=title, at="2,97", color="black", bgcolor="white", size=2.5)
    img.save(str(out_path))
    gs.run_command("g.remove", type="vector", name="insp_ero_pts,insp_dep_pts",
                   flags="f", quiet=True)
    print(f"wrote {out_path}")


def inspect_site(gs, site, ero_rows, dep_rows, context_rasters):
    rows_out = []
    for kind, rows in (("erosion", ero_rows), ("deposition", dep_rows)):
        for i, r in enumerate(rows, start=1):
            e, n = float(r["easting"]), float(r["northing"])
            ctx = sample_window(gs, e, n, WINDOW_HALF_M, context_rasters)
            row = {
                "site": site, "kind": kind, "rank": i,
                "area_m2": r["area_m2"], "mean_depth_m": r["mean_depth_m"],
                "volume_m3": r["volume_m3"], "pct_of_gross": r["pct_of_gross"],
                "easting": e, "northing": n,
            }
            for cname, u in ctx.items():
                if u is None or u.get("n") in (None, "0"):
                    row[cname] = ""
                else:
                    row[cname] = u.get("mean", "")
                    if cname.endswith("_max_ctx"):
                        row[cname] = u.get("max", "")
            rows_out.append(row)
    return rows_out


def zoom_feature(gs, gj, e, n, ortho_map, out_path, half_m=40):
    gs.run_command("g.region", n=n + half_m, s=n - half_m, e=e + half_m,
                   w=e - half_m, res=1, flags="a")
    img = gj.Map(width=700, height=700, use_region=True)
    img.d_rast(map=ortho_map)
    img.save(str(out_path))
    print(f"wrote {out_path}")


def main():
    # ---------------------------------------------------------- watershed
    gs_w, gj_w = grass_session(project="DEM_generation", mapset="for_codem")
    ero_w, dep_w = load_top8(TABLES / "dominant_features_watershed_d1_excluded.csv")

    # d1_cap_ortho_rgb was built at a small window in
    # 16_resolve_d1_deposition.py; rebuild it over the full cat34 extent here
    gs_w.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")
    gs_w.run_command("g.region", n="n+60", s="s-60", e="e+60", w="w-60", flags="a")
    gs_w.run_command("r.composite",
                     red="A0069A_data_upload_flightA0069_ortho_full.1@PERMANENT",
                     green="A0069A_data_upload_flightA0069_ortho_full.2@PERMANENT",
                     blue="A0069A_data_upload_flightA0069_ortho_full.3@PERMANENT",
                     output="d1_cap_ortho_rgb", overwrite=True, quiet=True)
    render_site_map(gs_w, gj_w, ero_w, dep_w, "d1_cap_ortho_rgb",
                    {"vector": "basins_90_v_cat34@for_codem"},
                    "Watershed (cat34), D1 EXCLUDED: top-8 erosion (red, E) / "
                    "deposition (blue, D)",
                    FIGURES / "dominant_features_watershed_map_d1_excluded.png")

    gs_w.mapcalc("_insp_chan2020 = if(discharge_2020_regis > 0.01, 1, null())",
                overwrite=True, quiet=True)
    ctx_rasters_w = {
        "canopy_mean_m": "coreg_canopy",
        "slope_mean_deg": "coreg_slope",
        "flow_accum_max_ctx": "accumulation",
        "discharge_2020_coverage_n": "discharge_2020_regis",
    }
    rows_w = inspect_site(gs_w, "watershed", ero_w, dep_w, ctx_rasters_w)
    gs_w.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")
    gs_w.run_command("g.remove", type="raster", name="_insp_chan2020", flags="f", quiet=True)

    # ---- watershed zoom crops (must happen before the session switches to
    # Lake Lure below -- grass_session() repoints the one global GRASS
    # session, so gs_w/gj_w stop resolving watershed maps once gs_l exists)
    zoom_feature(gs_w, gj_w, 313220.5, 193419.5, "d1_cap_ortho_rgb",
                DIAG / "feature_zoom_watershed_D7.png")
    zoom_feature(gs_w, gj_w, 313204.5, 193474.5, "d1_cap_ortho_rgb",
                DIAG / "feature_zoom_watershed_D8.png")

    # ------------------------------------------------------------ Lake Lure
    gs_l, gj_l = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    ero_l, dep_l = load_top8(TABLES / "dominant_features_lure.csv")

    gs_l.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")
    gs_l.run_command("i.group", group="_insp_lure_ortho_grp",
                     input="Ortho_aerial_2024.1,Ortho_aerial_2024.2,Ortho_aerial_2024.3",
                     quiet=True)
    gs_l.run_command("r.composite",
                     red="Ortho_aerial_2024.1", green="Ortho_aerial_2024.2",
                     blue="Ortho_aerial_2024.3", output="_insp_lure_ortho_rgb",
                     overwrite=True, quiet=True)
    render_site_map(gs_l, gj_l, ero_l, dep_l, "_insp_lure_ortho_rgb",
                    {"vector": "boundary@PERMANENT"},
                    "Lake Lure: top-8 erosion (red, E) / deposition (blue, D)",
                    FIGURES / "dominant_features_lure_map.png")

    gs_l.run_command("r.grow.distance", input="lure_water_mask",
                     distance="_insp_water_dist", overwrite=True, quiet=True)
    ctx_rasters_l = {
        "canopy_mean_m": "coreg_lure_lidar_canopy",
        "slope_mean_deg": "coreg_lure_lidar_slope",
        "dist_to_water_m": "_insp_water_dist",
    }
    rows_l = inspect_site(gs_l, "lure", ero_l, dep_l, ctx_rasters_l)

    # ---- Lake Lure zoom crops on features the screening pass flagged as
    # ambiguous (moderate-tall canopy + far from water) -- see log for the
    # reasoning per feature
    zoom_feature(gs_l, gj_l, 317947.5, 191426.5, "_insp_lure_ortho_rgb",
                DIAG / "feature_zoom_lure_D7.png")
    zoom_feature(gs_l, gj_l, 316768.5, 191523.5, "_insp_lure_ortho_rgb",
                DIAG / "feature_zoom_lure_E5.png")
    zoom_feature(gs_l, gj_l, 317910, 192055, "_insp_lure_ortho_rgb",
                DIAG / "feature_zoom_lure_E6_E7_E8.png", half_m=90)

    gs_l.run_command("g.remove", type="raster", name="_insp_water_dist,_insp_lure_ortho_rgb",
                     flags="f", quiet=True)
    gs_l.run_command("g.remove", type="group", name="_insp_lure_ortho_grp", flags="f", quiet=True)

    # ------------------------------------------------------------ combine
    out_csv = TABLES / "dominant_features_inspection_metrics.csv"
    fieldnames = ["site", "kind", "rank", "area_m2", "mean_depth_m", "volume_m3",
                 "pct_of_gross", "easting", "northing", "canopy_mean_m",
                 "slope_mean_deg", "flow_accum_max_ctx", "discharge_2020_coverage_n",
                 "dist_to_water_m"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_w + rows_l:
            w.writerow(r)
    print(f"wrote {out_csv}")

    log("19_dominant_features_inspection.py run. Top-8 erosion + top-8 "
        "deposition, both sites (watershed D1-excluded), context-sampled "
        f"in a {2*WINDOW_HALF_M:.0f}x{2*WINDOW_HALF_M:.0f} m window per "
        "feature centroid: canopy height, slope, and (watershed) flow "
        "accumulation/discharge coverage or (Lake Lure) distance to "
        "lure_water_mask. Regenerated dominant_features_watershed_map_"
        "d1_excluded.png and dominant_features_lure_map.png. Zoom crops "
        "saved for watershed D7/D8 (moderate canopy, near-zero discharge "
        "coverage, upper-west area, previously 'not visually confirmed') "
        "and lure D7/E5/E6-E7-E8 (far-east/south features flagged in the "
        "2026-08-27 identification write-up). "
        f"Written to {out_csv.name}.", run="dominant_features_inspection")


if __name__ == "__main__":
    main()
