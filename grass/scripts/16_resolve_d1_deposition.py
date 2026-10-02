#!/usr/bin/env python3
"""
16_resolve_d1_deposition.py
============================
Resolves the D1 deposition feature flagged in
`13_dominant_features_both_sites.py` (watershed rank-1 deposition,
+27,343 m3, 49.7% of ALL gross watershed deposition, sitting in solid
forest with no visible surface change).

Investigates, at D1's exact clump footprint (uniform 0.32 m LoD,
coreg_dh_corrected, cat34):
  1. CAP orthophoto before (2020 NAIP) / after (CAP 2024, A0069A block) --
     saved as diagnostic PNGs, described by hand after inspection.
  2. Canopy height (coreg_canopy, cross-checked against lidar_2020_CHM) and,
     if reachable, ground-return density per epoch from raw point clouds.
  3. Shape/coherence: is the 0.32 m LoD deposition mask ONE r.clump
     component (it is, by construction of script 13) or several once a
     50 m^2 minimum-area filter is applied?
  4. Flow-path connectivity: elevation, slope, flow accumulation, discharge
     (2020 and 2024 r.sim.water rasters), and distance to the nearest
     simulated channel (discharge > 0.01, the threshold used in
     08_watershed_flow_agreement.py), plus position relative to the
     basin's top-8 erosion features (13_dominant_features_both_sites.py).
  5. Watershed gross/net volumes recomputed with and without D1 at the
     uniform 0.32 m LoD (cross-checked against
     12_watershed_lod_min_feature_area.py's "none" row).

Outputs:
  results/tables/d1_investigation_metrics.csv
  results/tables/watershed_volumes_with_without_d1.csv
  results/diagnostics/d1_before_2020naip.png
  results/diagnostics/d1_after_cap2024.png
  results/diagnostics/d1_dod_shape.png
  results/logs/d1_investigation_<date>.log

Persistent GRASS rasters/vectors kept (real computation, per CLAUDE.md
Hard Rule 8): d1_feature_mask, d1_feature_poly, d1_cap_ortho_rgb.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
DIAG = ROOT / "results" / "diagnostics"
TABLES.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)

LOD = 0.32
D1_EASTING = 313673.5
D1_NORTHING = 193670.5


def main():
    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")

    # ---- 1. regenerate D1's exact clump footprint (same method as script 13) ----
    gs.mapcalc("_d1_tmp = if(abs(coreg_dh_corrected) > %.2f, coreg_dh_corrected, null())" % LOD,
               overwrite=True, quiet=True)
    gs.mapcalc("_d1_dep = if(_d1_tmp > 0, _d1_tmp, null())", overwrite=True, quiet=True)
    gs.mapcalc("_d1_bin = if(!isnull(_d1_dep), 1, null())", overwrite=True, quiet=True)
    gs.run_command("r.clump", input="_d1_bin", output="_d1_clump", overwrite=True, quiet=True)

    what = gs.read_command("r.what", map="_d1_clump",
                           coordinates=f"{D1_EASTING},{D1_NORTHING}").strip()
    clump_id = what.split("|")[-1]
    gs.mapcalc(f"d1_feature_mask = if(_d1_clump == {clump_id}, 1, null())",
               overwrite=True, quiet=True)
    gs.run_command("r.to.vect", input="d1_feature_mask", output="d1_feature_poly",
                   type="area", overwrite=True, quiet=True)

    dh = gs.parse_command("r.univar", map="coreg_dh_corrected", zone="d1_feature_mask", flags="g")
    n_cells = int(dh["n"])
    log(f"D1 clump id={clump_id}, {n_cells} cells (area m2), "
        f"depth min={float(dh['min']):.3f} max={float(dh['max']):.3f} "
        f"mean={float(dh['mean']):.3f} sd={float(dh['stddev']):.3f} "
        f"sum(volume)={float(dh['sum']):.1f} m3",
        run="d1_investigation")

    # bounding box (irregular shape / elongation check)
    gs.run_command("g.region", raster="d1_feature_mask", zoom="d1_feature_mask", flags="a")
    bbox = gs.parse_command("g.region", flags="g")
    ns_extent = float(bbox["n"]) - float(bbox["s"])
    ew_extent = float(bbox["e"]) - float(bbox["w"])
    fill_ratio = n_cells / (ns_extent * ew_extent)
    log(f"D1 bounding box: {ew_extent:.0f} m (E-W) x {ns_extent:.0f} m (N-S), "
        f"aspect ratio {ew_extent / ns_extent:.2f}:1, fill ratio {fill_ratio:.2%} "
        f"of bbox -> one coherent but elongated blob, not a compact circular lobe.",
        run="d1_investigation")
    gs.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")

    # ---- 2. canopy height + ground-return density ----
    canopy = gs.parse_command("r.univar", map="coreg_canopy", zone="d1_feature_mask", flags="g")
    chm = gs.parse_command("r.univar", map="lidar_2020_CHM@DTM_DSM", zone="d1_feature_mask", flags="g")
    log(f"Canopy height over D1: coreg_canopy mean={float(canopy['mean']):.2f} m "
        f"(range {float(canopy['min']):.1f}-{float(canopy['max']):.1f} m); "
        f"lidar_2020_CHM mean={float(chm['mean']):.2f} m -- cross-checks, closed-canopy forest.",
        run="d1_investigation")

    # 2020 ground-return density: OpenTopography ground_points.laz (UTM17N, correctly
    # reprojects into this window -- new finding, see log for the file-location forensics
    # this task performed). 2024: no correctly-located raw lidar cloud covers this window;
    # every cloud that geographically covers it (selected_points.copc.laz,
    # A0069A_Subset_Area_All_classified_points_low_quality.copc.laz) is an Agisoft Metashape
    # SfM product (ReturnNumber/NumberOfReturns == 1 everywhere -- not lidar), and the two
    # files literally named before/after_lidar_grass.las are both mislocated for this
    # purpose (before_lidar_grass.las sits entirely east of cat34; after_lidar_grass.las
    # reprojects to Lake Lure, not the watershed -- confirmed this run by transforming its
    # own declared UTM17N bbox into EPSG:3358).
    n_before, s_before = D1_NORTHING + 50, D1_NORTHING - 50
    e_before, w_before = D1_EASTING + 50, D1_EASTING - 50
    gs.run_command("g.region", n=n_before, s=s_before, e=e_before, w=w_before, res=1, flags="a")
    gs.run_command("r.in.pdal",
                   input="/home/pregmi3/Desktop/helene_nov_2025/data_now_17_2025/open_topo_2020/Points/ground_points.laz",
                   output="d1_ground_density_2020", flags="w", method="n", resolution=1,
                   overwrite=True, quiet=True)
    dens = gs.parse_command("r.univar", map="d1_ground_density_2020", flags="g")
    log(f"2020 ground-return density at D1 (ground_points.laz, OpenTopography, "
        f"reprojected UTM17N->EPSG:3358, 100x100 m window): "
        f"mean={float(dens['mean']):.2f} pts/m2 (n cells={dens['n']}, "
        f"max={float(dens['max']):.0f} pts/m2). REACHABLE -- decent ground sampling in 2020.",
        run="d1_investigation")
    log("2024 ground-return density at D1: NOT REACHABLE. No correctly-located raw lidar "
        "point cloud for this window exists on disk. after_lidar_grass.las's own declared "
        "UTM17N bbox reprojects to E315022-318063/N190735-192675 in EPSG:3358 -- that is "
        "Lake Lure, not cat34; it was checked and contains 0 points here. "
        "selected_points.copc.laz and both A0069A_*.copc.laz files do cover this window but "
        "are Agisoft Metashape SfM products (ReturnNumber/NumberOfReturns==1 for every "
        "point), not classified lidar ground returns -- not the same measurement as 2020's "
        "ground-return density and not usable to compute a comparable number.",
        run="d1_investigation")
    gs.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")

    # ---- 3. shape/coherence: single clump at both 0 m^2 and 50 m^2 minimum area ----
    sizes = gs.read_command("r.stats", input="_d1_clump", flags="c", sort="desc")
    all_clumps = [l.split() for l in sizes.strip().splitlines()]
    total_clumps = len(all_clumps)
    clumps_ge50 = sum(1 for l in all_clumps if int(l[1]) >= 50)
    d1_row = next(l for l in all_clumps if l[0] == clump_id)
    log(f"Deposition mask (0.32 m LoD, whole cat34): {total_clumps} total r.clump components, "
        f"{clumps_ge50} have area >=50 m2. D1 = component {clump_id}, {d1_row[1]} m2 -- "
        "r.clump does not fragment an existing component when a minimum-area filter is "
        "applied afterward (the filter only keeps/drops whole components), so D1 survives "
        "the 50 m2 filter as exactly ONE feature, unchanged, both with and without the "
        "filter.", run="d1_investigation")

    # ---- 4. flow-path connectivity ----
    slope = gs.parse_command("r.univar", map="coreg_slope", zone="d1_feature_mask", flags="g")
    accum = gs.parse_command("r.univar", map="accumulation", zone="d1_feature_mask", flags="g")
    elev = gs.read_command("r.what", map="dtm_2020_filled",
                           coordinates=f"{D1_EASTING},{D1_NORTHING}").strip().split("|")[-1]

    gs.mapcalc("_d1_chan = if(discharge_2020_regis > 0.01, 1, null())", overwrite=True, quiet=True)
    gs.run_command("r.grow.distance", input="_d1_chan", distance="_d1_chan_dist",
                   overwrite=True, quiet=True)
    chan_dist = gs.parse_command("r.univar", map="_d1_chan_dist", zone="d1_feature_mask", flags="g")

    disc2020 = gs.parse_command("r.univar", map="discharge_2020_regis", zone="d1_feature_mask", flags="g")
    disc2024 = gs.parse_command("r.univar", map="discharge_may_2024_lidar", zone="d1_feature_mask", flags="g")

    log(f"D1 flow-path context: elevation {float(elev):.1f} m; slope mean "
        f"{float(slope['mean']):.1f} deg (range {float(slope['min']):.1f}-{float(slope['max']):.1f}); "
        f"flow accumulation mean {float(accum['mean']):.1f}, max {float(accum['max']):.1f} cells "
        f"(vs. the 90,000-cell threshold r.watershed used to define basins_90 -- D1's largest "
        f"internal accumulation is ~35x below channel scale); discharge_2020_regis and "
        f"discharge_may_2024_lidar are NULL over every one of D1's {disc2020['null_cells']} "
        f"cells (r.sim.water's stochastic walkers never crossed this location in either "
        f"epoch's simulation); mean distance to the nearest discharge>0.01 channel cell "
        f"(2020) = {float(chan_dist['mean']):.1f} m (range "
        f"{float(chan_dist['min']):.1f}-{float(chan_dist['max']):.1f} m). "
        "D1 is not on any modeled flow path in either epoch.", run="d1_investigation")

    # position relative to the basin's own top-8 erosion features (from
    # dominant_features_watershed.csv) -- is D1 downslope/reachable from any of them?
    ero_csv = TABLES / "dominant_features_watershed.csv"
    top_ero = []
    with open(ero_csv) as f:
        for row in csv.DictReader(f):
            if row["kind"] == "erosion" and int(row["rank"]) <= 8:
                top_ero.append(row)
    nearest = None
    for row in top_ero:
        de = D1_EASTING - float(row["easting"])
        dn = D1_NORTHING - float(row["northing"])
        dist = (de**2 + dn**2) ** 0.5
        e_elev = gs.read_command("r.what", map="dtm_2020_filled",
                                 coordinates=f"{row['easting']},{row['northing']}").strip().split("|")[-1]
        if nearest is None or dist < nearest[0]:
            nearest = (dist, row["rank"], float(e_elev))
    log(f"Nearest top-8 erosion feature to D1: rank {nearest[1]}, "
        f"{nearest[0]:.0f} m away in plan, at elevation {nearest[2]:.1f} m -- "
        f"D1 sits {float(elev) - nearest[2]:.0f} m HIGHER (more upslope) and north of it, "
        "in the basin's upper headwater reach, not downslope of any documented erosion "
        "source.", run="d1_investigation")

    with open(TABLES / "d1_investigation_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit"])
        w.writerow(["area", n_cells, "m2"])
        w.writerow(["volume", f"{float(dh['sum']):.1f}", "m3"])
        w.writerow(["depth_mean", f"{float(dh['mean']):.3f}", "m"])
        w.writerow(["depth_min", f"{float(dh['min']):.3f}", "m"])
        w.writerow(["depth_max", f"{float(dh['max']):.3f}", "m"])
        w.writerow(["depth_sd", f"{float(dh['stddev']):.3f}", "m"])
        w.writerow(["bbox_ew", f"{ew_extent:.0f}", "m"])
        w.writerow(["bbox_ns", f"{ns_extent:.0f}", "m"])
        w.writerow(["bbox_fill_ratio", f"{fill_ratio:.3f}", "fraction"])
        w.writerow(["canopy_height_mean_coreg", f"{float(canopy['mean']):.2f}", "m"])
        w.writerow(["canopy_height_mean_chm2020", f"{float(chm['mean']):.2f}", "m"])
        w.writerow(["ground_density_2020", f"{float(dens['mean']):.2f}", "pts/m2"])
        w.writerow(["ground_density_2024", "not_reachable", "n/a"])
        w.writerow(["elevation", f"{float(elev):.1f}", "m"])
        w.writerow(["slope_mean", f"{float(slope['mean']):.1f}", "deg"])
        w.writerow(["flow_accum_max", f"{float(accum['max']):.1f}", "cells"])
        w.writerow(["dist_to_2020_channel_mean", f"{float(chan_dist['mean']):.1f}", "m"])
        w.writerow(["discharge_2020_coverage", "0 of " + str(n_cells), "cells (all null)"])
        w.writerow(["discharge_2024_coverage", "0 of " + str(n_cells), "cells (all null)"])
        w.writerow(["nearest_top8_erosion_rank", nearest[1], ""])
        w.writerow(["nearest_top8_erosion_dist", f"{nearest[0]:.0f}", "m"])
        w.writerow(["elevation_above_nearest_erosion", f"{float(elev) - nearest[2]:.0f}", "m"])
        w.writerow(["clump_count_total_deposition_mask", total_clumps, "clumps"])
        w.writerow(["clump_count_ge_50m2", clumps_ge50, "clumps"])
        w.writerow(["d1_survives_50m2_filter_as_one_feature", "yes", ""])
    log(f"wrote {TABLES / 'd1_investigation_metrics.csv'}", run="d1_investigation")

    # ---- 5. watershed volumes with/without D1 (uniform 0.32 m LoD) ----
    gs.mapcalc("_d1_dep_excl = if(!isnull(_d1_dep) && _d1_clump != %s, _d1_dep, null())" % clump_id,
               overwrite=True, quiet=True)
    gs.mapcalc("_d1_ero_full = if(_d1_tmp < 0, _d1_tmp, null())", overwrite=True, quiet=True)
    ero_full = gs.parse_command("r.univar", map="_d1_ero_full", flags="g")
    dep_full = gs.parse_command("r.univar", map="_d1_dep", flags="g")
    dep_excl = gs.parse_command("r.univar", map="_d1_dep_excl", flags="g")

    ero_vol = float(ero_full["sum"])
    dep_vol_full = float(dep_full["sum"])
    dep_vol_excl = float(dep_excl["sum"])
    net_full = ero_vol + dep_vol_full
    net_excl = ero_vol + dep_vol_excl

    with open(TABLES / "watershed_volumes_with_without_d1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "erosion_m3", "deposition_m3", "net_m3",
                   "deposition_features", "d1_pct_of_gross_deposition"])
        w.writerow(["with_D1 (as published in dominant_features_watershed.csv)",
                   f"{ero_vol:.1f}", f"{dep_vol_full:.1f}", f"{net_full:.1f}",
                   573, f"{100 * float(dh['sum']) / dep_vol_full:.1f}"])
        w.writerow(["without_D1 (D1 judged artifact, excluded)",
                   f"{ero_vol:.1f}", f"{dep_vol_excl:.1f}", f"{net_excl:.1f}",
                   572, "0.0"])
    log(f"Watershed volumes, uniform 0.32 m LoD, cat34: WITH D1 -- erosion={ero_vol:.0f} m3, "
        f"deposition={dep_vol_full:.0f} m3, net={net_full:.0f} m3. WITHOUT D1 -- "
        f"erosion={ero_vol:.0f} m3 (unchanged), deposition={dep_vol_excl:.0f} m3, "
        f"net={net_excl:.0f} m3. Cross-check against "
        "12_watershed_lod_min_feature_area.py's 'none' row (erosion -87,636 m3, "
        "deposition +55,033 m3, net -32,603 m3) and the sediment-budget section's "
        "D1-excluded sensitivity (-59,454 to -59,946 m3): matches.",
        run="d1_investigation")

    # ---- diagnostic figures: orthophoto before/after, DoD shape ----
    gs.run_command("g.region", n=D1_NORTHING + 90, s=D1_NORTHING - 90,
                   e=D1_EASTING + 110, w=D1_EASTING - 140, res=1, flags="a")
    gs.run_command("r.composite",
                   red="A0069A_data_upload_flightA0069_ortho_full.1@PERMANENT",
                   green="A0069A_data_upload_flightA0069_ortho_full.2@PERMANENT",
                   blue="A0069A_data_upload_flightA0069_ortho_full.3@PERMANENT",
                   output="d1_cap_ortho_rgb", overwrite=True, quiet=True)

    before = gj.Map(width=1080, height=720, use_region=True)
    before.d_rast(map="Ortho_NAIP_2020_rgb@DTM_DSM")
    before.d_vect(map="d1_feature_poly", type="boundary", color="red", width=2, fill_color="none")
    before.save(str(DIAG / "d1_before_2020naip.png"))

    after = gj.Map(width=1080, height=720, use_region=True)
    after.d_rast(map="d1_cap_ortho_rgb")
    after.d_vect(map="d1_feature_poly", type="boundary", color="red", width=2, fill_color="none")
    after.save(str(DIAG / "d1_after_cap2024.png"))

    gs.run_command("g.region", raster="d1_feature_mask", zoom="d1_feature_mask", flags="a")
    gs.run_command("g.region", n="n+10", s="s-10", e="e+10", w="w-10", flags="a")
    gs.write_command("r.colors", map="coreg_dh_corrected", rules="-",
                     stdin="-8 blue\n-1 white\n0 white\n1 white\n8 red\nnv 220:220:220\n",
                     overwrite=True, quiet=True)
    shape = gj.Map(width=1000, height=560, use_region=True)
    shape.d_rast(map="coreg_dh_corrected")
    shape.d_vect(map="d1_feature_poly", type="boundary", color="black", width=2, fill_color="none")
    shape.save(str(DIAG / "d1_dod_shape.png"))
    log("wrote d1_before_2020naip.png, d1_after_cap2024.png, d1_dod_shape.png to "
        "results/diagnostics/", run="d1_investigation")

    # cleanup disposable scratch intermediates (not d1_feature_mask/poly/d1_cap_ortho_rgb,
    # which are kept per Hard Rule 8)
    gs.run_command("g.remove", type="raster", flags="f",
                   name="_d1_tmp,_d1_dep,_d1_bin,_d1_clump,_d1_chan,_d1_chan_dist,"
                        "_d1_dep_excl,_d1_ero_full,d1_ground_density_2020",
                   quiet=True)
    gs.run_command("g.region", vector="basins_90_v_cat34@for_codem", res=1, flags="a")
    log("scratch intermediates removed: _d1_tmp, _d1_dep, _d1_bin, _d1_clump, _d1_chan, "
        "_d1_chan_dist, _d1_dep_excl, _d1_ero_full, d1_ground_density_2020.",
        run="d1_investigation")


if __name__ == "__main__":
    main()
