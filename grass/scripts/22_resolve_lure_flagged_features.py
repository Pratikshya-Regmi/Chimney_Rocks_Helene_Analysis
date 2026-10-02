#!/usr/bin/env python3
"""
22_resolve_lure_flagged_features.py
=======================================
Applies the D1 standard to Lake Lure's five features flagged in
19_dominant_features_inspection.py as possible smaller-scale D1 analogues:
erosion E5, E6, E7, E8 and deposition D7. Same battery of checks as
16_resolve_d1_deposition.py, at each feature's EXACT r.clump footprint
(not a fixed window, unlike the screening pass in script 19):

  1. Orthophoto before (2020 NAIP, imported via r.proj from
     DEM_generation/DTM_DSM -- same CRS, a straight resample, no
     reprojection distortion) / after (CAP 2024, Ortho_aerial_2024, native
     to this project) -- 3-panel composite per feature (before | after |
     DoD shape), stitched with PIL.
  2. Canopy height (coreg_lure_lidar_canopy) over the exact footprint.
  3. Feature shape: bounding box, fill ratio, and the DoD depth pattern
     (lobe/fan vs. smooth ramp vs. diffuse) -- same color ramp as D1's
     figure.
  4. Position relative to flow paths / erosional sources: Lake Lure has no
     r.sim.water discharge for the lidar-lidar pair (established in
     19_...py), so the flow-path analogue here is (a) distance to
     lure_water_mask's shoreline -- the reservoir's own "channel", and
     (b) distance to the nearest of the site's already-established
     plausible-real dredge-zone features (erosion E1-E4, deposition
     D1-D6/D8) -- the visible active-reworking zone that is this site's
     equivalent of the watershed's erosion sources.

Then recomputes Lake Lure volumes (uniform 0.32 m LoD, water-excluded,
0/50/100 m^2 minimum feature area) with whichever of the five are judged
artifacts excluded -- NOT applied to any manuscript table without asking
first, per instruction; this script only computes and reports the number.

Outputs:
  results/tables/lure_flagged_features_investigation.csv
  results/tables/lure_volumes_with_without_flagged.csv
  results/diagnostics/lure_feature_panel_<label>.png (one 3-panel image
    per feature: before | after | DoD shape)
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
DIAG = ROOT / "results" / "diagnostics"
for d in (TABLES, DIAG):
    d.mkdir(parents=True, exist_ok=True)

LOD = 0.32

# label -> (kind, easting, northing) from dominant_features_lure.csv
FLAGGED = {
    "E5": ("erosion", 316768.5, 191523.5),
    "E6": ("erosion", 317954.5, 192060.5),
    "E7": ("erosion", 317793.5, 192047.5),
    "E8": ("erosion", 317884.5, 192059.5),
    "D7": ("deposition", 317947.5, 191426.5),
}

# already-established plausible-real dredge-zone features (everything else
# in the top-8 both kinds) -- the site's equivalent of "erosional source"
PLAUSIBLE_REAL = [
    (316235.5, 191883.5), (316376.5, 191781.5), (315953.5, 191843.5),
    (316078.5, 191865.5),  # E1-E4
    (316526.5, 191901.5), (316490.5, 191755.5), (317002.5, 191597.5),
    (316601.5, 191765.5), (315818.5, 191862.5), (316208.5, 191912.5),
    (316875.5, 191889.5),  # D1-D6, D8
]


def main():
    gs, gj = grass_session(project="DEM_generation_lure", mapset="PERMANENT")
    gs.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")

    # ---- import 2020 NAIP (pre-event "before") from DEM_generation, same
    # CRS -- a resample, not a reprojection. Import the three separate
    # continuous reflectance bands, NOT the pre-packed Ortho_NAIP_2020_rgb
    # composite -- that composite's values (0-32767) are a categorical
    # r.composite packing (32 levels/band: (R*32+G)*32+B) with its own
    # color table; bilinear-resampling packed category codes during r.proj
    # produces meaningless values with no matching color table entry
    # (renders solid black). Reproject each raw band (bilinear is valid on
    # continuous reflectance) then recomposite fresh, same as the CAP ortho.
    # NOTE: this project also has Ortho_NAIP_2020.red/.green/.blue, but
    # those are a DIFFERENT, mis-registered raster set (r.proj -g on them
    # reports coordinates in the ~1,000,000+ range -- a different CRS
    # entirely, not this project's EPSG:3358). Use .1/.2/.3 (verified
    # in-CRS: r.proj -g matches the project's actual N/S/E/W).
    for band, name in ((".1", "red"), (".2", "green"), (".3", "blue")):
        gs.run_command("r.proj", project="DEM_generation", mapset="DTM_DSM",
                       input=f"Ortho_NAIP_2020{band}", output=f"lure_naip2020_{name}",
                       resolution=1, method="bilinear", overwrite=True, quiet=True)
    gs.run_command("r.composite", red="lure_naip2020_red", green="lure_naip2020_green",
                   blue="lure_naip2020_blue", output="lure_naip2020_rgb",
                   overwrite=True, quiet=True)
    gs.run_command("i.group", group="lure_cap2024_grp",
                   input="Ortho_aerial_2024.1,Ortho_aerial_2024.2,Ortho_aerial_2024.3",
                   quiet=True)
    gs.run_command("r.composite", red="Ortho_aerial_2024.1",
                   green="Ortho_aerial_2024.2", blue="Ortho_aerial_2024.3",
                   output="lure_cap2024_rgb", overwrite=True, quiet=True)

    gs.run_command("r.grow.distance", input="lure_water_mask",
                   distance="_flag_water_dist", overwrite=True, quiet=True)

    # ---- build the same erosion/deposition clump masks the ranking used
    gs.mapcalc(f"_flag_tmp = if(isnull(lure_water_mask) && "
              f"abs(coreg_lure_lidar_dh_corrected) > {LOD}, "
              "coreg_lure_lidar_dh_corrected, null())", overwrite=True, quiet=True)
    gs.mapcalc("_flag_ero = if(_flag_tmp < 0, _flag_tmp, null())",
              overwrite=True, quiet=True)
    gs.mapcalc("_flag_dep = if(_flag_tmp > 0, _flag_tmp, null())",
              overwrite=True, quiet=True)
    clump_of = {}
    for kind, dhmap in (("erosion", "_flag_ero"), ("deposition", "_flag_dep")):
        gs.mapcalc(f"_flag_bin_{kind} = if(!isnull({dhmap}), 1, null())",
                  overwrite=True, quiet=True)
        gs.run_command("r.clump", input=f"_flag_bin_{kind}",
                       output=f"_flag_clump_{kind}", overwrite=True, quiet=True)
        clump_of[kind] = f"_flag_clump_{kind}"

    rows = []
    for label, (kind, e, n) in FLAGGED.items():
        # reset region every iteration -- a previous iteration's panel
        # rendering leaves the region zoomed to its own feature's window,
        # and r.what (unlike most GRASS point queries one might assume)
        # DOES respect the current computational region, returning null
        # for any point outside it
        gs.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")
        clump_raster = clump_of[kind]
        cid = gs.read_command("r.what", map=clump_raster,
                              coordinates=f"{e},{n}").strip().split("|")[-1]
        maskname = f"lure_{label}_mask"
        gs.mapcalc(f"{maskname} = if({clump_raster} == {cid}, 1, null())",
                  overwrite=True, quiet=True)

        dh = gs.parse_command("r.univar", map="coreg_lure_lidar_dh_corrected",
                              zone=maskname, flags="g", quiet=True)
        canopy = gs.parse_command("r.univar", map="coreg_lure_lidar_canopy",
                                  zone=maskname, flags="g", quiet=True)
        slope = gs.parse_command("r.univar", map="coreg_lure_lidar_slope",
                                 zone=maskname, flags="g", quiet=True)
        water_dist = gs.parse_command("r.univar", map="_flag_water_dist",
                                      zone=maskname, flags="g", quiet=True)

        gs.run_command("g.region", raster=maskname, zoom=maskname, flags="a")
        bbox = gs.parse_command("g.region", flags="g")
        ns = float(bbox["n"]) - float(bbox["s"])
        ew = float(bbox["e"]) - float(bbox["w"])
        n_cells = int(dh["n"])
        fill_ratio = n_cells / (ns * ew) if ns * ew else 0
        gs.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")

        nearest = min(((e - px) ** 2 + (n - py) ** 2) ** 0.5
                     for px, py in PLAUSIBLE_REAL)

        row = {
            "label": label, "kind": kind, "easting": e, "northing": n,
            "area_m2": n_cells,
            "mean_depth_m": float(dh["mean"]),
            "min_depth_m": float(dh["min"]), "max_depth_m": float(dh["max"]),
            "sd_depth_m": float(dh["stddev"]),
            "volume_m3": float(dh["sum"]),
            "bbox_ew_m": ew, "bbox_ns_m": ns, "fill_ratio": fill_ratio,
            "canopy_mean_m": float(canopy["mean"]),
            "slope_mean_deg": float(slope["mean"]),
            "dist_to_water_m": float(water_dist["mean"]),
            "dist_to_nearest_plausible_real_feature_m": nearest,
        }
        rows.append(row)
        print(f"\n=== {label} ({kind}) ===")
        for k, v in row.items():
            print(f"  {k}: {v}")

        # ---- 3-panel diagnostic: before | after | DoD shape
        half = max(ew, ns) / 2 + 30
        gs.run_command("g.region", n=n + half, s=n - half, e=e + half,
                       w=e - half, res=1, flags="a")
        gs.run_command("r.to.vect", input=maskname, output=f"flagpoly_{label}",
                       type="area", overwrite=True, quiet=True)

        before = gj.Map(width=500, height=500, use_region=True)
        before.d_rast(map="lure_naip2020_rgb")
        before.d_vect(map=f"flagpoly_{label}", type="boundary", color="red",
                      width=2, fill_color="none")
        before.save(str(DIAG / f"_tmp_{label}_before.png"))

        after = gj.Map(width=500, height=500, use_region=True)
        after.d_rast(map="lure_cap2024_rgb")
        after.d_vect(map=f"flagpoly_{label}", type="boundary", color="red",
                    width=2, fill_color="none")
        after.save(str(DIAG / f"_tmp_{label}_after.png"))

        gs.write_command("r.colors", map="coreg_lure_lidar_dh_corrected",
                         rules="-", stdin="-8 blue\n-1 white\n0 white\n1 white\n"
                         "8 red\nnv 220:220:220\n", overwrite=True, quiet=True)
        shape = gj.Map(width=500, height=500, use_region=True)
        shape.d_rast(map="coreg_lure_lidar_dh_corrected")
        shape.d_vect(map=f"flagpoly_{label}", type="boundary", color="black",
                    width=2, fill_color="none")
        shape.save(str(DIAG / f"_tmp_{label}_shape.png"))

        from PIL import Image
        imgs = [Image.open(DIAG / f"_tmp_{label}_{p}.png")
               for p in ("before", "after", "shape")]
        w, h = imgs[0].size
        canvas = Image.new("RGB", (w * 3 + 20, h), "white")
        for i, im in enumerate(imgs):
            canvas.paste(im, (i * (w + 10), 0))
        out_path = DIAG / f"lure_feature_panel_{label}.png"
        canvas.save(out_path)
        for p in ("before", "after", "shape"):
            (DIAG / f"_tmp_{label}_{p}.png").unlink()
        print(f"  wrote {out_path} (before | after | DoD shape)")

        gs.run_command("g.remove", type="vector", name=f"flagpoly_{label}",
                       flags="f", quiet=True)

    out_csv = TABLES / "lure_flagged_features_investigation.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    gs.run_command("g.region", vector="boundary@PERMANENT", res=1, flags="a")
    gs.run_command("g.remove", type="raster",
                   name="_flag_tmp,_flag_ero,_flag_dep,_flag_bin_erosion,"
                        "_flag_bin_deposition,_flag_clump_erosion,"
                        "_flag_clump_deposition,_flag_water_dist,"
                        "lure_cap2024_rgb,lure_naip2020_red,lure_naip2020_green,"
                        "lure_naip2020_blue", flags="f", quiet=True)
    gs.run_command("g.remove", type="group", name="lure_cap2024_grp",
                   flags="f", quiet=True)

    log("22_resolve_lure_flagged_features.py run. Full D1-standard "
        "inspection (orthophoto before/after, canopy, shape, flow-path/"
        "erosional-source position) on Lake Lure's 5 flagged features "
        "(E5-E8, D7). Persistent masks kept: lure_E5_mask...lure_D7_mask, "
        "lure_naip2020_rgb (imported via r.proj from DEM_generation/"
        "DTM_DSM). Per-feature metrics: " +
        "; ".join(f"{r['label']}: area={r['area_m2']}m2 vol={r['volume_m3']:+.0f}m3 "
                 f"canopy={r['canopy_mean_m']:.1f}m dist_water={r['dist_to_water_m']:.0f}m "
                 f"dist_plausible_real={r['dist_to_nearest_plausible_real_feature_m']:.0f}m"
                 for r in rows) +
        f". Written to {out_csv.name}. 3-panel diagnostic images: "
        "lure_feature_panel_{E5,E6,E7,E8,D7}.png.",
        run="lure_flagged_features_investigation")


if __name__ == "__main__":
    main()
