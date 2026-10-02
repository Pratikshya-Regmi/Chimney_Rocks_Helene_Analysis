#!/usr/bin/env python3
"""
00_inventory.py — audit both GRASS projects before any analysis.

Answers, without changing anything:
  - which mapsets exist in each project
  - which rasters and vectors are present, and in which mapset
  - what the computational region is
  - which pipeline prerequisites are MISSING (notably the 2020 DSM)
  - how many points are in each stable-point vector (manuscript claims n = 9)
  - whether the two Lake Lure workflows produce different DoDs

Run first. Read the output. It tells you what can and cannot proceed.

    python3 scripts/00_inventory.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grass_env import grass_session, log, PROJECTS  # noqa: E402

# Maps the pipeline needs, by role. Missing entries are reported, not created.
REQUIRED = {
    "DEM_generation": {
        "watershed pre-event DTM": ["dtm_2020_filled", "lidar_2020_regis",
                                     "lidar_2020_DTM"],
        "watershed post-event DTM (lidar)": ["dtm_2024_filled", "lidar_2024_regis"],
        "watershed post-event DTM (SfM)": ["cap_dtm_2024_filled", "cap_sfm_regis"],
        "watershed 2020 DSM  << canopy height depends on this": [
            "dsm_2020", "dsm_2020_filled", "lidar_2020_DSM"],
        "analysis extent": ["hull_mask"],
        "selected basin": ["basins_90"],
        "NDVI difference": ["ndvi_diff", "ndvi_difference", "dNDVI"],
    },
    "DEM_generation_lure": {
        "Lure pre-event DTM": ["lidar_2017_DTM_lure", "dtm_2017"],
        "Lure pre-event DSM": ["lidar_2017_DSM_lure"],
        "Lure post-event DTM (lidar)": ["lidar_2024_DTM_corr_lure",
                                         "lidar_2024_dtm_allpoints_filled_lure"],
        "Lure post-event DTM (SfM)": ["sfm_DTM_corr_lure", "sfm_dtm_filled_lure"],
        "Lure post-event DSM (lidar)": ["lidar_2024_DSM_corr_lure",
                                         "lidar_2024_dsm_filled_lure"],
        "analysis extent": ["boundary"],
        "Morse Park extent": ["morse_park_rast"],
    },
}

STABLE_POINT_VECTORS = {
    "DEM_generation": ["more_stable_pts", "reg_points"],
    "DEM_generation_lure": ["reg_points"],
}

# The two competing Lake Lure DoDs (see CLAUDE.md section 4, item 5)
COMPETING_LURE_DODS = {
    "DEM_generation": "Change_in_elevation_lidar_may_lure",
    "DEM_generation_lure": "Change_in_DTM_lidar_lure",
}


def inventory_project(project):
    print("\n" + "=" * 72)
    print(f"PROJECT: {project}")
    print("=" * 72)

    report = {"project": project, "mapsets": {}, "missing": [], "notes": []}

    gs, _ = grass_session(project=project, mapset="PERMANENT")

    mapsets = gs.read_command("g.mapsets", flags="l").split()
    print(f"\nMapsets: {', '.join(mapsets)}")

    # make every mapset readable so cross-mapset references resolve
    gs.run_command("g.mapsets", mapset=",".join(mapsets), operation="set")

    all_rasters, all_vectors = {}, {}
    for ms in mapsets:
        rasters = gs.read_command("g.list", type="raster", mapset=ms).split()
        vectors = gs.read_command("g.list", type="vector", mapset=ms).split()
        all_rasters[ms] = rasters
        all_vectors[ms] = vectors
        report["mapsets"][ms] = {"n_rasters": len(rasters),
                                 "n_vectors": len(vectors)}
        print(f"  {ms:<16s} {len(rasters):>4d} rasters, {len(vectors):>3d} vectors")

    flat_rasters = {r: ms for ms, rs in all_rasters.items() for r in rs}

    # ---------------------------------------------------------- requirements
    print("\n--- Pipeline prerequisites")
    for role, candidates in REQUIRED.get(project, {}).items():
        hit = next((c for c in candidates if c in flat_rasters), None)
        if hit:
            print(f"  OK      {role}")
            print(f"          -> {hit}@{flat_rasters[hit]}")
        else:
            print(f"  MISSING {role}")
            print(f"          looked for: {', '.join(candidates)}")
            report["missing"].append(role)

    # ---------------------------------------------------------- stable points
    print("\n--- Stable-point vectors (manuscript claims n = 9)")
    flat_vectors = {v: ms for ms, vs in all_vectors.items() for v in vs}
    for vec in STABLE_POINT_VECTORS.get(project, []):
        if vec not in flat_vectors:
            print(f"  {vec}: not found")
            continue
        info = gs.vector_info_topo(f"{vec}@{flat_vectors[vec]}")
        n = info.get("points", 0)
        flag = "" if n == 9 else "   <-- does NOT match the manuscript's n = 9"
        print(f"  {vec}@{flat_vectors[vec]}: {n} points{flag}")
        report.setdefault("stable_points", {})[vec] = n

    # ---------------------------------------------------------- competing DoDs
    dod = COMPETING_LURE_DODS.get(project)
    if dod and dod in flat_rasters:
        st = gs.parse_command("r.univar", map=f"{dod}@{flat_rasters[dod]}",
                              flags="ge", quiet=True)
        print(f"\n--- Lake Lure DoD in this project: {dod}")
        print(f"    n={int(st.get('n', 0)):,}  median={float(st.get('median', 0)):+.3f}  "
              f"mean={float(st.get('mean', 0)):+.3f}  sd={float(st.get('stddev', 0)):.3f}")
        print(f"    min={float(st.get('min', 0)):+.2f}  max={float(st.get('max', 0)):+.2f}")
        report["lure_dod"] = {"map": dod,
                              "median": float(st.get("median", 0)),
                              "sd": float(st.get("stddev", 0))}
        print("    -> Compare against the other project. If these differ, you must")
        print("       determine which produced the figures in the submitted PDF.")

    return report


def main():
    reports = []
    for project in PROJECTS.values():
        try:
            reports.append(inventory_project(project))
        except SystemExit:
            raise
        except Exception as exc:                      # noqa: BLE001
            print(f"\n!! Could not inventory {project}: {exc}")
            reports.append({"project": project, "error": str(exc)})

    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "inventory.json", "w") as fh:
        json.dump(reports, fh, indent=2)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in reports:
        miss = r.get("missing", [])
        print(f"\n{r['project']}: {len(miss)} missing prerequisite(s)")
        for m in miss:
            print(f"    - {m}")

    print("\nIf the 2020 DSM is missing, generate it before anything else:")
    print("  v.in.pdal input=<2020.laz> output=pc_2020_all   # no class_filter")
    print("  v.surf.rst input=pc_2020_all elevation=dsm_2020 \\")
    print("             tension=20 smooth=1 npmin=300 dmin=2 segmax=40")
    print("  r.fillnulls input=dsm_2020 output=dsm_2020_filled method=rst")
    print("\nCanopy height = dsm_2020_filled - dtm_2020_filled, and the spatially")
    print("variable Level of Detection depends on it. Without it the analysis")
    print("falls back to slope binning, which is materially weaker.")

    log("inventory complete", run="inventory")
    print("\nwrote results/tables/inventory.json")


if __name__ == "__main__":
    main()
