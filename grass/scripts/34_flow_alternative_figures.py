#!/usr/bin/env python3
"""
34_flow_alternative_figures.py
===================================
Three ALTERNATIVE flow-comparison figures, for the analyst to choose from
(NOT auto-inserted into the manuscript):

  (a) fig_flow_network_overlay      -- channel networks from all three
      epochs (discharge > 0.01, same threshold as the Jaccard metric),
      thinned to 1-cell-wide skeletons (r.thin) and overlaid as three
      coloured lines on one hillshade panel, basin-wide.

  (b) fig_flow_change_map           -- discharge_may_2024_lidar minus
      discharge_2020_regis (2024-lidar-minus-2020-lidar, the pair the
      Jaccard headline number is built on), diverging red=lost/blue=gained,
      with the 2024 SfM channel network (thinned, discharge > 0.01)
      overlaid as a black line for a visual check against the two-lidar
      change pattern.

  (c) fig_flow_longitudinal_profile -- discharge sampled along
      corridor_scar_line (the main erosional/scar corridor already
      established in the DoD corridor analysis -- used here as the
      dominant flow-concentration corridor's path; median discharge along
      it near the downstream end, ~0.09-0.10 in 2020, confirms it tracks
      real channelised flow, not diffuse hillslope) for all three epochs
      on one axis, distance from corridor head (E313445/N193008) to its
      downstream end (E313333/N192641).

Requires 30_flow_sequence_setup.py's GeoTIFF exports and GRASS session
access (re-run of thinning done fresh here, not reused from probing).

Outputs:
  results/figures/fig_flow_network_overlay.pdf / .png
  results/figures/fig_flow_change_map.pdf / .png
  results/figures/fig_flow_longitudinal_profile.pdf / .png
  results/tables/flow_longitudinal_profile.csv
  results/logs/flow_alternative_figures_<date>.log
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402
from map_style import style_map_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

DISCHARGE_2020_TIF = FIGURES / "_flow_discharge_2020.tif"
DISCHARGE_CAP_TIF = FIGURES / "_flow_discharge_2024cap.tif"
DISCHARGE_2024_TIF = FIGURES / "_flow_discharge_2024lidar.tif"
HILLSHADE_TIF = FIGURES / "_flow_hillshade.tif"

MM = 1 / 25.4
FIG_W_MM = 174
CHANNEL_THRESH = 0.01


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        b = src.bounds
        extent = (b.left, b.right, b.bottom, b.top)
    return arr, extent


def main():
    if not DISCHARGE_2020_TIF.exists():
        sys.exit("Run 30_flow_sequence_setup.py first.")

    gs, gj = grass_session(project="DEM_generation", mapset="for_codem")
    gs.run_command("g.region", raster="discharge_2020_regis", flags="a")

    hs, hs_extent = read_raster(HILLSHADE_TIF)

    # ================================================================
    # (a) channel network overlay
    # ================================================================
    epochs = [
        ("2020 lidar", "discharge_2020_regis", "_ovl_2020", "#1b9e77"),
        ("2024 CAP SfM", "discharge_2024_cap", "_ovl_cap", "#d95f02"),
        ("2024 lidar", "discharge_may_2024_lidar", "_ovl_2024", "#7570b3"),
    ]
    thinned_tifs = {}
    for label, rmap, base, color in epochs:
        gs.mapcalc(f"{base} = if({rmap} > {CHANNEL_THRESH}, 1, null())",
                  overwrite=True, quiet=True)
        gs.run_command("r.thin", input=base, output=f"{base}_thin",
                       overwrite=True, quiet=True)
        tif = FIGURES / f"_{base}_thin.tif"
        gs.run_command("r.out.gdal", input=f"{base}_thin", output=str(tif),
                       format="GTiff", type="Byte", nodata=0, overwrite=True,
                       quiet=True)
        thinned_tifs[label] = (tif, color)

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM, 130 * MM))
    ax.imshow(hs, cmap="gray", extent=hs_extent, origin="upper", zorder=1)
    for label, (tif, color) in thinned_tifs.items():
        arr, extent = read_raster(tif)
        rgba = np.zeros((*arr.shape, 4))
        c = matplotlib.colors.to_rgba(color)
        mask = (~arr.mask) & (arr.filled(0) > 0)
        rgba[mask] = c
        rgba[..., 3] = mask.astype(float) * 0.9
        ax.imshow(rgba, extent=extent, origin="upper", zorder=3, interpolation="none")
        ax.plot([], [], color=color, linewidth=2, label=label)  # legend proxy
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85,
             title=f"Channel network (discharge > {CHANNEL_THRESH})",
             title_fontsize=7)
    style_map_panel(ax, hs_extent, graticule_step=500, scale_color="white")
    out_a_pdf = FIGURES / "fig_flow_network_overlay.pdf"
    out_a_png = FIGURES / "fig_flow_network_overlay.png"
    fig.savefig(out_a_pdf, bbox_inches="tight")
    fig.savefig(out_a_png, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"(a) wrote {out_a_pdf} and .png")

    # ================================================================
    # (b) flow-change map: 2024 lidar minus 2020 lidar, SfM network overlay
    # ================================================================
    gs.mapcalc("_flow_change = discharge_may_2024_lidar - discharge_2020_regis",
              overwrite=True, quiet=True)
    change_tif = FIGURES / "_flow_change.tif"
    gs.run_command("r.out.gdal", input="_flow_change", output=str(change_tif),
                   format="GTiff", type="Float64", nodata=-9999,
                   overwrite=True, quiet=True)
    change, change_extent = read_raster(change_tif)
    change_stats = gs.parse_command("r.univar", map="_flow_change", flags="ge")
    vlim = 3 * float(change_stats["stddev"])

    sfm_tif, sfm_color = thinned_tifs["2024 CAP SfM"]
    sfm_arr, sfm_extent = read_raster(sfm_tif)

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM, 130 * MM))
    ax.imshow(hs, cmap="gray", extent=hs_extent, origin="upper", zorder=1)
    im = ax.imshow(change, cmap="RdBu", vmin=-vlim, vmax=vlim,
                   extent=change_extent, origin="upper", alpha=0.85, zorder=2)
    sfm_mask = (~sfm_arr.mask) & (sfm_arr.filled(0) > 0)
    rgba = np.zeros((*sfm_arr.shape, 4))
    rgba[sfm_mask] = (0, 0, 0, 1)
    ax.imshow(rgba, extent=sfm_extent, origin="upper", zorder=3, interpolation="none")
    ax.plot([], [], color="black", linewidth=1.3,
           label=f"2024 SfM channel (discharge > {CHANNEL_THRESH})")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
    style_map_panel(ax, hs_extent, cax=None, mappable=im,
                    cbar_label="Discharge change, 2024 lidar $-$ 2020 lidar",
                    cbar_units="m$^3$ s$^{-1}$ m$^{-1}$-equivalent",
                    graticule_step=500, scale_color="white")
    out_b_pdf = FIGURES / "fig_flow_change_map.pdf"
    out_b_png = FIGURES / "fig_flow_change_map.png"
    fig.savefig(out_b_pdf, bbox_inches="tight")
    fig.savefig(out_b_png, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"(b) wrote {out_b_pdf} and .png (colour limit +/-{vlim:.3f}, 3 SD)")

    # ================================================================
    # (c) longitudinal discharge profile along corridor_scar_line
    # ================================================================
    gs.run_command("v.to.points", input="corridor_scar_line",
                   output="flowprof_pts", dmax=2, type="line",
                   overwrite=True, quiet=True)
    gs.run_command("v.db.addcolumn", map="flowprof_pts", layer=2,
                   columns="disch2020 double precision, "
                           "disch2024cap double precision, "
                           "disch2024lidar double precision", quiet=True)
    gs.run_command("v.what.rast", map="flowprof_pts", layer=2,
                   raster="discharge_2020_regis", column="disch2020", quiet=True)
    gs.run_command("v.what.rast", map="flowprof_pts", layer=2,
                   raster="discharge_2024_cap", column="disch2024cap", quiet=True)
    gs.run_command("v.what.rast", map="flowprof_pts", layer=2,
                   raster="discharge_may_2024_lidar", column="disch2024lidar",
                   quiet=True)
    out = gs.read_command("v.db.select", map="flowprof_pts", layer=2,
                          columns="along,disch2020,disch2024cap,disch2024lidar",
                          separator="comma")
    rows = [r.split(",") for r in out.strip().split("\n")[1:]]
    dist = np.array([float(r[0]) for r in rows])
    d2020 = np.array([float(r[1]) if r[1] not in ("", "*") else np.nan for r in rows])
    dcap = np.array([float(r[2]) if r[2] not in ("", "*") else np.nan for r in rows])
    d2024 = np.array([float(r[3]) if r[3] not in ("", "*") else np.nan for r in rows])

    with open(TABLES / "flow_longitudinal_profile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance_m", "discharge_2020_lidar", "discharge_2024_cap_sfm",
                   "discharge_2024_lidar"])
        for i in range(len(dist)):
            w.writerow([f"{dist[i]:.2f}", f"{d2020[i]:.6g}", f"{dcap[i]:.6g}",
                       f"{d2024[i]:.6g}"])

    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM, 70 * MM))
    ax.plot(dist, d2020, color="#1b9e77", linewidth=1.1, label="2020 lidar (pre-event)")
    ax.plot(dist, dcap, color="#d95f02", linewidth=1.1,
           label="2024 CAP SfM (7 Oct, +10 d)")
    ax.plot(dist, d2024, color="#7570b3", linewidth=1.1,
           label="2024 lidar (15-16 Nov, +7 wk)")
    ax.set_yscale("log")
    ax.set_xlabel("Distance along main scar corridor, from head [m]", fontsize=8)
    ax.set_ylabel("Unit discharge\n[m$^3$ s$^{-1}$ m$^{-1}$-equivalent]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.3, linewidth=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    out_c_pdf = FIGURES / "fig_flow_longitudinal_profile.pdf"
    out_c_png = FIGURES / "fig_flow_longitudinal_profile.png"
    fig.tight_layout()
    fig.savefig(out_c_pdf, bbox_inches="tight")
    fig.savefig(out_c_png, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"(c) wrote {out_c_pdf} and .png, "
          f"{TABLES / 'flow_longitudinal_profile.csv'}")

    gs.run_command("g.remove", type="raster",
                   name="_ovl_2020,_ovl_2020_thin,_ovl_cap,_ovl_cap_thin,"
                        "_ovl_2024,_ovl_2024_thin,_flow_change",
                   flags="f", quiet=True)
    gs.run_command("g.remove", type="vector", name="flowprof_pts", flags="f",
                   quiet=True)

    log("34_flow_alternative_figures.py run. (a) channel network overlay "
        f"(thinned, discharge>{CHANNEL_THRESH}) for all 3 epochs on one "
        "hillshade. (b) discharge_may_2024_lidar - discharge_2020_regis "
        f"(3 SD colour limit +/-{vlim:.3f}) with 2024 SfM channel overlaid. "
        "(c) discharge sampled every 2 m along corridor_scar_line "
        f"({len(dist)} points, {dist.max():.0f} m) for all 3 epochs, "
        "log y-axis. All three are ALTERNATIVES for the analyst to choose "
        "from -- none inserted into the manuscript.",
        run="flow_alternative_figures")


if __name__ == "__main__":
    main()
