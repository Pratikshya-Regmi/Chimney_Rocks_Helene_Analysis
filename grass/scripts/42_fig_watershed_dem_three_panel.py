#!/usr/bin/env python3
"""
42_fig_watershed_dem_three_panel.py
====================================
Watershed DEM comparison figure (manuscript Figure 7), rebuilt from the
GRASS rasters rather than from the existing JPGs.

Panels, left to right:
    (a) 2020 lidar DTM
    (b) 2024 lidar DTM
    (c) 2024 CAP SfM DTM

WHICH RASTERS, AND WHY THE *_regis SET
    The co-registered surfaces are used:
        lidar_2020_regis, lidar_2024_regis, cap_sfm_regis   (@for_codem)
    NOT the raw dtm_*_filled set. Over the cat 34 basin the raw 2024 surface
    has a mean of 545.25 m against 576.48 m for the raw 2020 surface -- a
    ~31 m offset, the NAVD88 / NAD83-ellipsoid geoid separation for western
    North Carolina (see RESULTS_FOR_PAPER.md, 2026-09-08). Putting the raw
    surfaces on a shared elevation colour scale would render the 2024 panel
    a uniformly different colour for a datum reason, not a terrain one. The
    *_regis surfaces are in a common datum and are what the analysis uses.

ONE COMMON COLOUR SCALE
    The combined min and max are computed across all three rasters over the
    basin, a single colour rules file is written spanning exactly that range,
    and it is applied to all three with `r.colors rules=`. The same rules
    file is then parsed to build the matplotlib colormap, so the rendered
    colours are literally that table: identical colour = identical elevation
    in every panel.

    The rules are applied to basin-masked COPIES in this script's own mapset.
    The originals in for_codem keep their colour tables untouched (hard rule
    1: never modify a map a published figure depends on).

SHADED RELIEF
    r.relief with identical parameters (azimuth 270, altitude 30, zscale 1)
    for all three, computed on the UNMASKED source so the basin edge carries
    no illumination artifact, then masked. Blended multiplicatively and
    identically in every panel.

FIGURE FURNITURE (per the brief)
    - one shared colourbar, "Elevation [m]", 5 ticks, no per-panel legends
    - one scale bar and one north arrow, on panel (c) only
    - watershed boundary drawn in black on every panel
    - panel labels (a)/(b)/(c) below each map
    - map_style.py graticule + frame on all three, so the treatment matches
      the other map figures

OUTPUTS
    results/figures/fig_watershed_dem_three_panel.pdf   (vector, Type 42)
    results/figures/fig_watershed_dem_three_panel.png   (600 dpi)
    results/figures/dem_common_colors.txt               (the shared rules file)
    results/figures/_dem_panel_*.tif, _dem_relief_*.tif (GRASS exports)
    results/logs/fig_dem_panels_<date>.log
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import grass_session, log  # noqa: E402
from map_style import finalize_frame, add_graticule  # noqa: E402

RUN = "fig_dem_panels"
ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

WORK_MAPSET = "figs_dem"
BASIN_VECT = "basins_90_v_cat34@for_codem"

PANELS = [
    ("a", "2020 lidar DTM", "lidar_2020_regis@for_codem", "2020"),
    ("b", "2024 lidar DTM", "lidar_2024_regis@for_codem", "2024"),
    ("c", "2024 CAP SfM DTM", "cap_sfm_regis@for_codem", "sfm"),
]

RELIEF = dict(azimuth=270, altitude=30, zscale=1)

RULES_FILE = FIGURES / "dem_common_colors.txt"

# GRASS-elevation-style ramp, as fractions of the common range. Matches the
# green -> yellow -> orange -> brown -> grey progression of the existing
# figure so the rebuild is recognisably the same map.
RAMP = [
    (0.00, (0, 191, 191)),
    (0.15, (0, 255, 0)),
    (0.35, (255, 255, 0)),
    (0.55, (255, 127, 0)),
    (0.75, (191, 127, 63)),
    (0.90, (163, 140, 140)),
    (1.00, (255, 255, 255)),
]

MM = 1 / 25.4
FIG_W_MM = 174.0          # double column


def write_rules(vmin, vmax):
    """One colour rules file spanning exactly [vmin, vmax], in map units."""
    lines = []
    for frac, (r, g, b) in RAMP:
        val = vmin + frac * (vmax - vmin)
        lines.append(f"{val:.4f} {r}:{g}:{b}")
    # no "nv" rule: nulls are rendered fully transparent by shade_rgb()
    RULES_FILE.write_text("\n".join(lines) + "\n")
    return RULES_FILE


def cmap_from_rules(vmin, vmax):
    """Build the matplotlib colormap from the SAME ramp written to the rules
    file, so the figure's colours are the rules file's colours."""
    stops = [(frac, tuple(c / 255.0 for c in rgb)) for frac, rgb in RAMP]
    return LinearSegmentedColormap.from_list("dem_common", stops, N=1024)



def draw_scale_north_inline(ax, scale_m, mm_per_m, axes_w_mm, color="black",
                            fontsize=6.5):
    """Scale bar + north arrow on the figure's legend line, not inside a map.

    The bar must still be geometrically correct, so its length is derived
    from the maps' own scale: `mm_per_m` is the figure millimetres each map
    metre occupies (map column width / map width in metres), and the bar is
    drawn that many millimetres long, expressed as a fraction of this axes'
    width. Moving the bar out of the map frame therefore does not decouple
    it from the map scale.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    bar_frac = (scale_m * mm_per_m) / axes_w_mm
    x0, y0 = 0.02, 0.42
    x1 = x0 + bar_frac

    ax.plot([x0, x1], [y0, y0], color=color, linewidth=1.8,
            solid_capstyle="butt", zorder=5)
    for xt in (x0, x1):
        ax.plot([xt, xt], [y0 - 0.10, y0 + 0.10], color=color,
                linewidth=1.2, zorder=5)
    ax.text((x0 + x1) / 2, y0 + 0.17, f"{scale_m:,.0f} m", ha="center",
            va="bottom", fontsize=fontsize, color=color, zorder=5)

    xn = x1 + 0.20
    ax.annotate("", xy=(xn, y0 + 0.42), xytext=(xn, y0 - 0.28),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.2),
                zorder=5)
    ax.text(xn, y0 + 0.48, "N", ha="center", va="bottom", fontsize=fontsize + 1,
            fontweight="bold", color=color, zorder=5)



def basin_rings(path):
    """Exterior + interior rings of the basin polygon, as (xs, ys) arrays.

    Read from the exported GeoJSON rather than traced off the mask raster so
    the line is the true vector boundary, not a stair-stepped cell edge.
    """
    import json
    gj = json.loads(Path(path).read_text())
    rings = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        polys = ([geom["coordinates"]] if geom.get("type") == "Polygon"
                 else geom.get("coordinates", []))
        for poly in polys:
            for ring in poly:
                arr = np.asarray(ring, dtype=float)
                if arr.ndim == 2 and len(arr) > 2:
                    rings.append((arr[:, 0], arr[:, 1]))
    return rings


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        b = src.bounds
    return arr, (b.left, b.right, b.bottom, b.top)


def prepare(gs):
    """Mask each surface to the basin, apply the shared colour table, build
    identical shaded relief, and export everything to GeoTIFF."""
    import grass.script as g

    g.run_command("g.region", vector=BASIN_VECT, res=1, flags="a")
    reg = g.parse_command("g.region", flags="g")
    log(f"region: w={reg['w']} e={reg['e']} s={reg['s']} n={reg['n']} "
        f"rows={reg['rows']} cols={reg['cols']} res=1", run=RUN)

    # basin polygon -> mask raster (no outline is drawn; this only clips)
    g.run_command("v.to.rast", input=BASIN_VECT, output="fig_basin_mask",
                  use="value", value=1, type="area", overwrite=True, quiet=True)

    # --- mask each surface, and build relief on the UNMASKED source -------
    for _, _, src, tag in PANELS:
        g.mapcalc(f"fig_dem_{tag} = if(!isnull(fig_basin_mask), {src}, null())",
                  overwrite=True, quiet=True)
        g.run_command("r.relief", input=src, output=f"_fig_relief_raw_{tag}",
                      overwrite=True, quiet=True, **RELIEF)
        g.mapcalc(f"fig_relief_{tag} = if(!isnull(fig_basin_mask), "
                  f"_fig_relief_raw_{tag}, null())", overwrite=True, quiet=True)
        g.run_command("g.remove", type="raster", name=f"_fig_relief_raw_{tag}",
                      flags="f", quiet=True)

    # --- combined elevation range across all three ------------------------
    mins, maxs = [], []
    for _, label, _, tag in PANELS:
        u = g.parse_command("r.univar", map=f"fig_dem_{tag}", flags="g")
        mn, mx, n = float(u["min"]), float(u["max"]), int(u["n"])
        mins.append(mn)
        maxs.append(mx)
        log(f"{label} (fig_dem_{tag}): n={n} min={mn:.2f} max={mx:.2f}", run=RUN)
    vmin, vmax = min(mins), max(maxs)
    log(f"COMMON ELEVATION RANGE: {vmin:.2f} to {vmax:.2f} m "
        f"(span {vmax - vmin:.2f} m)", run=RUN)

    # --- one rules file, applied to all three -----------------------------
    rules = write_rules(vmin, vmax)
    for _, _, _, tag in PANELS:
        g.run_command("r.colors", map=f"fig_dem_{tag}", rules=str(rules),
                      quiet=True)
    log(f"applied {rules.name} to " +
        ", ".join(f"fig_dem_{t}" for *_, t in PANELS), run=RUN)

    # --- basin outline, for drawing in black on every panel ---------------
    bnd = FIGURES / "_basin_boundary.geojson"
    if bnd.exists():
        bnd.unlink()
    g.run_command("v.out.ogr", input=BASIN_VECT, output=str(bnd),
                  format="GeoJSON", type="area", quiet=True)
    log(f"exported basin outline -> {bnd.name}", run=RUN)

    # --- export -----------------------------------------------------------
    for _, _, _, tag in PANELS:
        for kind in ("dem", "relief"):
            g.run_command("r.out.gdal", input=f"fig_{kind}_{tag}",
                          output=str(FIGURES / f"_{kind}_panel_{tag}.tif"),
                          # Float64: the surfaces are DCELL, and Float32
                          # would lose precision on ~890 m elevations
                          format="GTiff", type="Float64",
                          createopt="COMPRESS=LZW", nodata=-9999,
                          overwrite=True, quiet=True)
    return vmin, vmax


def shade_rgb(elev, relief, cmap, norm):
    """Colour the DEM by the shared table, then darken by the shaded relief.

    Identical treatment in every panel. Returns an RGBA array with nulls
    fully transparent.
    """
    rgba = cmap(norm(np.ma.filled(elev, np.nan)))
    s = np.ma.filled(relief, np.nan) / 255.0
    s = np.clip(s, 0.0, 1.0)
    bright = 0.45 + 0.65 * s              # darken with relief, never crush
    bright = np.clip(bright, 0.0, 1.15)
    rgba[..., :3] = np.clip(rgba[..., :3] * bright[..., None], 0, 1)
    invalid = np.ma.getmaskarray(elev) | ~np.isfinite(np.ma.filled(elev, np.nan))
    rgba[..., 3] = np.where(invalid, 0.0, 1.0)
    return rgba


def main():
    matplotlib.rcParams["pdf.fonttype"] = 42     # Type 42 (TrueType)
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    gs, _ = grass_session(project="DEM_generation", mapset="PERMANENT")
    import grass.script as g
    try:
        g.run_command("g.mapset", flags="c", mapset=WORK_MAPSET, quiet=True)
    except Exception:                                    # noqa: BLE001
        g.run_command("g.mapset", mapset=WORK_MAPSET, quiet=True)
    log(f"START mapset={WORK_MAPSET} panels={[p[1] for p in PANELS]} "
        f"relief={RELIEF}", run=RUN)

    vmin, vmax = prepare(g)
    cmap = cmap_from_rules(vmin, vmax)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # ---------------- figure ---------------------------------------------
    # Every element is placed in explicit millimetres via fig.add_axes rather
    # than a GridSpec: the map axes are given exactly the rasters' aspect, so
    # aspect='equal' has nothing to shrink and leaves no dead band anywhere.
    panels = []
    for letter, title, _, tag in PANELS:
        elev, extent = read_raster(FIGURES / f"_dem_panel_{tag}.tif")
        relief, _ = read_raster(FIGURES / f"_relief_panel_{tag}.tif")
        panels.append((letter, title, elev, relief, extent))
    ext0 = panels[0][4]
    map_w_m = ext0[1] - ext0[0]
    data_aspect = (ext0[3] - ext0[2]) / map_w_m

    # ---- horizontal budget (mm) ----
    left_mm, right_mm = 9.5, 2.5
    col_gap_mm = 4.5
    usable_w = FIG_W_MM - left_mm - right_mm
    col_w = (usable_w - 2 * col_gap_mm) / 3.0
    map_h = col_w * data_aspect
    mm_per_m = col_w / map_w_m           # figure mm per map metre

    # ---- vertical budget (mm) ----
    top_mm = 2.0
    decor_mm = 13.5          # rotated easting labels + the (a)/(b)/(c) caption
    gap_mm = 9.0             # clear gap between the captions and the legend
    legend_h_mm = 4.5        # the colourbar bar itself
    legend_decor_mm = 9.5    # its tick labels + "Elevation [m]"
    bot_mm = 3.0
    fig_h_mm = (top_mm + map_h + decor_mm + gap_mm + legend_h_mm
                + legend_decor_mm + bot_mm)

    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h_mm * MM))

    def rect(x_mm, y_top_mm, w_mm, h_mm):
        """Axes rect from millimetres measured down from the figure top."""
        return [x_mm / FIG_W_MM, 1.0 - (y_top_mm + h_mm) / fig_h_mm,
                w_mm / FIG_W_MM, h_mm / fig_h_mm]

    rings = basin_rings(FIGURES / "_basin_boundary.geojson")

    im = None
    for i, (letter, title, elev, relief, extent) in enumerate(panels):
        x_mm = left_mm + i * (col_w + col_gap_mm)
        ax = fig.add_axes(rect(x_mm, top_mm, col_w, map_h))

        rgba = shade_rgb(elev, relief, cmap, norm)
        ax.imshow(rgba, extent=extent, origin="upper", zorder=2,
                  interpolation="nearest")
        # invisible mappable carrying the shared scale, for the one colourbar
        im = ax.imshow(np.ma.masked_all(elev.shape), cmap=cmap, norm=norm,
                       extent=extent, origin="upper", zorder=0)

        # watershed boundary, black -- drawn over the DEM, under the frame
        for xs, ys in rings:
            ax.plot(xs, ys, color="black", linewidth=0.8,
                    solid_joinstyle="round", zorder=6)

        # frame + graticule left at map_style's own weights, so this figure
        # matches the other map figures
        finalize_frame(ax, extent)
        add_graticule(ax, extent, step=250, fontsize=5.2)

        ax.set_xlabel(f"({letter}) {title}", fontsize=7.5, fontweight="bold",
                      labelpad=5)

    # ---- legend line: colourbar on the left, scale bar + north arrow right --
    legend_top = top_mm + map_h + decor_mm + gap_mm
    scale_group_w = 34.0
    inner_gap = 6.0
    cbar_w = usable_w - scale_group_w - inner_gap

    cax = fig.add_axes(rect(left_mm, legend_top, cbar_w, legend_h_mm))
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:,.0f}" for v in ticks])
    cbar.ax.tick_params(labelsize=6.5, length=2.5, width=0.5)
    cbar.set_label("Elevation [m]", fontsize=7.5)
    cbar.outline.set_linewidth(0.5)

    # same line as the legend, spanning the legend bar + its labels vertically
    sax = fig.add_axes(rect(left_mm + cbar_w + inner_gap, legend_top,
                            scale_group_w, legend_h_mm + legend_decor_mm))
    draw_scale_north_inline(sax, scale_m=250, mm_per_m=mm_per_m,
                            axes_w_mm=scale_group_w)

    out_pdf = FIGURES / "fig_watershed_dem_three_panel.pdf"
    out_png = FIGURES / "fig_watershed_dem_three_panel.png"
    fig.savefig(out_pdf)          # exact 174 mm width, no bbox trimming
    fig.savefig(out_png, dpi=600)
    plt.close(fig)

    print(f"\nCommon elevation range: {vmin:.2f} to {vmax:.2f} m "
          f"(span {vmax - vmin:.2f} m)")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    log(f"wrote {out_pdf.name} + .png at {FIG_W_MM} mm, 600 dpi, Type 42. "
        f"Common elevation range {vmin:.2f}-{vmax:.2f} m applied to all three "
        f"panels via {RULES_FILE.name}. One colourbar, one scale bar + north "
        f"arrow on the legend line (right of the colourbar), basin "
        f"boundary in black on every panel.", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
