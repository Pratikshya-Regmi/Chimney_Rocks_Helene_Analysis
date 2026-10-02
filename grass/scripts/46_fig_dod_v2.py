#!/usr/bin/env python3
"""
46_fig_dod_v2.py
=================
Figure 10 (fig:dem_difference) rebuilt as THREE panels:
    (a) pre-Helene NAIP imagery
    (b) post-Helene NAIP imagery
    (c) DoD, lidar 2020 vs lidar 2024

All three clipped to the cat 34 watershed boundary, on ONE common extent and
scale. One north arrow, one scale bar, and the elevation-change legend --
which applies only to panel (c) -- share a single line beneath panel (c).

IMAGERY PROVENANCE (checked, not assumed)
    A full NAIP series exists in DEM_generation/DTM_DSM: 2018, 2020, 2022,
    2024. NAIP is normally flown leaf-on in summer, so "NAIP 2024" is not
    automatically post-event -- it was verified by rendering the basin from
    each year: NAIP 2022 is unbroken forest canopy, NAIP 2024 shows the bare
    landslide scars and debris corridors. NAIP 2024 is therefore genuinely
    post-Helene.

    PRE  = Ortho_NAIP_2020_rgb  -- chosen to match the DoD's own pre-epoch
           (the DoD is lidar 2020 vs 2024), so panels (a) and (c) refer to
           the same baseline year. Ortho_NAIP_2022_rgb is also available and
           is closer to the event; switching is a one-line change (PRE_MAP).
    POST = Ortho_NAIP_2024_1_rgb patched with Ortho_NAIP_2024_2_rgb. Tile _1
           alone leaves a no-data stripe down the east side of the basin;
           the two together cover 607,715 of 608,841 cells (99.8%).

    The `.red`/`.green`/`.blue` component rasters of these composites are
    empty in this region -- only the `_rgb` composites carry data -- so the
    imagery is rendered through GRASS's own display (which honours the
    composite colour table) rather than by stacking bands.

CLIPPING
    Each panel's image artist is clipped to the basin polygon with a
    matplotlib clip path, so the cut follows the true vector boundary rather
    than a stair-stepped raster mask.

LAYOUT VARIANTS (300 dpi PNG + vector PDF)
    a  three equal panels in a row
    b  the two imagery panels stacked left, DoD larger on the right
    c  a row of three with the DoD panel ~1.55x wider than each imagery panel

    The basin is 547 x 1108 m, aspect ~1.9, which drives these: in a row of
    three the panel width is fixed at a third of the page whatever the
    height, so (b) is the only way to make any panel materially bigger, and
    it costs total width (the figure ends up ~152 mm of the 174 mm column).

Outputs: results/diagnostics/fig_dod_v2_{a,b,c}.png / .pdf / .tex
         results/logs/fig_dod_v2_<date>.log
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from PIL import Image
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import grass_session, log                      # noqa: E402
from map_style import (finalize_frame, add_graticule,          # noqa: E402
                       add_scale_north_inline, add_north_scale)
from make_transect_figure_v2 import (                          # noqa: E402
    export_map_rasters, HILLSHADE_TIF, DOD_TIF, C_EROSION, C_DEPOSITION, LOD,
)

RUN = "fig_dod_v2"
ROOT = HERE.parents[1]
DIAG = ROOT / "results" / "diagnostics"
TABLES = ROOT / "results" / "tables"
DIAG.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM = 174
MARGIN_M = 40                     # around the basin, same on every panel
PRE_MAP = "Ortho_NAIP_2020_rgb@DTM_DSM"
POST_1 = "Ortho_NAIP_2024_1_rgb@DTM_DSM"
POST_2 = "Ortho_NAIP_2024_2_rgb@DTM_DSM"
PRE_PNG = DIAG / "_naip_pre.png"
POST_PNG = DIAG / "_naip_post.png"

TITLES = ["(a) Pre-Helene NAIP (2020)", "(b) Post-Helene NAIP (2024)",
          "(c) DoD: lidar 2020 vs 2024"]


def basin_ring():
    import pandas as pd
    b = pd.read_csv(TABLES / "cat34_boundary.csv").sort_values("vertex_order")
    return np.column_stack([b["easting"].to_numpy(), b["northing"].to_numpy()])


def tight_extent(ring):
    return (ring[:, 0].min() - MARGIN_M, ring[:, 0].max() + MARGIN_M,
            ring[:, 1].min() - MARGIN_M, ring[:, 1].max() + MARGIN_M)


def render_imagery(extent):
    """Render both NAIP epochs over the common extent via GRASS display."""
    if PRE_PNG.exists() and POST_PNG.exists():
        log("reusing rendered NAIP panels", run=RUN)
        return
    gs, gj = grass_session(project="DEM_generation", mapset="DTM_DSM")
    import grass.script as g
    ms = g.read_command("g.mapsets", flags="l").split()
    g.run_command("g.mapsets", mapset=",".join(ms), operation="set")
    g.run_command("g.region", w=extent[0], e=extent[1], s=extent[2],
                  n=extent[3], res=0.6, flags="a")
    reg = g.parse_command("g.region", flags="g")
    log(f"imagery region {reg['w']}..{reg['e']} x {reg['s']}..{reg['n']} "
        f"res={reg['nsres']} ({reg['cols']}x{reg['rows']})", run=RUN)

    # Patch the two 2024 NAIP tiles: tile _1 alone leaves a no-data stripe.
    # The two do not quite abut -- 4,010 cells (a ~2-cell column) are null in
    # BOTH, which renders as a black line down the mosaic. r.grow expands the
    # surrounding imagery across that gap; it is a seam artifact of the tile
    # boundary, not missing survey data.
    g.mapcalc(f"_naip_post_raw = if(isnull({POST_1}), {POST_2}, {POST_1})",
              overwrite=True, quiet=True)
    gap = int(float(g.parse_command(
        "r.univar", map="_naip_post_raw", flags="g")["null_cells"]))
    g.run_command("r.grow", input="_naip_post_raw", output="_naip_post_grown",
                  radius=3.01, overwrite=True, quiet=True)
    g.mapcalc("_naip_post = if(isnull(_naip_post_raw), _naip_post_grown, "
              "_naip_post_raw)", overwrite=True, quiet=True)
    left = int(float(g.parse_command(
        "r.univar", map="_naip_post", flags="g")["null_cells"]))
    log(f"NAIP 2024 mosaic seam: {gap} cells null in both tiles, "
        f"{left} still null after r.grow", run=RUN)
    g.run_command("r.colors", map="_naip_post", raster=POST_1, quiet=True)
    for src, out in ((PRE_MAP, PRE_PNG), ("_naip_post", POST_PNG)):
        m = gj.Map(use_region=True, filename=str(out), width=1400)
        m.d_rast(map=src)
        m.show()
        log(f"rendered {src} -> {out.name}", run=RUN)
    g.run_command("g.remove", type="raster",
                  name="_naip_post,_naip_post_raw,_naip_post_grown",
                  flags="f", quiet=True)


def read_tif(path):
    with rasterio.open(path) as s:
        arr = s.read(1, masked=True)
        b = s.bounds
    return arr, (b.left, b.right, b.bottom, b.top)


def clip_to_basin(artist, ax, ring):
    artist.set_clip_path(PathPatch(MplPath(ring), transform=ax.transData))


def panel_imagery(ax, png, extent, ring):
    img = np.asarray(Image.open(png).convert("RGB"))
    art = ax.imshow(img, extent=extent, origin="upper", zorder=2,
                    interpolation="bilinear")
    clip_to_basin(art, ax, ring)
    ax.plot(ring[:, 0], ring[:, 1], color="black", linewidth=0.9, zorder=6)


def panel_dod(ax, ring, extent):
    hs, hs_ext = read_tif(HILLSHADE_TIF)
    dod, dod_ext = read_tif(DOD_TIF)
    a1 = ax.imshow(hs, cmap="gray", vmin=0, vmax=255, extent=hs_ext,
                   origin="upper", zorder=1)
    cmap = LinearSegmentedColormap.from_list(
        "erosion_deposition", [C_EROSION, "#f7f7f7", C_DEPOSITION])
    cmap.set_bad(color=(0, 0, 0, 0))
    a2 = ax.imshow(dod, extent=dod_ext, origin="upper", cmap=cmap,
                   norm=Normalize(vmin=-3, vmax=3), zorder=3, alpha=0.92)
    for art in (a1, a2):
        clip_to_basin(art, ax, ring)
    ax.plot(ring[:, 0], ring[:, 1], color="black", linewidth=0.9, zorder=6)
    return a2


def style(ax, extent, xlabels=True, ylabels=True):
    """All panels share one extent, so one set of coordinate labels suffices;
    the rest keep their graticule lines and tick marks but drop the text."""
    finalize_frame(ax, extent)
    add_graticule(ax, extent, step=250, fontsize=5.0)
    if not xlabels:
        ax.set_xticklabels([])
    if not ylabels:
        ax.set_yticklabels([])


def legend_line(fig, rect_mm, mappable, extent, panel_w_mm, fig_w, fig_h):
    """Colourbar + one scale bar + one north arrow, on a single line."""
    x_mm, y_mm, w_mm, h_mm = rect_mm
    SCALE_FRAC, GAP = 0.38, 0.04
    cb_w = w_mm * (1 - SCALE_FRAC - GAP)
    cax = fig.add_axes([x_mm / fig_w, y_mm / fig_h, cb_w / fig_w, h_mm / fig_h])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_ticks([-3, -1.5, 0, 1.5, 3])
    cbar.ax.tick_params(labelsize=6.0, length=2.5, width=0.5)
    cbar.set_label("Elevation change [m]  (erosion red, deposition blue)",
                   fontsize=6.6)
    cbar.outline.set_linewidth(0.5)

    sx = x_mm + cb_w + w_mm * GAP
    sw = w_mm * SCALE_FRAC
    sax = fig.add_axes([sx / fig_w, (y_mm - h_mm * 1.0) / fig_h,
                        sw / fig_w, (h_mm * 3.0) / fig_h])

    # Pick the longest round bar that still leaves room for the north arrow
    # beside it. A fixed 200 m bar was WIDER than this axes (16.4 mm of a
    # 15.4 mm slot), so it overflowed and pushed the arrow off the axes,
    # where it was clipped and simply did not appear.
    mm_per_m = panel_w_mm / (extent[1] - extent[0])
    scale_m = next((c for c in (500, 200, 100, 50, 25)
                    if c * mm_per_m <= 0.52 * sw), 25)
    add_scale_north_inline(sax, scale_m=scale_m, mm_per_m=mm_per_m,
                           axes_w_mm=sw, color="black")
    return cax, sax, scale_m


def build(variant, ring, extent):
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    left, right, gap = 10.0, 3.0, 4.0
    top, xlab, legend_h, bot = 4.0, 12.0, 4.5, 14.0

    if variant == "a":
        pw = (FIG_W_MM - left - right - 2 * gap) / 3.0
        widths = [pw, pw, pw]
        ph = pw * aspect
        fig_h = top + ph + xlab + legend_h + bot
        fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
        xs, axes = [left + i * (pw + gap) for i in range(3)], []
        for i, x in enumerate(xs):
            axes.append(fig.add_axes([x / FIG_W_MM, 1 - (top + ph) / fig_h,
                                      pw / FIG_W_MM, ph / fig_h]))
        legend_rect = (xs[2], bot + legend_h * 0.6, widths[2], legend_h)
        panel_w_for_scale = pw

    elif variant == "b":
        fig_h = 205.0
        avail = fig_h - top - xlab - legend_h - bot
        H = avail
        wi = (H - gap) / (2 * aspect)
        wd = H / aspect
        fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
        x0 = left
        h_i = wi * aspect
        axes = [
            fig.add_axes([x0 / FIG_W_MM, 1 - (top + h_i) / fig_h,
                          wi / FIG_W_MM, h_i / fig_h]),
            fig.add_axes([x0 / FIG_W_MM, 1 - (top + 2 * h_i + gap) / fig_h,
                          wi / FIG_W_MM, h_i / fig_h]),
            fig.add_axes([(x0 + wi + 6) / FIG_W_MM, 1 - (top + H) / fig_h,
                          wd / FIG_W_MM, H / fig_h]),
        ]
        legend_rect = (x0 + wi + 6, bot + legend_h * 0.6, wd, legend_h)
        panel_w_for_scale = wd

    else:  # c -- DoD given more width than the imagery panels
        r = 1.55
        total = FIG_W_MM - left - right - 2 * gap
        wi = total / (2 + r)
        wd = wi * r
        ph = wd * aspect
        fig_h = top + ph + xlab + legend_h + bot
        fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
        h_i = wi * aspect
        axes = [
            fig.add_axes([left / FIG_W_MM, 1 - (top + h_i) / fig_h,
                          wi / FIG_W_MM, h_i / fig_h]),
            fig.add_axes([(left + wi + gap) / FIG_W_MM, 1 - (top + h_i) / fig_h,
                          wi / FIG_W_MM, h_i / fig_h]),
            fig.add_axes([(left + 2 * (wi + gap)) / FIG_W_MM,
                          1 - (top + ph) / fig_h, wd / FIG_W_MM, ph / fig_h]),
        ]
        legend_rect = (left + 2 * (wi + gap), bot + legend_h * 0.6, wd, legend_h)
        panel_w_for_scale = wd

    panel_imagery(axes[0], PRE_PNG, extent, ring)
    panel_imagery(axes[1], POST_PNG, extent, ring)
    im = panel_dod(axes[2], ring, extent)
    # In (b) the imagery panels are stacked, so panel (a)'s easting labels
    # would land on panel (b)'s title -- put the x labels on the LOWER left
    # panel there instead.
    if variant == "b":
        label_x, label_y = 1, [0, 1]
    else:
        label_x, label_y = 0, [0]
    for i, ax in enumerate(axes):
        style(ax, extent, xlabels=(i == label_x), ylabels=(i in label_y))
        ax.set_title(TITLES[i], fontsize=7.2, fontweight="bold", pad=3)

    _, _, scale_m = legend_line(fig, legend_rect, im, extent,
                                panel_w_for_scale, FIG_W_MM, fig_h)

    stem = DIAG / f"fig_dod_v2_{variant}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return stem, fig_h, scale_m




def draw_scale_north_stacked(ax, scale_m, mm_per_m, axes_w_mm, color="black",
                             fontsize=9.0, lw=2.6):
    """North arrow above a scale bar, stacked, for a narrow vertical slot.

    NOT currently called: the imagery figure carries no scale bar or north
    arrow. Kept because it is the only stacked variant in the codebase and
    the layout has changed several times.

    add_scale_north_inline puts the bar and arrow side by side, which needs a
    wide slot. Between two side-by-side maps the slot is tall and narrow, so
    they stack instead. Bar length still comes from the maps' own
    mm-per-metre, so it stays true to their scale.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    bar = (scale_m * mm_per_m) / axes_w_mm
    x0, x1 = 0.5 - bar / 2.0, 0.5 + bar / 2.0

    xn, y_arrow0, y_arrow1 = 0.5, 0.60, 0.80
    ax.annotate("", xy=(xn, y_arrow1), xytext=(xn, y_arrow0),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw * 0.6,
                                mutation_scale=17), zorder=5)
    ax.text(xn, y_arrow1 + 0.015, "N", ha="center", va="bottom",
            fontsize=fontsize + 1, fontweight="bold", color=color, zorder=5)

    y_bar = 0.44
    ax.plot([x0, x1], [y_bar, y_bar], color=color, linewidth=lw,
            solid_capstyle="butt", zorder=5)
    for xt in (x0, x1):
        ax.plot([xt, xt], [y_bar - 0.022, y_bar + 0.022], color=color,
                linewidth=lw * 0.6, zorder=5)
    ax.text(0.5, y_bar - 0.055, f"{scale_m:,.0f} m", ha="center", va="top",
            fontsize=fontsize, color=color, zorder=5)


def build_imagery_only(ring, extent):
    """Two-panel version: pre- and post-Helene imagery only, no DoD.

    With the DoD gone the elevation-change legend goes too -- it described
    that panel and nothing else -- so the furniture line carries just the
    scale bar and north arrow. Dropping from three panels to two also widens
    each from ~51 mm to ~78 mm, so the imagery is markedly more legible than
    in the three-panel variant.
    """
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    # Bare figure: no frames, no titles, no captions, no coordinate labels.
    # The gap between the two maps is widened so the scale bar and north
    # arrow sit in it, clear of both maps -- nothing overlaps map area.
    # Nothing sits between the maps now, so the gap goes back to a plain
    # separator; the images take the width that freed up (82.5 mm each,
    # against 70 mm when the scale group lived in the gap).
    left = right = 2.0
    gap, top, bot = 5.0, 3.0, 3.0
    pw = (FIG_W_MM - left - right - gap) / 2.0
    ph = pw * aspect
    fig_h = top + ph + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    axes = []
    for i in range(2):
        x = left + i * (pw + gap)
        axes.append(fig.add_axes([x / FIG_W_MM, 1 - (top + ph) / fig_h,
                                  pw / FIG_W_MM, ph / fig_h]))
    panel_imagery(axes[0], PRE_PNG, extent, ring)
    panel_imagery(axes[1], POST_PNG, extent, ring)
    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.set_axis_off()          # no box around either map

    scale_m = None          # no scale bar or north arrow on this figure

    stem = DIAG / "fig_dod_v2_a_imagery_only"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return stem, fig_h, scale_m


LATEX = {
    "a": r"""% Figure 10 -- variant (a): three equal panels in a row.
% The three panels are composited into ONE image file, so the old
% subfigure block collapses to a single \includegraphics; the panel
% letters live in the image and are referenced from the caption.
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{fig_dod_v2_a.png}
\caption{Pre- and post-Helene NAIP imagery and the co-registered,
bias-corrected DEM of Difference for the tributary watershed:
(a) NAIP 2020, pre-event; (b) NAIP 2024, post-event; (c) DoD, lidar 2020
vs.\ lidar 2024. All three panels are clipped to the watershed boundary and
share one extent and scale, so the single scale bar and north arrow apply to
all of them. The elevation-change scale applies to panel (c) only: erosion
red, deposition blue, diverging about zero, shown where $|\Delta z|$ exceeds
the 0.32~m level of detection.}
\label{fig:dem_difference}
\end{figure}""",
    "b": r"""% Figure 10 -- variant (b): imagery stacked left, DoD larger right.
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{fig_dod_v2_b.png}
\caption{Pre- and post-Helene NAIP imagery and the co-registered,
bias-corrected DEM of Difference for the tributary watershed: (a) NAIP 2020,
pre-event, and (b) NAIP 2024, post-event, stacked at left; (c) DoD, lidar
2020 vs.\ lidar 2024, at right. All three panels are clipped to the
watershed boundary and share one extent and scale, so the single scale bar
and north arrow apply to all of them. The elevation-change scale applies to
panel (c) only: erosion red, deposition blue, diverging about zero, shown
where $|\Delta z|$ exceeds the 0.32~m level of detection.}
\label{fig:dem_difference}
\end{figure}""",
    "c": r"""% Figure 10 -- variant (c): row of three, DoD panel widened.
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{fig_dod_v2_c.png}
\caption{Pre- and post-Helene NAIP imagery and the co-registered,
bias-corrected DEM of Difference for the tributary watershed: (a) NAIP 2020,
pre-event; (b) NAIP 2024, post-event; (c) DoD, lidar 2020 vs.\ lidar 2024,
shown at larger size. All three panels are clipped to the watershed boundary
and share one extent and scale, so the single scale bar and north arrow apply
to all of them. The elevation-change scale applies to panel (c) only: erosion
red, deposition blue, diverging about zero, shown where $|\Delta z|$ exceeds
the 0.32~m level of detection.}
\label{fig:dem_difference}
\end{figure}""",
}


LATEX_IMAGERY = r"""% Pre/post-Helene imagery only (DoD removed).
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{fig_dod_v2_a_imagery_only.png}
\caption{Pre- and post-Helene NAIP imagery of the tributary watershed:
(a) NAIP 2020, pre-event; (b) NAIP 2024, post-event. Both panels are clipped
to the watershed boundary and share one extent and scale, so the single
scale bar and north arrow apply to both. Bare ground in (b) marks the
landslide scars and debris corridors opened by the storm.}
\label{fig:pre_post_imagery}
\end{figure}"""


def main():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    if not HILLSHADE_TIF.exists() or not DOD_TIF.exists():
        export_map_rasters()
    ring = basin_ring()
    extent = tight_extent(ring)
    render_imagery(extent)
    log(f"START extent {extent} (basin + {MARGIN_M} m), pre={PRE_MAP}, "
        f"post={POST_1}+{POST_2}, LoD={LOD}", run=RUN)
    for v in ("a", "b", "c"):
        stem, fig_h, scale_m = build(v, ring, extent)
        (stem.with_suffix(".tex")).write_text(LATEX[v] + "\n")
        print(f"({v}) {stem.name}.png  {FIG_W_MM:.0f} x {fig_h:.0f} mm  "
              f"scale bar {scale_m} m")
        log(f"variant ({v}) -> {stem.name}.png (300 dpi) + .pdf + .tex, "
            f"{FIG_W_MM:.0f}x{fig_h:.0f} mm, scale bar {scale_m} m", run=RUN)
    stem, fig_h, scale_m = build_imagery_only(ring, extent)
    (stem.with_suffix(".tex")).write_text(LATEX_IMAGERY + "\n")
    print(f"(imagery-only) {stem.name}.png  {FIG_W_MM:.0f} x {fig_h:.0f} mm  "
          f"scale bar: {scale_m if scale_m else 'none'}")
    log(f"imagery-only (no DoD, no elevation legend) -> {stem.name}.png "
        f"300 dpi + .pdf + .tex, {FIG_W_MM:.0f}x{fig_h:.0f} mm", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
