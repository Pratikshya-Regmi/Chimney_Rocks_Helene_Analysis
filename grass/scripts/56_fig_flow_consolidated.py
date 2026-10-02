#!/usr/bin/env python3
"""
56_fig_flow_consolidated.py
===========================
Consolidate Figures 12, 13 and 14 (fig_flow_sequence_basinwide,
fig_flow_sequence_zoom, fig_flow_change_map) into ONE full-page figure that
carries the drainage-reorganisation argument in a single view.

Five layout variants are produced, for the analyst to choose from. NONE is
auto-inserted into the manuscript.

  a  basinwide3_zoom3col_nochange
       basin row = 3 epochs; zoom = 3 zone rows x 3 cols
       (2020 lidar | 2024 lidar | change). SfM appears only basin-wide.
       Each zoom row reads left-to-right as before -> after -> what changed.
  b  basinwide3_zoom2col_overlay
       basin row = 3 epochs; zoom = 3 zone rows x 2 cols (2020 | 2024), the
       change raster folded onto the 2024 panel as a diverging overlay, so
       there is no change column and no change row at all.
  c  basinwide3_zoom3epoch_changerow
       conservative merge that drops nothing: basin row = 3 epochs; zoom =
       3 zone rows x 3 epochs (2020 | SfM | 2024); full-width basin-wide
       change map as its own bottom row.
  d  basinwide4_zoom3col
       basin row = 4 panels (3 epochs + basin-wide change), so Fig 14 is
       folded into the top row beside the epochs it is built from; zoom =
       3 zone rows x 3 cols (2020 | 2024 | change).
  e  basinwide4_zonecols_overlay
       basin row = 4 panels as in (d); zoom transposed -- zones as COLUMNS,
       epochs as rows (2020 / 2024) with the change as a diverging overlay
       on the 2024 row. The most compact arrangement.

SYMBOLOGY, identical in every variant and every zoom panel
  magenta   2020 flow course with no 2024 channel within 2 m -- abandoned
  orange    2024 flow course with no 2020 channel within 2 m -- new
  open dot  split point: the cell carrying channel in BOTH epochs that is
            simultaneously closest to an abandoned and to a new segment,
            i.e. the junction where the course divides. Drawn only where
            both of those distances are <= 3 m, so it is never invented for
            a zone that has no real junction.
Courses are taken from the r.thin skeletons of the two lidar epochs at the
discharge > 0.01 threshold already used for the Jaccard metric, so the
annotation is derived from the data, not drawn by hand.

COLOUR SCALES (one of each, shared by every panel that uses them)
  discharge  YlGnBu, log, 1e-4 to 0.48 -- identical to Figures 12 and 13
  change     RdBu, +/- 3 SD of the basin-wide change raster (+/-0.0819),
             red = discharge lost, blue = discharge gained -- as Figure 14

Requires the GeoTIFF exports from 30_flow_sequence_setup.py,
31_fig_flow_sequence_basinwide.py and 34_flow_alternative_figures.py.

Outputs:
  results/figures/fig_flow_consolidated/fig_flow_consolidated_<key>.png/.pdf
  results/figures/fig_flow_consolidated/VARIANTS.md
  results/logs/flow_consolidated_<date>.log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patheffects

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import log  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
OUT = FIGURES / "fig_flow_consolidated"
OUT.mkdir(parents=True, exist_ok=True)

TIF = {
    "2020": FIGURES / "_flow_discharge_2020.tif",
    "sfm": FIGURES / "_flow_discharge_2024cap.tif",
    "2024": FIGURES / "_flow_discharge_2024lidar.tif",
    "change": FIGURES / "_flow_change.tif",
    "hs": FIGURES / "_flow_hillshade.tif",
    "thin2020": FIGURES / "__ovl_2020_thin.tif",
    "thin2024": FIGURES / "__ovl_2024_thin.tif",
    "thinsfm": FIGURES / "__ovl_cap_thin.tif",
}

MM = 1 / 25.4
FIG_W_MM = 174.0
MAX_H_MM = 232.0          # full page, sn-jnl text block

VMIN, VMAX = 1e-4, 0.48   # discharge, log, as Figs 12/13
CMAP_Q = "YlGnBu"
CMAP_D = "RdBu"

C_ABANDON = "#C51B7D"     # 2020 course lost
C_NEW = "#E66100"         # 2024 course gained
C_SPLIT = "#000000"
C_ZONE = "#D01C1C"        # Z1-Z3 boxes, as Fig 12

FS = dict(letter=8.0, header=7.5, zone=7.5, cbar=7.0, tick=7.0,
          key=7.0, north=8.5, scale=7.0)

MIN_COURSE_CELLS = 20     # shortest course stretch kept, in 1 m cells
PAD_M = 8.0               # zoom padding beyond the zone box
LETTER_H = 4.2            # mm strip under each panel for its (x) letter
HEADER_H = 4.4            # mm strip above a zoom block for column headers
ZONE_W = 5.6              # mm left gutter for the Z1/Z2/Z3 row labels
LEGEND_H = 22.0           # mm bottom strip: two colourbars + key + north

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6,
})


# --------------------------------------------------------------------- data
def read_window(path, ext):
    """ext = (e0, e1, n0, n1). Returns (masked array, extent actually read)."""
    e0, e1, n0, n1 = ext
    with rasterio.open(path) as src:
        win = rasterio.windows.from_bounds(e0, n0, e1, n1,
                                           transform=src.transform)
        arr = src.read(1, window=win, masked=True, boundless=True)
        b = rasterio.windows.bounds(win, src.transform)
    return arr, (b[0], b[2], b[1], b[3])


def thin_points(path, ext):
    a, e = read_window(path, ext)
    filled = a.filled(0)
    r, c = np.nonzero((~np.ma.getmaskarray(a)) & (filled > 0))
    if r.size == 0:
        return np.empty((0, 2))
    res = (e[1] - e[0]) / a.shape[1]
    x = e[0] + (c + 0.5) * res
    y = e[3] - (r + 0.5) * res
    return np.column_stack([x, y])


def _mask_window(path, ext):
    a, e = read_window(path, ext)
    return (~np.ma.getmaskarray(a)) & (a.filled(0) > 0), e


def _cells_to_xy(mask, e):
    r, c = np.nonzero(mask)
    res = (e[1] - e[0]) / mask.shape[1]
    return np.column_stack([e[0] + (c + 0.5) * res, e[3] - (r + 0.5) * res])


def zone_annotation(ext, box):
    """Abandoned / new course and split point for one zoom window.

    A cell is abandoned if it carried 2020 channel and no 2024 channel lies
    within 2 m, and new on the mirror test. Both sets are then reduced to
    8-connected components of at least MIN_COURSE_CELLS cells, so only
    coherent stretches of course survive and the scattered one- and two-cell
    hillslope threads -- which are simulation noise, not reorganisation --
    are dropped.

    box = (e_min, e_max, n_min, n_max) of the zone proper; the split point is
    searched only inside it, never out in the padding."""
    m20, e = _mask_window(TIF["thin2020"], ext)
    m24, _ = _mask_window(TIF["thin2024"], ext)
    out = dict(abandon=np.empty((0, 2)), new=np.empty((0, 2)), split=None,
               n_abandon=0, n_new=0)
    if not m20.any() or not m24.any():
        return out
    d24 = ndi.distance_transform_edt(~m24)
    d20 = ndi.distance_transform_edt(~m20)
    st = np.ones((3, 3), bool)

    def keep_components(mask):
        lab, n = ndi.label(mask, structure=st)
        if n == 0:
            return np.zeros_like(mask)
        sizes = np.bincount(lab.ravel())
        big = np.where(sizes >= MIN_COURSE_CELLS)[0]
        big = big[big > 0]
        return np.isin(lab, big), int(big.size)

    ab_mask, n_ab = keep_components(m20 & (d24 > 2.0))
    nw_mask, n_nw = keep_components(m24 & (d20 > 2.0))
    out["abandon"] = _cells_to_xy(ab_mask, e)
    out["new"] = _cells_to_xy(nw_mask, e)
    out["n_abandon"], out["n_new"] = n_ab, n_nw

    shared = m20 & (d24 <= 1.5)
    if ab_mask.any() and nw_mask.any() and shared.any():
        da = ndi.distance_transform_edt(~ab_mask)
        dn = ndi.distance_transform_edt(~nw_mask)
        inbox = np.zeros_like(shared)
        xy = _cells_to_xy(np.ones_like(shared), e).reshape(shared.shape + (2,))
        inbox = ((xy[..., 0] >= box[0]) & (xy[..., 0] <= box[1]) &
                 (xy[..., 1] >= box[2]) & (xy[..., 1] <= box[3]) & shared)
        if inbox.any():
            score = np.where(inbox, np.maximum(da, dn), np.inf)
            r, c = np.unravel_index(int(np.argmin(score)), score.shape)
            if score[r, c] <= 3.0:
                out["split"] = (xy[r, c], float(da[r, c]), float(dn[r, c]))
    return out


# ------------------------------------------------------------------ drawing
def bare(ax, ext):
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("black")


def draw_discharge(ax, key, ext):
    hs, he = read_window(TIF["hs"], ext)
    ax.imshow(hs, cmap="gray", extent=he, origin="upper", zorder=1,
              interpolation="bilinear")
    q, qe = read_window(TIF[key], ext)
    qm = np.ma.masked_where(q.filled(0) <= 0, q)
    im = ax.imshow(qm, cmap=CMAP_Q, norm=LogNorm(VMIN, VMAX), extent=qe,
                   origin="upper", alpha=0.85, zorder=2,
                   interpolation="nearest")
    return im


def draw_change(ax, ext, vlim, fade=0.25, overlay=False):
    """fade: |change| below fade*vlim is fully transparent, ramping to opaque
    at vlim, so near-zero cells never paint over the terrain."""
    if not overlay:
        hs, he = read_window(TIF["hs"], ext)
        ax.imshow(hs, cmap="gray", extent=he, origin="upper", zorder=1,
                  interpolation="bilinear")
    d, de = read_window(TIF["change"], ext)
    z = np.asarray(d.filled(np.nan), dtype=float)
    cmap = plt.get_cmap(CMAP_D)
    rgba = cmap(Normalize(-vlim, vlim)(z))
    lo, hi = fade * vlim, vlim
    a = np.clip((np.abs(z) - lo) / (hi - lo), 0, 1)
    rgba[..., 3] = np.where(np.isfinite(z), a * (0.95 if overlay else 1.0), 0)
    ax.imshow(rgba, extent=de, origin="upper", zorder=4 if overlay else 2,
              interpolation="nearest")
    return plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(-vlim, vlim))


def draw_watershed_tint(ax, ext, alpha=0.50, zorder=2):
    """Lighten the watershed on the basin-wide CHANGE panel.

    The discharge panels show the basin as a filled colour field, so its
    outline is obvious; the change field is sparse, which left the basin
    indistinguishable from the surrounding hillshade. The discharge raster is
    clipped to the watershed, so its valid cells ARE the basin -- tint those
    white to lift the basin out of the terrain without hiding the relief."""
    q, qe = read_window(TIF["2020"], ext)
    inside = ~np.ma.getmaskarray(q)
    rgba = np.zeros(inside.shape + (4,), dtype=float)
    rgba[..., :3] = 1.0
    rgba[..., 3] = np.where(inside, alpha, 0.0)
    ax.imshow(rgba, extent=qe, origin="upper", zorder=zorder,
              interpolation="nearest")


def draw_sfm_network(ax, ext):
    p = thin_points(TIF["thinsfm"], ext)
    if len(p):
        ax.plot(p[:, 0], p[:, 1], ls="none", marker="s", ms=0.45,
                mfc="black", mec="none", zorder=6)


def draw_annotation(ax, ann, mm_per_m, show_split=True):
    """Cell markers sized so 1 m cells just touch -> the skeletons read as
    continuous courses rather than as dot scatter."""
    ms = max(0.7, mm_per_m / 25.4 * 72 * 1.35)
    for pts, col in ((ann["abandon"], C_ABANDON), (ann["new"], C_NEW)):
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], ls="none", marker="s", ms=ms,
                    mfc=col, mec="none", zorder=7, alpha=0.95)
    if show_split and ann["split"] is not None:
        (sx, sy), _, _ = ann["split"]
        ax.plot([sx], [sy], marker="o", ms=4.2, mfc="none", mec=C_SPLIT,
                mew=1.1, zorder=9)
        ax.plot([sx], [sy], marker="o", ms=4.2, mfc="none", mec="white",
                mew=2.4, zorder=8)


def draw_zone_boxes(ax, zones, label=True):
    for _, z in zones.iterrows():
        ax.add_patch(Rectangle((z.e_min, z.n_min), z.width_m, z.height_m,
                               fill=False, edgecolor=C_ZONE, lw=0.9, zorder=8))
        if label:
            # inside the box's own top-left corner and clipped to the axes,
            # so a label never spills out over the neighbouring panel
            t = ax.annotate(z.zone_id, xy=(z.e_min, z.n_max),
                            xytext=(1.5, -1.5), textcoords="offset points",
                            fontsize=FS["zone"], color=C_ZONE,
                            fontweight="bold", ha="left", va="top", zorder=10,
                            path_effects=[matplotlib.patheffects.withStroke(
                                linewidth=2.0, foreground="white")])
            t.set_clip_on(True)


def draw_scale_bar(ax, ext, length_m, mm_per_m, corner="br"):
    """Cased bar in a panel corner. `corner` picks which one, so the basin
    panels can keep it up on the bare hillshade instead of over the Z boxes,
    which all sit low in the basin."""
    e0, e1, n0, n1 = ext
    w = e1 - e0
    h = n1 - n0
    x1 = e1 - 0.045 * w
    x0 = x1 - length_m
    y = n0 + 0.055 * h if corner.startswith("b") else n1 - 0.045 * h
    ax.plot([x0, x1], [y, y], color="black", lw=2.4, solid_capstyle="butt",
            zorder=11,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=4.2,
                                                            foreground="white")])
    top = not corner.startswith("b")
    ax.annotate(f"{length_m:g} m", xy=((x0 + x1) / 2, y),
                xytext=(0, -2.5 if top else 2.0), textcoords="offset points",
                ha="center", va="top" if top else "bottom",
                fontsize=FS["scale"], color="black", zorder=12,
                path_effects=[matplotlib.patheffects.withStroke(
                    linewidth=2.0, foreground="white")])



# ------------------------------------------------------------------- layout
def size_row(panels, avail_w, gap, mode):
    """Return (widths_mm, height_mm) for one row of panels.

    uniform  every panel the same width (panels in the row share an extent)
    matched  every panel the same HEIGHT, widths set by each one's aspect
             (used when the columns are the three differently shaped zones)
    """
    n = len(panels)
    free = avail_w - gap * (n - 1)
    if mode == "uniform":
        w = free / n
        h = w * max(p["aspect"] for p in panels)
        return [w] * n, h
    h = free / sum(1.0 / p["aspect"] for p in panels)
    return [h / p["aspect"] for p in panels], h


def build_layout(blocks, avail_w, gap_x, gap_y, block_gap):
    """Walk the blocks, size every row, return geometry + total height."""
    rows_out = []
    total = 0.0
    for bi, blk in enumerate(blocks):
        if bi:
            total += block_gap
        if blk.get("headers"):
            total += HEADER_H
        for ri, row in enumerate(blk["rows"]):
            inset = ZONE_W if blk.get("row_labels") else 0.0
            widths, h = size_row(row, avail_w - inset, gap_x, blk["mode"])
            rows_out.append(dict(block=bi, row=ri, widths=widths, h=h,
                                 inset=inset, panels=row,
                                 headers=blk.get("headers") if ri == 0 else None,
                                 row_label=(blk["row_labels"][ri]
                                            if blk.get("row_labels") else None)))
            total += h + LETTER_H
            if ri < len(blk["rows"]) - 1:
                total += gap_y
    return rows_out, total


# ------------------------------------------------------------------ panels
def panel_spec(kind, ext, aspect, zone=None, ann=None, scale_m=None,
               header=None, overlay=False, sfm_net=False, zone_boxes=False,
               scale_corner="br", watershed=False):
    return dict(kind=kind, ext=ext, aspect=aspect, zone=zone, ann=ann,
                scale_m=scale_m, header=header, overlay=overlay,
                sfm_net=sfm_net, zone_boxes=zone_boxes,
                scale_corner=scale_corner, watershed=watershed)


def render_panel(fig, rect_fn, x_mm, y_mm, w_mm, h_mm, p, letter, zones,
                 vlim, state):
    ax = fig.add_axes(rect_fn(x_mm, y_mm, w_mm, h_mm))
    ext = p["ext"]
    mm_per_m = w_mm / (ext[1] - ext[0])

    if p["kind"] in ("2020", "sfm", "2024"):
        im = draw_discharge(ax, p["kind"], ext)
        state.setdefault("q_mappable", im)
        if p["overlay"]:
            sm = draw_change(ax, ext, vlim, overlay=True)
            state.setdefault("d_mappable", sm)
    elif p["kind"] == "change":
        if p.get("watershed"):
            hs, he = read_window(TIF["hs"], ext)
            ax.imshow(hs, cmap="gray", extent=he, origin="upper", zorder=1,
                      interpolation="bilinear")
            draw_watershed_tint(ax, ext)
            sm = draw_change(ax, ext, vlim, overlay=True)
        else:
            sm = draw_change(ax, ext, vlim)
        state.setdefault("d_mappable", sm)
    if p["sfm_net"]:
        draw_sfm_network(ax, ext)
    if p["zone_boxes"]:
        draw_zone_boxes(ax, zones)
    if p["ann"] is not None:
        draw_annotation(ax, p["ann"], mm_per_m)
    bare(ax, ext)
    if p["scale_m"]:
        draw_scale_bar(ax, ext, p["scale_m"], mm_per_m,
                       corner=p.get("scale_corner", "br"))

    # panel letter in its own strip BELOW the panel -- never over the data
    fig.text((x_mm + w_mm / 2) / FIG_W_MM,
             1 - (y_mm + h_mm + LETTER_H * 0.72) / state["fig_h"],
             f"({letter})", ha="center", va="baseline",
             fontsize=FS["letter"], fontweight="bold")
    return ax


# ----------------------------------------------------------------- variants
def zoom_panels(zones, cols, ann_by_zone, exts, scale_m, overlay_col=None):
    """One zoom row per zone; `cols` names the epoch/change column contents."""
    rows, letters_headers = [], []
    for _, z in zones.iterrows():
        ext = exts[z.zone_id]
        aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
        row = []
        for c in cols:
            row.append(panel_spec(
                c, ext, aspect, zone=z.zone_id, ann=ann_by_zone[z.zone_id],
                scale_m=scale_m, overlay=(c == overlay_col)))
        rows.append(row)
    return rows


HEAD = {"2020": "2020 lidar (pre-event)",
        "sfm": "2024 CAP SfM (5 Oct)",
        "2024": "2024 lidar (+7 wk)",
        "change": "change (2024 $-$ 2020)"}


# ------------------------------------------------------------------ legend
def draw_legend(fig, rect_fn, x, y, w, h, state, vlim, stacked):
    """Shared legend: one discharge scale, one change scale, the annotation
    key, and the figure's single north arrow.

    stacked = a tall narrow block beside the basin row; otherwise an explicit
    four-column strip across the page foot, so the key can never land on top
    of a colourbar."""
    Q_LABEL = ("unit discharge\n(m$^3$ s$^{-1}$ m$^{-1}$-equivalent)")
    D_LABEL = ("discharge change, 2024 $-$ 2020\n"
               "lost $\\leftarrow$   $\\rightarrow$ gained")
    if stacked:
        # compact block that fits BESIDE the basin row: two colourbars
        # stacked, then the key and the north arrow side by side beneath
        cb_w = min(w - 4.0, 52.0)
        cax1 = fig.add_axes(rect_fn(x, y + 2.0, cb_w, 3.0))
        cax2 = fig.add_axes(rect_fn(x, y + 15.0, cb_w, 3.0))
        kx, ky, kw = x, y + 29.0, min(cb_w - 14.0, 40.0)
        nx, ny = x + cb_w - 10.0, y + 30.0
        Q_LABEL = "unit discharge (m$^3$ s$^{-1}$ m$^{-1}$-eq.)"
    else:
        cb_w = 42.0
        cax1 = fig.add_axes(rect_fn(x + 1.0, y + 4.0, cb_w, 3.0))
        cax2 = fig.add_axes(rect_fn(x + 1.0 + cb_w + 12.0, y + 4.0, cb_w, 3.0))
        kx = x + 1.0 + 2 * (cb_w + 12.0)
        ky, kw = y + 0.5, 42.0
        nx, ny = x + w - 11.0, y + 1.0

    cb1 = fig.colorbar(state["q_mappable"], cax=cax1, orientation="horizontal",
                       extend="max")
    cb1.set_label(Q_LABEL, fontsize=FS["cbar"], labelpad=2)
    cb1.ax.tick_params(labelsize=FS["tick"], length=2.2, width=0.5)
    cb1.outline.set_linewidth(0.5)

    cb2 = fig.colorbar(state["d_mappable"], cax=cax2, orientation="horizontal",
                       extend="both")
    cb2.set_label(D_LABEL, fontsize=FS["cbar"], labelpad=2)
    cb2.set_ticks([-vlim, 0, vlim])
    cb2.set_ticklabels([f"$-${vlim:.2f}", "0", f"{vlim:.2f}"])
    cb2.ax.tick_params(labelsize=FS["tick"], length=2.2, width=0.5)
    cb2.outline.set_linewidth(0.5)

    handles = [
        Line2D([], [], ls="none", marker="s", ms=4.5, mfc=C_ABANDON,
               mec="none", label="removed flow (2020)"),
        Line2D([], [], ls="none", marker="s", ms=4.5, mfc=C_NEW, mec="none",
               label="new flow (2024)"),
        Line2D([], [], ls="none", marker="o", ms=5.0, mfc="none",
               mec=C_SPLIT, mew=1.1, label="split point"),
    ]
    kax = fig.add_axes(rect_fn(kx, ky, kw, 14.0))
    kax.axis("off")
    kax.legend(handles=handles, loc="center left", frameon=False,
               fontsize=FS["key"], handletextpad=0.5, labelspacing=0.5,
               borderpad=0.0)

    nax = fig.add_axes(rect_fn(nx, ny, 10.0, 13.0))
    nax.axis("off")
    nax.set_xlim(0, 1)
    nax.set_ylim(0, 1)
    nax.annotate("", xy=(0.5, 0.70), xytext=(0.5, 0.06),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5,
                                 mutation_scale=9))
    nax.text(0.5, 0.80, "N", ha="center", va="bottom",
             fontsize=FS["north"], fontweight="bold")


# ---------------------------------------------------------------- variants
def variant_spec(key, zones, exts, ann, basin_ext):
    """Return (blocks, legend_mode, basin_frac, zoom_frac, description)."""
    ba = (basin_ext[3] - basin_ext[2]) / (basin_ext[1] - basin_ext[0])

    def basin_row(cols, sfm_net_on_change=True):
        """All basin panels share one extent and scale, so the bar is drawn
        once, on the last panel, and up on the hillshade clear of the Z
        boxes."""
        row = []
        for i, c in enumerate(cols):
            row.append(panel_spec(
                c, basin_ext, ba,
                scale_m=200 if i == len(cols) - 1 else None,
                scale_corner="tr",
                zone_boxes=(c != "change"),
                watershed=(c == "change"),
                sfm_net=(c == "change" and sfm_net_on_change)))
        return row

    hdr3 = [HEAD["2020"], HEAD["sfm"], HEAD["2024"]]
    hdr4 = hdr3 + [HEAD["change"]]
    # the basin panels are narrow; full epoch names collide there, so the
    # basin row takes short headers and the dates live in the caption
    short4 = ["2020 lidar", "2024 SfM", "2024 lidar", "change"]

    if key == "a":
        blocks = [
            dict(rows=[basin_row(["2020", "sfm", "2024"])], mode="uniform",
                 headers=hdr3, frac=0.60),
            dict(rows=zoom_panels(zones, ["2020", "2024", "change"], ann,
                                  exts, 25),
                 mode="uniform", headers=[HEAD["2020"], HEAD["2024"],
                                          HEAD["change"]],
                 row_labels=list(zones.zone_id), frac=1.0),
        ]
        return blocks, "stacked", ("basin row = the three epochs; each zoom "
                                   "row reads before / after / change; SfM "
                                   "basin-wide only; no change row")
    if key == "b":
        blocks = [
            dict(rows=[basin_row(["2020", "sfm", "2024"])], mode="uniform",
                 headers=hdr3, frac=0.86),
            dict(rows=zoom_panels(zones, ["2020", "2024"], ann, exts, 25,
                                  overlay_col="2024"),
                 mode="uniform",
                 headers=[HEAD["2020"], "2024 lidar + change overlay"],
                 row_labels=list(zones.zone_id), frac=0.64),
        ]
        return blocks, "stacked_zoom", ("basin row = the three epochs; zoom is two "
                                   "columns only, the change folded onto the "
                                   "2024 panel as a diverging overlay")
    if key == "c":
        blocks = [
            dict(rows=[basin_row(["2020", "sfm", "2024", "change"])],
                 mode="uniform", headers=short4, frac=0.62),
            dict(rows=zoom_panels(zones, ["2020", "sfm", "2024"], ann,
                                  exts, 25),
                 mode="uniform", headers=hdr3,
                 row_labels=list(zones.zone_id), frac=1.0),
        ]
        return blocks, "stacked", ("nothing dropped: all three epochs in the "
                                  "zooms, change map kept as the 4th basin "
                                  "panel")
    if key == "d":
        blocks = [
            dict(rows=[basin_row(["2020", "sfm", "2024", "change"])],
                 mode="uniform", headers=short4, frac=0.62),
            dict(rows=zoom_panels(zones, ["2020", "2024", "change"], ann,
                                  exts, 25),
                 mode="uniform", headers=[HEAD["2020"], HEAD["2024"],
                                          HEAD["change"]],
                 row_labels=list(zones.zone_id), frac=1.0),
        ]
        return blocks, "stacked", ("change map is the 4th basin panel AND the "
                                  "3rd zoom column; SfM basin-wide only")
    if key == "e":
        rows = []
        for epoch in ("2020", "sfm", "2024"):
            row = []
            for _, z in zones.iterrows():
                ext = exts[z.zone_id]
                row.append(panel_spec(
                    epoch, ext, (ext[3] - ext[2]) / (ext[1] - ext[0]),
                    zone=z.zone_id, ann=ann[z.zone_id], scale_m=25,
                    overlay=(epoch == "2024")))
            rows.append(row)
        blocks = [
            dict(rows=[basin_row(["2020", "sfm", "2024", "change"])],
                 mode="uniform", headers=short4, frac=0.62),
            dict(rows=rows, mode="matched", headers=list(zones.zone_id),
                 row_labels=[HEAD["2020"], HEAD["sfm"],
                             HEAD["2024"] + " + change"],
                 frac=1.0),
        ]
        return blocks, "stacked", ("zooms transposed -- zones as columns, all "
                                  "three epochs as rows, change as an overlay "
                                  "on the 2024 row")
    raise ValueError(key)


# -------------------------------------------------------------------- build
MARGIN_L = MARGIN_R = 3.0
MARGIN_T = 3.0
MARGIN_B = 3.0
GAP_X = 2.5
GAP_Y = 3.0
BLOCK_GAP = 6.0


def compute(blocks, legend_mode, shrink):
    """Size every row at the given shrink factor; return (geom, total_h)."""
    avail_full = (FIG_W_MM - MARGIN_L - MARGIN_R) * shrink
    geom = []
    y = MARGIN_T
    for bi, blk in enumerate(blocks):
        if bi:
            y += BLOCK_GAP
        if blk.get("headers"):
            y += HEADER_H
        inset = ZONE_W if blk.get("row_labels") else 0.0
        for ri, row in enumerate(blk["rows"]):
            avail = avail_full * blk.get("frac", 1.0) - inset
            widths, h = size_row(row, avail, GAP_X, blk["mode"])
            geom.append(dict(block=bi, row=ri, y=y, widths=widths, h=h,
                             inset=inset, panels=row, align="center",
                             headers=blk.get("headers") if ri == 0 else None,
                             row_label=(blk["row_labels"][ri]
                                        if blk.get("row_labels") else None)))
            y += h + LETTER_H
            if ri < len(blk["rows"]) - 1:
                y += GAP_Y
    if legend_mode == "bottom":
        y += BLOCK_GAP + LEGEND_H
    else:
        # the block the legend sits beside stays left-anchored, so the pair
        # of them spans the page; every other row is centred
        anchor = 1 if legend_mode == "stacked_zoom" else 0
        for g in geom:
            if g["block"] == anchor:
                g["align"] = "left"
    return geom, y + MARGIN_B


def fit_shrink(blocks, legend_mode):
    lo, hi = 0.30, 1.0
    if compute(blocks, legend_mode, hi)[1] <= MAX_H_MM:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if compute(blocks, legend_mode, mid)[1] <= MAX_H_MM:
            lo = mid
        else:
            hi = mid
    return lo


def build(key, zones, exts, ann, basin_ext, vlim):
    blocks, legend_mode, desc = variant_spec(key, zones, exts, ann, basin_ext)
    shrink = fit_shrink(blocks, legend_mode)
    geom, fig_h = compute(blocks, legend_mode, shrink)
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
    state = dict(fig_h=fig_h)

    def rect_fn(x, ytop, w, h):
        return [x / FIG_W_MM, 1 - (ytop + h) / fig_h, w / FIG_W_MM, h / fig_h]

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    smallest_mm_per_m = 1e9

    # Pass 1: horizontal placement. Rows are centred on the page rather than
    # anchored left, because fit_shrink narrows them to make the height fit
    # and a left anchor would leave all the slack on the right.
    for g in geom:
        row_w = g["inset"] + sum(g["widths"]) + GAP_X * (len(g["widths"]) - 1)
        g["x0"] = (MARGIN_L if g["align"] == "left"
                   else (FIG_W_MM - row_w) / 2)
    # Every block should sit inside the same left and right edges. Take those
    # from the centred blocks' PANEL area (excluding the row-label gutter) and
    # pull the legend-anchored block out to meet them.
    spans = [(g["x0"] + g["inset"],
              g["x0"] + g["inset"] + sum(g["widths"]) +
              GAP_X * (len(g["widths"]) - 1))
             for g in geom if g["align"] == "center"]
    if spans:
        ref_l, ref_r = min(a for a, _ in spans), max(b for _, b in spans)
        for g in geom:
            if g["align"] == "left":
                g["x0"] = ref_l - g["inset"]
        state["ref_r"] = ref_r

    # Pass 2: draw
    for g in geom:
        x = g["x0"] + g["inset"]
        for i, (p, w) in enumerate(zip(g["panels"], g["widths"])):
            if g["headers"] is not None:
                fig.text((x + w / 2) / FIG_W_MM,
                         1 - (g["y"] - HEADER_H * 0.30) / fig_h,
                         g["headers"][i], ha="center", va="baseline",
                         fontsize=FS["header"])
            render_panel(fig, rect_fn, x, g["y"], w, g["h"], p,
                         next(letters), zones, vlim, state)
            smallest_mm_per_m = min(smallest_mm_per_m,
                                    w / (p["ext"][1] - p["ext"][0]))
            x += w + GAP_X
        if g["row_label"]:
            fig.text((g["x0"] + g["inset"] / 2) / FIG_W_MM,
                     1 - (g["y"] + g["h"] / 2) / fig_h, g["row_label"],
                     ha="center", va="center", rotation=90,
                     fontsize=FS["zone"], fontweight="bold")

    # legend
    if legend_mode == "bottom":
        widest = max(g["inset"] + sum(g["widths"]) +
                     GAP_X * (len(g["widths"]) - 1) for g in geom)
        draw_legend(fig, rect_fn, (FIG_W_MM - widest) / 2,
                    fig_h - MARGIN_B - LEGEND_H, widest, LEGEND_H, state,
                    vlim, stacked=False)
    else:
        anchor = 1 if legend_mode == "stacked_zoom" else 0
        g0 = next(g for g in geom if g["block"] == anchor)
        x0 = g0["x0"] + g0["inset"] + sum(g0["widths"]) + \
            GAP_X * (len(g0["widths"]) - 1) + 9.0
        right = state.get("ref_r", FIG_W_MM - MARGIN_R)
        # The legend needs a usable width; if aligning its right edge to the
        # panel block would squeeze it, let it run to the page margin instead
        # and accept a few mm of asymmetry rather than an overlapping key.
        if right - x0 < 48.0:
            right = FIG_W_MM - MARGIN_R
        draw_legend(fig, rect_fn, x0, g0["y"] + 2.0, right - x0, g0["h"],
                    state, vlim, stacked=True)

    name = f"fig_flow_consolidated_{key}"
    png, pdf = OUT / f"{name}.png", OUT / f"{name}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return dict(key=key, desc=desc, png=png.name, h=fig_h, shrink=shrink,
                mm_per_m=smallest_mm_per_m,
                npanels=sum(len(g["panels"]) for g in geom))


def main():
    zones = pd.read_csv(TABLES / "flow_change_zones.csv")
    with rasterio.open(TIF["2020"]) as src:
        a = src.read(1, masked=True)
        t = src.transform
    r, c = np.nonzero(~np.ma.getmaskarray(a))
    basin_ext = (t.c + c.min() * t.a, t.c + (c.max() + 1) * t.a,
                 t.f + (r.max() + 1) * t.e, t.f + r.min() * t.e)

    ch, _ = read_window(TIF["change"], basin_ext)
    v = np.asarray(ch.filled(np.nan), dtype=float)
    vlim = 3 * np.nanstd(v)

    exts, ann = {}, {}
    for _, z in zones.iterrows():
        exts[z.zone_id] = (z.e_min - PAD_M, z.e_max + PAD_M,
                           z.n_min - PAD_M, z.n_max + PAD_M)
        ann[z.zone_id] = zone_annotation(exts[z.zone_id],
                                         (z.e_min, z.e_max, z.n_min, z.n_max))

    print(f"basin extent {basin_ext[1]-basin_ext[0]:.0f} x "
          f"{basin_ext[3]-basin_ext[2]:.0f} m   change limit +/-{vlim:.4f}")
    for zid, a_ in ann.items():
        s = a_["split"]
        print(f"  {zid}: abandoned {len(a_['abandon']):4d} cells, "
              f"new {len(a_['new']):4d} cells, split "
              + (f"E{s[0][0]:.1f} N{s[0][1]:.1f} "
                 f"(d_abandoned {s[1]:.1f} m, d_new {s[2]:.1f} m)"
                 if s else "none within 3 m"))

    rows = []
    for key in "abcde":
        rows.append(build(key, zones, exts, ann, basin_ext, vlim))
        r_ = rows[-1]
        print(f"  {r_['key']}  {FIG_W_MM:.0f} x {r_['h']:.1f} mm  "
              f"{r_['npanels']:2d} panels  panel scale {r_['shrink']:.2f}  "
              f"finest {r_['mm_per_m']*1000:.0f} um per map metre")
        log(f"fig_flow_consolidated_{key}: {FIG_W_MM:.0f}x{r_['h']:.1f} mm, "
            f"{r_['npanels']} panels, shrink {r_['shrink']:.2f}",
            run="flow_consolidated")

    md = ["# Consolidated flow figure (Figs 12+13+14) -- variants", "",
          f"All 174 mm wide, 300 dpi PNG + vector PDF, Type 42, sans-serif.",
          f"Discharge: {CMAP_Q} log {VMIN:g}-{VMAX:g}. "
          f"Change: {CMAP_D} +/-{vlim:.4f} (3 SD).", "",
          "| variant | file | size (mm) | panels | what distinguishes it |",
          "|---|---|---|---|---|"]
    for r_ in rows:
        md.append(f"| {r_['key']} | `{r_['png']}` | 174 x {r_['h']:.0f} | "
                  f"{r_['npanels']} | {r_['desc']} |")
    (OUT / "VARIANTS_AUTO.md").write_text("\n".join(md) + "\n")
    # NB: the curated VARIANTS.md / E_OPTIONS.md alongside this file is
    # hand-written and must NOT be overwritten by a re-run.
    log("DONE", run="flow_consolidated")


if __name__ == "__main__":
    main()
