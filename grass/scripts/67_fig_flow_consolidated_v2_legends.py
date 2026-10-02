#!/usr/bin/env python3
"""
67_fig_flow_consolidated_v2_legends.py
======================================
Refine fig_flow_consolidated_a: close the unused white space between the
legend block and the zoom rows by ENLARGING the legend and colour keys to
fill it, rather than by tightening the spacing.

THE PROBLEM, MEASURED
    In variant (a) the legend sits in the column to the right of the basin
    row. That column is about 56 mm wide and 68 mm tall. draw_legend() in
    56_fig_flow_consolidated.py places its contents at FIXED mm offsets
    (colourbars 3.0 mm tall at +2 and +15, key at +29), so it uses only the
    top ~42 mm and leaves ~26 mm of the column empty above the zoom block.
    The bars are also only 3 mm tall and 52 mm long against panels 58 mm
    tall, so they read as an afterthought at print size.

WHAT IS KEPT IN EVERY VARIANT
    Structure  basin-wide row (2020 lidar | 2024 SfM | 2024 lidar), then
               three zoom rows (Z1/Z2/Z3), each 2020 | 2024 | change.
    Content    every panel, overlay, zone box, scale bar and the derived
               magenta/orange/split-point annotation, unchanged -- all of it
               comes from 56_fig_flow_consolidated.py, imported not copied.
    Ramps      discharge YlGnBu log 1e-4..0.48; change RdBu +/-3 SD. Both
               colourblind-safe, unchanged.
    Legibility every tick label and unit at >= 7.5 pt (was 7.0), swatches
               enlarged, and no label placed over data.

THE FIVE VARIANTS
    a  legends enlarged in place: same column, contents scaled and spaced to
       fill its full height exactly.
    b  horizontal colourbars spanning the full figure width beneath the row
       each one serves -- discharge under the basin row, change under the
       last zoom row; the categorical key takes over the side column.
    c  colourbars turned vertical and moved alongside the panels: discharge
       in a right-hand gutter spanning the whole zoom block, change in the
       side column. Reclaims the vertical space the horizontal bars and
       their under-slung labels used.
    d  one shared legend block at the figure foot, generously sized, the two
       scales side by side; the basin row widens to use the reclaimed side
       space.
    e  my own: the side column becomes a proper key column with two TALL
       vertical scales side by side, and the north arrow moves out of the
       legend onto the basin hillshade where it belongs -- a north arrow is a
       map element, not a legend entry, and moving it frees legend space.

Outputs (results/figures/fig_flow_consolidated/):
  fig_flow_consolidated_v2_<a|b|c|d|e>.png   300 dpi
  fig_flow_consolidated_v2_<a|b|c|d|e>.pdf   vector, Type 42
  VARIANTS_V2.md
The existing fig_flow_consolidated_<a..e>.* are never touched.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import rasterio

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "f56", HERE / "56_fig_flow_consolidated.py")
f = importlib.util.module_from_spec(spec)
sys.modules["f56"] = f
spec.loader.exec_module(f)

sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402

RUN = "flow_consolidated_v2"
OUT = f.OUT
FIG_W_MM, MM, MAX_H_MM = f.FIG_W_MM, f.MM, f.MAX_H_MM
HEADER_H, LETTER_H, ZONE_W = f.HEADER_H, f.LETTER_H, f.ZONE_W
MARGIN_L, MARGIN_R = f.MARGIN_L, f.MARGIN_R
MARGIN_T, MARGIN_B = f.MARGIN_T, f.MARGIN_B
GAP_X, GAP_Y, BLOCK_GAP = f.GAP_X, f.GAP_Y, f.BLOCK_GAP
HEAD, C_ABANDON, C_NEW, C_SPLIT = f.HEAD, f.C_ABANDON, f.C_NEW, f.C_SPLIT

# Enlarged type. Every value here is >= the 7 pt print floor.
FS = dict(letter=8.0, header=7.5, zone=7.5,
          cbar=8.0,      # colourbar axis titles (was 7.0)
          tick=7.5,      # colourbar tick labels (was 7.0)
          key=8.5,       # categorical key text (was 7.0)
          north=9.5, scale=7.0)
KEY_MS = 7.0             # categorical swatch size (was 4.5)
BAR_T = 5.5              # horizontal colourbar thickness mm (was 3.0)
BAR_W_V = 5.0            # vertical colourbar width mm

Q_LABEL = "unit discharge (m$^3$ s$^{-1}$ m$^{-1}$-eq.)"
Q_LABEL_SHORT = "unit discharge\n(m$^3$ s$^{-1}$ m$^{-1}$-eq.)"
D_LABEL = "discharge change, 2024 $-$ 2020"
D_SUB = "lost $\\leftarrow$    $\\rightarrow$ gained"


# --------------------------------------------------------------- key pieces
def key_handles():
    return [
        Line2D([], [], ls="none", marker="s", ms=KEY_MS, mfc=C_ABANDON,
               mec="none", label="removed flow (2020)"),
        Line2D([], [], ls="none", marker="s", ms=KEY_MS, mfc=C_NEW,
               mec="none", label="new flow (2024)"),
        Line2D([], [], ls="none", marker="o", ms=KEY_MS + 0.8, mfc="none",
               mec=C_SPLIT, mew=1.3, label="split point"),
    ]


def draw_key(fig, rect_fn, x, y, w, h, ncol=1, title=None):
    ax = fig.add_axes(rect_fn(x, y, w, h))
    ax.axis("off")
    lg = ax.legend(handles=key_handles(), loc="upper left", frameon=False,
                   fontsize=FS["key"], handletextpad=0.7, labelspacing=0.85,
                   borderpad=0.0, ncol=ncol, columnspacing=1.6)
    if title:
        lg.set_title(title, prop=dict(size=FS["key"], weight="bold"))
        lg.get_title().set_ha("left")
    return ax


def draw_north(fig, rect_fn, x, y, w, h, scale=1.0):
    ax = fig.add_axes(rect_fn(x, y, w, h))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate("", xy=(0.5, 0.68), xytext=(0.5, 0.04),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=1.7 * scale, mutation_scale=11 * scale))
    ax.text(0.5, 0.76, "N", ha="center", va="bottom",
            fontsize=FS["north"], fontweight="bold")
    return ax


def north_on_panel(ax, ext, frac_x=0.13, frac_y=0.86):
    """North arrow drawn on a panel's own axes, with a white halo so it
    stays readable on the grey hillshade. Used by variant (e)."""
    x = ext[0] + frac_x * (ext[1] - ext[0])
    y0 = ext[2] + (frac_y - 0.11) * (ext[3] - ext[2])
    y1 = ext[2] + frac_y * (ext[3] - ext[2])
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    ax.annotate("", xy=(x, y1), xytext=(x, y0), zorder=12,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6,
                                mutation_scale=10,
                                path_effects=halo))
    t = ax.text(x, y1 + 0.012 * (ext[3] - ext[2]), "N", ha="center",
                va="bottom", fontsize=FS["north"], fontweight="bold",
                zorder=12)
    t.set_path_effects(halo)


def hbar(fig, rect_fn, mappable, x, y, w, kind, vlim, sub=False,
         short=False):
    """Horizontal colourbar, label beneath, ticks beneath the bar.

    `short` wraps the discharge title onto two lines, for slots where the
    one-line form would be wider than the bar and could reach a neighbour.
    """
    cax = fig.add_axes(rect_fn(x, y, w, BAR_T))
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal",
                      extend="max" if kind == "q" else "both")
    lab = (Q_LABEL_SHORT if short else Q_LABEL) if kind == "q" else (
        D_LABEL + ("\n" + D_SUB if sub else ""))
    cb.set_label(lab, fontsize=FS["cbar"], labelpad=2.5)
    if kind == "d":
        cb.set_ticks([-vlim, 0, vlim])
        cb.set_ticklabels([f"$-${vlim:.2f}", "0", f"{vlim:.2f}"])
    cb.ax.tick_params(labelsize=FS["tick"], length=2.6, width=0.6)
    cb.outline.set_linewidth(0.6)
    return cb


def vbar(fig, rect_fn, mappable, x, y, h, kind, vlim, label=True):
    """Vertical colourbar, ticks and rotated title to its right."""
    cax = fig.add_axes(rect_fn(x, y, BAR_W_V, h))
    cb = fig.colorbar(mappable, cax=cax, orientation="vertical",
                      extend="max" if kind == "q" else "both")
    if label:
        cb.set_label(Q_LABEL_SHORT if kind == "q"
                     else D_LABEL.replace(", ", ",\n"),
                     fontsize=FS["cbar"], labelpad=3.0)
    if kind == "d":
        cb.set_ticks([-vlim, 0, vlim])
        cb.set_ticklabels([f"$-${vlim:.2f}", "0", f"{vlim:.2f}"])
    cb.ax.tick_params(labelsize=FS["tick"], length=2.6, width=0.6)
    cb.outline.set_linewidth(0.6)
    return cb


# ------------------------------------------------------------------- layout
PLANS = {
    "a": dict(basin_frac=0.60, side=True, gap_after_basin=0.0, foot_h=0.0,
              gutter=0.0,
              desc="legends enlarged in place, spaced to fill the side "
                   "column exactly"),
    "b": dict(basin_frac=0.60, side=True, gap_after_basin=15.0, foot_h=20.0,
              gutter=0.0,
              desc="full-width horizontal colourbar beneath the row each "
                   "serves; key fills the side column"),
    "c": dict(basin_frac=0.60, side=True, gap_after_basin=0.0, foot_h=0.0,
              gutter=21.0,
              desc="colourbars vertical and alongside the panels: discharge "
                   "in a right-hand gutter, change in the side column"),
    "d": dict(basin_frac=1.00, side=False, gap_after_basin=0.0, foot_h=27.0,
              gutter=0.0,
              desc="one shared legend at the foot, scales side by side; "
                   "basin row widened to the full text width to use the "
                   "reclaimed side space"),
    "e": dict(basin_frac=0.60, side=True, gap_after_basin=0.0, foot_h=0.0,
              gutter=0.0,
              desc="key column with two tall vertical scales; north arrow "
                   "moved onto the basin hillshade"),
}


def blocks_for(zones, exts, ann, basin_ext, plan):
    ba = (basin_ext[3] - basin_ext[2]) / (basin_ext[1] - basin_ext[0])
    basin = []
    for i, c in enumerate(("2020", "sfm", "2024")):
        basin.append(f.panel_spec(
            c, basin_ext, ba,
            scale_m=200 if i == 2 else None, scale_corner="tr",
            zone_boxes=True))
    return [
        dict(rows=[basin], mode="uniform",
             headers=[HEAD["2020"], HEAD["sfm"], HEAD["2024"]],
             frac=plan["basin_frac"]),
        dict(rows=f.zoom_panels(zones, ["2020", "2024", "change"], ann,
                                exts, 25),
             mode="uniform",
             headers=[HEAD["2020"], HEAD["2024"], HEAD["change"]],
             row_labels=list(zones.zone_id), frac=1.0),
    ]


def compute(blocks, plan, shrink):
    avail_full = (FIG_W_MM - MARGIN_L - MARGIN_R) * shrink
    geom = []
    y = MARGIN_T
    for bi, blk in enumerate(blocks):
        if bi:
            y += BLOCK_GAP + plan["gap_after_basin"]
        if blk.get("headers"):
            y += HEADER_H
        # The zoom block reserves ZONE_W on the left for its Z1/Z2/Z3
        # labels. Where no legend sits beside the basin row, that row is
        # centred too, so it must reserve the same gutter -- otherwise the
        # two blocks compute different panel widths and their panel areas
        # end up offset from each other by ZONE_W.
        inset = ZONE_W if (blk.get("row_labels") or not plan["side"]) else 0.0
        gutter = plan["gutter"] if bi == 1 else 0.0
        for ri, row in enumerate(blk["rows"]):
            avail = avail_full * blk["frac"] - inset - gutter
            widths, h = f.size_row(row, avail, GAP_X, blk["mode"])
            geom.append(dict(block=bi, row=ri, y=y, widths=widths, h=h,
                             inset=inset, panels=row,
                             align="left" if (bi == 0 and plan["side"])
                             else "center",
                             headers=blk.get("headers") if ri == 0 else None,
                             row_label=(blk["row_labels"][ri]
                                        if blk.get("row_labels") else None)))
            y += h + LETTER_H
            if ri < len(blk["rows"]) - 1:
                y += GAP_Y
    if plan["foot_h"]:
        y += BLOCK_GAP + plan["foot_h"]
    return geom, y + MARGIN_B


def fit_shrink(blocks, plan):
    lo, hi = 0.30, 1.0
    if compute(blocks, plan, hi)[1] <= MAX_H_MM:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if compute(blocks, plan, mid)[1] <= MAX_H_MM:
            lo = mid
        else:
            hi = mid
    return lo


def stack_fill(y0, y1, heights, min_gap=2.5):
    """Vertical positions that spread `heights` to fill y0..y1 exactly.

    This is what closes the white space: the gap is derived from the space
    available rather than fixed, so the block always reaches y1.
    """
    n = len(heights)
    if n == 1:
        return [y0]
    gap = max(min_gap, (y1 - y0 - sum(heights)) / (n - 1))
    ys, y = [], y0
    for h in heights:
        ys.append(y)
        y += h + gap
    return ys


def side_stack(fig, rect_fn, box, items):
    """Draw `items` as full-width rows stacked down the side column.

    Every legend element gets the whole column width and its own vertical
    band, so nothing can ever sit beside anything else and overlap it --
    which is exactly how the first pass put the north arrow's "N" on top of
    "removed flow (2020)". Bands are spread by stack_fill, so the stack
    still reaches the bottom of the column and the white space stays closed.

    items = [(height_mm, draw(x, y, w, h)), ...]
    """
    x0, x1, y0, y1 = box
    w = x1 - x0
    ys = stack_fill(y0, y1, [h for h, _ in items])
    for (h, draw), y in zip(items, ys):
        draw(x0, y, w, h)


# ------------------------------------------------------------------ legends
H_QBAR = BAR_T + 8.5       # bar + tick labels + one-line title
H_DBAR = BAR_T + 13.0      # bar + tick labels + two-line title
H_KEY = 21.5               # three key rows at 8.5 pt
H_NORTH = 15.0


def legend_a(fig, rect_fn, box, state, vlim, geom, fig_h):
    """Enlarged in place: full-width bars, big key, spread to fill."""
    side_stack(fig, rect_fn, box, [
        (H_QBAR, lambda x, y, w, h: hbar(fig, rect_fn, state["q_mappable"],
                                         x, y, w, "q", vlim)),
        (H_DBAR, lambda x, y, w, h: hbar(fig, rect_fn, state["d_mappable"],
                                         x, y, w, "d", vlim, sub=True)),
        (H_KEY, lambda x, y, w, h: draw_key(fig, rect_fn, x, y, w, h)),
        (H_NORTH, lambda x, y, w, h: draw_north(fig, rect_fn, x, y, 12.0, h,
                                                scale=1.2)),
    ])


def legend_b(fig, rect_fn, box, state, vlim, geom, fig_h):
    """Full-width bar beneath the row it serves; key in the side column."""
    g_basin, g_last = geom[0], geom[-1]
    pl = min(g["x0"] + g["inset"] for g in geom)
    span = state["ref_r"] - pl

    # discharge: immediately beneath the basin row, spanning the page. It
    # serves the basin row and the 2020/2024 zoom columns below it.
    y_q = g_basin["y"] + g_basin["h"] + LETTER_H + 3.5
    hbar(fig, rect_fn, state["q_mappable"], pl, y_q, span, "q", vlim)
    # change: at the foot, beneath the change column it serves
    y_d = g_last["y"] + g_last["h"] + LETTER_H + BLOCK_GAP + 1.5
    hbar(fig, rect_fn, state["d_mappable"], pl, y_d, span, "d", vlim,
         sub=True)
    # the side column now carries only the categorical key -- give it room
    side_stack(fig, rect_fn, box, [
        (H_KEY + 6.0, lambda x, y, w, h: draw_key(fig, rect_fn, x, y, w, h,
                                                  title="flow course")),
        (H_NORTH + 3.0, lambda x, y, w, h: draw_north(fig, rect_fn, x, y,
                                                      13.0, h, scale=1.3)),
    ])


def legend_c(fig, rect_fn, box, state, vlim, geom, fig_h):
    """Vertical bars alongside the panels they serve."""
    x0, x1, y0, y1 = box
    zoom = [g for g in geom if g["block"] == 1]
    zt, zb = zoom[0]["y"], zoom[-1]["y"] + zoom[-1]["h"]
    # discharge: right-hand gutter, spanning the whole zoom block
    vbar(fig, rect_fn, state["q_mappable"], state["ref_r"] + 5.0, zt,
         zb - zt, "q", vlim)
    # change: side column, vertical, taking the height the column allows
    h_v = (y1 - y0) - H_KEY - H_NORTH - 8.0
    side_stack(fig, rect_fn, box, [
        (h_v, lambda x, y, w, h: vbar(fig, rect_fn, state["d_mappable"],
                                      x + 1.0, y, h, "d", vlim)),
        (H_KEY, lambda x, y, w, h: draw_key(fig, rect_fn, x, y, w, h)),
        (H_NORTH, lambda x, y, w, h: draw_north(fig, rect_fn, x, y, 12.0, h,
                                                scale=1.2)),
    ])


def legend_d(fig, rect_fn, box, state, vlim, geom, fig_h):
    """One shared block at the foot, the two scales side by side.

    Spans the full text width rather than the panel span, so both bars stay
    generous. Slot widths are explicit and the discharge title is wrapped,
    so no label can reach its neighbour.
    """
    g_last = geom[-1]
    pl = MARGIN_L
    span = FIG_W_MM - MARGIN_L - MARGIN_R
    y = g_last["y"] + g_last["h"] + LETTER_H + BLOCK_GAP

    key_w, north_w, gap = 44.0, 13.0, 9.0
    bar_w = (span - key_w - north_w - 3 * gap) / 2.0
    hbar(fig, rect_fn, state["q_mappable"], pl, y + 3.0, bar_w, "q", vlim,
         short=True)
    hbar(fig, rect_fn, state["d_mappable"], pl + bar_w + gap, y + 3.0,
         bar_w, "d", vlim, sub=True)
    kx = pl + 2 * (bar_w + gap)
    draw_key(fig, rect_fn, kx, y, key_w, H_KEY + 3.0)
    draw_north(fig, rect_fn, kx + key_w + gap, y + 1.0, north_w, 17.0,
               scale=1.3)


def legend_e(fig, rect_fn, box, state, vlim, geom, fig_h):
    """Key column: two tall vertical scales side by side, north on the map."""
    x0, x1, y0, y1 = box
    h_k = 22.0
    h_bar = (y1 - y0) - h_k - 7.0
    # two vertical bars side by side; each gets its own tick/title strip
    col_w = (x1 - x0) / 2.0
    vbar(fig, rect_fn, state["q_mappable"], x0 + 1.0, y0, h_bar, "q", vlim)
    vbar(fig, rect_fn, state["d_mappable"], x0 + col_w + 2.0, y0, h_bar,
         "d", vlim)
    draw_key(fig, rect_fn, x0, y1 - h_k, x1 - x0, h_k, ncol=1)
    # north arrow goes onto the last basin panel instead of the legend
    ax = state["axes"].get((0, 0, 2))
    if ax is not None:
        north_on_panel(ax, state["basin_ext"])


LEGENDS = dict(a=legend_a, b=legend_b, c=legend_c, d=legend_d, e=legend_e)


# -------------------------------------------------------------------- build
def build(key, zones, exts, ann, basin_ext, vlim):
    plan = PLANS[key]
    blocks = blocks_for(zones, exts, ann, basin_ext, plan)
    shrink = fit_shrink(blocks, plan)
    geom, fig_h = compute(blocks, plan, shrink)

    # "2020 lidar (pre-event)" sets the type at ~29.6 mm wide at 7.5 pt.
    # With a 2.5 mm gutter between panels the full names still clear each
    # other down to a panel width of about 25 mm -- that is what the
    # approved variant (a) does at 27.8 mm. Below that they touch, so fall
    # back to short names and let the dates live in the caption, the same
    # convention 56_fig_flow_consolidated.py uses for its 4-panel basin rows.
    SHORT = ["2020 lidar", "2024 SfM", "2024 lidar"]
    for g in geom:
        if g["block"] == 0 and g["headers"] and g["widths"][0] < 25.0:
            g["headers"] = SHORT

    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
    state = dict(fig_h=fig_h, axes={}, basin_ext=basin_ext)

    def rect_fn(x, ytop, w, h):
        return [x / FIG_W_MM, 1 - (ytop + h) / fig_h, w / FIG_W_MM, h / fig_h]

    # horizontal placement, then pull the left-anchored block out to the
    # same edges as the centred ones (as in 56)
    for g in geom:
        row_w = g["inset"] + sum(g["widths"]) + GAP_X * (len(g["widths"]) - 1)
        if g["block"] == 1:
            row_w += plan["gutter"]
        g["x0"] = (MARGIN_L if g["align"] == "left"
                   else (FIG_W_MM - row_w) / 2)
    spans = [(g["x0"] + g["inset"],
              g["x0"] + g["inset"] + sum(g["widths"]) +
              GAP_X * (len(g["widths"]) - 1))
             for g in geom if g["align"] == "center"]
    ref_l, ref_r = min(a for a, _ in spans), max(b for _, b in spans)
    for g in geom:
        if g["align"] == "left":
            g["x0"] = ref_l - g["inset"]
    state["ref_r"] = ref_r

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for g in geom:
        x = g["x0"] + g["inset"]
        for i, (p, w) in enumerate(zip(g["panels"], g["widths"])):
            if g["headers"] is not None:
                fig.text((x + w / 2) / FIG_W_MM,
                         1 - (g["y"] - HEADER_H * 0.30) / fig_h,
                         g["headers"][i], ha="center", va="baseline",
                         fontsize=FS["header"])
            ax = f.render_panel(fig, rect_fn, x, g["y"], w, g["h"], p,
                                next(letters), zones, vlim, state)
            state["axes"][(g["block"], g["row"], i)] = ax
            x += w + GAP_X
        if g["row_label"]:
            fig.text((g["x0"] + g["inset"] / 2) / FIG_W_MM,
                     1 - (g["y"] + g["h"] / 2) / fig_h, g["row_label"],
                     ha="center", va="center", rotation=90,
                     fontsize=FS["zone"], fontweight="bold")

    # the side column: from the right edge of the basin row to the right
    # edge of the zoom block, and from the basin row top down to just above
    # the zoom headers. Filling this box IS the fix.
    g0 = geom[0]
    g1 = next(g for g in geom if g["block"] == 1)
    bx = g0["x0"] + g0["inset"] + sum(g0["widths"]) + \
        GAP_X * (len(g0["widths"]) - 1)
    bottom = g1["y"] - HEADER_H - 1.5
    if plan["gap_after_basin"]:
        # a full-width colourbar occupies that gap and runs under the side
        # column too, so the column must stop above it
        bottom = g0["y"] + g0["h"] + LETTER_H
    box = (bx + 8.0, ref_r, g0["y"], bottom)
    state["side_box"] = box
    LEGENDS[key](fig, rect_fn, box, state, vlim, geom, fig_h)

    name = f"fig_flow_consolidated_v2_{key}"
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    fill = (box[3] - box[2]) if plan["side"] else 0.0
    return dict(key=key, desc=plan["desc"], png=f"{name}.png", h=fig_h,
                shrink=shrink, side_h=fill,
                basin_w=g0["widths"][0], zoom_w=g1["widths"][0],
                npanels=sum(len(g["panels"]) for g in geom))


def main():
    zones = pd.read_csv(f.TABLES / "flow_change_zones.csv")
    with rasterio.open(f.TIF["2020"]) as src:
        a = src.read(1, masked=True)
        t = src.transform
    r, c = np.nonzero(~np.ma.getmaskarray(a))
    basin_ext = (t.c + c.min() * t.a, t.c + (c.max() + 1) * t.a,
                 t.f + (r.max() + 1) * t.e, t.f + r.min() * t.e)
    ch, _ = f.read_window(f.TIF["change"], basin_ext)
    vlim = 3 * np.nanstd(np.asarray(ch.filled(np.nan), dtype=float))

    exts, ann = {}, {}
    for _, z in zones.iterrows():
        exts[z.zone_id] = (z.e_min - f.PAD_M, z.e_max + f.PAD_M,
                           z.n_min - f.PAD_M, z.n_max + f.PAD_M)
        ann[z.zone_id] = f.zone_annotation(
            exts[z.zone_id], (z.e_min, z.e_max, z.n_min, z.n_max))

    print(f"change limit +/-{vlim:.4f}\n")
    rows = []
    for key in "abcde":
        rows.append(build(key, zones, exts, ann, basin_ext, vlim))
        r_ = rows[-1]
        print(f"  v2_{r_['key']}  174 x {r_['h']:.1f} mm  "
              f"{r_['npanels']} panels  shrink {r_['shrink']:.2f}  "
              f"side column {r_['side_h']:.1f} mm")
        log(f"fig_flow_consolidated_v2_{key}: 174x{r_['h']:.1f} mm, "
            f"shrink {r_['shrink']:.2f}", run=RUN)

    md = ["# Consolidated flow figure -- v2, legend and colour-key variants",
          "",
          "All 174 mm wide, 300 dpi PNG + vector PDF, Type 42 fonts.",
          "Panel content, overlays and annotation identical to "
          "`fig_flow_consolidated_a`; only the legends and colour keys "
          "differ.",
          f"Discharge {f.CMAP_Q} log {f.VMIN:g}-{f.VMAX:g}; "
          f"change {f.CMAP_D} +/-{vlim:.4f} (3 SD). Both colourblind-safe.",
          "Tick labels 7.5 pt, key text 8.5 pt, bars "
          f"{BAR_T:g} mm thick (was 3.0 mm).",
          "",
          "Panel widths are the cost of each legend placement: a legend in "
          "the side column beside the basin row is free, because that row "
          "has to be narrow for the page to fit at all. A legend in the "
          "vertical flow (b, d) is paid for out of panel area.",
          "",
          "| file | basin panel (mm) | zoom panel (mm) | what distinguishes "
          "it |", "|---|---|---|---|"]
    for r_ in rows:
        md.append(f"| `{r_['png']}` | {r_['basin_w']:.1f} | "
                  f"{r_['zoom_w']:.1f} | {r_['desc']} |")
    (OUT / "VARIANTS_V2.md").write_text("\n".join(md) + "\n")
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
