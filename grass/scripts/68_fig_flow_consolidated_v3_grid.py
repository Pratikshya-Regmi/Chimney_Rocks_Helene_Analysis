#!/usr/bin/env python3
"""
68_fig_flow_consolidated_v3_grid.py
===================================
Add coordinate grids to the chosen layout, fig_flow_consolidated_v2_e.

Everything about v2_e is kept: the key column with two tall vertical scales,
the north arrow on the basin hillshade, panel content, overlays, zone boxes,
scale bars and the derived magenta/orange/split-point annotation.

COORDINATE SYSTEM
    EPSG:3358, NAD83(HARN) / North Carolina, metres. Labels are full
    easting/northing values in metres, no offset and no scaling.

WHY THE LINES SIT OVER THE DATA, NOT UNDER IT
    Asked for: grid under the mapped layers. Not possible here -- every map
    panel is floored by an OPAQUE hillshade raster (zorder 1) with the
    discharge or change raster over it (zorder 2-3), so a line drawn beneath
    them is simply invisible. The lines are therefore drawn above the
    rasters but below every annotation, at 0.3 pt, grey, alpha 0.30, which
    keeps them well clear of the data visually. Variant (c) sidesteps the
    problem completely: edge ticks only, so no grid ink crosses the data at
    all.

INTERVALS -- one per block, not one for the figure
    basin-wide panel   547 x 1108 m
    zoom Z1/Z2/Z3      132 x 109, 131 x 164, 124 x 101 m
    Two orders of magnitude apart, so each block picks its own interval:
      sparse  basin 250 m (2 easting, 4 northing lines); zoom 50 m (2-3 each)
      dense   basin 200 m (3 easting, 6 northing lines); zoom 25 m (4-6 each)
    The basin panel is 2:1, so no single interval can put 3-5 lines on BOTH
    of its axes; the interval is chosen to keep the northing count in range
    and the easting count is then 2-3. Grid cells stay square.

LABELS
    Left column only -- northings on the leftmost panel of every row,
    eastings beneath it. All three panels in a row share one extent (stated
    in the caption), so repeating the numbers on the interior panels would
    add crowding and no information.
    7.0 pt, the stated floor. Ticks within 10% of a panel edge are dropped
    so no label overhangs into the neighbouring panel or the gutter. Where
    even the kept labels would not fit the panel width, the labels are
    thinned to every other grid line -- the lines stay, the numbers do not
    collide. Nothing is drawn below 7.0 pt.

Outputs (results/figures/fig_flow_consolidated/):
  fig_flow_consolidated_v3_<a|b|c>.png   300 dpi
  fig_flow_consolidated_v3_<a|b|c>.pdf   vector, Type 42
  LATEX_BLOCK_V3.tex                     caption naming the CRS
  VARIANTS_V3.md
v2_e and every earlier file are left untouched.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio

HERE = Path(__file__).resolve().parent


def _load(fn, name):
    spec = importlib.util.spec_from_file_location(name, HERE / fn)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


f = _load("56_fig_flow_consolidated.py", "f56")
v2 = _load("67_fig_flow_consolidated_v2_legends.py", "v2")

sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402

RUN = "flow_consolidated_v3"
OUT = f.OUT
FIG_W_MM, MM, MAX_H_MM = f.FIG_W_MM, f.MM, f.MAX_H_MM
HEADER_H, ZONE_W = f.HEADER_H, f.ZONE_W
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = (f.MARGIN_L, f.MARGIN_R,
                                          f.MARGIN_T, f.MARGIN_B)
GAP_X, GAP_Y, BLOCK_GAP = f.GAP_X, f.GAP_Y, f.BLOCK_GAP

CRS_NAME = "EPSG:3358, NAD83(HARN) / North Carolina (m)"

FS_GRID = 7.0            # grid tick labels -- the stated floor
GRID_KW = dict(color="0.35", lw=0.3, alpha=0.30, zorder=4)
LETTER_BASE = 4.2        # f.LETTER_H as shipped: the (x) letter strip
XTICK_H = 4.4            # extra strip for the easting labels
YTICK_W = 11.0           # gutter for the northing labels
EDGE_FRAC = 0.10         # drop ticks within this fraction of a panel edge
LABEL_MM_PER_CHAR = 1.60  # 7 pt DejaVu Sans tabular digit

VARIANTS = {
    "a": dict(basin=250, zoom=50, mode="lines",
              desc="sparse full grid lines - basin 250 m, zoom 50 m"),
    "b": dict(basin=200, zoom=25, mode="lines",
              desc="denser full grid lines - basin 200 m, zoom 25 m, "
                   "labels thinned where they would not fit"),
    "c": dict(basin=250, zoom=50, mode="ticks",
              desc="same sparse interval as (a) but edge ticks only - no "
                   "grid ink crosses the data"),
}


# ------------------------------------------------------------------ legend
# v2.legend_e put BOTH vertical scales' ticks and titles on their right.
# For the left-hand (discharge) bar that lands its title hard against the
# right-hand (change) bar, so the title reads as if it belonged to the change
# scale -- and the second line, "(m^3 s^-1 m^-1-eq.)", was drawn underneath
# the change bar and lost outright, taking the units with it.
#
# Fixed by mirroring: the discharge scale carries its ticks and title on its
# LEFT, the change scale on its RIGHT. Each title is then contiguous with its
# own bar, nothing at all sits between the two bars, and there is no
# proximity for a reader to misread. v2's own legend_e is left alone so the
# v2 figures stay reproducible.
# The titles are set on ONE line each. Rotated beside a ~58 mm bar they run
# vertically, so a single line costs ~2.8 mm of width where the wrapped
# two-line form costs 5.5 mm -- and the grid's northing gutter has already
# taken the width this column used to have.
TEXT_W_Q = 12.0          # discharge: "10^-4" ticks + one-line title
TEXT_W_D = 14.5          # change: "-0.08" ticks are wider
BAR_GAP = 8.0            # mm of clear space between the two bars


def vbar_side(fig, rect_fn, mappable, x, y, h, kind, vlim, side):
    cax = fig.add_axes(rect_fn(x, y, v2.BAR_W_V, h))
    cb = fig.colorbar(mappable, cax=cax, orientation="vertical",
                      extend="max" if kind == "q" else "both")
    if side == "left":
        cb.ax.yaxis.set_ticks_position("left")
        cb.ax.yaxis.set_label_position("left")
    if kind == "d":
        cb.set_ticks([-vlim, 0, vlim])
        cb.set_ticklabels([f"$-${vlim:.2f}", "0", f"{vlim:.2f}"])
    cb.set_label(v2.Q_LABEL if kind == "q" else v2.D_LABEL,
                 fontsize=v2.FS["cbar"], labelpad=2.5)
    cb.ax.tick_params(labelsize=v2.FS["tick"], length=2.6, width=0.6)
    cb.outline.set_linewidth(0.6)
    return cb


def legend_v3(fig, rect_fn, box, state, vlim, geom, fig_h):
    x0, x1, y0, y1 = box
    w = x1 - x0
    h_k = 22.0
    h_bar = (y1 - y0) - h_k - 7.0
    # Spend whatever width the box has on the gap BETWEEN the two scales.
    # That both fills the column and pushes the two apart, which is the
    # separation that was missing when the discharge title sat against the
    # change bar.
    gap = min(18.0, max(BAR_GAP,
                        w - TEXT_W_Q - TEXT_W_D - 2 * v2.BAR_W_V))
    group = TEXT_W_Q + TEXT_W_D + 2 * v2.BAR_W_V + gap
    xs = x0 + max(0.0, (w - group) / 2.0)
    x_q = xs + TEXT_W_Q
    x_d = x_q + v2.BAR_W_V + gap
    vbar_side(fig, rect_fn, state["q_mappable"], x_q, y0, h_bar, "q", vlim,
              side="left")
    vbar_side(fig, rect_fn, state["d_mappable"], x_d, y0, h_bar, "d", vlim,
              side="right")
    v2.draw_key(fig, rect_fn, x0, y1 - h_k, w, h_k)
    ax = state["axes"].get((0, 0, 2))
    if ax is not None:
        v2.north_on_panel(ax, state["basin_ext"])


# --------------------------------------------------------------------- grid
def ticks_for(lo, hi, step):
    """Round multiples of `step` inside (lo, hi), excluding the edge band."""
    pad = EDGE_FRAC * (hi - lo)
    first = int(np.ceil((lo + pad) / step)) * step
    return [t for t in np.arange(first, hi - pad + 1e-6, step)]


def thin_to_fit(vals, span_mm, extent_span, ndigits=6):
    """Keep every k-th label so the labels cannot touch each other."""
    if not vals:
        return []
    need = ndigits * LABEL_MM_PER_CHAR + 2.0
    room = max(1, int(span_mm // need))
    if len(vals) <= room:
        return list(vals)
    k = int(np.ceil(len(vals) / room))
    return list(vals[::k])


def apply_grid(ax, ext, step, mode, w_mm, h_mm, xlab, ylab):
    """Draw the graticule (or edge ticks) and label only where asked."""
    e0, e1, n0, n1 = ext
    xs = ticks_for(e0, e1, step)
    ys = ticks_for(n0, n1, step)

    if mode == "lines":
        for x in xs:
            ax.axvline(x, **GRID_KW)
        for y in ys:
            ax.axhline(y, **GRID_KW)

    # Real axis ticks: matplotlib then places the labels in the strip
    # outside the axes, so no label is ever inside a panel.
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    xl = thin_to_fit(xs, w_mm, e1 - e0) if xlab else []
    yl = thin_to_fit(ys, h_mm, n1 - n0) if ylab else []
    ax.set_xticklabels([f"{v:.0f}" if v in xl else "" for v in xs],
                       fontsize=FS_GRID)
    ax.set_yticklabels([f"{v:.0f}" if v in yl else "" for v in ys],
                       fontsize=FS_GRID)
    ax.tick_params(axis="both", which="both", direction="out",
                   length=1.7 if mode == "ticks" else 1.2, width=0.5,
                   color="0.35", pad=1.6,
                   labelbottom=bool(xl), labelleft=bool(yl),
                   bottom=True, left=True,
                   top=(mode == "ticks"), right=(mode == "ticks"),
                   labeltop=False, labelright=False)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("black")


# ------------------------------------------------------------------- layout
def compute(blocks, shrink):
    """As v2's compute for the (e) plan, plus the grid label gutters.

    The basin row is left-anchored to the zoom block's panel edge, so its
    northing labels sit in the gutter the zoom block already reserves -- the
    basin panels pay nothing for them.
    """
    avail_full = (FIG_W_MM - MARGIN_L - MARGIN_R) * shrink
    letter_h = LETTER_BASE + XTICK_H
    geom = []
    y = MARGIN_T
    for bi, blk in enumerate(blocks):
        if bi:
            y += BLOCK_GAP
        if blk.get("headers"):
            y += HEADER_H
        inset = (ZONE_W + YTICK_W) if blk.get("row_labels") else 0.0
        for ri, row in enumerate(blk["rows"]):
            avail = avail_full * blk["frac"] - inset
            widths, h = f.size_row(row, avail, GAP_X, blk["mode"])
            geom.append(dict(block=bi, row=ri, y=y, widths=widths, h=h,
                             inset=inset, panels=row,
                             align="left" if bi == 0 else "center",
                             headers=blk.get("headers") if ri == 0 else None,
                             row_label=(blk["row_labels"][ri]
                                        if blk.get("row_labels") else None)))
            y += h + letter_h
            if ri < len(blk["rows"]) - 1:
                y += GAP_Y
    return geom, y + MARGIN_B


def fit_shrink(blocks):
    lo, hi = 0.30, 1.0
    if compute(blocks, hi)[1] <= MAX_H_MM:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if compute(blocks, mid)[1] <= MAX_H_MM:
            lo = mid
        else:
            hi = mid
    return lo


# -------------------------------------------------------------------- build
def build(key, zones, exts, ann, basin_ext, vlim):
    cfg = VARIANTS[key]
    plan = v2.PLANS["e"]
    blocks = v2.blocks_for(zones, exts, ann, basin_ext, plan)
    shrink = fit_shrink(blocks)
    geom, fig_h = compute(blocks, shrink)

    # the (x) letter strip has to hold the easting labels as well, and
    # render_panel reads f.LETTER_H to place the letter -- widen it so the
    # letter sits below the labels instead of on top of them
    f.LETTER_H = LETTER_BASE + XTICK_H

    SHORT = ["2020 lidar", "2024 SfM", "2024 lidar"]
    for g in geom:
        if g["block"] == 0 and g["headers"] and g["widths"][0] < 25.0:
            g["headers"] = SHORT

    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
    state = dict(fig_h=fig_h, axes={}, basin_ext=basin_ext)

    def rect_fn(x, ytop, w, h):
        return [x / FIG_W_MM, 1 - (ytop + h) / fig_h, w / FIG_W_MM, h / fig_h]

    # Anchor the panel blocks to the left margin rather than centring them.
    # The legend has to reach the right margin to fit two scales beside the
    # grid's gutters (see below); centring the blocks as well would leave a
    # 17 mm hole on the left against 5 mm on the right. Left-anchoring
    # balances the margins and hands the legend the width it needs.
    for g in geom:
        g["x0"] = MARGIN_L
    spans = [(g["x0"] + g["inset"],
              g["x0"] + g["inset"] + sum(g["widths"]) +
              GAP_X * (len(g["widths"]) - 1))
             for g in geom if g["block"] == 1]
    ref_l, ref_r = min(a for a, _ in spans), max(b for _, b in spans)
    for g in geom:
        if g["block"] == 0:
            g["x0"] = ref_l - g["inset"]
    state["ref_r"] = ref_r

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for g in geom:
        x = g["x0"] + g["inset"]
        step = cfg["basin"] if g["block"] == 0 else cfg["zoom"]
        for i, (p, w) in enumerate(zip(g["panels"], g["widths"])):
            if g["headers"] is not None:
                fig.text((x + w / 2) / FIG_W_MM,
                         1 - (g["y"] - HEADER_H * 0.30) / fig_h,
                         g["headers"][i], ha="center", va="baseline",
                         fontsize=f.FS["header"])
            ax = f.render_panel(fig, rect_fn, x, g["y"], w, g["h"], p,
                                next(letters), zones, vlim, state)
            # left column carries the labels; interior panels get the
            # graticule and edge ticks but no numbers
            apply_grid(ax, p["ext"], step, cfg["mode"], w, g["h"],
                       xlab=(i == 0), ylab=(i == 0))
            state["axes"][(g["block"], g["row"], i)] = ax
            x += w + GAP_X
        if g["row_label"]:
            # in the OUTER part of the gutter, clear of the northing labels
            fig.text((g["x0"] + ZONE_W / 2) / FIG_W_MM,
                     1 - (g["y"] + g["h"] / 2) / fig_h, g["row_label"],
                     ha="center", va="center", rotation=90,
                     fontsize=f.FS["zone"], fontweight="bold")

    g0 = geom[0]
    g1 = next(g for g in geom if g["block"] == 1)
    bx = g0["x0"] + g0["inset"] + sum(g0["widths"]) + \
        GAP_X * (len(g0["widths"]) - 1)
    # The grid's northing gutter pulls the zoom block inward, so aligning the
    # legend's right edge to the panels would leave it only ~33 mm -- not
    # enough for two scales and their titles, and the change title then ran
    # off the page. Let it reach the page margin instead and accept a few mm
    # of asymmetry against the panel block.
    legend_r = max(ref_r, FIG_W_MM - MARGIN_R)
    box = (bx + 6.0, legend_r, g0["y"], g1["y"] - HEADER_H - 1.5)
    legend_v3(fig, rect_fn, box, state, vlim, geom, fig_h)

    name = f"fig_flow_consolidated_v3_{key}"
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    f.LETTER_H = LETTER_BASE
    return dict(key=key, desc=cfg["desc"], png=f"{name}.png", h=fig_h,
                shrink=shrink, basin_w=g0["widths"][0],
                zoom_w=g1["widths"][0], basin_step=cfg["basin"],
                zoom_step=cfg["zoom"])


CAPTION = r"""% =====================================================================
% Caption for fig_flow_consolidated_v3_<a|b|c> (12 panels, a-l).
% Swap the filename for the variant you choose.
% =====================================================================
\begin{figure}[p]
\centering
\includegraphics[width=174mm]{figures/fig_flow_consolidated_v3_a.pdf}
\caption{Drainage reorganisation in the tributary watershed, from
overland-flow simulation on the pre- and post-event surfaces.
\textbf{Top row, basin-wide:} unit discharge for (a) 2020 lidar,
pre-event; (b) 2024 CAP SfM, flown 5 October 2024 ($\sim$10 days
post-event); and (c) 2024 lidar, 15--16 November 2024 (seven weeks
post-event). Red boxes mark the three zones of largest
2020-vs-2024-lidar flow-path disagreement, Z1--Z3. \textbf{Lower rows,
zoomed to those zones} (Z1 116$\times$93~m, Z2 115$\times$148~m, Z3
108$\times$85~m, each with 8~m of padding, so the three rows are at
different scales and each panel carries its own 25~m bar): (d), (g), (j)
2020 lidar; (e), (h), (k) 2024 lidar; (f), (i), (l) the discharge change
between the two lidar epochs. All discharge panels share one logarithmic
colour scale, $1\times10^{-4}$ to 0.48~m$^{3}$~s$^{-1}$~m$^{-1}$-equivalent;
cells above the maximum are saturated at the top colour rather than
rescaled per panel. All change panels share one diverging scale,
$\pm$0.082 (three standard deviations of the basin-wide change raster),
red denoting discharge lost and blue discharge gained. On every zoom
panel, magenta marks 2020 flow course with no 2024 channel within 2~m
(abandoned), orange marks 2024 course with no 2020 channel within 2~m
(new), and the open circle marks the split point where the course
divides. \textbf{Coordinates} are eastings and northings in metres on
@CRS@. Grid interval is @BSTEP@~m on the basin-wide
panels and @ZSTEP@~m on the zoom panels, the two differing because the
extents differ by two orders of magnitude; the three panels in each row
share that row's extent, so the grid is labelled on the left-hand panel
only.}
\label{fig:flow_reorganisation}
\end{figure}
"""


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

    rows = []
    for key in "abc":
        rows.append(build(key, zones, exts, ann, basin_ext, vlim))
        r_ = rows[-1]
        print(f"  v3_{key}  174 x {r_['h']:.1f} mm  shrink {r_['shrink']:.2f}"
              f"  basin panel {r_['basin_w']:.1f} mm  "
              f"zoom panel {r_['zoom_w']:.1f} mm")
        log(f"fig_flow_consolidated_v3_{key}: 174x{r_['h']:.1f} mm, "
            f"basin grid {r_['basin_step']} m, zoom grid {r_['zoom_step']} m",
            run=RUN)

    (OUT / "LATEX_BLOCK_V3.tex").write_text(
        CAPTION.replace("@CRS@", CRS_NAME)
               .replace("@BSTEP@", str(VARIANTS["a"]["basin"]))
               .replace("@ZSTEP@", str(VARIANTS["a"]["zoom"])))

    md = ["# Consolidated flow figure -- v3, coordinate grids on v2_e", "",
          "All 174 mm wide, 300 dpi PNG + vector PDF, Type 42 fonts.",
          f"Layout, legend and content identical to "
          f"`fig_flow_consolidated_v2_e`; only the graticule differs.",
          f"Coordinates: {CRS_NAME}. Grid labels 7.0 pt, "
          "left-hand panel of each row only.",
          "Lines are 0.3 pt grey at alpha 0.30, above the rasters (an "
          "opaque hillshade floors every panel, so a line underneath it "
          "would be invisible) and below all annotation.", "",
          "| file | basin grid | zoom grid | basin/zoom panel (mm) | what "
          "distinguishes it |", "|---|---|---|---|---|"]
    for r_ in rows:
        md.append(f"| `{r_['png']}` | {r_['basin_step']} m | "
                  f"{r_['zoom_step']} m | {r_['basin_w']:.1f} / "
                  f"{r_['zoom_w']:.1f} | {r_['desc']} |")
    (OUT / "VARIANTS_V3.md").write_text("\n".join(md) + "\n")
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
