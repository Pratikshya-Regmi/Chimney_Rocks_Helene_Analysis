#!/usr/bin/env python3
"""
43_transect_label_variants.py
==============================
Four legibility treatments for the transect endpoint labels (A/A', B/B',
C/C') on the map panel of fig_transects_v4, for the advisor to choose from:
"very hard to read the letters representing the profiles on the map."

    (a) white text, thick dark stroke, larger type, offset further along
        each transect so the mid-basin cluster separates
    (b) dark text in small opaque rounded boxes
    (c) colour-coded transects + labels, with a legend keyed to the profile
        panel titles (Okabe-Ito, colour-blind safe)
    (d) filled circle markers at each endpoint with the letter inside

Everything else in the figure is unchanged: same layout, same map content,
same three profile pairs, same 174 x 205 mm page. Only the transect line and
label rendering differs, by monkeypatching the `draw_transects` symbol that
make_transect_figure_v3.draw_locator_v3 calls, so no shared module is edited
and fig_transects_v4.py keeps producing the current figure untouched.

TWO DEFECTS FOUND IN THE CURRENT LABELS (both fixed in every variant):

1. The prime is an ASCII apostrophe "'" and the label carries
   patheffects.withStroke(linewidth=2). On a 7 pt glyph a 2 pt stroke is
   wider than the apostrophe itself, so the stroke closes over it and it
   renders as a solid block: "A'" reads as "A#" at print size. Every variant
   uses the true prime U+2032 and a stroke sized to the glyph.
2. The C label sits directly on the C-C' line's first segment.

CURRENT SIZE, FOR REFERENCE: fontsize=7 pt bold (map_style.draw_transects
default). The figure is saved without bbox_inches, and the PDF measures
exactly 174.00 x 205.00 mm, so 7 pt is the true final print size.

Each variant is checked programmatically for overlap of every label against
every transect line, the watershed boundary, and every other label; the
report prints per variant.

Outputs (results/diagnostics/, these are choices to pick from, not yet a
manuscript deliverable):
    fig_transects_v4_labels_a.png / .pdf   ... _b, _c, _d
    results/logs/transect_label_variants_<date>.log
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grass_env import log                                    # noqa: E402
import map_style                                             # noqa: E402
from map_style import style_map_panel, _endpoint_label_pos   # noqa: E402
import make_transect_figure_v3 as v3                         # noqa: E402
from make_transect_figure_v2 import (                        # noqa: E402
    export_map_rasters, load_tables, draw_profile_pair, LOD, FIGURES,
)

RUN = "transect_label_variants"
ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM = 174
FIG_H_MM = 205
MAP_COL_W_MM = 96
PROFILE_COL_W_MM = FIG_W_MM - MAP_COL_W_MM - 6

PRIME = "′"        # true prime, not an ASCII apostrophe
SPECS = [("transect1_crosssection", "A", f"A{PRIME}"),
         ("transect2_longitudinal", "B", f"B{PRIME}"),
         ("transect3_westcorridor", "C", f"C{PRIME}")]

# Okabe-Ito, chosen to stay clear of the DoD ramp's vermillion (#D55E00)
# and blue (#0072B2) so a transect line is never mistaken for change data.
TRANSECT_COLORS = {"transect1_crosssection": "#009E73",   # bluish green
                   "transect2_longitudinal": "#CC79A7",   # reddish purple
                   "transect3_westcorridor": "#F0E442"}   # yellow
PANEL_TITLES = {"transect1_crosssection": f"A-A{PRIME}  cross-section",
                "transect2_longitudinal": f"B-B{PRIME}  longitudinal profile",
                "transect3_westcorridor": f"C-C{PRIME}  western corridor"}

VARIANTS = {
    # clearance_m ~ the label's half-diagonal in map metres (the map spans
    # ~948 m across a ~96 mm column, so 1 pt ~ 3.5 m) plus a little air.
    "a": dict(fontsize=10.0, offset_m=58, side_m=24, min_sep_m=78,
              clearance_m=34, max_shift_m=120),
    "b": dict(fontsize=9.0, offset_m=48, side_m=20, min_sep_m=66,
              clearance_m=34, max_shift_m=115),
    "c": dict(fontsize=9.5, offset_m=50, side_m=20, min_sep_m=70,
              clearance_m=34, max_shift_m=118),
    "d": dict(fontsize=7.5, offset_m=30, side_m=12, min_sep_m=58,
              marker_pt=15.0, clearance_m=32, max_shift_m=100),
}


# ----------------------------------------------------------------- drawing
def _repel(entries, min_sep_m, iters=60):
    """Push apart any two label anchors closer than min_sep_m."""
    for _ in range(iters):
        moved = False
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                xi, yi = entries[i]["pos"]
                xj, yj = entries[j]["pos"]
                dx, dy = xj - xi, yj - yi
                d = (dx * dx + dy * dy) ** 0.5
                if d < min_sep_m:
                    if d == 0:
                        ux, uy, push = 1.0, 0.0, min_sep_m / 2
                    else:
                        push = (min_sep_m - d) / 2
                        ux, uy = dx / d, dy / d
                    entries[i]["pos"][0] -= ux * push
                    entries[i]["pos"][1] -= uy * push
                    entries[j]["pos"][0] += ux * push
                    entries[j]["pos"][1] += uy * push
                    moved = True
        if not moved:
            break
    return entries


def _densify(pts, step_m=4.0):
    """Resample a polyline so nearest-point tests can't slip between vertices."""
    out = []
    for k in range(len(pts) - 1):
        p0, p1 = pts[k], pts[k + 1]
        d = float(np.hypot(*(p1 - p0)))
        n = max(2, int(d / step_m) + 1)
        out.append(np.column_stack([np.linspace(p0[0], p1[0], n),
                                    np.linspace(p0[1], p1[1], n)]))
    return np.vstack(out) if out else np.asarray(pts, dtype=float)


def _avoid_lines(entries, obstacles, clearance_m, min_sep_m, max_shift_m,
                 iters=40):
    """Push labels off the lines they sit on, then re-separate them.

    The endpoint offset alone cannot fix this: A's outward direction points
    straight at the C-C' path and the basin boundary, so scaling the offset
    just slides A further along the collision. Displacement from the label's
    own endpoint is capped at max_shift_m so a label never drifts far enough
    to lose its association with the endpoint (the leader line carries the
    rest).
    """
    if not obstacles:
        return entries
    cloud = np.vstack(obstacles)
    for _ in range(iters):
        moved = False
        for e in entries:
            pos = np.array(e["pos"], dtype=float)
            d2 = np.sum((cloud - pos) ** 2, axis=1)
            k = int(np.argmin(d2))
            dist = float(np.sqrt(d2[k]))
            if dist < clearance_m:
                away = pos - cloud[k]
                n = float(np.hypot(*away))
                ux, uy = (away / n) if n > 1e-9 else (0.0, 1.0)
                pos = pos + np.array([ux, uy]) * (clearance_m - dist + 0.5)
                anchor = np.array(e["anchor"], dtype=float)
                delta = pos - anchor
                dn = float(np.hypot(*delta))
                if dn > max_shift_m:            # keep it tied to its endpoint
                    pos = anchor + delta / dn * max_shift_m
                e["pos"] = [float(pos[0]), float(pos[1])]
                moved = True
        _repel(entries, min_sep_m, iters=20)
        if not moved:
            break
    return entries


def draw_transects_variant(ax, geom, specs, variant="a", **_ignored):
    """Draw the transect lines and endpoint labels in one of four styles.

    Signature matches map_style.draw_transects so it can stand in for it.
    Records the drawn Text objects and line geometries on the axes for the
    later overlap check.
    """
    cfg = VARIANTS[variant]
    fs = cfg["fontsize"]
    specs = SPECS                      # always the U+2032 labels

    entries, polylines, handles = [], [], []
    for name, start_lab, end_lab in specs:
        sub = geom[geom["transect"] == name].sort_values("vertex_order")
        xs = sub["easting"].to_numpy()
        ys = sub["northing"].to_numpy()
        polylines.append(np.column_stack([xs, ys]))

        if variant == "c":
            col = TRANSECT_COLORS[name]
            ax.plot(xs, ys, color="black", linewidth=3.0, zorder=6,
                    solid_capstyle="round")
            ax.plot(xs, ys, color=col, linewidth=1.6, zorder=7,
                    solid_capstyle="round")
            handles.append(Line2D([], [], color=col, linewidth=1.8,
                                  label=PANEL_TITLES[name]))
        else:
            col = "black"
            ax.plot(xs, ys, color="white", linewidth=2.6, zorder=6,
                    solid_capstyle="round")
            ax.plot(xs, ys, color="black", linewidth=1.2, zorder=7,
                    solid_capstyle="round")

        x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
        x0p, y0p = (xs[1], ys[1]) if len(xs) > 1 else (x1, y1)
        x1p, y1p = (xs[-2], ys[-2]) if len(xs) > 1 else (x0, y0)
        for lab, (ex, ey), (px, py) in (
                (start_lab, (x0, y0), (x0p, y0p)),
                (end_lab, (x1, y1), (x1p, y1p))):
            entries.append({
                "label": lab, "anchor": (ex, ey), "color": col,
                "pos": _endpoint_label_pos(ex, ey, px, py,
                                           cfg["offset_m"], cfg["side_m"])})

    _repel(entries, cfg["min_sep_m"])

    # keep labels off the transect lines and the watershed boundary
    obstacles = [_densify(p) for p in polylines]
    bnd = getattr(ax, "_variant_boundary", None)
    if bnd is not None:
        obstacles.append(_densify(bnd))
    _avoid_lines(entries, obstacles, clearance_m=cfg["clearance_m"],
                 min_sep_m=cfg["min_sep_m"], max_shift_m=cfg["max_shift_m"])

    texts = []
    for e in entries:
        lx, ly = e["pos"]
        ax_, ay_ = e["anchor"]

        # hairline leader from the true endpoint toward the label, stopping
        # well short of the glyph so it never runs into the lettering
        lead_x = ax_ + 0.62 * (lx - ax_)
        lead_y = ay_ + 0.62 * (ly - ay_)
        ax.plot([ax_, lead_x], [ay_, lead_y],
                color="black" if variant != "c" else e["color"],
                linewidth=0.5, zorder=8, solid_capstyle="butt",
                path_effects=[pe.withStroke(linewidth=1.6,
                                            foreground="white")])

        if variant == "a":
            t = ax.text(lx, ly, e["label"], fontsize=fs, fontweight="bold",
                        color="white", ha="center", va="center", zorder=10,
                        path_effects=[pe.withStroke(linewidth=3.0,
                                                    foreground="black")])
        elif variant == "b":
            t = ax.text(lx, ly, e["label"], fontsize=fs, fontweight="bold",
                        color="black", ha="center", va="center", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                                  alpha=0.88, edgecolor="0.25", linewidth=0.6))
        elif variant == "c":
            t = ax.text(lx, ly, e["label"], fontsize=fs, fontweight="bold",
                        color=e["color"], ha="center", va="center", zorder=10,
                        path_effects=[pe.withStroke(linewidth=2.6,
                                                    foreground="black")])
        else:  # (d) circle marker with the letter inside
            ax.plot([lx], [ly], marker="o", markersize=cfg["marker_pt"],
                    markerfacecolor="white", markeredgecolor="black",
                    markeredgewidth=1.0, zorder=9, linestyle="none")
            t = ax.text(lx, ly, e["label"], fontsize=fs, fontweight="bold",
                        color="black", ha="center", va="center", zorder=10)
        texts.append(t)

    if variant == "c" and handles:
        leg = ax.legend(handles=handles, loc="upper left", fontsize=6.2,
                        frameon=True, framealpha=0.92, borderpad=0.4,
                        handlelength=1.6, labelspacing=0.35)
        leg.get_frame().set_linewidth(0.5)
        leg.set_zorder(12)

    ax._variant_texts = texts
    ax._variant_polylines = polylines
    return texts


# ----------------------------------------------------------------- checking
def _seg_hits_bbox(p0, p1, bbox, step_px=1.0):
    """Does the display-space segment p0->p1 pass through bbox?"""
    d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    n = max(2, int(d / step_px) + 1)
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    return bool(np.any((xs >= bbox.x0) & (xs <= bbox.x1) &
                       (ys >= bbox.y0) & (ys <= bbox.y1)))


def check_overlaps(fig, ax, boundary, pad_px=1.0):
    """Report any label overlapping a transect line, the basin boundary, or
    another label. Works in display space on the rendered figure, so it
    measures what actually prints, not the intended geometry."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = getattr(ax, "_variant_texts", [])
    polys = getattr(ax, "_variant_polylines", [])

    boxes = []
    for t in texts:
        bb = t.get_window_extent(renderer=rend).expanded(1.0, 1.0)
        bb = bb.padded(pad_px)
        boxes.append((t.get_text(), bb))

    problems = []
    named = [("transect line", ax.transData.transform(p)) for p in polys]
    named.append(("watershed boundary", ax.transData.transform(
        np.column_stack([boundary["easting"].to_numpy(),
                         boundary["northing"].to_numpy()]))))
    for lab, bb in boxes:
        for what, pts in named:
            hit = any(_seg_hits_bbox(pts[k], pts[k + 1], bb)
                      for k in range(len(pts) - 1))
            if hit:
                problems.append(f"label {lab!r} overlaps a {what}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                problems.append(
                    f"labels {boxes[i][0]!r} and {boxes[j][0]!r} overlap")
    return problems


# ----------------------------------------------------------------- figure
def build_figure(variant, tables):
    """Reproduce fig_transects_v4's layout exactly, with one label style."""
    t1, t2, t3, geom, boundary = tables

    orig = v3.draw_transects
    v3.draw_transects = (lambda ax, g, specs, **kw:
                         draw_transects_variant(ax, g, specs, variant=variant))
    try:
        fig = plt.figure(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))
        outer = fig.add_gridspec(1, 2,
                                 width_ratios=[MAP_COL_W_MM, PROFILE_COL_W_MM],
                                 wspace=0.30)
        ax_map = fig.add_subplot(outer[0, 0])
        ax_map._variant_boundary = np.column_stack(
            [boundary["easting"].to_numpy(), boundary["northing"].to_numpy()])
        im, extent = v3.draw_locator_v3(ax_map, geom, boundary)
        avoid = (boundary["easting"].min(), boundary["easting"].max(),
                 boundary["northing"].min(), boundary["northing"].max())
        style_map_panel(ax_map, extent, mappable=im,
                        cbar_label="Elevation change", cbar_units="m",
                        basin_label=None, avoid_bbox=avoid)

        right = outer[0, 1].subgridspec(3, 1, height_ratios=[1, 1, 1],
                                        hspace=0.45)
        pairs = [right[i, 0].subgridspec(2, 1, height_ratios=[2, 1],
                                         hspace=0.08) for i in range(3)]
        axes = []
        for p in pairs:
            ax_e = fig.add_subplot(p[0, 0])
            ax_d = fig.add_subplot(p[1, 0], sharex=ax_e)
            axes.append((ax_e, ax_d))
        titles = [f"A-A{PRIME}  cross-section",
                  f"B-B{PRIME}  longitudinal profile",
                  f"C-C{PRIME}  western corridor"]
        for (ax_e, ax_d), df, title in zip(axes, [t1, t2, t3], titles):
            draw_profile_pair(ax_e, ax_d, df, title)
        axes[0][0].legend(loc="upper left", ncol=2, handlelength=1.4,
                          columnspacing=1.0)

        problems = check_overlaps(fig, ax_map, boundary)

        stem = DIAG / f"fig_transects_v4_labels_{variant}"
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"), dpi=600)
        plt.close(fig)
        return stem, problems
    finally:
        v3.draw_transects = orig


def main():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    print("Exporting GRASS map-panel rasters (once, shared by all variants)...")
    export_map_rasters()
    tables = load_tables()

    log(f"START current label size = 7.0 pt bold (map_style.draw_transects "
        f"default), figure {FIG_W_MM}x{FIG_H_MM} mm saved without "
        f"bbox_inches, so 7.0 pt is the true print size", run=RUN)

    for v in ("a", "b", "c", "d"):
        cfg = VARIANTS[v]
        stem, problems = build_figure(v, tables)
        status = "no overlaps" if not problems else "; ".join(sorted(set(problems)))
        print(f"\n({v}) {stem.name}.png  fontsize={cfg['fontsize']} pt  "
              f"offset={cfg['offset_m']} m  min_sep={cfg['min_sep_m']} m")
        print(f"     overlap check: {status}")
        log(f"variant ({v}): fontsize={cfg['fontsize']} pt offset={cfg['offset_m']} m "
            f"side={cfg['side_m']} m min_sep={cfg['min_sep_m']} m -> "
            f"{stem.name}.png/.pdf | overlap check: {status}", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
