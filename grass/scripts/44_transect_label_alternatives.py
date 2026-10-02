#!/usr/bin/env python3
"""
44_transect_label_alternatives.py
==================================
Second round of transect-label treatments for the fig_transects_v4 map panel,
to the brief's four designs. Everything else in the figure is identical --
traced C-C', scrim, masked DoD, graticule, north arrow and scale bar, and the
three profile panels.

    (a) halo    -- larger white letters, thicker dark stroke
    (b) boxed   -- white letters in a semi-transparent dark box
    (c) margin  -- coloured lines, filled circular endpoint markers, letters
                   OUTSIDE the map frame with leader lines
    (d) number  -- one numbered disc per transect at its midpoint, keyed to
                   the profile panel titles

Geometry and collision machinery is imported from 43_transect_label_variants
rather than re-implemented, so both rounds resolve label placement the same
way (endpoint offset -> pairwise repulsion -> push off lines, displacement
capped so a label stays tied to its own transect).

WHY (b) USES A DARK BOX: the map is pale outside the basin (white scrim at
alpha 0.55) and dark inside (orthophoto at alpha 0.55 over hillshade). A
white box with dark text vanishes into the scrim wherever a label lands
outside the boundary -- C' does. A dark box with white text holds contrast
on both, so that is the direction taken.

TRADE-OFF IN (d), FLAGGED: numbering at the midpoint removes the start/end
distinction that A -> A' carries, so the profiles' distance origin is no
longer readable from the map. A small open tick is drawn at each transect's
distance-0 end to keep that recoverable, but if the reader needs to know
which end is which at a glance, (a)-(c) do it better.

Outputs (results/diagnostics/):
    fig_transects_v4_alt_a.png / .pdf ... _b, _c, _d
    results/logs/transect_label_alternatives_<date>.log
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log                                    # noqa: E402
from map_style import style_map_panel, _endpoint_label_pos   # noqa: E402
import make_transect_figure_v3 as v3                         # noqa: E402
from make_transect_figure_v2 import (                        # noqa: E402
    export_map_rasters, load_tables, draw_profile_pair, LOD,
)

# reuse round 1's placement/collision helpers (module name starts with a digit)
_spec = importlib.util.spec_from_file_location(
    "v43", HERE / "43_transect_label_variants.py")
v43 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v43)

RUN = "transect_label_alternatives"
ROOT = HERE.parents[1]
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM, FIG_H_MM = 174, 205
MAP_COL_W_MM = 96
PROFILE_COL_W_MM = FIG_W_MM - MAP_COL_W_MM - 6

PRIME = "′"
SPECS = [("transect1_crosssection", "A", f"A{PRIME}"),
         ("transect2_longitudinal", "B", f"B{PRIME}"),
         ("transect3_westcorridor", "C", f"C{PRIME}")]

# Okabe-Ito, kept clear of the DoD ramp's vermillion #D55E00 / blue #0072B2
COLORS = {"transect1_crosssection": "#009E73",
          "transect2_longitudinal": "#CC79A7",
          "transect3_westcorridor": "#E69F00"}
NUMBERS = {"transect1_crosssection": "1",
           "transect2_longitudinal": "2",
           "transect3_westcorridor": "3"}

TITLES = {
    "letters": [f"A-A{PRIME}  cross-section",
                f"B-B{PRIME}  longitudinal profile",
                f"C-C{PRIME}  western corridor"],
    "numbers": ["1   cross-section", "2   longitudinal profile",
                "3   western corridor"],
}

CFG = {
    "a": dict(fontsize=11.0, offset_m=58, side_m=24, min_sep_m=80,
              clearance_m=44, max_shift_m=140),
    "b": dict(fontsize=9.5, offset_m=50, side_m=20, min_sep_m=72,
              clearance_m=36, max_shift_m=120),
    "c": dict(fontsize=9.5, offset_m=0, side_m=0, min_sep_m=0,
              clearance_m=0, max_shift_m=0),
    "d": dict(fontsize=9.5, offset_m=0, side_m=0, min_sep_m=0,
              clearance_m=0, max_shift_m=0, marker_pt=18.0),
}


def _polyline(geom, name):
    sub = geom[geom["transect"] == name].sort_values("vertex_order")
    return np.column_stack([sub["easting"].to_numpy(),
                            sub["northing"].to_numpy()])


def _midpoint(pts):
    """Point at half the cumulative path length (not the middle vertex --
    C-C' is a traced sinuous path with uneven vertex spacing)."""
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    half = cum[-1] / 2.0
    k = int(np.searchsorted(cum, half))
    k = max(1, min(k, len(pts) - 1))
    t = (half - cum[k - 1]) / max(seg[k - 1], 1e-9)
    return pts[k - 1] + t * (pts[k] - pts[k - 1])



def _point_at_fraction(pts, f):
    """Point at fraction f of the polyline's cumulative length."""
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.clip(f, 0.0, 1.0) * cum[-1]
    k = int(np.clip(np.searchsorted(cum, target), 1, len(pts) - 1))
    t = (target - cum[k - 1]) / max(seg[k - 1], 1e-9)
    return pts[k - 1] + t * (pts[k] - pts[k - 1])


def _separate_along_paths(polys, boundary=None, sep_markers_m=75.0,
                          clear_lines_m=42.0, lo=0.18, hi=0.82,
                          n_grid=65, passes=6):
    """Choose one point per polyline, ON that polyline, maximising clearance.

    Two constraints, both real: the discs must not collide with each other
    (A-A' and B-B' cross near mid-basin, so their midpoints coincide), and a
    disc must not sit on top of a DIFFERENT transect (a disc straddling two
    lines is exactly the ambiguity this design is meant to remove). Sliding
    along the line satisfies both without breaking the association, which
    moving the disc off its line would.

    Coordinate descent over a grid of path fractions, scoring each candidate
    by the binding constraint. Deterministic: same input, same placement.
    """
    # obstacles for disc i: every OTHER transect, plus the watershed
    # boundary (a disc half off the basin edge reads as badly as one on
    # another line)
    extra = [_densify_local(np.asarray(boundary, dtype=float))] if boundary is not None else []
    others = [np.vstack([_densify_local(polys[j]) for j in range(len(polys))
                         if j != i] + extra) for i in range(len(polys))]
    grid = np.linspace(lo, hi, n_grid)
    fracs = [0.5] * len(polys)

    def score(i, f, cur):
        pt = _point_at_fraction(polys[i], f)
        d_line = float(np.sqrt(np.min(np.sum((others[i] - pt) ** 2, axis=1))))
        d_mark = min((float(np.hypot(*(_point_at_fraction(polys[j], cur[j]) - pt)))
                      for j in range(len(polys)) if j != i), default=1e9)
        # cap at 1: once both clearances are met, extra distance buys nothing.
        # Without the cap every safe candidate ties and the grid maximum is
        # picked arbitrarily -- which drove all three discs to the range end.
        met = min(1.0, d_line / clear_lines_m, d_mark / sep_markers_m)
        return met - 0.002 * abs(f - 0.5)      # tie-break toward the midpoint

    for _ in range(passes):
        changed = False
        for i in range(len(polys)):
            best_f = max(grid, key=lambda f: score(i, f, fracs))
            if abs(best_f - fracs[i]) > 1e-9:
                fracs[i] = float(best_f)
                changed = True
        if not changed:
            break
    return [_point_at_fraction(polys[i], fracs[i]) for i in range(len(polys))], fracs


def _densify_local(pts, step_m=4.0):
    out = []
    for k in range(len(pts) - 1):
        d = float(np.hypot(*(pts[k + 1] - pts[k])))
        n = max(2, int(d / step_m) + 1)
        out.append(np.column_stack([np.linspace(pts[k, 0], pts[k + 1, 0], n),
                                    np.linspace(pts[k, 1], pts[k + 1, 1], n)]))
    return np.vstack(out) if out else np.asarray(pts, dtype=float)


def draw_variant(ax, geom, specs, variant="a", **_ignored):
    cfg = CFG[variant]
    fs = cfg["fontsize"]
    polylines, texts, handles, own = [], [], [], []

    coloured = variant in ("c",)
    for name, start_lab, end_lab in SPECS:
        pts = _polyline(geom, name)
        polylines.append(pts)
        col = COLORS[name] if coloured else "black"
        if coloured:
            ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=3.0,
                    zorder=6, solid_capstyle="round")
            ax.plot(pts[:, 0], pts[:, 1], color=col, linewidth=1.6, zorder=7,
                    solid_capstyle="round")
            handles.append(Line2D([], [], color=col, linewidth=1.8,
                                  label=f"{start_lab}-{end_lab}"))
        else:
            ax.plot(pts[:, 0], pts[:, 1], color="white", linewidth=2.6,
                    zorder=6, solid_capstyle="round")
            ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.2,
                    zorder=7, solid_capstyle="round")

    # ------------------------------------------------ (d) numbered midpoints
    if variant == "d":
        # a disc is ~18 pt across; the map spans ~948 m over the column, so
        # ~3.5 m per pt -> keep centres ~75 m apart to leave clear air
        centres, fracs = _separate_along_paths(
            polylines, boundary=getattr(ax, "_variant_boundary", None))
        for idx, (name, start_lab, _) in enumerate(SPECS):
            pts = polylines[idx]
            mx, my = centres[idx]
            ax.plot([mx], [my], marker="o", markersize=cfg["marker_pt"],
                    markerfacecolor="#111111", markeredgecolor="white",
                    markeredgewidth=1.2, zorder=9, linestyle="none")
            t = ax.text(mx, my, NUMBERS[name], fontsize=fs, fontweight="bold",
                        color="white", ha="center", va="center", zorder=10)
            texts.append(t)
            own.append(idx)
            # keep the distance-0 end recoverable (see module docstring)
            ax.plot([pts[0, 0]], [pts[0, 1]], marker="o", markersize=4.0,
                    markerfacecolor="white", markeredgecolor="black",
                    markeredgewidth=0.9, zorder=9, linestyle="none")
        print("     (d) disc positions along each path: "
              + ", ".join(f"{NUMBERS[n]}@{f:.2f}"
                          for (n, _, _), f in zip(SPECS, fracs)))
        ax._variant_texts, ax._variant_polylines = texts, polylines
        ax._variant_own = own
        return texts

    # ------------------------------------------------ (c) margin labels
    if variant == "c":
        ends = []
        for name, start_lab, end_lab in SPECS:
            pts = _polyline(geom, name)
            ends.append((start_lab, pts[0], COLORS[name]))
            ends.append((end_lab, pts[-1], COLORS[name]))
        for _, (ex, ey), col in ((e[0], e[1], e[2]) for e in ends):
            ax.plot([ex], [ey], marker="o", markersize=5.0,
                    markerfacecolor=col, markeredgecolor="black",
                    markeredgewidth=0.8, zorder=9, linestyle="none")

        # stack the letters down the right margin, ordered by northing, then
        # spread them so no two collide vertically
        y0, y1 = ax.get_ylim()
        order = sorted(range(len(ends)), key=lambda i: -ends[i][1][1])
        ys = [(ends[i][1][1] - y0) / (y1 - y0) for i in order]
        min_gap = 0.075
        for _ in range(200):
            moved = False
            for k in range(len(ys) - 1):
                if ys[k] - ys[k + 1] < min_gap:
                    d = (min_gap - (ys[k] - ys[k + 1])) / 2
                    ys[k] += d
                    ys[k + 1] -= d
                    moved = True
            ys = [min(0.99, max(0.01, y)) for y in ys]
            if not moved:
                break
        for slot, i in enumerate(order):
            lab, (ex, ey), col = ends[i]
            t = ax.annotate(
                lab, xy=(ex, ey), xycoords="data",
                xytext=(1.012, ys[slot]), textcoords=ax.transAxes,
                fontsize=fs, fontweight="bold", color=col,
                ha="left", va="center", annotation_clip=False, zorder=12,
                arrowprops=dict(arrowstyle="-", color=col, linewidth=0.6,
                                shrinkA=0, shrinkB=3,
                                connectionstyle="arc3,rad=0.0"))
            texts.append(t)
        ax._variant_texts, ax._variant_polylines = [], polylines
        ax._variant_own = []
        return texts

    # ------------------------------------------------ (a) halo / (b) boxed
    entries = []
    for name, start_lab, end_lab in SPECS:
        pts = _polyline(geom, name)
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        x0p, y0p = pts[1] if len(pts) > 1 else pts[-1]
        x1p, y1p = pts[-2] if len(pts) > 1 else pts[0]
        for lab, (ex, ey), (px, py) in ((start_lab, (x0, y0), (x0p, y0p)),
                                        (end_lab, (x1, y1), (x1p, y1p))):
            entries.append({"label": lab, "anchor": (ex, ey),
                            "pos": _endpoint_label_pos(ex, ey, px, py,
                                                       cfg["offset_m"],
                                                       cfg["side_m"])})
    v43._repel(entries, cfg["min_sep_m"])
    obstacles = [v43._densify(p) for p in polylines]
    bnd = getattr(ax, "_variant_boundary", None)
    if bnd is not None:
        obstacles.append(v43._densify(bnd))
    v43._avoid_lines(entries, obstacles, clearance_m=cfg["clearance_m"],
                     min_sep_m=cfg["min_sep_m"], max_shift_m=cfg["max_shift_m"])

    for e in entries:
        lx, ly = e["pos"]
        axx, axy = e["anchor"]
        ax.plot([axx, axx + 0.62 * (lx - axx)], [axy, axy + 0.62 * (ly - axy)],
                color="black", linewidth=0.5, zorder=8, solid_capstyle="butt",
                path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
        if variant == "a":
            t = ax.text(lx, ly, e["label"], fontsize=fs, fontweight="bold",
                        color="white", ha="center", va="center", zorder=10,
                        path_effects=[pe.withStroke(linewidth=4.0,
                                                    foreground="black")])
        else:
            t = ax.text(lx, ly, e["label"], fontsize=fs, fontweight="bold",
                        color="white", ha="center", va="center", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.30",
                                  facecolor="#141414", alpha=0.85,
                                  edgecolor="white", linewidth=0.7))
        texts.append(t)

    ax._variant_texts, ax._variant_polylines = texts, polylines
    ax._variant_own = [None] * len(texts)
    return texts



def check_overlaps_aware(fig, ax, boundary, pad_px=1.0):
    """Like v43.check_overlaps, but a label may legitimately sit on the line
    it belongs to -- that is the whole design in (d). Its OWN polyline is
    therefore skipped; every other line, the boundary, and every other label
    are still checked."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = getattr(ax, "_variant_texts", [])
    polys = getattr(ax, "_variant_polylines", [])
    own = getattr(ax, "_variant_own", [None] * len(texts))

    boxes = [(t.get_text(), t.get_window_extent(renderer=rend).padded(pad_px))
             for t in texts]
    disp = [ax.transData.transform(p) for p in polys]
    bnd = ax.transData.transform(
        np.column_stack([boundary["easting"].to_numpy(),
                         boundary["northing"].to_numpy()]))

    problems = []
    for n, (lab, bb) in enumerate(boxes):
        for pi, pts in enumerate(disp):
            if own[n] is not None and pi == own[n]:
                continue                       # its own line, by design
            if any(v43._seg_hits_bbox(pts[k], pts[k + 1], bb)
                   for k in range(len(pts) - 1)):
                problems.append(f"label {lab!r} overlaps another transect line")
        if any(v43._seg_hits_bbox(bnd[k], bnd[k + 1], bb)
               for k in range(len(bnd) - 1)):
            problems.append(f"label {lab!r} overlaps the watershed boundary")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                problems.append(
                    f"labels {boxes[i][0]!r} and {boxes[j][0]!r} overlap")
    return problems


def build(variant, tables):
    t1, t2, t3, geom, boundary = tables
    orig = v3.draw_transects
    v3.draw_transects = (lambda ax, g, specs, **kw:
                         draw_variant(ax, g, specs, variant=variant))
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
        titles = TITLES["numbers" if variant == "d" else "letters"]
        for (ax_e, ax_d), df, title in zip(axes, [t1, t2, t3], titles):
            draw_profile_pair(ax_e, ax_d, df, title)
        axes[0][0].legend(loc="upper left", ncol=2, handlelength=1.4,
                          columnspacing=1.0)

        problems = check_overlaps_aware(fig, ax_map, boundary)
        # (c) places letters outside the map frame, so they can collide with
        # the neighbouring profile column -- which the map-only check cannot
        # see. Check them against the profile axes and their labels too.
        if variant == "c":
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            prof_boxes = [a.get_tightbbox(rend) for pair in axes for a in pair]
            for t in ax_map.texts:
                if not t.get_text():
                    continue
                tb = t.get_window_extent(renderer=rend)
                if any(tb.overlaps(pb) for pb in prof_boxes):
                    problems.append(
                        f"margin label {t.get_text()!r} overlaps the profile column")

        stem = DIAG / f"fig_transects_v4_alt_{variant}"
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
    log(f"START four alternatives; figure {FIG_W_MM}x{FIG_H_MM} mm, no "
        f"bbox_inches, so quoted point sizes are true print sizes "
        f"(current published labels are 7.0 pt)", run=RUN)
    for v in ("a", "b", "c", "d"):
        stem, problems = build(v, tables)
        status = "no overlaps" if not problems else "; ".join(sorted(set(problems)))
        print(f"\n({v}) {stem.name}.png  fontsize={CFG[v]['fontsize']} pt")
        print(f"     overlap check: {status}")
        log(f"({v}) fontsize={CFG[v]['fontsize']} pt -> {stem.name}.png/.pdf "
            f"| overlap check: {status}", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
