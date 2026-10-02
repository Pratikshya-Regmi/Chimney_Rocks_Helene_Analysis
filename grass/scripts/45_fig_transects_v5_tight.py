#!/usr/bin/env python3
"""
45_fig_transects_v5_tight.py
=============================
fig_transects_v5 -- the transect figure with the LEFT PANEL RECROPPED to the
watershed, so A-A', B-B' and C-C' are legible at print size.

WHAT CHANGES FROM v4
    v4's locator extends MAP_BUFFER_M = 200 m past the cat 34 boundary on
    every side and dims that surround with a white scrim. The basin is
    547 x 1108 m inside a 948 x 1509 m frame, so ~58% of the panel area is
    faded context and the transects are drawn small.

    v5 crops to the basin bounding box plus MARGIN_M and replaces the scrim
    with an OPAQUE white mask, so the context is removed rather than dimmed.
    Same column width, 627 m across instead of 948 m: the map scale improves
    by ~1.5x and every transect grows with it.

    Nothing else about the figure changes -- same rasters, same masked DoD at
    the 0.32 m LoD, same traced C-C', same graticule, north arrow and scale
    bar, same three profile pairs on the right.

VARIANTS (all 300 dpi PNG, plus a vector PDF alongside)
    a  tight crop, context removed, 9.5 pt haloed labels at the line ends
    b  as (a) plus a small inset locator showing the crop in its wider setting
    c  as (a) with thicker transect lines and larger labels
    d  as (a) with colour-coded transects and matching coloured labels

The scale bar auto-sizes to the new extent (map_style picks 200 m rather
than v4's 500 m); that is the crop working, not a settings change.

Outputs: results/diagnostics/fig_transects_v5_{a,b,c,d}.png / .pdf
         results/logs/transects_v5_<date>.log
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log                                    # noqa: E402
from map_style import (style_map_panel, _endpoint_label_pos,   # noqa: E402
                       add_scale_north_inline)
from make_transect_figure_v2 import (                        # noqa: E402
    export_map_rasters, load_tables, draw_profile_pair, read_raster,
    ORTHO_TIF, HILLSHADE_TIF, DOD_TIF, ORTHO_ALPHA, C_EROSION, C_DEPOSITION,
    LOD,
)

_spec = importlib.util.spec_from_file_location(
    "v43", HERE / "43_transect_label_variants.py")
v43 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v43)

RUN = "transects_v5"
ROOT = HERE.parents[1]
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM, FIG_H_MM = 174, 205
MAP_COL_W_MM = 90                       # basin aspect ~1.90 -> ~171 mm tall
PROFILE_COL_W_MM = FIG_W_MM - MAP_COL_W_MM - 6
# 40 m was too tight: C-C' ends at the basin's SW corner and its label had
# nowhere outside the boundary to go, so the clamp pushed it back onto the
# line. 70 m still crops 687 x 1248 m against v4's 948 x 1509 m.
MARGIN_M = 70                           # breathing room around the basin

PRIME = "′"
SPECS = [("transect1_crosssection", "A", f"A{PRIME}"),
         ("transect2_longitudinal", "B", f"B{PRIME}"),
         ("transect3_westcorridor", "C", f"C{PRIME}")]
COLORS = {"transect1_crosssection": "#009E73",
          "transect2_longitudinal": "#CC79A7",
          "transect3_westcorridor": "#E69F00"}

# offsets are in map metres; the tight crop roughly halves metres-per-mm, so
# these are ~0.6x the values v4 used for the same visual separation
CFG = {
    "a": dict(fontsize=9.5, line_w=1.2, halo_w=2.6, offset_m=34, side_m=14,
              min_sep_m=48, clearance_m=23, max_shift_m=95, inset=False,
              coloured=False),
    "b": dict(fontsize=9.5, line_w=1.2, halo_w=2.6, offset_m=34, side_m=14,
              min_sep_m=48, clearance_m=23, max_shift_m=95, inset=True,
              coloured=False),
    "c": dict(fontsize=12.0, line_w=1.9, halo_w=3.6, offset_m=40, side_m=17,
              min_sep_m=58, clearance_m=29, max_shift_m=100, inset=False,
              coloured=False),
    "d": dict(fontsize=10.5, line_w=1.9, halo_w=3.4, offset_m=36, side_m=15,
              min_sep_m=52, clearance_m=25, max_shift_m=98, inset=False,
              coloured=True),
}


def basin_ring(boundary):
    b = boundary.sort_values("vertex_order")
    return np.column_stack([b["easting"].to_numpy(), b["northing"].to_numpy()])


def tight_extent(ring):
    return (ring[:, 0].min() - MARGIN_M, ring[:, 0].max() + MARGIN_M,
            ring[:, 1].min() - MARGIN_M, ring[:, 1].max() + MARGIN_M)


def draw_map(ax, geom, boundary, cfg):
    """Map content, cropped to the basin with the surround masked out."""
    hill, hill_ext = read_raster(HILLSHADE_TIF)
    ortho, ortho_ext = read_raster(ORTHO_TIF)
    dod, dod_ext = read_raster(DOD_TIF)
    assert hill_ext == ortho_ext == dod_ext, "map rasters are not co-registered"

    ax.imshow(hill[0], cmap="gray", vmin=0, vmax=255, extent=hill_ext,
              origin="upper", zorder=1)
    ax.imshow(np.transpose(ortho, (1, 2, 0)).astype(float) / 255.0,
              extent=ortho_ext, origin="upper", alpha=ORTHO_ALPHA, zorder=2)

    cmap = LinearSegmentedColormap.from_list(
        "erosion_deposition", [C_EROSION, "#f7f7f7", C_DEPOSITION])
    cmap.set_bad(color=(0, 0, 0, 0))
    im = ax.imshow(dod[0], extent=dod_ext, origin="upper", cmap=cmap,
                   norm=Normalize(vmin=-3, vmax=3), zorder=3, alpha=0.9)

    # OPAQUE mask outside the basin -- v4 dimmed this area with a 0.55-alpha
    # scrim and kept it visible; here the context is removed outright, which
    # is the point of the recrop.
    ring = basin_ring(boundary)
    signed = np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
    outer = np.array([[hill_ext[0], hill_ext[2]], [hill_ext[1], hill_ext[2]],
                      [hill_ext[1], hill_ext[3]], [hill_ext[0], hill_ext[3]],
                      [hill_ext[0], hill_ext[2]]])
    inner = ring if signed < 0 else ring[::-1]
    verts = np.concatenate([outer, inner])
    codes = ([MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
             + [MplPath.MOVETO] + [MplPath.LINETO] * (len(inner) - 2)
             + [MplPath.CLOSEPOLY])
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="white",
                           alpha=1.0, edgecolor="none", zorder=4))
    ax.plot(ring[:, 0], ring[:, 1], color="black", linewidth=1.1, zorder=5)
    return im, ring




def _label_clearance_m(fontsize_pt, extent_w_m, map_w_mm=MAP_COL_W_MM,
                       longest_label_chars=2, safety=1.08, pad_m=5.0):
    """Half-diagonal of the rendered label, in MAP METRES, plus a pad.

    `safety` allows for the map axes being a little narrower than its column
    once aspect='equal' has shrunk it (which increases metres per point).
    """
    m_per_pt = (extent_w_m / map_w_mm) * 0.3528 * safety
    w_pt = 0.62 * fontsize_pt * longest_label_chars
    h_pt = 1.30 * fontsize_pt          # ascent + descent, not cap height
    return 0.5 * float(np.hypot(w_pt, h_pt)) * m_per_pt + pad_m


def _place_labels(entries, obstacles, ext, cfg):
    """Choose each label's position by SEARCHING around its endpoint.

    The push-based placement used elsewhere moves a label directly away from
    whatever is nearest, which cannot get around an obstacle. Inside this
    tight crop that failed on A: its endpoint sits 40 m from the C-C' path
    with the basin edge 65 m off, and the only roomy spot (58 m clear) is
    ~85 m to the south-west -- a direction the push never explores because it
    is not "away from the nearest point".

    Scoring each candidate on the ring around the endpoint finds it. The
    score is the binding constraint (clearance from lines and boundary,
    separation from other labels), capped at 1 so that once a position is
    good enough the tie-break takes over and prefers the SMALLEST
    displacement -- keeping every label as close to its own line end as the
    geometry allows.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(np.vstack(obstacles))
    pad = cfg["clearance_m"] * 0.9
    radii = np.arange(12.0, cfg["max_shift_m"] + 0.1, 4.0)
    angs = np.radians(np.arange(0.0, 360.0, 8.0))
    ring_dx = np.outer(radii, np.cos(angs)).ravel()
    ring_dy = np.outer(radii, np.sin(angs)).ravel()
    ring_r = np.repeat(radii, len(angs))

    for _ in range(4):
        for i, e in enumerate(entries):
            ax_, ay_ = e["anchor"]
            cx, cy = ax_ + ring_dx, ay_ + ring_dy
            ok = ((cx > ext[0] + pad) & (cx < ext[1] - pad)
                  & (cy > ext[2] + pad) & (cy < ext[3] - pad))
            if not ok.any():
                continue
            pts = np.column_stack([cx[ok], cy[ok]])
            rr = ring_r[ok]
            d_obs, _ = tree.query(pts)
            others = np.array([entries[j]["pos"] for j in range(len(entries))
                               if j != i], dtype=float)
            d_lab = (cKDTree(others).query(pts)[0] if len(others)
                     else np.full(len(pts), 1e9))
            score = np.minimum(1.0, np.minimum(d_obs / cfg["clearance_m"],
                                               d_lab / cfg["min_sep_m"]))
            score = score - 0.0015 * rr          # prefer staying close
            e["pos"] = [float(pts[np.argmax(score)][0]),
                        float(pts[np.argmax(score)][1])]
    return entries


def draw_transects(ax, geom, boundary, cfg):
    polylines, entries = [], []
    for name, s_lab, e_lab in SPECS:
        sub = geom[geom["transect"] == name].sort_values("vertex_order")
        pts = np.column_stack([sub["easting"].to_numpy(),
                               sub["northing"].to_numpy()])
        polylines.append(pts)
        col = COLORS[name] if cfg["coloured"] else "black"
        ax.plot(pts[:, 0], pts[:, 1], color="black" if cfg["coloured"] else "white",
                linewidth=cfg["halo_w"], zorder=6, solid_capstyle="round")
        ax.plot(pts[:, 0], pts[:, 1], color=col, linewidth=cfg["line_w"],
                zorder=7, solid_capstyle="round")
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        x0p, y0p = pts[1] if len(pts) > 1 else pts[-1]
        x1p, y1p = pts[-2] if len(pts) > 1 else pts[0]
        for lab, (ex, ey), (px, py) in ((s_lab, (x0, y0), (x0p, y0p)),
                                        (e_lab, (x1, y1), (x1p, y1p))):
            entries.append({"label": lab, "anchor": (ex, ey), "color": col,
                            "pos": _endpoint_label_pos(ex, ey, px, py,
                                                       cfg["offset_m"],
                                                       cfg["side_m"])})

    obstacles = [v43._densify(p) for p in polylines] + [v43._densify(
        basin_ring(boundary))]
    ext = tight_extent(basin_ring(boundary))

    # Derive the required clearance from the label's own rendered size rather
    # than hand-setting it: a text bbox includes ascent and descent, so it is
    # taller than the visible glyph, and the prime widens it. Hand-set values
    # (23 m at 9.5 pt) were smaller than the real half-diagonal, which is why
    # C and C' still overlapped even with 99 m of clear space available.
    cfg = dict(cfg)
    cfg["clearance_m"] = _label_clearance_m(cfg["fontsize"],
                                            ext[1] - ext[0])
    cfg["min_sep_m"] = 1.9 * cfg["clearance_m"]
    _place_labels(entries, obstacles, ext, cfg)

    texts = []
    for e in entries:
        lx, ly = e["pos"]
        axx, axy = e["anchor"]
        ax.plot([axx, axx + 0.6 * (lx - axx)], [axy, axy + 0.6 * (ly - axy)],
                color=e["color"], linewidth=0.5, zorder=8,
                path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
        t = ax.text(lx, ly, e["label"], fontsize=cfg["fontsize"],
                    fontweight="bold",
                    color="white" if not cfg["coloured"] else e["color"],
                    ha="center", va="center", zorder=10,
                    path_effects=[pe.withStroke(
                        linewidth=3.4 if not cfg["coloured"] else 2.8,
                        foreground="black")])
        texts.append(t)
    ax._variant_texts, ax._variant_polylines = texts, polylines
    return texts


def add_inset(fig, ax, boundary, extent):
    """Small locator: the full exported frame, with the v5 crop outlined."""
    hill, hill_ext = read_raster(HILLSHADE_TIF)
    ins = ax.inset_axes([0.655, 0.762, 0.335, 0.225])
    ins.imshow(hill[0], cmap="gray", vmin=0, vmax=255, extent=hill_ext,
               origin="upper", zorder=1)
    ring = basin_ring(boundary)
    ins.plot(ring[:, 0], ring[:, 1], color="black", linewidth=0.7, zorder=3)
    ins.add_patch(Rectangle((extent[0], extent[2]),
                            extent[1] - extent[0], extent[3] - extent[2],
                            fill=False, edgecolor="red", linewidth=0.9,
                            zorder=4))
    ins.set_xlim(hill_ext[0], hill_ext[1])
    ins.set_ylim(hill_ext[2], hill_ext[3])
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_linewidth(0.7)
    ins.set_facecolor("white")
    return ins


def build(variant, tables):
    cfg = CFG[variant]
    t1, t2, t3, geom, boundary = tables
    ring = basin_ring(boundary)
    extent = tight_extent(ring)

    fig = plt.figure(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))
    outer = fig.add_gridspec(1, 2,
                             width_ratios=[MAP_COL_W_MM, PROFILE_COL_W_MM],
                             wspace=0.30)
    # Own legend row for the map column: cropping to the basin puts the
    # lower-right corner INSIDE the watershed, so map_style's in-panel scale
    # bar would sit on terrain. Colourbar and the scale bar / north arrow
    # both go below the map instead.
    left = outer[0, 0].subgridspec(2, 1, height_ratios=[1, 0.05], hspace=0.30)
    ax_map = fig.add_subplot(left[0, 0])
    im, _ = draw_map(ax_map, geom, boundary, cfg)
    draw_transects(ax_map, geom, boundary, cfg)
    if cfg["inset"]:
        add_inset(fig, ax_map, boundary, extent)

    fig.canvas.draw()
    row = left[1, 0].get_position(fig)
    SCALE_FRAC, GAP_FRAC = 0.30, 0.04
    cb_w = row.width * (1.0 - SCALE_FRAC - GAP_FRAC)
    cax = fig.add_axes([row.x0, row.y0, cb_w, row.height])
    sax = fig.add_axes([row.x0 + cb_w + row.width * GAP_FRAC,
                        row.y0 - row.height * 1.0,
                        row.width * SCALE_FRAC, row.height * 3.0])

    style_map_panel(ax_map, extent, mappable=im, cax=cax,
                    cbar_label="Elevation change", cbar_units="m",
                    basin_label=None, graticule_step=200,
                    draw_scale=False, draw_north=False)

    fig.canvas.draw()
    panel_w_mm = (ax_map.get_window_extent().width / fig.dpi) * 25.4
    sax_w_mm = (sax.get_window_extent().width / fig.dpi) * 25.4
    add_scale_north_inline(sax, scale_m=200,
                           mm_per_m=panel_w_mm / (extent[1] - extent[0]),
                           axes_w_mm=sax_w_mm, color="black")

    right = outer[0, 1].subgridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.45)
    pairs = [right[i, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
             for i in range(3)]
    axes = []
    for p in pairs:
        ax_e = fig.add_subplot(p[0, 0])
        ax_d = fig.add_subplot(p[1, 0], sharex=ax_e)
        axes.append((ax_e, ax_d))
    titles = [f"A-A{PRIME}  cross-section", f"B-B{PRIME}  longitudinal profile",
              f"C-C{PRIME}  western corridor"]
    for (ax_e, ax_d), df, title in zip(axes, [t1, t2, t3], titles):
        draw_profile_pair(ax_e, ax_d, df, title)
    axes[0][0].legend(loc="upper left", ncol=2, handlelength=1.4,
                      columnspacing=1.0)

    problems = v43.check_overlaps(fig, ax_map, boundary)

    stem = DIAG / f"fig_transects_v5_{variant}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return stem, problems, extent


def main():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    print("Exporting GRASS map-panel rasters (once, shared by all variants)...")
    export_map_rasters()
    tables = load_tables()

    for v in ("a", "b", "c", "d"):
        stem, problems, extent = build(v, tables)
        w = extent[1] - extent[0]
        h = extent[3] - extent[2]
        status = "no overlaps" if not problems else "; ".join(sorted(set(problems)))
        clr = _label_clearance_m(CFG[v]["fontsize"], w)
        print(f"\n({v}) {stem.name}.png  crop {w:.0f} x {h:.0f} m  "
              f"label {CFG[v]['fontsize']} pt  line {CFG[v]['line_w']} pt  "
              f"clearance {clr:.0f} m")
        print(f"     overlap check: {status}")
        log(f"v5 ({v}): crop {w:.0f}x{h:.0f} m (basin + {MARGIN_M} m), "
            f"fontsize={CFG[v]['fontsize']} pt line={CFG[v]['line_w']} pt "
            f"inset={CFG[v]['inset']} coloured={CFG[v]['coloured']} -> "
            f"{stem.name}.png (300 dpi) + .pdf | overlap: {status}", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
