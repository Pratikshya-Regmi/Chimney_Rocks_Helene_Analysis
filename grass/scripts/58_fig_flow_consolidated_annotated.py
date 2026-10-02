#!/usr/bin/env python3
"""
58_fig_flow_consolidated_annotated.py
=====================================
The consolidated flow figure (56, variant e) reads as noisy: the course
annotation draws EVERY retained skeleton cell -- five or six separate
components per class per zone -- as pixel squares, on all nine zoom panels.
The reader sees a scatter of magenta and orange and has to work out which
bit is the argument.

This rebuild says what changed, in words, on the panels:

  * each class is reduced to its DOMINANT course -- the largest connected
    component -- and that component to its longest path (the graph diameter
    of the skeleton), so branch stubs disappear and one clean line remains
  * the line is drawn cased in white, so it reads over dark channels and
    pale hillslope alike
  * short leader-arrow labels name the features directly: "removed flow"
    on the abandoned 2020 course, "new flow" on the 2024 course, "split"
    at the divergence point
  * labels are placed in whichever panel corner carries the least simulated
    discharge, so text never lands on a channel

Three options:
  n1  labels on the 2020 row only (the geometry is identical on every row,
      so naming it once is enough); dominant courses on all rows
  n2  labels on every row -- fully explicit, more text
  n3  no course lines at all: just labelled arrows pointing at where the
      change is, over clean discharge panels. Minimum ink.

Layout, scales, watershed tint, scale bars and north arrow are inherited
unchanged from 56 variant (e).

Outputs:
  results/figures/fig_flow_consolidated/annotated/fig_flow_annot_<key>.png/.pdf
"""

import importlib.util
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patheffects

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import ndimage as ndi
import rasterio

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "fc56", HERE / "56_fig_flow_consolidated.py")
m = importlib.util.module_from_spec(spec)
sys.modules["fc56"] = m
spec.loader.exec_module(m)

sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402

OUT = m.OUT / "annotated"
OUT.mkdir(parents=True, exist_ok=True)
FS = m.FS
FS_ANNOT = 7.0


# ------------------------------------------------------- skeleton -> path
def _neighbours(cell, cells):
    r, c = cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                n = (r + dr, c + dc)
                if n in cells:
                    yield n


def _bfs(start, cells):
    seen = {start: None}
    q = deque([start])
    last = start
    while q:
        cur = q.popleft()
        last = cur
        for n in _neighbours(cur, cells):
            if n not in seen:
                seen[n] = cur
                q.append(n)
    return last, seen


def longest_path(cells):
    """Graph diameter of an 8-connected skeleton component: the longest of
    its shortest paths. Branch stubs are dropped, leaving the main course."""
    if not cells:
        return []
    start = next(iter(cells))
    a, _ = _bfs(start, cells)
    b, parents = _bfs(a, cells)
    path, cur = [], b
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    return path[::-1]


def smooth(xy, k=5):
    if len(xy) < k:
        return xy
    out = xy.copy()
    for i in range(2):
        out[:, i] = np.convolve(xy[:, i], np.ones(k) / k, mode="same")
        out[:k // 2, i] = xy[:k // 2, i]
        out[-(k // 2):, i] = xy[-(k // 2):, i]
    return out


def dominant_course(mask, e):
    """Largest component of `mask`, reduced to its longest path, in map
    coordinates. Returns (xy, n_cells_in_component)."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.empty((0, 2)), 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    k = int(np.argmax(sizes))
    cells = set(map(tuple, np.argwhere(lab == k)))
    path = longest_path(cells)
    if not path:
        return np.empty((0, 2)), int(sizes[k])
    res = (e[1] - e[0]) / mask.shape[1]
    xy = np.array([[e[0] + (c + 0.5) * res, e[3] - (r + 0.5) * res]
                   for r, c in path], dtype=float)
    return smooth(xy), int(sizes[k])


def zone_courses(ext, box):
    """Dominant abandoned and new course, plus the split point, for one zone.

    Same definitions as 56 -- a cell is abandoned if it carried 2020 channel
    and no 2024 channel lies within 2 m, new on the mirror test -- but only
    the single largest component of each survives, reduced to its longest
    path."""
    m20, e = m._mask_window(m.TIF["thin2020"], ext)
    m24, _ = m._mask_window(m.TIF["thin2024"], ext)
    out = dict(abandon=np.empty((0, 2)), new=np.empty((0, 2)), split=None,
               n_ab=0, n_new=0)
    if not m20.any() or not m24.any():
        return out
    d24 = ndi.distance_transform_edt(~m24)
    d20 = ndi.distance_transform_edt(~m20)
    ab_mask = m20 & (d24 > 2.0)
    nw_mask = m24 & (d20 > 2.0)
    out["abandon"], out["n_ab"] = dominant_course(ab_mask, e)
    out["new"], out["n_new"] = dominant_course(nw_mask, e)

    shared = m20 & (d24 <= 1.5)
    if ab_mask.any() and nw_mask.any() and shared.any():
        da = ndi.distance_transform_edt(~ab_mask)
        dn = ndi.distance_transform_edt(~nw_mask)
        res = (e[1] - e[0]) / shared.shape[1]
        rr, cc = np.mgrid[0:shared.shape[0], 0:shared.shape[1]]
        xx = e[0] + (cc + 0.5) * res
        yy = e[3] - (rr + 0.5) * res
        inbox = (shared & (xx >= box[0]) & (xx <= box[1]) &
                 (yy >= box[2]) & (yy <= box[3]))
        if inbox.any():
            score = np.where(inbox, np.maximum(da, dn), np.inf)
            r, c = np.unravel_index(int(np.argmin(score)), score.shape)
            if score[r, c] <= 3.0:
                out["split"] = (np.array([xx[r, c], yy[r, c]]),
                                float(da[r, c]), float(dn[r, c]))
    return out


# ------------------------------------------------------------- annotation
DIRS = [(1, 0), (0.7, 0.7), (0, 1), (-0.7, 0.7), (-1, 0), (-0.7, -0.7),
        (0, -1), (0.7, -0.7)]


def ink_field(ext):
    q, qe = m.read_window(m.TIF["2020"], ext)
    return np.asarray(q.filled(0), dtype=float), qe


def _ink_at(z, qe, x, y, halo_m=9.0):
    """Mean discharge in a box around a point -- how much would a label put
    here cover up."""
    ny, nx = z.shape
    res_x = (qe[1] - qe[0]) / nx
    res_y = (qe[3] - qe[2]) / ny
    c = int((x - qe[0]) / res_x)
    r = int((qe[3] - y) / res_y)
    hx, hy = max(1, int(halo_m / res_x)), max(1, int(halo_m / res_y))
    r0, r1 = max(0, r - hy), min(ny, r + hy)
    c0, c1 = max(0, c - hx), min(nx, c + hx)
    if r0 >= r1 or c0 >= c1:
        return 1e9
    return float(z[r0:r1, c0:c1].mean())


def place_label(ext, target, z, qe, taken, scale_zone, courses_xy,
                text, mm_per_m):
    """Anchor the label a short step from its feature, in whichever direction
    covers least discharge, stays inside the panel, keeps clear of the scale
    bar and does not collide with a label already placed."""
    w, h = ext[1] - ext[0], ext[3] - ext[2]
    step = 0.19 * w
    # keep the whole label BOX inside the panel, not just its anchor point:
    # approximate its half-width from the string at ~0.6 em per character
    half_mm = 0.5 * (len(text) * FS_ANNOT * 0.6 / 72 * 25.4) + 1.2
    pad_x = max(0.16 * w, half_mm / mm_per_m)
    pad_y = max(0.10 * h, 3.0 / mm_per_m)
    best, best_score = None, None
    for dx, dy in DIRS:
        x = target[0] + dx * step
        y = target[1] + dy * step
        score = _ink_at(z, qe, x, y) * 1000.0
        if not (ext[0] + pad_x <= x <= ext[1] - pad_x):
            score += 50.0
        if not (ext[2] + pad_y <= y <= ext[3] - pad_y):
            score += 50.0
        if (x > scale_zone[0] and y < scale_zone[1]):
            score += 60.0
        for tx, ty in taken:
            d = np.hypot(x - tx, y - ty)
            if d < 0.30 * w:
                score += 40.0 * (1 - d / (0.30 * w))
        # keep the label box off the drawn courses themselves
        for pts in courses_xy:
            if len(pts):
                d = float(np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))
                if d < 0.13 * w:
                    score += 35.0 * (1 - d / (0.13 * w))
        if best_score is None or score < best_score:
            best, best_score = (x, y), score
    x = float(np.clip(best[0], ext[0] + pad_x, ext[1] - pad_x))
    y = float(np.clip(best[1], ext[2] + pad_y, ext[3] - pad_y))
    return x, y


def draw_course(ax, xy, color, mm_per_m, zorder=7):
    if len(xy) < 2:
        return
    lw = max(1.0, min(1.8, mm_per_m * 2.2))
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, solid_capstyle="round",
            zorder=zorder,
            path_effects=[matplotlib.patheffects.withStroke(
                linewidth=lw + 1.6, foreground="white")])


def draw_split(ax, pt, zorder=9):
    ax.plot([pt[0]], [pt[1]], marker="o", ms=4.6, mfc="none", mec="white",
            mew=2.6, zorder=zorder - 1)
    ax.plot([pt[0]], [pt[1]], marker="o", ms=4.6, mfc="none",
            mec=m.C_SPLIT, mew=1.2, zorder=zorder)


def annotate_panel(ax, ext, cz, mm_per_m, draw_lines, labels):
    """labels: subset of {"removed", "new", "split"} to name on this panel."""
    if draw_lines:
        draw_course(ax, cz["abandon"], m.C_ABANDON, mm_per_m)
        draw_course(ax, cz["new"], m.C_NEW, mm_per_m)
    # the split marker belongs with the courses: if a panel is deliberately
    # left clean of them, it gets no marker either, unless it carries the
    # "split" label itself
    if cz["split"] is not None and (draw_lines or "split" in labels):
        draw_split(ax, cz["split"][0])
    if not labels:
        return
    items = []
    if "removed" in labels and len(cz["abandon"]):
        items.append(("removed flow", cz["abandon"][len(cz["abandon"]) // 2],
                      m.C_ABANDON))
    if "new" in labels and len(cz["new"]):
        items.append(("new flow", cz["new"][len(cz["new"]) // 2], m.C_NEW))
    if "split" in labels and cz["split"] is not None:
        items.append(("split", cz["split"][0], "black"))
    if not items:
        return
    z, qe = ink_field(ext)
    scale_zone = (ext[0] + 0.62 * (ext[1] - ext[0]),
                  ext[2] + 0.20 * (ext[3] - ext[2]))
    taken = []
    courses_xy = [cz["abandon"], cz["new"]]
    for text, target, col in items:
        x, y = place_label(ext, target, z, qe, taken, scale_zone, courses_xy,
                           text, mm_per_m)
        taken.append((x, y))
        ax.annotate(
            text, xy=(target[0], target[1]), xytext=(x, y),
            ha="center", va="center", fontsize=FS_ANNOT, color="black",
            zorder=14,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor=col, linewidth=0.7, alpha=0.93),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9,
                            shrinkA=1.0, shrinkB=2.0, mutation_scale=7,
                            path_effects=[matplotlib.patheffects.withStroke(
                                linewidth=2.0, foreground="white")]))


# ---------------------------------------------------------------- options
# Which labels go on which zoom row (0 = 2020, 1 = SfM, 2 = 2024 + change).
# Naming each feature on the row where it belongs -- what was removed on the
# pre-event panel, what is new on the post-event panel -- spreads the text
# out and keeps any one panel to a single label.
OPTIONS = {
    "n1": dict(lines=True,
               rows={0: {"removed"}, 1: set(), 2: {"new", "split"}},
               desc="labels distributed by meaning: 'removed flow' on the "
                    "2020 panel, 'new flow' and 'split' on the 2024 panel; "
                    "at most two labels on any panel"),
    "n2": dict(lines=True,
               rows={0: {"removed", "new", "split"}, 1: set(), 2: set()},
               desc="all three labels together on the 2020 row, the other "
                    "rows left clean"),
    "n3": dict(lines=False,
               rows={0: {"removed"}, 1: set(), 2: {"new", "split"}},
               desc="no course lines at all -- just labelled arrows pointing "
                    "at where the change is, over clean discharge panels"),
}


def build(key, zones, exts, courses, basin_ext, vlim):
    opt = OPTIONS[key]
    blocks, legend_mode, _ = m.variant_spec("e", zones, exts,
                                            {z: None for z in exts}, basin_ext)
    shrink = m.fit_shrink(blocks, legend_mode)
    geom, fig_h = m.compute(blocks, legend_mode, shrink)
    fig = plt.figure(figsize=(m.FIG_W_MM * m.MM, fig_h * m.MM))
    state = dict(fig_h=fig_h)

    def rect_fn(x, ytop, w, h):
        return [x / m.FIG_W_MM, 1 - (ytop + h) / fig_h,
                w / m.FIG_W_MM, h / fig_h]

    for g in geom:
        row_w = g["inset"] + sum(g["widths"]) + m.GAP_X * (len(g["widths"]) - 1)
        g["x0"] = (m.MARGIN_L if g["align"] == "left"
                   else (m.FIG_W_MM - row_w) / 2)
    spans = [(g["x0"] + g["inset"],
              g["x0"] + g["inset"] + sum(g["widths"]) +
              m.GAP_X * (len(g["widths"]) - 1))
             for g in geom if g["align"] == "center"]
    ref_l, ref_r = min(a for a, _ in spans), max(b for _, b in spans)
    for g in geom:
        if g["align"] == "left":
            g["x0"] = ref_l - g["inset"]

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for g in geom:
        x = g["x0"] + g["inset"]
        for i, (p, w) in enumerate(zip(g["panels"], g["widths"])):
            if g["headers"] is not None:
                fig.text((x + w / 2) / m.FIG_W_MM,
                         1 - (g["y"] - m.HEADER_H * 0.30) / fig_h,
                         g["headers"][i], ha="center", va="baseline",
                         fontsize=FS["header"])
            ax = m.render_panel(fig, rect_fn, x, g["y"], w, g["h"], p,
                                next(letters), zones, vlim, state)
            if g["block"] == 1:
                mm_per_m = w / (p["ext"][1] - p["ext"][0])
                annotate_panel(ax, p["ext"], courses[p["zone"]], mm_per_m,
                               draw_lines=opt["lines"],
                               labels=opt["rows"].get(g["row"], set()))
            x += w + m.GAP_X
        if g["row_label"]:
            fig.text((g["x0"] + g["inset"] / 2) / m.FIG_W_MM,
                     1 - (g["y"] + g["h"] / 2) / fig_h, g["row_label"],
                     ha="center", va="center", rotation=90,
                     fontsize=FS["zone"], fontweight="bold")

    g0 = next(g for g in geom if g["block"] == 0)
    x0 = g0["x0"] + g0["inset"] + sum(g0["widths"]) + \
        m.GAP_X * (len(g0["widths"]) - 1) + 9.0
    right = ref_r if ref_r - x0 >= 48.0 else m.FIG_W_MM - m.MARGIN_R
    m.draw_legend(fig, rect_fn, x0, g0["y"] + 2.0, right - x0, g0["h"],
                  state, vlim, stacked=True)

    name = f"fig_flow_annot_{key}"
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    return dict(key=key, name=name, h=fig_h, desc=opt["desc"])


def main():
    zones = pd.read_csv(m.TABLES / "flow_change_zones.csv")
    with rasterio.open(m.TIF["2020"]) as src:
        a = src.read(1, masked=True)
        t = src.transform
    r, c = np.nonzero(~np.ma.getmaskarray(a))
    basin_ext = (t.c + c.min() * t.a, t.c + (c.max() + 1) * t.a,
                 t.f + (r.max() + 1) * t.e, t.f + r.min() * t.e)
    ch, _ = m.read_window(m.TIF["change"], basin_ext)
    vlim = 3 * np.nanstd(np.asarray(ch.filled(np.nan), dtype=float))

    exts, courses = {}, {}
    for _, z in zones.iterrows():
        exts[z.zone_id] = (z.e_min - m.PAD_M, z.e_max + m.PAD_M,
                           z.n_min - m.PAD_M, z.n_max + m.PAD_M)
        courses[z.zone_id] = zone_courses(
            exts[z.zone_id], (z.e_min, z.e_max, z.n_min, z.n_max))
        cz = courses[z.zone_id]
        print(f"  {z.zone_id}: removed course {cz['n_ab']:3d} cells -> "
              f"{len(cz['abandon']):3d}-point path; new course "
              f"{cz['n_new']:3d} cells -> {len(cz['new']):3d}-point path; "
              f"split {'yes' if cz['split'] is not None else 'none'}")

    for k in OPTIONS:
        r_ = build(k, zones, exts, courses, basin_ext, vlim)
        print(f"  {k}  174 x {r_['h']:.0f} mm  {r_['desc']}")
        log(f"fig_flow_annot_{k}: {r_['desc']}", run="flow_annotated")
    log("DONE", run="flow_annotated")


if __name__ == "__main__":
    main()
