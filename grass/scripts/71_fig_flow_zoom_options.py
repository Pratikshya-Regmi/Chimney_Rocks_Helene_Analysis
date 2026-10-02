#!/usr/bin/env python3
"""
71_fig_flow_zoom_options.py
===========================
Five annotation approaches for the zoom figure (Fig. 10), rendered as
mock-ups on the same data so the author can choose one. None is inserted in
the manuscript.

  1 callouts   2020 | SfM | 2024. Letter tags with leader arrows -- A
               (abandoned) on the 2020 panel, N (new) and S (split) on the
               2024 panel; SfM left clean. Nothing is traced.
  2 change     2020 | SfM | 2024 | change. A fourth column of discharge
               change, 2024 lidar minus 2020 lidar, RdBu +/- 3 SD of the
               basin-wide change raster, near-zero faded out (56.draw_change).
  3 network    2020 | SfM | 2024 | network. A fourth column showing only the
               two channel networks, classified persisted / abandoned / new,
               over faded hillshade.
  4 pair       2020 | 2024 only, larger panels at one common scale, with the
               abandoned 2020 course traced as a dashed ghost on the 2024
               panel. The SfM zoom is dropped (it stays in Fig. 9).
  5 schematic  2020 | SfM | 2024 | schematic. Rasters untouched; a narrow line
               diagram per zone traced from the computed courses, with a
               one-word tag.

Shared by all five, so they compare on the approach alone: script 32's
discharge rendering (YlGnBu, log 1e-4 to 0.48, clipped so values above the
maximum saturate), the zone extents, one 25 m bar per row and one north
arrow. The graticule is left off every option to give the annotation room;
it can go back on whichever is chosen.

Course definitions are 70's (tests from 56): old = 2020 lidar skeleton with
no 2024 skeleton within 2 m, new = the mirror, persisted = within 2 m of the
other epoch; stretches of >= 70.MIN_ANNOT_CELLS (30) cells are the named
changes. The schematic's tag words are the manuscript's own descriptions of
each zone.

Run from the repository root so results/ resolves to the canonical tree.

Outputs:
  results/figures/fig_flow_zoom_options/opt<N>_<name>.png (300 dpi) / .pdf
  results/figures/fig_flow_zoom_options/OPTIONS.md
  results/tables/flow_zoom_options_cvd.csv   network-map palette separation
  results/logs/flow_zoom_options_<date>.log
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import rasterio
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 70 carries 32 (rendering) and 56 (course tests); 58 carries the course
# tracing and label placement. 56/58 set rcParams at import -- keep them out.
with matplotlib.rc_context():
    a70 = _load("fz70", "70_fig_flow_sequence_zoom_annotated.py")
    an = _load("fc58", "58_fig_flow_consolidated_annotated.py")
z32, m = a70.z32, a70.m

OUT = z32.FIGURES / "fig_flow_zoom_options"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM = 174.0
MAX_H_MM = 225.0          # sn-jnl text block at 174 mm width is 232
ML = MR = 3.0
MT = MB = 3.0
ZONE_W = 6.0
GAP_X = 2.5
GAP_Y = 3.5
HEADER_H = 5.0
KEY_H = 17.0

VMIN, VMAX = z32.VMIN, z32.VMAX
C_OLD = a70.C_OLD          # vermillion
C_NEW = a70.C_NEW          # black
C_KEEP = "#0072B2"         # Okabe-Ito blue -- persisted channel (option 3)
C_CONTEXT = "#C4C4C4"      # persisted channel as context (option 5)
FAINT = 0.35               # alpha for stretches < MIN_ANNOT_CELLS
FS = dict(header=7.0, zone=9.0, key=7.0, cbar=7.0, tick=6.5, tag=7.0,
          north=8.0)
CASE = [pe.withStroke(linewidth=2.2, foreground="white")]

TAGS = {"Z1": "split", "Z2": "abandoned → new", "Z3": "corridor shift"}
HEAD = {"2020": z32.COLUMNS[0][0], "sfm": z32.COLUMNS[1][0],
        "2024": z32.COLUMNS[2][0], "change": "change (2024 − 2020)",
        "network": "channel networks", "schematic": "schematic"}
TIF = {"2020": z32.DISCHARGE_2020_TIF, "sfm": z32.DISCHARGE_CAP_TIF,
       "2024": z32.DISCHARGE_2024_TIF}


# --------------------------------------------------------------------- data
def centrelines(mask, e, k=5, min_cells=0):
    """One smoothed centreline per stretch: the longest path through each
    8-connected component (58.longest_path), so branch stubs drop out."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel())
    res = (e[1] - e[0]) / mask.shape[1]
    out = []
    for i in range(1, n + 1):
        if sizes[i] < min_cells:
            continue
        path = an.longest_path(set(map(tuple, np.argwhere(lab == i))))
        if len(path) < 2:
            continue
        xy = np.array([[e[0] + (c + 0.5) * res, e[3] - (r + 0.5) * res]
                       for r, c in path], dtype=float)
        out.append(an.smooth(xy, k=k))
    return out


def zone_data(z):
    box = (z.e_min, z.e_max, z.n_min, z.n_max)
    pad = (z.e_min - m.PAD_M, z.e_max + m.PAD_M,
           z.n_min - m.PAD_M, z.n_max + m.PAD_M)
    d = a70.zone_changes(pad, box)
    m20, _ = m._mask_window(m.TIF["thin2020"], pad)
    m24, _ = m._mask_window(m.TIF["thin2024"], pad)
    d24 = ndi.distance_transform_edt(~m24)
    d20 = ndi.distance_transform_edt(~m20)
    d["keep"] = (m20 & (d24 <= 2.0)) | (m24 & (d20 <= 2.0))
    d["paths_old"] = centrelines(d["old"], d["e"])
    d["paths_new"] = centrelines(d["new"], d["e"])
    d["sketch_old"] = centrelines(d["old"], d["e"], k=9)
    d["sketch_new"] = centrelines(d["new"], d["e"], k=9)
    d["sketch_keep"] = centrelines(d["keep"], d["e"], k=9, min_cells=10)
    if z.zone_id not in a70.SPLIT_ZONES:
        d["split"] = None
    d["box"] = box
    return d


# ------------------------------------------------------------------ drawing
def draw_q(ax, kind, box):
    """Script 32's rendering, unchanged."""
    hs, he = z32.read_window(z32.HILLSHADE_TIF, *box)
    ax.imshow(hs, cmap="gray", extent=he, origin="upper", zorder=1)
    arr, ex = z32.read_window(TIF[kind], *box)
    return ax.imshow(np.clip(arr, VMIN, VMAX), cmap="YlGnBu",
                     norm=LogNorm(vmin=VMIN, vmax=VMAX), extent=ex,
                     origin="upper", alpha=0.88, zorder=2)


def _paint(ax, mask, e, color, alpha, grow=True, zorder=3):
    if not mask.any():
        return
    if grow:
        mask = ndi.binary_dilation(mask, structure=np.ones((3, 3), bool))
    rgba = np.zeros(mask.shape + (4,))
    rgba[mask] = (*to_rgb(color), alpha)
    ax.imshow(rgba, extent=(e[0], e[1], e[2], e[3]), origin="upper",
              interpolation="nearest", zorder=zorder)


def draw_network(ax, d, box):
    """Both skeletons, classified, each grown to 3 cells so a 1 m course
    reads at print size. Stretches below the display threshold stay, faint,
    so the panel still accounts for every channel cell."""
    hs, he = z32.read_window(z32.HILLSHADE_TIF, *box)
    ax.imshow(hs, cmap="gray", extent=he, origin="upper", zorder=1,
              alpha=0.30)
    e = d["e"]
    _paint(ax, d["keep"], e, C_KEEP, 1.0)
    _paint(ax, d["old_all"] & ~d["old"], e, C_OLD, FAINT, zorder=4)
    _paint(ax, d["new_all"] & ~d["new"], e, C_NEW, FAINT, zorder=4)
    _paint(ax, d["old"], e, C_OLD, 1.0, zorder=5)
    _paint(ax, d["new"], e, C_NEW, 1.0, zorder=5)


def draw_schematic(ax, d, tag):
    """Persisted channels as thin grey centrelines for context, the changed
    courses over them, and the zone's tag word above the box."""
    for xy in d["sketch_keep"]:
        ax.plot(xy[:, 0], xy[:, 1], color=C_CONTEXT, lw=1.0,
                solid_capstyle="round", zorder=2)
    for xy in d["sketch_old"]:
        ax.plot(xy[:, 0], xy[:, 1], color=C_OLD, lw=1.3, ls=(0, (3, 1.8)),
                solid_capstyle="round", zorder=5)
    for xy in d["sketch_new"]:
        ax.plot(xy[:, 0], xy[:, 1], color=C_NEW, lw=1.3,
                solid_capstyle="round", zorder=6)
    if d["split"] is not None:
        a70.draw_split(ax, d["split"])
    ax.text(0.5, 1.04, tag, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=FS["tag"] - 0.5, fontweight="bold",
            clip_on=False, zorder=10)


def draw_ghost(ax, d):
    for xy in d["paths_old"]:
        ax.plot(xy[:, 0], xy[:, 1], color=C_OLD, lw=1.1, ls=(0, (3.2, 1.8)),
                zorder=7, path_effects=CASE)


def draw_tags(ax, d, kind, box, mm_per_m):
    """A on the 2020 panel, N and S on the 2024 panel. Targets closer
    together than 0.45 of the panel width share one tag with an arrow each,
    so the braided pair in Z1 reads as a pair."""
    groups = []
    if kind == "2020":
        groups.append(("A", [xy[len(xy) // 2] for xy in d["paths_old"]],
                       C_OLD))
    if kind == "2024":
        groups.append(("N", [xy[len(xy) // 2] for xy in d["paths_new"]],
                       C_NEW))
        if d["split"] is not None:
            groups.append(("S", [np.asarray(d["split"])], C_NEW))
    ext = box
    w = ext[1] - ext[0]
    arr, qe = z32.read_window(TIF[kind], *box)
    ink = np.asarray(arr.filled(0), dtype=float)
    scale_zone = (ext[0] + 0.62 * w, ext[2] + 0.20 * (ext[3] - ext[2]))
    courses = d["paths_old"] + d["paths_new"]
    taken = []
    for letter, targets, col in groups:
        clusters = []
        for t in targets:
            for c in clusters:
                if np.hypot(*(np.mean(c, axis=0) - t)) < 0.45 * w:
                    c.append(t)
                    break
            else:
                clusters.append([t])
        for c in clusters:
            anchor = np.mean(c, axis=0)
            x, y = an.place_label(ext, anchor, ink, qe, taken, scale_zone,
                                  courses, letter, mm_per_m)
            taken.append((x, y))
            for t in c:
                ax.annotate(
                    letter, xy=(t[0], t[1]), xytext=(x, y), ha="center",
                    va="center", fontsize=FS["tag"], fontweight="bold",
                    color="black", zorder=14,
                    bbox=dict(boxstyle="circle,pad=0.25", facecolor="white",
                              edgecolor=col, linewidth=0.9),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9,
                                    shrinkA=0.0, shrinkB=1.5,
                                    mutation_scale=7, path_effects=CASE))


def draw_north(ax):
    """Once per figure, in the key strip: kept off the panels so it cannot
    be read as option 1's N (new) tag."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate("", xy=(0.5, 0.72), xytext=(0.5, 0.10),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4,
                                mutation_scale=9))
    ax.text(0.5, 0.80, "N", ha="center", va="bottom", fontsize=FS["north"],
            fontweight="bold")


def badge(ax, x, y, letter, color):
    ax.text(x, y, letter, ha="center", va="center", fontsize=FS["tag"],
            fontweight="bold", transform=ax.transAxes,
            bbox=dict(boxstyle="circle,pad=0.25", facecolor="white",
                      edgecolor=color, linewidth=0.9))


# ------------------------------------------------------------------- layout
OPTIONS = {
    1: dict(name="callouts",
            cols=[("2020", 1.0), ("sfm", 1.0), ("2024", 1.0)]),
    2: dict(name="change_column",
            cols=[("2020", 1.0), ("sfm", 1.0), ("2024", 1.0),
                  ("change", 1.0)]),
    3: dict(name="network_map",
            cols=[("2020", 1.0), ("sfm", 1.0), ("2024", 1.0),
                  ("network", 1.0)]),
    4: dict(name="before_after_pair",
            cols=[("2020", 1.0), ("2024", 1.0)]),
    5: dict(name="schematic",
            cols=[("2020", 1.0), ("sfm", 1.0), ("2024", 1.0),
                  ("schematic", 0.55)]),
}


def build(key, zones, data, vlim):
    opt = OPTIONS[key]
    cols = opt["cols"]
    wsum = sum(wf for _, wf in cols)
    aspects = [(z.n_max - z.n_min) / (z.e_max - z.e_min)
               for _, z in zones.iterrows()]
    fixed_h = MT + HEADER_H + GAP_Y * (len(aspects) - 1) + KEY_H + MB
    pw_w = (FIG_W_MM - ML - MR - ZONE_W - GAP_X * (len(cols) - 1)) / wsum
    pw_h = (MAX_H_MM - fixed_h) / sum(aspects)
    pw = min(pw_w, pw_h)
    maps_w = pw * wsum + GAP_X * (len(cols) - 1)
    fig_w = max(ML + ZONE_W + maps_w + MR, 120.0)
    fig_h = fixed_h + pw * sum(aspects)
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    def rect(x, ytop, w, h):
        return [x / fig_w, 1 - (ytop + h) / fig_h, w / fig_w, h / fig_h]

    x0 = ML + ZONE_W
    y = MT + HEADER_H
    q_im = None
    last_map = max(i for i, (k, _) in enumerate(cols) if k != "schematic")
    for row, ((_, z), asp) in enumerate(zip(zones.iterrows(), aspects)):
        d = data[z.zone_id]
        box = d["box"]
        h = pw * asp
        mm_per_m = pw / (box[1] - box[0])
        x = x0
        for ci, (kind, wf) in enumerate(cols):
            w = pw * wf
            ax = fig.add_axes(rect(x, y, w, h))
            if kind in TIF:
                q_im = draw_q(ax, kind, box)
                if key == 1:
                    draw_tags(ax, d, kind, box, mm_per_m)
                if key == 4 and kind == "2024":
                    draw_ghost(ax, d)
            elif kind == "change":
                c_sm = m.draw_change(ax, box, vlim)
            elif kind == "network":
                draw_network(ax, d, box)
            elif kind == "schematic":
                draw_schematic(ax, d, TAGS[z.zone_id])
            m.bare(ax, box)
            if kind == "schematic":
                for s in ax.spines.values():
                    s.set_color("0.6")
            if ci == last_map:
                m.draw_scale_bar(ax, box, 25, mm_per_m)
            if row == 0:
                fig.text((x + w / 2) / fig_w, 1 - (y - 1.2) / fig_h,
                         HEAD[kind], ha="center", va="bottom",
                         fontsize=FS["header"], fontweight="bold")
            x += w + GAP_X
        fig.text((ML + ZONE_W / 2) / fig_w, 1 - (y + h / 2) / fig_h,
                 z.zone_id, ha="center", va="center", rotation=90,
                 fontsize=FS["zone"], fontweight="bold", color="red")
        y += h + GAP_Y

    # ---- key strip: the discharge scale, then this option's own key -----
    ky = fig_h - MB - KEY_H
    cax = fig.add_axes(rect(x0, ky + 3.0, 52.0, 2.8))
    cb = fig.colorbar(q_im, cax=cax, orientation="horizontal")
    cb.set_label("unit discharge (m$^3$ s$^{-1}$ m$^{-1}$-eq., log)",
                 fontsize=FS["cbar"], labelpad=2)
    cb.ax.tick_params(labelsize=FS["tick"], length=2.2, width=0.5)
    cb.outline.set_linewidth(0.5)
    draw_north(fig.add_axes(rect(fig_w - MR - 7.0, ky + 1.0, 7.0,
                                 KEY_H - 2.0)))
    kx = x0 + 60.0
    kw = fig_w - MR - 9.0 - kx
    kax = fig.add_axes(rect(kx, ky, kw, KEY_H))
    kax.axis("off")
    kw_args = dict(loc="center left", frameon=False, fontsize=FS["key"],
                   handlelength=2.4, handletextpad=0.6, labelspacing=0.45,
                   borderaxespad=0.0)
    if key == 1:
        for i, (letter, col, text) in enumerate([
                ("A", C_OLD, "abandoned 2020 path (on 2020 panel)"),
                ("N", C_NEW, "new 2024 path (on 2024 panel)"),
                ("S", C_NEW, "split point (Z1, on 2024 panel)")]):
            yy = 0.80 - i * 0.30
            badge(kax, 0.03, yy, letter, col)
            kax.text(0.09, yy, text, transform=kax.transAxes, ha="left",
                     va="center", fontsize=FS["key"])
    elif key == 2:
        cax2 = fig.add_axes(rect(kx + 8.0, ky + 3.0, 50.0, 2.8))
        cb2 = fig.colorbar(c_sm, cax=cax2, orientation="horizontal")
        cb2.set_ticks([-vlim, 0, vlim])
        cb2.set_ticklabels([f"−{vlim:.2f}", "0", f"+{vlim:.2f}"])
        cb2.set_label("discharge change, 2024 − 2020 (±3 SD)",
                      fontsize=FS["cbar"], labelpad=2)
        cb2.ax.tick_params(labelsize=FS["tick"], length=2.2, width=0.5)
        cb2.outline.set_linewidth(0.5)
    elif key == 3:
        kax.legend(handles=[
            Patch(color=C_KEEP, label="channel in both epochs (persisted)"),
            Patch(color=C_OLD, label="2020 only (abandoned)"),
            Patch(color=C_NEW, label="2024 only (new)"),
            Patch(color="0.55", alpha=FAINT,
                  label="faint: stretch shorter than 30 m")], ncol=2,
            **kw_args)
    elif key == 4:
        kax.legend(handles=[
            Line2D([], [], color=C_OLD, lw=1.1, ls=(0, (3.2, 1.8)),
                   path_effects=CASE,
                   label="2020 path abandoned by 2024\n(ghost, on the "
                         "2024 panel)")], **kw_args)
    elif key == 5:
        kax.legend(handles=[
            Line2D([], [], color=C_OLD, lw=1.3, ls=(0, (3, 1.8)),
                   label="2020 path, abandoned"),
            Line2D([], [], color=C_NEW, lw=1.3, label="2024 path, new"),
            Patch(color=C_CONTEXT, label="channel in both epochs"),
            Line2D([], [], ls="none", marker="o", ms=6, mfc="none",
                   mec="black", mew=1.0, label="split point (Z1)")],
            ncol=2, **kw_args)

    stem = f"opt{key}_{opt['name']}"
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    return dict(key=key, stem=stem, w=fig_w, h=fig_h, pw=pw,
                mm_per_m=pw / 116.0)


# --------------------------------------------------------------------- run
DESC = {
    1: ("Callout labels", "letter tags with leader arrows -- A on the 2020 "
        "panel, N and S on the 2024 panel; SfM clean; nothing traced",
        "minor overlay"),
    2: ("Discharge-change column", "4th column: 2024 minus 2020 discharge, "
        "RdBu +/-3 SD, near-zero faded; second colour bar",
        "new panel + re-layout"),
    3: ("Channel-network map", "4th column: both channel networks "
        "classified persisted (blue) / abandoned (vermillion) / new (black) "
        "over faded hillshade; short stretches faint",
        "new panel"),
    4: ("Before/after pair", "2020 and 2024 lidar only, larger panels at one common "
        "scale; abandoned 2020 path as a dashed ghost on the 2024 panel; "
        "SfM zoom dropped", "re-plot, new layout"),
    5: ("Schematic beside each row", "rasters untouched; narrow line "
        "diagram per zone (grey = persisted, dashed = abandoned, solid = "
        "new, ring = split) with a one-word tag", "new drawing code"),
}


def change_limit():
    """Exactly as 56: 3 SD of the change raster over the basin's bounding
    box (the extent of valid 2020 discharge)."""
    with rasterio.open(m.TIF["2020"]) as src:
        a = src.read(1, masked=True)
        t = src.transform
    r, c = np.nonzero(~np.ma.getmaskarray(a))
    basin_ext = (t.c + c.min() * t.a, t.c + (c.max() + 1) * t.a,
                 t.f + (r.max() + 1) * t.e, t.f + r.min() * t.e)
    v, _ = m.read_window(m.TIF["change"], basin_ext)
    return 3 * float(np.nanstd(np.asarray(v.filled(np.nan), dtype=float)))


def main():
    zones = pd.read_csv(z32.TABLES / "flow_change_zones.csv")
    data = {z.zone_id: zone_data(z) for _, z in zones.iterrows()}

    vlim = change_limit()
    rows = [build(k, zones, data, vlim) for k in OPTIONS]

    pal = {"persisted": C_KEEP, "abandoned": C_OLD, "new": C_NEW}
    cvd = []
    names = list(pal)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for vis in ("normal", "protan", "deutan", "tritan"):
                cvd.append(dict(pair=f"{names[i]} {pal[names[i]]} vs "
                                     f"{names[j]} {pal[names[j]]}",
                                vision=vis,
                                delta_e_oklab_x100=round(a70.delta_e(
                                    pal[names[i]], pal[names[j]], vis), 1)))
    cvd = pd.DataFrame(cvd)
    cvd_csv = z32.TABLES / "flow_zoom_options_cvd.csv"
    cvd.to_csv(cvd_csv, index=False)
    worst = cvd.loc[cvd.vision != "normal", "delta_e_oklab_x100"].min()

    md = ["# Zoom figure (Fig. 10) -- five annotation approaches", "",
          "Mock-ups for choosing an approach, not final figures. All share "
          "script 32's discharge rendering (YlGnBu, log 1e-4 to 0.48, "
          "saturated above), the zone extents, a 25 m bar per row and one "
          "north arrow; the graticule is off in all five. Old/new courses "
          "are script 70's (stretches >= 30 m).", "",
          "| # | approach | file | size (mm) | panel width | what is drawn "
          "| work to finish |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        name, what, work = DESC[r["key"]]
        md.append(f"| {r['key']} | {name} | `{r['stem']}.png` | "
                  f"{r['w']:.0f} x {r['h']:.0f} | {r['pw']:.1f} mm | {what} "
                  f"| {work} |")
    md += ["", f"Change scale (option 2): +/-{vlim:.4f}, 3 SD of the "
           "basin-wide change raster.",
           f"Network palette (option 3): worst CVD pair dE {worst:.1f} "
           f"(OKLab x100; see `{cvd_csv.name}`)."]
    (OUT / "OPTIONS.md").write_text("\n".join(md) + "\n")

    for r in rows:
        print(f"  opt{r['key']}: {r['w']:.0f} x {r['h']:.0f} mm, panel "
              f"{r['pw']:.1f} mm")
    print(f"  change limit +/-{vlim:.4f}; network palette worst CVD dE "
          f"{worst:.1f}")
    log("71_fig_flow_zoom_options.py run. Five annotation mock-ups for the "
        "zoom figure, same data and rendering as 32: "
        + "; ".join(f"opt{r['key']} {r['stem']} {r['w']:.0f}x{r['h']:.0f} mm "
                    f"panel {r['pw']:.1f} mm" for r in rows)
        + f". Change +/-{vlim:.4f} (3 SD). Network palette worst CVD dE "
        f"{worst:.1f}. Written to {OUT.name}/ and {cvd_csv.name}.",
        run="flow_zoom_options")


if __name__ == "__main__":
    main()
