#!/usr/bin/env python3
"""
72_fig_flow_consolidated_zoomchange.py
======================================
Fig. 9 (basin-wide overland flow, three epochs) and the chosen zoom figure --
option 2 of 71, Z1-Z3 x (2020 | SfM | 2024 | change) -- consolidated into one
figure. Three layouts for the author to choose from; none is inserted in the
manuscript.

  A stacked_key          basin row (a-c) above the zoom block, columns
                         aligned so each epoch reads straight down; the empty
                         fourth slot of the basin row holds the colour bars
                         and north arrow.
  B stacked_basinchange  as A, but the fourth basin slot is the basin-wide
                         change map, so the change column also runs top to
                         bottom (the raster of Fig. 11, without its SfM
                         network overlay); key strip at the foot.
  C epoch_rows           one row per epoch plus one for the change; columns
                         are the basin, then Z1, Z2, Z3 -- each row reads as
                         "this epoch everywhere".

Rendering is unchanged from the source figures: basin panels as 31 (Fig. 9's
full raster extent, alpha 0.85, red zone boxes labelled above), zoom panels
as 32 (alpha 0.88), change as 56.draw_change (RdBu, +/- 3 SD of the
basin-wide change, |change| below a quarter of that transparent). Every
discharge panel shares the log scale 1e-4 to 0.48, saturated above.
Graticules are off; panel letters sit in each panel's top-left corner.

Run from the repository root.

Outputs:
  results/figures/fig_flow_consolidated_zoomchange/opt<A|B|C>_<name>.png/.pdf
  results/figures/fig_flow_consolidated_zoomchange/OPTIONS.md
  results/logs/flow_consolidated_zoomchange_<date>.log
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 71 carries the zoom rendering (32), the change raster (56) and the key
# helpers; the scripts behind it set rcParams at import -- keep them out.
with matplotlib.rc_context():
    o71 = _load("fz71", "71_fig_flow_zoom_options.py")
z32, m = o71.z32, o71.m

OUT = z32.FIGURES / "fig_flow_consolidated_zoomchange"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM = 174.0
ML = MR = MT = MB = 3.0
GAP_X = 2.5
GAP_Y = 3.0
HEADER_H = 7.0            # two-line column header: name, then date
KEY_H = 15.0
VMIN, VMAX = z32.VMIN, z32.VMAX
C_ZONE = "red"            # as Figs 9 and 10
FS = dict(head=7.0, sub=6.3, row=7.5, zone=8.5, letter=7.0, cbar=6.8,
          tick=6.3)
HEAD = {"2020": ("2020 lidar", "pre-event"),
        "sfm": ("2024 CAP SfM", "5 Oct, +8 d"),
        "2024": ("2024 lidar", "15–16 Nov, +7 wk"),
        "change": ("change", "2024 − 2020 lidar")}
DISCHARGE_LABEL = "unit discharge (m$^3$ s$^{-1}$ m$^{-1}$-eq., log)"
CHANGE_LABEL = "discharge change, 2024 − 2020 (±3 SD)"


def basin_extent():
    """Fig. 9's extent: the full discharge raster, basin plus context."""
    with rasterio.open(z32.DISCHARGE_2020_TIF) as src:
        b = src.bounds
    return (b.left, b.right, b.bottom, b.top)


# ------------------------------------------------------------------ drawing
def draw_boxes(ax, zones, fs):
    """Fig. 9's zone boxes: red outline, label centred above the box."""
    for _, z in zones.iterrows():
        ax.add_patch(Rectangle((z.e_min, z.n_min), z.width_m, z.height_m,
                               fill=False, edgecolor=C_ZONE, linewidth=0.9,
                               zorder=6))
        ax.text(z.e_min + z.width_m / 2, z.n_max + 12, z.zone_id,
                color=C_ZONE, fontsize=fs, fontweight="bold", ha="center",
                va="bottom", zorder=7)


def draw_basin_q(ax, kind, ext, zones, box_fs):
    """Script 31's rendering, unchanged."""
    hs, he = z32.read_window(z32.HILLSHADE_TIF, *ext)
    ax.imshow(hs, cmap="gray", extent=he, origin="upper", zorder=1)
    arr, ex = z32.read_window(o71.TIF[kind], *ext)
    im = ax.imshow(np.clip(arr, VMIN, VMAX), cmap="YlGnBu",
                   norm=LogNorm(vmin=VMIN, vmax=VMAX), extent=ex,
                   origin="upper", alpha=0.85, zorder=2)
    draw_boxes(ax, zones, box_fs)
    return im


def draw_basin_change(ax, ext, vlim, zones, box_fs):
    """56's basin-wide change panel: hillshade, the watershed tinted so the
    sparse change field has an outline to sit in, then the change."""
    hs, he = m.read_window(m.TIF["hs"], ext)
    ax.imshow(hs, cmap="gray", extent=he, origin="upper", zorder=1,
              interpolation="bilinear")
    m.draw_watershed_tint(ax, ext)
    sm = m.draw_change(ax, ext, vlim, overlay=True)
    draw_boxes(ax, zones, box_fs)
    return sm


def letter(ax, ch):
    ax.text(0.03, 0.97, f"({ch})", transform=ax.transAxes, ha="left",
            va="top", fontsize=FS["letter"], fontweight="bold", zorder=20,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.85))


def header(fig, x, ytop, w, kind, fig_w, fig_h, color="black"):
    name, sub = HEAD[kind]
    cx = (x + w / 2) / fig_w
    fig.text(cx, 1 - (ytop - 3.6) / fig_h, name, ha="center", va="bottom",
             fontsize=FS["head"], fontweight="bold", color=color)
    fig.text(cx, 1 - (ytop - 0.8) / fig_h, sub, ha="center", va="bottom",
             fontsize=FS["sub"])


def cbar_q(fig, rect, im, label=DISCHARGE_LABEL):
    cb = fig.colorbar(im, cax=fig.add_axes(rect), orientation="horizontal")
    cb.set_label(label, fontsize=FS["cbar"], labelpad=2)
    cb.ax.tick_params(labelsize=FS["tick"], length=2.2, width=0.5)
    cb.outline.set_linewidth(0.5)


def cbar_d(fig, rect, sm, vlim, label=CHANGE_LABEL):
    cb = fig.colorbar(sm, cax=fig.add_axes(rect), orientation="horizontal")
    cb.set_ticks([-vlim, 0, vlim])
    cb.set_ticklabels([f"−{vlim:.2f}", "0", f"+{vlim:.2f}"])
    cb.set_label(label, fontsize=FS["cbar"], labelpad=2)
    cb.ax.tick_params(labelsize=FS["tick"], length=2.2, width=0.5)
    cb.outline.set_linewidth(0.5)


def key_strip(fig, rect, x0, ky, fig_w, q_im, d_sm, vlim):
    cbar_q(fig, rect(x0, ky + 3.0, 52.0, 2.6), q_im)
    cbar_d(fig, rect(x0 + 62.0, ky + 3.0, 50.0, 2.6), d_sm, vlim)
    o71.draw_north(fig.add_axes(rect(fig_w - MR - 7.0, ky + 0.5, 7.0,
                                     KEY_H - 1.0)))


def zoom_row(fig, rect, d, x0, y, pw, h, vlim, letters, state):
    """One zone across 2020 | SfM | 2024 | change, as option 2 of 71."""
    box = d["box"]
    mm_per_m = pw / (box[1] - box[0])
    x = x0
    for ci, kind in enumerate(("2020", "sfm", "2024", "change")):
        ax = fig.add_axes(rect(x, y, pw, h))
        if kind == "change":
            state["d_sm"] = m.draw_change(ax, box, vlim)
        else:
            state["q_im"] = o71.draw_q(ax, kind, box)
        m.bare(ax, box)
        letter(ax, next(letters))
        if ci == 3:
            m.draw_scale_bar(ax, box, 25, mm_per_m)
        x += pw + GAP_X


# ------------------------------------------------------------------ layouts
def build_stacked(key, zones, data, vlim, basin_change):
    """A (basin_change=False) and B (True)."""
    bext = basin_extent()
    basin_asp = (bext[3] - bext[2]) / (bext[1] - bext[0])
    row_w = 6.0
    pw = (FIG_W_MM - ML - MR - row_w - 3 * GAP_X) / 4
    zasp = [(z.n_max - z.n_min) / (z.e_max - z.e_min)
            for _, z in zones.iterrows()]
    basin_h = pw * basin_asp
    strip = KEY_H + 2.0 if basin_change else 0.0
    fig_h = (MT + HEADER_H + basin_h + HEADER_H + pw * sum(zasp)
             + GAP_Y * (len(zasp) - 1) + strip + MB)
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
    fig_w = FIG_W_MM

    def rect(x, ytop, w, h):
        return [x / fig_w, 1 - (ytop + h) / fig_h, w / fig_w, h / fig_h]

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    state = {}
    x0 = ML + row_w
    y = MT + HEADER_H
    kinds = ["2020", "sfm", "2024"] + (["change"] if basin_change else [])
    mm_per_m_basin = pw / (bext[1] - bext[0])
    for ci, kind in enumerate(kinds):
        x = x0 + ci * (pw + GAP_X)
        ax = fig.add_axes(rect(x, y, pw, basin_h))
        if kind == "change":
            state["d_sm"] = draw_basin_change(ax, bext, vlim, zones, 6.0)
        else:
            state["q_im"] = draw_basin_q(ax, kind, bext, zones, 6.0)
        m.bare(ax, bext)
        letter(ax, next(letters))
        if ci == len(kinds) - 1:
            m.draw_scale_bar(ax, bext, 500, mm_per_m_basin, corner="tr")
        header(fig, x, y, pw, kind, fig_w, fig_h)
    fig.text((ML + row_w / 2) / fig_w, 1 - (y + basin_h / 2) / fig_h,
             "watershed", ha="center", va="center", rotation=90,
             fontsize=FS["row"], fontweight="bold")
    key_top = y
    y += basin_h + HEADER_H
    if not basin_change:
        header(fig, x0 + 3 * (pw + GAP_X), y, pw, "change", fig_w, fig_h)

    for (_, z), asp in zip(zones.iterrows(), zasp):
        h = pw * asp
        zoom_row(fig, rect, data[z.zone_id], x0, y, pw, h, vlim,
                 letters, state)
        fig.text((ML + row_w / 2) / fig_w, 1 - (y + h / 2) / fig_h,
                 z.zone_id, ha="center", va="center", rotation=90,
                 fontsize=FS["zone"], fontweight="bold", color=C_ZONE)
        y += h + GAP_Y

    if basin_change:
        key_strip(fig, rect, x0, fig_h - MB - KEY_H, fig_w, state["q_im"],
                  state["d_sm"], vlim)
    else:
        # the empty fourth slot of the basin row carries the key
        kx = x0 + 3 * (pw + GAP_X) + 2.0
        kw = pw - 4.0
        cbar_q(fig, rect(kx, key_top + 8.0, kw, 2.6), state["q_im"],
               label="unit discharge\n(m$^3$ s$^{-1}$ m$^{-1}$-eq., log)")
        cbar_d(fig, rect(kx, key_top + 25.0, kw, 2.6), state["d_sm"], vlim,
               label="discharge change\n2024 − 2020 (±3 SD)")
        o71.draw_north(fig.add_axes(rect(kx + kw / 2 - 4.0, key_top + 40.0,
                                         8.0, 15.0)))
    return fig, fig_w, fig_h, dict(basin_w=pw, zoom_w=pw)


def build_epoch_rows(key, zones, data, vlim):
    """C: rows = 2020, SfM, 2024, change; columns = basin, Z1, Z2, Z3. Row
    height is set so the four columns fill the width at their own aspect."""
    bext = basin_extent()
    exts = [bext] + [data[z.zone_id]["box"] for _, z in zones.iterrows()]
    inv_asp = [(e[1] - e[0]) / (e[3] - e[2]) for e in exts]
    row_w = 9.0
    avail = FIG_W_MM - ML - MR - row_w - GAP_X * (len(exts) - 1)
    h = avail / sum(inv_asp)
    widths = [h * ia for ia in inv_asp]
    kinds = ["2020", "sfm", "2024", "change"]
    fig_h = (MT + HEADER_H + h * len(kinds) + GAP_Y * (len(kinds) - 1)
             + KEY_H + 2.0 + MB)
    fig_w = FIG_W_MM
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    def rect(x, ytop, w, hh):
        return [x / fig_w, 1 - (ytop + hh) / fig_h, w / fig_w, hh / fig_h]

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    state = {}
    x0 = ML + row_w
    y = MT + HEADER_H
    for ri, kind in enumerate(kinds):
        x = x0
        for ci, (ext, w) in enumerate(zip(exts, widths)):
            ax = fig.add_axes(rect(x, y, w, h))
            mm_per_m = w / (ext[1] - ext[0])
            if ci == 0:
                if kind == "change":
                    state["d_sm"] = draw_basin_change(ax, ext, vlim, zones,
                                                      5.0)
                else:
                    state["q_im"] = draw_basin_q(ax, kind, ext, zones, 5.0)
            elif kind == "change":
                state["d_sm"] = m.draw_change(ax, ext, vlim)
            else:
                state["q_im"] = o71.draw_q(ax, kind, ext)
            m.bare(ax, ext)
            letter(ax, next(letters))
            if ri == 0:
                m.draw_scale_bar(ax, ext, 500 if ci == 0 else 25, mm_per_m,
                                 corner="tr" if ci == 0 else "br")
                label = "watershed" if ci == 0 else zones.iloc[ci - 1].zone_id
                fig.text((x + w / 2) / fig_w, 1 - (y - 1.2) / fig_h, label,
                         ha="center", va="bottom", fontweight="bold",
                         fontsize=FS["head"] if ci == 0 else FS["zone"],
                         color="black" if ci == 0 else C_ZONE)
            x += w + GAP_X
        name, sub = HEAD[kind]
        fig.text((ML + row_w / 2) / fig_w, 1 - (y + h / 2) / fig_h,
                 f"{name}\n{sub}", ha="center", va="center", rotation=90,
                 fontsize=FS["sub"], fontweight="bold", linespacing=1.15)
        y += h + GAP_Y

    key_strip(fig, rect, x0, fig_h - MB - KEY_H, fig_w, state["q_im"],
              state["d_sm"], vlim)
    return fig, fig_w, fig_h, dict(basin_w=widths[0], zoom_w=widths[1])


LAYOUTS = {
    "A": ("stacked_key", "basin row (a-c) above the zoom block (d-o), "
          "columns aligned; key in the basin row's fourth slot",
          lambda z, d, v: build_stacked("A", z, d, v, basin_change=False)),
    "B": ("stacked_basinchange", "as A, with the basin-wide change map "
          "(d) in the fourth slot so the change column runs top to bottom; "
          "zoom panels (e-p); key strip at the foot",
          lambda z, d, v: build_stacked("B", z, d, v, basin_change=True)),
    "C": ("epoch_rows", "rows = 2020 / SfM / 2024 / change; columns = "
          "watershed, Z1, Z2, Z3 (a-p, row by row); key strip at the foot",
          lambda z, d, v: build_epoch_rows("C", z, d, v)),
}


def main():
    zones = pd.read_csv(z32.TABLES / "flow_change_zones.csv")
    data = {z.zone_id: o71.zone_data(z) for _, z in zones.iterrows()}
    vlim = o71.change_limit()

    rows = []
    for key, (name, desc, fn) in LAYOUTS.items():
        fig, fw, fh, sizes = fn(zones, data, vlim)
        stem = f"opt{key}_{name}"
        fig.savefig(OUT / f"{stem}.png", dpi=300)
        fig.savefig(OUT / f"{stem}.pdf")
        plt.close(fig)
        rows.append(dict(key=key, stem=stem, desc=desc, w=fw, h=fh, **sizes))
        print(f"  opt{key}: {fw:.0f} x {fh:.0f} mm, basin panel "
              f"{sizes['basin_w']:.1f} mm, Z1 panel {sizes['zoom_w']:.1f} mm")

    md = ["# Fig. 9 + zoom figure (option 2) consolidated -- layouts", "",
          "Mock-ups for choosing a layout. Rendering unchanged from the "
          "source figures: basin panels as script 31 (Fig. 9's extent), zoom "
          "panels as script 32, change as script 56 (RdBu, "
          f"+/-{vlim:.4f} = 3 SD of the basin-wide change). Graticules off. "
          "In the manuscript a 174 mm figure is reduced to 131 mm, so a "
          "225 mm figure leaves about 25 mm of the page for the caption.", "",
          "| layout | file | size (mm) | basin panel width | Z1 panel width "
          "| arrangement |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['key']} | `{r['stem']}.png` | {r['w']:.0f} x "
                  f"{r['h']:.0f} | {r['basin_w']:.1f} mm | {r['zoom_w']:.1f} "
                  f"mm | {r['desc']} |")
    (OUT / "OPTIONS.md").write_text("\n".join(md) + "\n")
    log("72_fig_flow_consolidated_zoomchange.py run. Fig. 9 + zoom option 2 "
        "consolidated, three layouts: "
        + "; ".join(f"opt{r['key']} {r['stem']} {r['w']:.0f}x{r['h']:.0f} mm "
                    f"(basin {r['basin_w']:.1f} mm, Z1 {r['zoom_w']:.1f} mm)"
                    for r in rows)
        + f". Change +/-{vlim:.4f} (3 SD). Written to {OUT.name}/.",
        run="flow_consolidated_zoomchange")


if __name__ == "__main__":
    main()
