#!/usr/bin/env python3
"""
57_fig_flow_consolidated_e_options.py
=====================================
Follow-up to 56_fig_flow_consolidated.py, variant (e), which the author
selected. The change layer there is painted over the coloured discharge
panel, so mid-magnitude change blends into the YlGnBu ramp and the
hillshade beneath it -- worst where gained discharge (blue) sits on a
channel that is already dark blue.

Eight options, all keeping variant (e)'s layout -- basin row of four panels
above, zoom block transposed so the three zones are COLUMNS -- and varying
only how the change is shown.

  PLACEMENT
    own row   the change gets its own zoom row over a pale base, so it never
              competes with the discharge ramp (3 zoom rows)
    overlay   variant (e)'s arrangement kept (2 zoom rows), but the base
              under it is lightened or the ramp hue moved off blue

  e1  ownrow_rdbu        own row, RdBu, pale grey relief base
  e2  ownrow_puor        own row, PuOr -- orange/purple, no hue shared with
                         the blue-green discharge ramp
  e3  ownrow_brbg        own row, BrBG -- brown/teal
  e4  ownrow_prgn        own row, PRGn -- purple/green
  e5  ownrow_rdbu_white  own row, RdBu on a near-white base: maximum
                         figure-ground contrast, terrain barely present
  e6  overlay_palebase   overlay kept, but the 2024 discharge under it is
                         desaturated to pale grey-blue so the change reads
  e7  overlay_puor       overlay kept, ramp moved to PuOr so nothing shares
                         a hue with the discharge beneath it
  e8  ownrow_classes     own row, two flat classes -- lost / gained beyond
                         the 3 SD limit -- on a pale base. Maximum clarity,
                         magnitude information deliberately discarded

Every option keeps the shared scales, the derived abandoned / new / split
annotation, the per-zoom-panel scale bars, the single north arrow and the
7 pt floor from 56.

Outputs:
  results/figures/fig_flow_consolidated/e_options/fig_flow_e_<key>.png/.pdf
  results/figures/fig_flow_consolidated/e_options/E_OPTIONS.md
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patheffects

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
import rasterio

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "fc56", HERE / "56_fig_flow_consolidated.py")
m = importlib.util.module_from_spec(spec)
sys.modules["fc56"] = m
spec.loader.exec_module(m)

sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402

OUT = m.OUT / "e_options"
OUT.mkdir(parents=True, exist_ok=True)
FS = m.FS


def pale_relief(ax, ext, lo=0.62, hi=1.0):
    """Hillshade mapped into a pale grey band, so terrain is present as
    texture but sits well below the change colours in contrast."""
    hs, he = m.read_window(m.TIF["hs"], ext)
    cmap = LinearSegmentedColormap.from_list(
        "pale", [str(lo), str(hi)], N=256)
    ax.imshow(hs, cmap=cmap, extent=he, origin="upper", zorder=1,
              interpolation="bilinear", vmin=0, vmax=255)


def desaturated_discharge(ax, ext):
    """The discharge field in a pale grey-blue ramp: channel geometry stays
    readable as context, but the panel's colour budget is left for the
    change layer on top of it."""
    hs, he = m.read_window(m.TIF["hs"], ext)
    ax.imshow(hs, cmap="gray", extent=he, origin="upper", zorder=1,
              interpolation="bilinear")
    q, qe = m.read_window(m.TIF["2024"], ext)
    qm = np.ma.masked_where(q.filled(0) <= 0, q)
    cmap = LinearSegmentedColormap.from_list(
        "palequench", ["#FFFFFF", "#D8DFE6", "#AEBDCA"], N=256)
    ax.imshow(qm, cmap=cmap, norm=LogNorm(m.VMIN, m.VMAX), extent=qe,
              origin="upper", alpha=0.92, zorder=2, interpolation="nearest")


def change_layer(ax, ext, vlim, cmap_name, fade=0.18, zorder=4, alpha=1.0):
    d, de = m.read_window(m.TIF["change"], ext)
    z = np.asarray(d.filled(np.nan), dtype=float)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(Normalize(-vlim, vlim)(z))
    lo, hi = fade * vlim, vlim
    a = np.clip((np.abs(z) - lo) / (hi - lo), 0, 1)
    rgba[..., 3] = np.where(np.isfinite(z), a * alpha, 0.0)
    ax.imshow(rgba, extent=de, origin="upper", zorder=zorder,
              interpolation="nearest")
    return plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(-vlim, vlim))


def change_contours(ax, ext, vlim, zorder=5):
    """The +/- limit isolines of the change field, drawn as outlines over the
    discharge panel. Nothing underneath is hidden."""
    d, de = m.read_window(m.TIF["change"], ext)
    z = np.asarray(d.filled(np.nan), dtype=float)
    ny, nx = z.shape
    xs = np.linspace(de[0], de[1], nx)
    ys = np.linspace(de[3], de[2], ny)
    # loss dashed, gain solid, both white-cased so they read over the dark
    # blue channels as well as over pale hillslope
    for lvl, col, ls in ((-vlim, "#8B1A1A", "--"), (vlim, "#0B3D91", "-")):
        try:
            cs = ax.contour(xs, ys, z, levels=[lvl], colors=[col],
                            linewidths=0.9, linestyles=ls, zorder=zorder)
            for coll in cs.collections if hasattr(cs, "collections") else []:
                coll.set_path_effects([
                    matplotlib.patheffects.withStroke(linewidth=2.1,
                                                      foreground="white")])
            if not hasattr(cs, "collections"):
                cs.set(path_effects=[matplotlib.patheffects.withStroke(
                    linewidth=2.1, foreground="white")])
        except Exception:
            pass
    return None


def change_classes(ax, ext, vlim, zorder=4):
    """Two flat classes: change beyond +/- the 3 SD limit. Everything inside
    the limit is left transparent, so the panel shows only what the figure
    claims is real."""
    d, de = m.read_window(m.TIF["change"], ext)
    z = np.asarray(d.filled(np.nan), dtype=float)
    rgba = np.zeros(z.shape + (4,), dtype=float)
    lost, gained = z <= -vlim, z >= vlim
    rgba[lost] = matplotlib.colors.to_rgba("#B2182B")
    rgba[gained] = matplotlib.colors.to_rgba("#2166AC")
    rgba[..., 3] = np.where(lost | gained, 0.95, 0.0)
    ax.imshow(rgba, extent=de, origin="upper", zorder=zorder,
              interpolation="nearest")
    return None


# ------------------------------------------------------------------ options
OPTIONS = {
    "e1": dict(mode="ownrow", cmap="RdBu", base="pale",
               desc="change on its own row, RdBu, pale grey relief base"),
    "e2": dict(mode="ownrow", cmap="PuOr_r", base="pale",
               desc="change on its own row, PuOr -- orange/purple, shares no "
                    "hue with the blue-green discharge ramp"),
    "e3": dict(mode="ownrow", cmap="BrBG", base="pale",
               desc="change on its own row, BrBG -- brown/teal"),
    "e4": dict(mode="ownrow", cmap="PRGn", base="pale",
               desc="change on its own row, PRGn -- purple/green"),
    "e5": dict(mode="ownrow", cmap="RdBu", base="white",
               desc="change on its own row, RdBu on a near-white base: "
                    "maximum figure-ground contrast, terrain barely present"),
    "e6": dict(mode="overlay", cmap="RdBu", base="quench",
               desc="overlay kept, but the 2024 discharge beneath is "
                    "desaturated to pale grey-blue so the change reads"),
    "e7": dict(mode="overlay", cmap="PuOr_r", base="normal",
               desc="overlay kept, ramp moved to PuOr so nothing shares a "
                    "hue with the discharge beneath it"),
    "e8": dict(mode="ownrow", cmap=None, base="pale",
               desc="change on its own row as two flat classes, lost / "
                    "gained beyond the 3 SD limit; magnitude discarded"),
    "e9": dict(mode="ownrow", cmap="RdBu", base="pale", ann_rows="epochs",
               desc="as e1, but the course annotation is confined to the two "
                    "discharge rows -- the change row shows the change field "
                    "and the split point only, so no magenta line ever sits "
                    "on a red loss patch"),
    "e10": dict(mode="ownrow", cmap="RdBu", base="white", ann_rows="epochs",
                fade=0.40,
                desc="as e9 on a near-white base, and only change beyond 40% "
                     "of the limit is painted at all -- the quietest, "
                     "highest-contrast option"),
    "e11": dict(mode="contour", cmap="RdBu", base="normal",
                desc="style change: two zoom rows kept (largest panels), the "
                     "change drawn as +/- limit CONTOUR OUTLINES over the "
                     "2024 discharge instead of a fill, so nothing is "
                     "obscured"),
}

ROW_LABEL = {"2020": "2020 lidar (pre-event)",
             "2024": "2024 lidar (+7 wk)",
             "change": "change (2024 $-$ 2020)"}


def render(fig, rect_fn, x, y, w, h, p, letter, zones, vlim, state, opt):
    ax = fig.add_axes(rect_fn(x, y, w, h))
    ext = p["ext"]
    mm_per_m = w / (ext[1] - ext[0])
    kind = p["kind"]

    if kind in ("2020", "sfm", "2024"):
        if kind == "2024" and opt["mode"] == "overlay" and \
                opt["base"] == "quench":
            desaturated_discharge(ax, ext)
        else:
            im = m.draw_discharge(ax, kind, ext)
            state.setdefault("q_mappable", im)
        if kind == "2024" and opt["mode"] == "overlay":
            sm = change_layer(ax, ext, vlim, opt["cmap"], fade=0.20,
                              alpha=0.95)
            state.setdefault("d_mappable", sm)
        if kind == "2024" and opt["mode"] == "contour":
            change_contours(ax, ext, vlim)
            state.setdefault("d_mappable", plt.cm.ScalarMappable(
                cmap=plt.get_cmap(opt["cmap"]),
                norm=Normalize(-vlim, vlim)))
            state["contour"] = True
    elif kind == "change":
        if opt["base"] == "white":
            pale_relief(ax, ext, lo=0.88, hi=1.0)
        else:
            pale_relief(ax, ext)
        if opt["cmap"] is None:
            change_classes(ax, ext, vlim)
            state["classes"] = True
        else:
            sm = change_layer(ax, ext, vlim, opt["cmap"],
                              fade=opt.get("fade", 0.12))
            state.setdefault("d_mappable", sm)
    elif kind == "change_basin":
        hs, he = m.read_window(m.TIF["hs"], ext)
        ax.imshow(hs, cmap="gray", extent=he, origin="upper", zorder=1,
                  interpolation="bilinear")
        m.draw_watershed_tint(ax, ext)
        m.draw_change(ax, ext, vlim, overlay=True)
        state.setdefault("d_basin", True)
        m.draw_sfm_network(ax, ext)

    if p["zone_boxes"]:
        m.draw_zone_boxes(ax, zones)
    if p["ann"] is not None:
        if opt.get("ann_rows") == "epochs" and kind == "change":
            m.draw_annotation(ax, dict(abandon=np.empty((0, 2)),
                                       new=np.empty((0, 2)),
                                       split=p["ann"]["split"]), mm_per_m)
        else:
            m.draw_annotation(ax, p["ann"], mm_per_m)
    m.bare(ax, ext)
    if p["scale_m"]:
        m.draw_scale_bar(ax, ext, p["scale_m"], mm_per_m,
                         corner=p.get("scale_corner", "br"))
    fig.text((x + w / 2) / m.FIG_W_MM,
             1 - (y + h + m.LETTER_H * 0.72) / state["fig_h"],
             f"({letter})", ha="center", va="baseline",
             fontsize=FS["letter"], fontweight="bold")


def legend(fig, rect_fn, x, y, w, state, vlim, opt):
    cb_w = 42.0
    cax1 = fig.add_axes(rect_fn(x + 1.0, y + 4.0, cb_w, 3.0))
    cb1 = fig.colorbar(state["q_mappable"], cax=cax1,
                       orientation="horizontal", extend="max")
    cb1.set_label("unit discharge\n(m$^3$ s$^{-1}$ m$^{-1}$-equivalent)",
                  fontsize=FS["cbar"], labelpad=2)
    cb1.ax.tick_params(labelsize=FS["tick"], length=2.2, width=0.5)
    cb1.outline.set_linewidth(0.5)

    x2 = x + 1.0 + cb_w + 12.0
    if opt["cmap"] is None:
        kax = fig.add_axes(rect_fn(x2, y + 1.0, cb_w, 13.0))
        kax.axis("off")
        kax.legend(handles=[
            Line2D([], [], ls="none", marker="s", ms=5.5, mfc="#B2182B",
                   mec="none", label=f"discharge lost ($<-${vlim:.2f})"),
            Line2D([], [], ls="none", marker="s", ms=5.5, mfc="#2166AC",
                   mec="none", label=f"discharge gained ($>${vlim:.2f})")],
            loc="center left", frameon=False, fontsize=FS["key"],
            handletextpad=0.5, labelspacing=0.5, borderpad=0.0)
    else:
        cax2 = fig.add_axes(rect_fn(x2, y + 4.0, cb_w, 3.0))
        cb2 = fig.colorbar(state["d_mappable"], cax=cax2,
                           orientation="horizontal", extend="both")
        cb2.set_label("discharge change, 2024 $-$ 2020\n"
                      "lost $\\leftarrow$   $\\rightarrow$ gained",
                      fontsize=FS["cbar"], labelpad=2)
        cb2.set_ticks([-vlim, 0, vlim])
        cb2.set_ticklabels([f"$-${vlim:.2f}", "0", f"{vlim:.2f}"])
        cb2.ax.tick_params(labelsize=FS["tick"], length=2.2, width=0.5)
        cb2.outline.set_linewidth(0.5)

    kx = x2 + cb_w + 12.0
    kax = fig.add_axes(rect_fn(kx, y + 0.5, 42.0, 14.0))
    kax.axis("off")
    kax.legend(handles=[
        Line2D([], [], ls="none", marker="s", ms=4.5, mfc=m.C_ABANDON,
               mec="none", label="2020 course abandoned"),
        Line2D([], [], ls="none", marker="s", ms=4.5, mfc=m.C_NEW,
               mec="none", label="2024 course new"),
        Line2D([], [], ls="none", marker="o", ms=5.0, mfc="none",
               mec=m.C_SPLIT, mew=1.1, label="split point")],
        loc="center left", frameon=False, fontsize=FS["key"],
        handletextpad=0.5, labelspacing=0.5, borderpad=0.0)

    nax = fig.add_axes(rect_fn(x + w - 11.0, y + 1.0, 10.0, 13.0))
    nax.axis("off")
    nax.set_xlim(0, 1)
    nax.set_ylim(0, 1)
    nax.annotate("", xy=(0.5, 0.70), xytext=(0.5, 0.06),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5,
                                 mutation_scale=9))
    nax.text(0.5, 0.80, "N", ha="center", va="bottom",
             fontsize=FS["north"], fontweight="bold")


def build(key, zones, exts, ann, basin_ext, vlim):
    opt = OPTIONS[key]
    ba = (basin_ext[3] - basin_ext[2]) / (basin_ext[1] - basin_ext[0])
    basin = []
    cols = ["2020", "sfm", "2024", "change_basin"]
    for i, c in enumerate(cols):
        basin.append(m.panel_spec(
            c, basin_ext, ba,
            scale_m=200 if i == len(cols) - 1 else None, scale_corner="tr",
            zone_boxes=(c != "change_basin")))

    epochs = ["2020", "2024"] + (["change"]
                                 if opt["mode"] == "ownrow" else [])
    zoom_rows = []
    for e in epochs:
        row = []
        for _, z in zones.iterrows():
            ext = exts[z.zone_id]
            row.append(m.panel_spec(
                e, ext, (ext[3] - ext[2]) / (ext[1] - ext[0]),
                zone=z.zone_id, ann=ann[z.zone_id], scale_m=25))
        zoom_rows.append(row)

    labels = [ROW_LABEL[e] for e in epochs]
    if opt["mode"] == "overlay":
        labels[1] = "2024 lidar + change"
    elif opt["mode"] == "contour":
        labels[1] = "2024 lidar + change contours"
    blocks = [
        dict(rows=[basin], mode="uniform", frac=1.0,
             headers=["2020 lidar (pre-event)", "2024 CAP SfM (+8 d)",
                      "2024 lidar (+7 wk)", "change (2024 $-$ 2020)"]),
        dict(rows=zoom_rows, mode="matched", frac=1.0,
             headers=list(zones.zone_id), row_labels=labels),
    ]

    shrink = m.fit_shrink(blocks, "bottom")
    geom, fig_h = m.compute(blocks, "bottom", shrink)
    fig = plt.figure(figsize=(m.FIG_W_MM * m.MM, fig_h * m.MM))
    state = dict(fig_h=fig_h)

    def rect_fn(x, ytop, w, h):
        return [x / m.FIG_W_MM, 1 - (ytop + h) / fig_h,
                w / m.FIG_W_MM, h / fig_h]

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    widths = {}
    for g in geom:
        x = m.MARGIN_L + g["inset"]
        for i, (p, w) in enumerate(zip(g["panels"], g["widths"])):
            if g["headers"] is not None:
                fig.text((x + w / 2) / m.FIG_W_MM,
                         1 - (g["y"] - m.HEADER_H * 0.30) / fig_h,
                         g["headers"][i], ha="center", va="baseline",
                         fontsize=FS["header"])
            render(fig, rect_fn, x, g["y"], w, g["h"], p, next(letters),
                   zones, vlim, state, opt)
            if g["block"] == 1:
                widths[p["zone"]] = w
            x += w + m.GAP_X
        if g["row_label"]:
            fig.text((m.MARGIN_L + 1.2) / m.FIG_W_MM,
                     1 - (g["y"] + g["h"] / 2) / fig_h, g["row_label"],
                     ha="center", va="center", rotation=90,
                     fontsize=FS["zone"], fontweight="bold")

    legend(fig, rect_fn, m.MARGIN_L, fig_h - m.MARGIN_B - m.LEGEND_H,
           m.FIG_W_MM - m.MARGIN_L - m.MARGIN_R, state, vlim, opt)

    name = f"fig_flow_e_{key}_{opt['mode']}"
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    return dict(key=key, name=name, h=fig_h, desc=opt["desc"],
                widths=widths, shrink=shrink)


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

    exts, ann = {}, {}
    for _, z in zones.iterrows():
        exts[z.zone_id] = (z.e_min - m.PAD_M, z.e_max + m.PAD_M,
                           z.n_min - m.PAD_M, z.n_max + m.PAD_M)
        ann[z.zone_id] = m.zone_annotation(
            exts[z.zone_id], (z.e_min, z.e_max, z.n_min, z.n_max))

    rows = []
    for k in OPTIONS:
        r_ = build(k, zones, exts, ann, basin_ext, vlim)
        rows.append(r_)
        w = r_["widths"]
        print(f"  {k}  174 x {r_['h']:.0f} mm  zoom widths "
              f"Z1 {w['Z1']:.1f} / Z2 {w['Z2']:.1f} / Z3 {w['Z3']:.1f} mm")
        log(f"fig_flow_e_{k}: 174x{r_['h']:.0f} mm, {r_['desc']}",
            run="flow_e_options")

    md = ["# Variant (e) -- change-layer options", "",
          "All 174 mm wide, 300 dpi PNG + vector PDF, Type 42, sans-serif.",
          "Layout is variant (e) throughout: basin row of four panels above,",
          "zoom block transposed so the three zones are COLUMNS. Only the",
          f"change layer differs. Change limit +/-{vlim:.4f} (3 SD).", "",
          "| option | file | placement | zoom panel widths (Z1/Z2/Z3) | "
          "what distinguishes it |", "|---|---|---|---|---|"]
    for r_ in rows:
        w = r_["widths"]
        md.append(f"| {r_['key']} | `{r_['name']}.png` | "
                  f"{OPTIONS[r_['key']]['mode']} | "
                  f"{w['Z1']:.1f} / {w['Z2']:.1f} / {w['Z3']:.1f} mm | "
                  f"{r_['desc']} |")
    (OUT / "E_OPTIONS_AUTO.md").write_text("\n".join(md) + "\n")
    # NB: the curated VARIANTS.md / E_OPTIONS.md alongside this file is
    # hand-written and must NOT be overwritten by a re-run.
    log("DONE", run="flow_e_options")


if __name__ == "__main__":
    main()
