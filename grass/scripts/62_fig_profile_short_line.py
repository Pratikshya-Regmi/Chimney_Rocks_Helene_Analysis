#!/usr/bin/env python3
"""
62_fig_profile_short_line.py
============================
Figures for the new short profile line: two location maps, four before/after
profile layout options, and one combined figure that replaces the current
Figures 2, 3 and 5.

THE DATUM PROBLEM, AND HOW EACH LAYOUT HANDLES IT
    The raw 2024 lidar surface (dtm_2024_filled) is delivered on a different
    vertical datum and sits 31.16 m below the 2020 surface along this line --
    the NAVD88 / NAD83-ellipsoid geoid separation in western North Carolina.
    Drawn raw on a shared axis it forces a 35 m range, in which the other
    three surfaces collapse into a single line. Nothing here rescales it
    silently:
      (a), (d)  before panel uses a BROKEN y-axis, so the raw value is shown
                in its own segment and the break is visible
      (b)       shared y-axis as asked, covering the three surfaces already
                on the right datum; the raw 2024 lidar is marked with an
                off-axis arrow giving its true offset
      (c)       inset covers the three on-datum surfaces, with the 2024 lidar
                offset stated in the inset title

Colours are fixed per surface across every panel (Okabe-Ito, colourblind
safe). No internal raster name appears in any label.

Outputs (results/figures/profile_short_line/):
  fig_pl_map_zoom.png/.pdf        location map, tight on the line
  fig_pl_map_wide.png/.pdf        location map, wider corridor
  fig_pl_profiles_a.png/.pdf      stacked before / after, shared x
  fig_pl_profiles_b.png/.pdf      side by side, shared y
  fig_pl_profiles_c.png/.pdf      after only, before as inset
  fig_pl_profiles_d.png/.pdf      before / after / residual
  fig_pl_combined.png/.pdf        location map + before/after together
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patheffects

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.image as mpimg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402
from map_style import add_north_scale  # noqa: E402

RUN = "profile_short_line"
ROOT = HERE.parents[1]
TABLES = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
OUT = FIG / "profile_short_line"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM = 174.0
DPI = 600

SURF = ["2017 lidar", "2020 lidar", "2024 lidar", "2024 SfM"]
COLORS = {"2017 lidar": "#0072B2", "2020 lidar": "#009E73",
          "2024 lidar": "#D55E00", "2024 SfM": "#CC79A7"}
STYLE = {"2017 lidar": "-", "2020 lidar": "-", "2024 lidar": "-",
         "2024 SfM": "-"}
REFERENCE = "2020 lidar"
LW = 1.1

FS = dict(label=8.0, tick=7.0, legend=7.0, panel=8.5, note=7.0, map=7.0)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
})


def load():
    b = pd.read_csv(TABLES / "profile_short_line_before.csv")
    a = pd.read_csv(TABLES / "profile_short_line_after.csv")
    return b, a


def pgw_extent(png):
    w = png.with_suffix(".pgw").read_text().split()
    px, _, _, py, ox, oy = (float(v) for v in w[:6])
    img = mpimg.imread(png)
    h, wd = img.shape[:2]
    e0 = ox - px / 2
    n1 = oy - py / 2
    return img, (e0, e0 + wd * px, n1 + h * py, n1)


def style_axis(ax, xlab=None, ylab=None):
    ax.grid(True, lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    if xlab:
        ax.set_xlabel(xlab, fontsize=FS["label"])
    if ylab:
        ax.set_ylabel(ylab, fontsize=FS["label"])
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def plot_surfaces(ax, df, cols=None, ls=None, alpha=1.0, lw=LW):
    for s in (cols or SURF):
        ax.plot(df["distance_m"], df[s], color=COLORS[s],
                ls=ls or STYLE[s], lw=lw, alpha=alpha, label=s,
                solid_capstyle="round")


def break_marks(ax_hi, ax_lo):
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], ms=5, ls="none", color="k",
              mec="k", mew=0.8, clip_on=False)
    ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **kw)
    ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **kw)
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(bottom=False, labelbottom=False)


def legend_below(fig, ax, ncol=4, y=None):
    handles = [Line2D([], [], color=COLORS[s], lw=1.6, label=s) for s in SURF]
    return fig.legend(handles=handles, loc="lower center", ncol=ncol,
                      frameon=False, fontsize=FS["legend"],
                      bbox_to_anchor=(0.5, y if y is not None else 0.0),
                      handlelength=2.0, columnspacing=1.8)


# --------------------------------------------------------------- location
def line_xy():
    gj = json.loads((FIG / "_pl_line.geojson").read_text())
    c = gj["features"][0]["geometry"]["coordinates"]
    while isinstance(c[0][0], list):
        c = c[0]
    return np.asarray(c, dtype=float)


def lure_rings():
    p = FIG / "_ndvi_bnd_lure.geojson"
    if not p.exists():
        return []
    gj = json.loads(p.read_text())
    out = []
    for ft in gj.get("features", [gj]):
        g = ft["geometry"]
        polys = ([g["coordinates"]] if g["type"] == "Polygon"
                 else g["coordinates"])
        for poly in polys:
            out.append(np.asarray(poly[0], dtype=float))
    return out


def draw_line_on(ax, xy, label_fs=None, show_labels=True):
    ax.plot(xy[:, 0], xy[:, 1], color="#FFE000", lw=2.6, solid_capstyle="round",
            zorder=6,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=4.4,
                                                            foreground="black")])
    for pt, txt, dx in ((xy[0], "A", -1), (xy[-1], "B", 1)):
        ax.plot([pt[0]], [pt[1]], marker="o", ms=4.0, mfc="white",
                mec="black", mew=1.0, zorder=8)
        if show_labels:
            ax.annotate(txt, xy=(pt[0], pt[1]),
                        xytext=(12 * dx, 9), textcoords="offset points",
                        ha="center", va="center", zorder=9,
                        fontsize=label_fs or FS["panel"], fontweight="bold",
                        color="black",
                        bbox=dict(boxstyle="circle,pad=0.22",
                                  facecolor="white", edgecolor="black",
                                  linewidth=0.8))


def location_map(view, scale_m):
    img, ext = pgw_extent(FIG / f"_pl_ortho_{view}.png")
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    left = right = 3.0
    top, bot = 3.0, 3.0
    w = FIG_W_MM - left - right
    h = w * aspect
    fig_h = top + h + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
    ax = fig.add_axes([left / FIG_W_MM, bot / fig_h, w / FIG_W_MM, h / fig_h])
    ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear",
              zorder=1)

    if view == "wide":
        for ring in lure_rings():
            ax.plot(ring[:, 0], ring[:, 1], color="#FF3B3B", lw=1.4,
                    zorder=5)
        rings = lure_rings()
        if rings:
            bx = min(r[:, 0].min() for r in rings)
            ax.annotate("Lake Lure\nanalysis area",
                        xy=(bx - 0.012 * (ext[1] - ext[0]),
                            ext[3] - 0.035 * (ext[3] - ext[2])),
                        ha="right", va="top", fontsize=FS["map"],
                        color="white", zorder=9, linespacing=1.25,
                        path_effects=[matplotlib.patheffects.withStroke(
                            linewidth=2.2, foreground="black")])

    draw_line_on(ax, line_xy())
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    add_north_scale(ax, ext, scale_m=scale_m, fontsize=FS["map"],
                    color="white")
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("black")

    name = f"fig_pl_map_{view}"
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    return name, fig_h, ext



# ---------------------------------------------------------------- layouts
def ranges(b, a):
    """y-limits: the on-datum cluster, and the raw 2024 lidar segment."""
    on = [s for s in SURF if s != "2024 lidar"]
    lo_hi = np.concatenate([b[on].values.ravel(), a[SURF].values.ravel()])
    hi = (lo_hi.min() - 0.35, lo_hi.max() + 0.35)
    raw = b["2024 lidar"].values
    lo = (raw.min() - 0.35, raw.max() + 0.35)
    return hi, lo


def layout_a(b, a):
    """Stacked, before on top (broken axis) and after below, shared x."""
    hi, lo = ranges(b, a)
    left, right, top, bot = 16.0, 3.0, 4.5, 16.0
    w = FIG_W_MM - left - right
    h_hi, h_lo, h_after, gap, brk = 26.0, 12.0, 40.0, 9.0, 1.6
    fig_h = top + h_hi + brk + h_lo + gap + h_after + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def R(y, hh):
        return [left / FIG_W_MM, 1 - (y + hh) / fig_h, w / FIG_W_MM, hh / fig_h]

    ax_hi = fig.add_axes(R(top, h_hi))
    ax_lo = fig.add_axes(R(top + h_hi + brk, h_lo), sharex=ax_hi)
    ax_af = fig.add_axes(R(top + h_hi + brk + h_lo + gap, h_after),
                         sharex=ax_hi)

    plot_surfaces(ax_hi, b)
    plot_surfaces(ax_lo, b)
    ax_hi.set_ylim(*hi)
    ax_lo.set_ylim(*lo)
    break_marks(ax_hi, ax_lo)
    for ax in (ax_hi, ax_lo):
        style_axis(ax)
    ax_lo.tick_params(labelbottom=False)
    fig.text(left * 0.30 / FIG_W_MM,
             1 - (top + (h_hi + brk + h_lo) / 2) / fig_h,
             "elevation (m)", rotation=90, ha="center", va="center",
             fontsize=FS["label"])
    ax_hi.set_title("(a)  before correction — raw surfaces as first built",
                    fontsize=FS["panel"], loc="left", pad=3)

    plot_surfaces(ax_af, a)
    ax_af.set_ylim(*hi)
    style_axis(ax_af, "distance along profile A–B (m)", "elevation (m)")
    ax_af.set_title("(b)  after datum harmonisation and co-registration",
                    fontsize=FS["panel"], loc="left", pad=3)
    ax_af.set_xlim(0, b["distance_m"].max())
    legend_below(fig, ax_af, y=2.0 / fig_h)
    return fig, "fig_pl_profiles_a", fig_h


def layout_b(b, a):
    """Side by side, before | after, shared y over the on-datum range."""
    hi, _ = ranges(b, a)
    left, right, top, bot = 15.0, 3.0, 5.0, 16.0
    gapx = 5.0
    w = (FIG_W_MM - left - right - gapx) / 2
    h = 56.0
    fig_h = top + h + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
    axs = []
    for i, (df, title) in enumerate(((b, "(a)  before correction"),
                                     (a, "(b)  after correction"))):
        ax = fig.add_axes([(left + i * (w + gapx)) / FIG_W_MM, bot / fig_h,
                           w / FIG_W_MM, h / fig_h])
        plot_surfaces(ax, df)
        ax.set_ylim(*hi)
        ax.set_xlim(0, df["distance_m"].max())
        style_axis(ax, "distance along profile A–B (m)",
                   "elevation (m)" if i == 0 else None)
        if i:
            ax.tick_params(labelleft=False)
        ax.set_title(title, fontsize=FS["panel"], loc="left", pad=3)
        axs.append(ax)

    # the raw 2024 lidar is 31 m below this shared range: say so on the panel
    off = float((b["2024 lidar"] - b[REFERENCE]).mean())
    axs[0].annotate(f"2024 lidar raw surface lies\n{abs(off):.1f} m below this range\n"
                    f"(vertical datum)",
                    xy=(0.975, 0.965), xycoords="axes fraction",
                    ha="right", va="top", fontsize=FS["note"],
                    color=COLORS["2024 lidar"], linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=COLORS["2024 lidar"], lw=0.7,
                              alpha=0.95))
    axs[0].annotate("", xy=(0.50, 0.015), xytext=(0.50, 0.085),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                    color=COLORS["2024 lidar"]))
    legend_below(fig, axs[0], y=2.0 / fig_h)
    return fig, "fig_pl_profiles_b", fig_h


def layout_c(b, a):
    """After only, with the before state as a small inset."""
    hi, _ = ranges(b, a)
    left, right, top, bot = 15.0, 3.0, 5.0, 16.0
    w = FIG_W_MM - left - right
    h = 62.0
    fig_h = top + h + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
    ax = fig.add_axes([left / FIG_W_MM, bot / fig_h, w / FIG_W_MM, h / fig_h])
    plot_surfaces(ax, a)
    ax.set_ylim(*hi)
    ax.set_xlim(0, a["distance_m"].max())
    style_axis(ax, "distance along profile A–B (m)", "elevation (m)")

    on = [s for s in SURF if s != "2024 lidar"]
    iw, ih = 58.0, 26.0
    ix = left + w - iw - 4.0
    iy = bot + h - ih - 5.0
    axi = fig.add_axes([ix / FIG_W_MM, iy / fig_h, iw / FIG_W_MM, ih / fig_h])
    plot_surfaces(axi, b, cols=on, lw=0.9)
    axi.set_xlim(0, b["distance_m"].max())
    axi.tick_params(labelsize=6.0, length=2.0)
    axi.set_ylabel("elev. (m)", fontsize=6.5, labelpad=1.5)
    axi.grid(True, lw=0.25, color="0.88")
    axi.set_axisbelow(True)
    for s in axi.spines.values():
        s.set_linewidth(0.5)
    off = float((b["2024 lidar"] - b[REFERENCE]).mean())
    axi.set_title(f"before correction (2024 lidar {off:+.1f} m, off scale)",
                  fontsize=6.5, pad=2)
    axi.patch.set_alpha(0.95)
    legend_below(fig, ax, y=2.0 / fig_h)
    return fig, "fig_pl_profiles_c", fig_h


def layout_d(b, a):
    """Before, after, and the residual against the reference surface."""
    hi, lo = ranges(b, a)
    left, right, top, bot = 16.0, 3.0, 4.5, 16.0
    w = FIG_W_MM - left - right
    h_hi, h_lo, h_af, h_rhi, h_rlo = 20.0, 10.0, 32.0, 20.0, 9.0
    gap, brk = 8.5, 1.6
    fig_h = (top + h_hi + brk + h_lo + gap + h_af + gap + h_rhi + brk +
             h_rlo + bot)
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def R(y, hh):
        return [left / FIG_W_MM, 1 - (y + hh) / fig_h, w / FIG_W_MM, hh / fig_h]

    y = top
    ax_hi = fig.add_axes(R(y, h_hi))
    ax_lo = fig.add_axes(R(y + h_hi + brk, h_lo), sharex=ax_hi)
    plot_surfaces(ax_hi, b)
    plot_surfaces(ax_lo, b)
    ax_hi.set_ylim(*hi)
    ax_lo.set_ylim(*lo)
    break_marks(ax_hi, ax_lo)
    for ax in (ax_hi, ax_lo):
        style_axis(ax)
    ax_lo.tick_params(labelbottom=False)
    ax_hi.set_title("(a)  before correction", fontsize=FS["panel"],
                    loc="left", pad=3)

    y += h_hi + brk + h_lo + gap
    ax_af = fig.add_axes(R(y, h_af), sharex=ax_hi)
    plot_surfaces(ax_af, a)
    ax_af.set_ylim(*hi)
    style_axis(ax_af, None, "elevation (m)")
    ax_af.tick_params(labelbottom=False)
    ax_af.set_title("(b)  after correction", fontsize=FS["panel"],
                    loc="left", pad=3)

    y += h_af + gap
    res_b = {s: b[s] - b[REFERENCE] for s in SURF}
    res_a = {s: a[s] - a[REFERENCE] for s in SURF}
    rlo_v = float(min(res_b["2024 lidar"].min(), -1)) - 0.4
    rlo_hi_v = float(res_b["2024 lidar"].max()) + 0.4
    tight = np.concatenate([res_a[s].values for s in SURF] +
                           [res_b[s].values for s in SURF if s != "2024 lidar"])
    rhi = (tight.min() - 0.3, tight.max() + 0.3)

    ax_rhi = fig.add_axes(R(y, h_rhi), sharex=ax_hi)
    ax_rlo = fig.add_axes(R(y + h_rhi + brk, h_rlo), sharex=ax_hi)
    for axr in (ax_rhi, ax_rlo):
        for s in SURF:
            axr.plot(b["distance_m"], res_b[s], color=COLORS[s], lw=0.9,
                     ls="--", alpha=0.9)
            axr.plot(a["distance_m"], res_a[s], color=COLORS[s], lw=LW)
        axr.axhline(0, color="0.35", lw=0.5, zorder=1)
        style_axis(axr)
    ax_rhi.set_ylim(*rhi)
    ax_rlo.set_ylim(rlo_v, rlo_hi_v)
    break_marks(ax_rhi, ax_rlo)
    ax_rlo.set_xlabel("distance along profile A–B (m)", fontsize=FS["label"])
    ax_rhi.set_title("(c)  residual against the 2020 lidar reference "
                     "(dashed before, solid after)",
                     fontsize=FS["panel"], loc="left", pad=3)
    fig.text(left * 0.28 / FIG_W_MM,
             1 - (y + (h_rhi + brk + h_rlo) / 2) / fig_h,
             "residual (m)", rotation=90, ha="center", va="center",
             fontsize=FS["label"])
    ax_hi.set_xlim(0, b["distance_m"].max())
    legend_below(fig, ax_rlo, y=2.0 / fig_h)
    return fig, "fig_pl_profiles_d", fig_h


def layout_combined(b, a):
    """Location map on top, before/after profiles below -- the single figure
    that replaces the present Figures 2, 3 and 5."""
    hi, lo = ranges(b, a)
    img, ext = pgw_extent(FIG / "_pl_ortho_zoom.png")
    left, right, top, bot = 16.0, 3.0, 4.5, 16.0
    w = FIG_W_MM - left - right
    map_h = w * (ext[3] - ext[2]) / (ext[1] - ext[0])
    h_hi, h_lo, h_af, gap, brk = 22.0, 10.0, 34.0, 9.0, 1.6
    fig_h = top + map_h + 4.0 + gap + h_hi + brk + h_lo + gap + h_af + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def R(y, hh):
        return [left / FIG_W_MM, 1 - (y + hh) / fig_h, w / FIG_W_MM, hh / fig_h]

    axm = fig.add_axes(R(top, map_h))
    axm.imshow(img, extent=ext, origin="upper", interpolation="bilinear",
               zorder=1)
    draw_line_on(axm, line_xy())
    axm.set_xlim(ext[0], ext[1])
    axm.set_ylim(ext[2], ext[3])
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    add_north_scale(axm, ext, scale_m=50, fontsize=FS["map"], color="white")
    for s in axm.spines.values():
        s.set_linewidth(0.6)
    axm.set_title("(a)  profile line A–B on the 2024 CAP orthophoto",
                  fontsize=FS["panel"], loc="left", pad=3)

    y = top + map_h + 4.0 + gap
    ax_hi = fig.add_axes(R(y, h_hi))
    ax_lo = fig.add_axes(R(y + h_hi + brk, h_lo), sharex=ax_hi)
    plot_surfaces(ax_hi, b)
    plot_surfaces(ax_lo, b)
    ax_hi.set_ylim(*hi)
    ax_lo.set_ylim(*lo)
    break_marks(ax_hi, ax_lo)
    for ax in (ax_hi, ax_lo):
        style_axis(ax)
    ax_lo.tick_params(labelbottom=False)
    ax_hi.set_title("(b)  before correction", fontsize=FS["panel"],
                    loc="left", pad=3)

    y += h_hi + brk + h_lo + gap
    ax_af = fig.add_axes(R(y, h_af), sharex=ax_hi)
    plot_surfaces(ax_af, a)
    ax_af.set_ylim(*hi)
    style_axis(ax_af, "distance along profile A–B (m)", "elevation (m)")
    ax_af.set_title("(c)  after correction", fontsize=FS["panel"],
                    loc="left", pad=3)
    ax_hi.set_xlim(0, b["distance_m"].max())
    fig.text(left * 0.28 / FIG_W_MM,
             1 - (top + map_h + 4.0 + gap + (h_hi + brk + h_lo) / 2) / fig_h,
             "elevation (m)", rotation=90, ha="center", va="center",
             fontsize=FS["label"])
    legend_below(fig, ax_af, y=2.0 / fig_h)
    return fig, "fig_pl_combined", fig_h


def main():
    b, a = load()
    d = b["distance_m"].values

    print("location maps:")
    for view, sm in (("zoom", 50), ("wide", 250)):
        name, h, ext = location_map(view, sm)
        print(f"  {name}.png/.pdf   174 x {h:.0f} mm   extent "
              f"E {ext[0]:.0f}-{ext[1]:.0f}  N {ext[2]:.0f}-{ext[3]:.0f}  "
              f"scale bar {sm} m")

    print("\nprofile layouts:")
    for fn in (layout_a, layout_b, layout_c, layout_d, layout_combined):
        fig, name, h = fn(b, a)
        fig.savefig(OUT / f"{name}.png", dpi=DPI)
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)
        print(f"  {name}.png/.pdf   174 x {h:.0f} mm")
        log(f"{name}: 174x{h:.0f} mm", run=RUN)

    print("\n" + "=" * 70)
    print("CAPTION NUMBERS -- do not retype these")
    print("=" * 70)
    print(f"profile length                 {d.max():.1f} m "
          f"({len(d)} samples at 1 m)")
    allv = np.concatenate([a[s].values for s in SURF])
    print(f"elevation range, after (all 4) {allv.min():.2f} to "
          f"{allv.max():.2f} m  (span {allv.max()-allv.min():.2f} m)")
    print(f"\n{'surface':12s} {'offset vs 2020 before':>22s} "
          f"{'offset vs 2020 after':>21s} {'correction applied':>19s}")
    for s in SURF:
        ob = float((b[s] - b[REFERENCE]).mean())
        oa = float((a[s] - a[REFERENCE]).mean())
        print(f"{s:12s} {ob:22.3f} {oa:21.3f} {oa-ob:19.3f}")
    print(f"\n{'surface':12s} {'mean elev before':>17s} "
          f"{'mean elev after':>16s} {'shift':>9s}")
    for s in SURF:
        print(f"{s:12s} {b[s].mean():17.3f} {a[s].mean():16.3f} "
              f"{a[s].mean()-b[s].mean():9.3f}")
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
