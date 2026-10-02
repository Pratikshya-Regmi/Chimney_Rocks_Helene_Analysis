#!/usr/bin/env python3
"""
64_fig_profile_combined_variants.py
===================================
Variants of fig_pl_combined, the location map + before/after profile figure.

WHAT THE MARKS ON THE Y-AXIS WERE
    The two small diagonal double-ticks sitting on the left and right of the
    "before" panel's y-axis were AXIS-BREAK MARKERS. The before panel spans
    two disjoint elevation ranges -- the three surfaces already on the right
    datum near 311-319 m, and the raw 2024 lidar near 280-283 m -- so the
    axis is cut and the two segments stacked. The marks were the conventional
    way of saying "the axis is not continuous here". They read as stray
    quotation marks at print size, so they are gone in every variant below.
    The break itself is still obvious: the tick values jump from 312.5
    straight to 282.5, and there is a visible gap between the two segments.

VARIANTS
    v1  the figure as it was, with the break marks removed and more space
        between the legend and the x-axis label. Nothing else changed.
    v2  the break presented as two clearly LABELLED sub-panels rather than
        one cut axis -- the raw 2024 lidar gets its own small strip with its
        own axis, captioned, so there is no break convention to read at all.
    v3  no break anywhere: the before panel is replaced by the offset of each
        surface from the 2020 lidar reference, on a symmetric-log axis. One
        continuous axis holds -31 m and ±0.1 m at once, and the panel states
        the size of each correction directly rather than leaving the reader
        to subtract two elevations.

Outputs: results/figures/profile_short_line/fig_pl_combined_v{1,2,3}.png/.pdf
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "fp62", HERE / "62_fig_profile_short_line.py")
f = importlib.util.module_from_spec(spec)
sys.modules["fp62"] = f
spec.loader.exec_module(f)

sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402
from map_style import add_north_scale  # noqa: E402

RUN = "profile_short_line"
OUT = f.OUT
MM, FIG_W_MM, DPI = f.MM, f.FIG_W_MM, f.DPI
SURF, COLORS, REFERENCE = f.SURF, f.COLORS, f.REFERENCE
FS = f.FS

LEGEND_GAP = 7.0     # mm between the x-axis label and the legend row


def map_panel(fig, rect_fn, y, w, left, map_h, fig_h, letter):
    img, ext = f.pgw_extent(f.FIG / "_pl_ortho_zoom.png")
    ax = fig.add_axes(rect_fn(y, map_h))
    ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear",
              zorder=1)
    f.draw_line_on(ax, f.line_xy())
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    add_north_scale(ax, ext, scale_m=50, fontsize=FS["map"], color="white")
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.set_title(f"({letter})  profile line A–B on the 2024 CAP orthophoto",
                 fontsize=FS["panel"], loc="left", pad=3)
    return map_h


def hide_break_spines(ax_hi, ax_lo):
    """The two segments of a cut axis, without the diagonal break marks."""
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(bottom=False, labelbottom=False)


def build_v1(b, a, labelled=False):
    """Cut axis, break marks removed; v2 adds captions to the two segments."""
    hi, lo = f.ranges(b, a)
    left, right, top = 16.0, 3.0, 4.5
    bot = 16.0 + LEGEND_GAP
    w = FIG_W_MM - left - right
    img, ext = f.pgw_extent(f.FIG / "_pl_ortho_zoom.png")
    map_h = w * (ext[3] - ext[2]) / (ext[1] - ext[0])
    h_hi, h_lo, h_af, gap, brk = 22.0, 10.0, 34.0, 9.0, (4.0 if labelled
                                                         else 1.6)
    fig_h = top + map_h + 4.0 + gap + h_hi + brk + h_lo + gap + h_af + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def R(y, hh):
        return [left / FIG_W_MM, 1 - (y + hh) / fig_h, w / FIG_W_MM, hh / fig_h]

    map_panel(fig, R, top, w, left, map_h, fig_h, "a")

    y = top + map_h + 4.0 + gap
    ax_hi = fig.add_axes(R(y, h_hi))
    ax_lo = fig.add_axes(R(y + h_hi + brk, h_lo), sharex=ax_hi)
    f.plot_surfaces(ax_hi, b)
    f.plot_surfaces(ax_lo, b)
    ax_hi.set_ylim(*hi)
    ax_lo.set_ylim(*lo)
    hide_break_spines(ax_hi, ax_lo)
    for ax in (ax_hi, ax_lo):
        f.style_axis(ax)
    ax_lo.tick_params(labelbottom=False)
    ax_hi.set_title("(b)  before correction", fontsize=FS["panel"],
                    loc="left", pad=3)
    if labelled:
        ax_hi.annotate("surfaces on the project datum",
                       xy=(0.985, 0.06), xycoords="axes fraction",
                       ha="right", va="bottom", fontsize=6.5, color="0.30")
        ax_lo.annotate("2024 lidar, as delivered — separate vertical datum",
                       xy=(0.985, 0.10), xycoords="axes fraction",
                       ha="right", va="bottom", fontsize=6.5,
                       color=COLORS["2024 lidar"])

    y += h_hi + brk + h_lo + gap
    ax_af = fig.add_axes(R(y, h_af), sharex=ax_hi)
    f.plot_surfaces(ax_af, a)
    ax_af.set_ylim(*hi)
    f.style_axis(ax_af, "distance along profile A–B (m)", "elevation (m)")
    ax_af.set_title("(c)  after correction", fontsize=FS["panel"],
                    loc="left", pad=3)
    ax_hi.set_xlim(0, b["distance_m"].max())
    fig.text(left * 0.28 / FIG_W_MM,
             1 - (top + map_h + 4.0 + gap + (h_hi + brk + h_lo) / 2) / fig_h,
             "elevation (m)", rotation=90, ha="center", va="center",
             fontsize=FS["label"])
    f.legend_below(fig, ax_af, y=2.0 / fig_h)
    return fig, fig_h


def build_v3(b, a):
    """No break: before panel becomes offset-from-reference on a symlog axis."""
    hi, _ = f.ranges(b, a)
    left, right, top = 16.0, 3.0, 4.5
    bot = 16.0 + LEGEND_GAP
    w = FIG_W_MM - left - right
    img, ext = f.pgw_extent(f.FIG / "_pl_ortho_zoom.png")
    map_h = w * (ext[3] - ext[2]) / (ext[1] - ext[0])
    h_off, h_af, gap = 36.0, 34.0, 9.0
    fig_h = top + map_h + 4.0 + gap + h_off + gap + h_af + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def R(y, hh):
        return [left / FIG_W_MM, 1 - (y + hh) / fig_h, w / FIG_W_MM, hh / fig_h]

    map_panel(fig, R, top, w, left, map_h, fig_h, "a")

    y = top + map_h + 4.0 + gap
    ax = fig.add_axes(R(y, h_off))
    for s in SURF:
        ax.plot(b["distance_m"], b[s] - b[REFERENCE], color=COLORS[s],
                lw=f.LW, ls="--", alpha=0.95)
        ax.plot(a["distance_m"], a[s] - a[REFERENCE], color=COLORS[s],
                lw=f.LW)
    ax.axhline(0, color="0.35", lw=0.5, zorder=1)
    ax.set_yscale("symlog", linthresh=0.3, linscale=0.8)
    ticks = [-30, -10, -3, -0.3, 0, 0.3, 3, 10]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}" if abs(v) >= 1 or v == 0 else f"{v:g}"))
    ax.set_ylim(-45, 12)
    f.style_axis(ax, None, "offset from 2020 lidar (m)")
    ax.tick_params(labelbottom=False)
    ax.set_title("(b)  offset from the 2020 lidar reference "
                 "(dashed before, solid after; symmetric-log axis)",
                 fontsize=FS["panel"], loc="left", pad=3)

    y += h_off + gap
    ax_af = fig.add_axes(R(y, h_af), sharex=ax)
    f.plot_surfaces(ax_af, a)
    ax_af.set_ylim(*hi)
    f.style_axis(ax_af, "distance along profile A–B (m)", "elevation (m)")
    ax_af.set_title("(c)  after correction", fontsize=FS["panel"],
                    loc="left", pad=3)
    ax.set_xlim(0, b["distance_m"].max())
    f.legend_below(fig, ax_af, y=2.0 / fig_h)
    return fig, fig_h


def main():
    b, a = f.load()
    for key, fn in (("v1", lambda: build_v1(b, a, labelled=False)),
                    ("v2", lambda: build_v1(b, a, labelled=True)),
                    ("v3", lambda: build_v3(b, a))):
        fig, fig_h = fn()
        name = f"fig_pl_combined_{key}"
        fig.savefig(OUT / f"{name}.png", dpi=DPI)
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)
        print(f"  {name}.png/.pdf   174 x {fig_h:.0f} mm")
        log(f"{name}: 174x{fig_h:.0f} mm", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
