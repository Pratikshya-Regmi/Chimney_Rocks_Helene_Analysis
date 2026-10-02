#!/usr/bin/env python3
"""
66_fig_profile_short_line_datum_fixed.py
========================================
The combined location-map + before/after profile figure, rebuilt on the
datum-harmonised surfaces written by 65_profile_short_line_datum_fixed.py.

WHAT CHANGED AGAINST fig_pl_combined_v1
    The before panel no longer needs a BROKEN Y-AXIS. In v1 the before panel
    was cut in two because the 2024 lidar was drawn from the as-delivered
    NAD83(2011) ellipsoidal surface, ~31 m below everything else. That was
    the geoid separation, not a vertical bias, and Methods 2.3.3 removes it
    in an earlier step. With every surface on NAVD88/GEOID18 the whole figure
    fits one continuous axis.

    Both panels are drawn on the SAME y-limits so the two are directly
    comparable, which is what the broken axis previously made impossible.

    Styling, colours, fonts, width (174 mm), dpi (600) and the location map
    are inherited unchanged from 62_fig_profile_short_line.py.

Outputs (results/figures/profile_short_line/):
  fig_pl_combined_datumfixed.png / .pdf
Existing figure files are left in place; these are new names.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "fp62", HERE / "62_fig_profile_short_line.py")
f = importlib.util.module_from_spec(spec)
sys.modules["fp62"] = f
spec.loader.exec_module(f)

sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402

RUN = "profile_short_line_datumfixed"
OUT, TABLES = f.OUT, f.TABLES
MM, FIG_W_MM, DPI = f.MM, f.FIG_W_MM, f.DPI
SURF, COLORS, REFERENCE, FS = f.SURF, f.COLORS, f.REFERENCE, f.FS

LEGEND_GAP = 7.0
NAME = "fig_pl_combined_datumfixed"


def load():
    b = pd.read_csv(TABLES / "profile_short_line_datumfixed_before.csv")
    a = pd.read_csv(TABLES / "profile_short_line_datumfixed_after.csv")
    return b, a


def shared_ylim(b, a, pad=0.35):
    v = np.concatenate([b[SURF].values.ravel(), a[SURF].values.ravel()])
    return v.min() - pad, v.max() + pad


def build(b, a):
    ylim = shared_ylim(b, a)
    left, right, top = 16.0, 3.0, 4.5
    bot = 16.0 + LEGEND_GAP
    w = FIG_W_MM - left - right
    img, ext = f.pgw_extent(f.FIG / "_pl_ortho_zoom.png")
    map_h = w * (ext[3] - ext[2]) / (ext[1] - ext[0])
    h_bf, h_af, gap = 34.0, 34.0, 9.0
    fig_h = top + map_h + 4.0 + gap + h_bf + gap + h_af + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def R(y, hh):
        return [left / FIG_W_MM, 1 - (y + hh) / fig_h,
                w / FIG_W_MM, hh / fig_h]

    # (a) location map -- unchanged from v1
    ax_map = fig.add_axes(R(top, map_h))
    ax_map.imshow(img, extent=ext, origin="upper", interpolation="bilinear",
                  zorder=1)
    f.draw_line_on(ax_map, f.line_xy())
    ax_map.set_xlim(ext[0], ext[1])
    ax_map.set_ylim(ext[2], ext[3])
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    from map_style import add_north_scale
    add_north_scale(ax_map, ext, scale_m=50, fontsize=FS["map"], color="white")
    for s in ax_map.spines.values():
        s.set_linewidth(0.6)
    ax_map.set_title("(a)  profile line A–B on the 2024 CAP orthophoto",
                     fontsize=FS["panel"], loc="left", pad=3)

    # (b) before -- one continuous axis, no break
    y = top + map_h + 4.0 + gap
    ax_b = fig.add_axes(R(y, h_bf))
    f.plot_surfaces(ax_b, b)
    ax_b.set_ylim(*ylim)
    f.style_axis(ax_b, None, "elevation (m)")
    ax_b.tick_params(labelbottom=False)
    ax_b.set_title("(b)  before correction — all surfaces on NAVD88 (GEOID18)",
                   fontsize=FS["panel"], loc="left", pad=3)

    # (c) after
    y += h_bf + gap
    ax_a = fig.add_axes(R(y, h_af), sharex=ax_b)
    f.plot_surfaces(ax_a, a)
    ax_a.set_ylim(*ylim)
    f.style_axis(ax_a, "distance along profile A–B (m)", "elevation (m)")
    ax_a.set_title("(c)  after co-registration and vertical bias correction",
                   fontsize=FS["panel"], loc="left", pad=3)

    ax_b.set_xlim(0, b["distance_m"].max())
    f.legend_below(fig, ax_a, y=2.0 / fig_h)
    return fig, fig_h


def main():
    b, a = load()
    fig, fig_h = build(b, a)
    fig.savefig(OUT / f"{NAME}.png", dpi=DPI)
    fig.savefig(OUT / f"{NAME}.pdf")
    plt.close(fig)
    print(f"  {NAME}.png/.pdf   174 x {fig_h:.0f} mm")
    for s in SURF:
        print(f"    {s:12s} offset before {float((b[s]-b[REFERENCE]).mean()):+.4f} m"
              f"   after {float((a[s]-a[REFERENCE]).mean()):+.4f} m")
    log(f"{NAME}: 174x{fig_h:.0f} mm", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
