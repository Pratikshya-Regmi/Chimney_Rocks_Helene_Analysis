#!/usr/bin/env python3
"""
59_fig_flow_zoom_annotated.py
=============================
Rebuild of the STANDALONE zoom figure (Figure 13, fig_flow_sequence_zoom)
with the flow change named directly on the panels. The consolidated figure
is not touched.

Layout is the original Figure 13 arrangement -- one row per zone (Z1-Z3),
one column per epoch (2020 lidar / 2024 CAP SfM / 2024 lidar), common
logarithmic discharge scale -- with three things added:

  * the DOMINANT removed and new flow courses, each reduced to the longest
    path of its largest connected component so one clean line is drawn
    rather than a scatter of skeleton cells
  * the split point where the course divides
  * short leader-arrow labels naming them: "removed flow", "new flow",
    "split", placed automatically where they cover least discharge

Course definitions are unchanged from the consolidated work: a cell is
removed if it carried 2020 channel and no 2024 channel lies within 2 m, new
on the mirror test, both from the r.thin skeletons of the two lidar epochs
at the discharge > 0.01 threshold used for the Jaccard metric. The SfM epoch
takes no part in defining them -- it is a benchmark surface -- but the same
courses are drawn over it so the reader can see how it compares.

Three options:
  z1  courses on all three columns; labels distributed by meaning --
      "removed flow" on the 2020 panel, "new flow" and "split" on the 2024
      lidar panel
  z2  courses on all three columns; all labels on the 2020 column, the other
      two columns left free of text
  z3  courses on the two LIDAR columns only, SfM left clean; labels
      distributed as z1

Outputs:
  results/figures/fig_flow_zoom_annotated/fig_flow_zoom_annot_<key>.png/.pdf
  results/figures/fig_flow_zoom_annotated/ZOOM_OPTIONS.md
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
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import rasterio

HERE = Path(__file__).resolve().parent
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

m = _load("fc56", "56_fig_flow_consolidated.py")
an = _load("fc58", "58_fig_flow_consolidated_annotated.py")

sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402

OUT = m.FIGURES / "fig_flow_zoom_annotated"
OUT.mkdir(parents=True, exist_ok=True)

FIG_W_MM = m.FIG_W_MM
MM = m.MM
FS = dict(letter=8.0, header=7.5, zone=7.5, cbar=7.0, tick=7.0, key=7.0,
          north=8.5)

COLS = [("2020", "2020 lidar (pre-event)"),
        ("sfm", "2024 CAP SfM (5 Oct)"),
        ("2024", "2024 lidar (+7 wk)")]

MARGIN_L = MARGIN_R = 3.0
MARGIN_T = 3.0
MARGIN_B = 3.0
GAP_X = 2.5
GAP_Y = 3.5
HEADER_H = 4.4
LETTER_H = 4.2
ZONE_W = 5.6
LEGEND_H = 15.0

OPTIONS = {
    "z1": dict(course_cols={"2020", "sfm", "2024"},
               labels={"2020": {"removed"}, "sfm": set(),
                       "2024": {"new", "split"}},
               desc="courses on all three columns; labels distributed by "
                    "meaning -- 'removed flow' on the 2020 panel, 'new flow' "
                    "and 'split' on the 2024 lidar panel"),
    "z2": dict(course_cols={"2020", "sfm", "2024"},
               labels={"2020": {"removed", "new", "split"}, "sfm": set(),
                       "2024": set()},
               desc="courses on all three columns; all labels on the 2020 "
                    "column, the SfM and 2024 columns left free of text"),
    "z3": dict(course_cols={"2020", "2024"},
               labels={"2020": {"removed"}, "sfm": set(),
                       "2024": {"new", "split"}},
               desc="courses on the two LIDAR columns only, SfM left clean "
                    "as the benchmark surface; labels distributed as z1"),
}


def build(key, zones, exts, courses):
    opt = OPTIONS[key]

    usable = FIG_W_MM - MARGIN_L - MARGIN_R - ZONE_W
    pw = (usable - GAP_X * (len(COLS) - 1)) / len(COLS)
    heights = []
    for _, z in zones.iterrows():
        e = exts[z.zone_id]
        heights.append(pw * (e[3] - e[2]) / (e[1] - e[0]))

    fig_h = (MARGIN_T + HEADER_H + sum(heights) + LETTER_H * len(heights) +
             GAP_Y * (len(heights) - 1) + LEGEND_H + MARGIN_B)
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def rect_fn(x, ytop, w, h):
        return [x / FIG_W_MM, 1 - (ytop + h) / fig_h, w / FIG_W_MM, h / fig_h]

    x_left = MARGIN_L + ZONE_W
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    y = MARGIN_T + HEADER_H
    q_mappable = None

    for (_, z), h in zip(zones.iterrows(), heights):
        ext = exts[z.zone_id]
        cz = courses[z.zone_id]
        mm_per_m = pw / (ext[1] - ext[0])
        x = x_left
        for ci, (kind, header) in enumerate(COLS):
            ax = fig.add_axes(rect_fn(x, y, pw, h))
            im = m.draw_discharge(ax, kind, ext)
            if q_mappable is None:
                q_mappable = im
            an.annotate_panel(ax, ext, cz, mm_per_m,
                              draw_lines=(kind in opt["course_cols"]),
                              labels=opt["labels"][kind])
            m.bare(ax, ext)
            # every zoom panel carries its own bar: the three zones are
            # different sizes, so the rendered scale differs row to row
            m.draw_scale_bar(ax, ext, 25, mm_per_m)
            if y == MARGIN_T + HEADER_H:
                fig.text((x + pw / 2) / FIG_W_MM,
                         1 - (y - HEADER_H * 0.30) / fig_h, header,
                         ha="center", va="baseline", fontsize=FS["header"])
            fig.text((x + pw / 2) / FIG_W_MM,
                     1 - (y + h + LETTER_H * 0.72) / fig_h,
                     f"({next(letters)})", ha="center", va="baseline",
                     fontsize=FS["letter"], fontweight="bold")
            x += pw + GAP_X
        fig.text((MARGIN_L + ZONE_W / 2) / FIG_W_MM,
                 1 - (y + h / 2) / fig_h, z.zone_id, ha="center", va="center",
                 rotation=90, fontsize=FS["zone"], fontweight="bold")
        y += h + LETTER_H + GAP_Y

    # ---- legend: one discharge scale, the key, one north arrow ----------
    ly = fig_h - MARGIN_B - LEGEND_H
    cax = fig.add_axes(rect_fn(x_left, ly + 4.0, 52.0, 3.0))
    cb = fig.colorbar(q_mappable, cax=cax, orientation="horizontal",
                      extend="max")
    cb.set_label("unit discharge (m$^3$ s$^{-1}$ m$^{-1}$-equivalent)",
                 fontsize=FS["cbar"], labelpad=2)
    cb.ax.tick_params(labelsize=FS["tick"], length=2.2, width=0.5)
    cb.outline.set_linewidth(0.5)

    handles = [
        Line2D([], [], color=m.C_ABANDON, lw=1.6, label="removed flow (2020)"),
        Line2D([], [], color=m.C_NEW, lw=1.6, label="new flow (2024)"),
        Line2D([], [], ls="none", marker="o", ms=5.0, mfc="none",
               mec=m.C_SPLIT, mew=1.1, label="split point"),
    ]
    kax = fig.add_axes(rect_fn(x_left + 66.0, ly + 0.5, 70.0, 13.0))
    kax.axis("off")
    kax.legend(handles=handles, loc="center left", frameon=False,
               fontsize=FS["key"], handletextpad=0.6, columnspacing=1.4,
               labelspacing=0.4, borderpad=0.0, ncol=2)

    nax = fig.add_axes(rect_fn(FIG_W_MM - MARGIN_R - 11.0, ly + 1.0,
                               10.0, 13.0))
    nax.axis("off")
    nax.set_xlim(0, 1)
    nax.set_ylim(0, 1)
    nax.annotate("", xy=(0.5, 0.70), xytext=(0.5, 0.06),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5,
                                 mutation_scale=9))
    nax.text(0.5, 0.80, "N", ha="center", va="bottom",
             fontsize=FS["north"], fontweight="bold")

    name = f"fig_flow_zoom_annot_{key}"
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    return dict(key=key, name=name, h=fig_h, pw=pw, desc=opt["desc"])


def main():
    zones = pd.read_csv(m.TABLES / "flow_change_zones.csv")
    exts, courses = {}, {}
    for _, z in zones.iterrows():
        exts[z.zone_id] = (z.e_min - m.PAD_M, z.e_max + m.PAD_M,
                           z.n_min - m.PAD_M, z.n_max + m.PAD_M)
        courses[z.zone_id] = an.zone_courses(
            exts[z.zone_id], (z.e_min, z.e_max, z.n_min, z.n_max))
        cz = courses[z.zone_id]
        print(f"  {z.zone_id}: removed {cz['n_ab']:3d} cells, new "
              f"{cz['n_new']:3d} cells, split "
              f"{'yes' if cz['split'] is not None else 'none'}")

    rows = []
    for k in OPTIONS:
        r = build(k, zones, exts, courses)
        rows.append(r)
        print(f"  {k}  {FIG_W_MM:.0f} x {r['h']:.0f} mm  "
              f"panel width {r['pw']:.1f} mm")
        log(f"fig_flow_zoom_annot_{k}: {FIG_W_MM:.0f}x{r['h']:.0f} mm, "
            f"panel {r['pw']:.1f} mm, {r['desc']}", run="flow_zoom_annot")

    md = ["# Standalone zoom figure (Figure 13) with the change named",
          "", "Original Figure 13 arrangement -- one row per zone, one column",
          "per epoch, common log discharge scale 1e-4 to 0.48, a 25 m bar on",
          "every panel, one north arrow. 174 mm wide, 300 dpi PNG + vector",
          "PDF, 7.0 pt label floor.", "",
          f"Panel width {rows[0]['pw']:.1f} mm.", "",
          "| option | file | size | what distinguishes it |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['key']} | `{r['name']}.png` | "
                  f"174 x {r['h']:.0f} mm | {r['desc']} |")
    (OUT / "ZOOM_OPTIONS.md").write_text("\n".join(md) + "\n")
    log("DONE", run="flow_zoom_annot")


if __name__ == "__main__":
    main()
