#!/usr/bin/env python3
"""
OBSOLETE as of the 3-transect rebuild of make_transect_figure_v4.py.

This script used to render D-D' (western corridor, then a straight line
between endpoints) as a standalone supplementary figure, because four
profile pairs did not fit legibly in the main figure's two-column layout.
Since then: (1) the old C-C' channel cross-section was dropped from the
main figure entirely, freeing a slot; (2) the western corridor was re-traced
as a genuine sinuous path and renamed into that freed C-C' slot. All three
transects (A-A', B-B', C-C') now fit in the main figure -- no supplementary
figure is needed, per instruction.

Left in place for provenance only. Do not run it: its input,
transect4_westcorridor.csv, no longer exists (superseded by
transect3_westcorridor.csv, which make_transect_figure_v4.py now plots
directly). Its previous outputs (fig_supplementary_dd.pdf/.png) have been
removed from results/figures/ so they don't linger as stale, mislabelled
deliverables.
"""

import sys

print(__doc__)
sys.exit(1)

"""
Original docstring, kept for reference:

make_supplementary_figure.py -- D-D' (western corridor) profile pair, moved
here from the main geomorphic-transects figure (make_transect_figure_v4.py):
four profile pairs did not fit legibly in a 174 x ~205 mm two-column layout
(tested and confirmed -- axis labels collided), so this transect gets its
own supplementary figure at a size that lets it breathe. D-D' still appears
on the main figure's locator map, labelled, so the reader knows where it is.

Single-column width (84 mm) is plenty for one pair; no map panel needed here
since the main figure's locator already shows D-D's location.

Usage (needs the project's conda env):
    /home/pregmi3/mambaforge/envs/helene_dec1/bin/python3 \\
        agent/scripts/make_supplementary_figure.py
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_transect_figure_v2 import draw_profile_pair, LOD, FIGURES, TABLES  # noqa: E402

MM = 1 / 25.4
SINGLE_COL = 84 * MM


def main():
    t4 = pd.read_csv(TABLES / "transect4_westcorridor.csv")

    fig = plt.figure(figsize=(SINGLE_COL, SINGLE_COL * 0.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.10)
    ax_e = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[1, 0], sharex=ax_e)

    stats = draw_profile_pair(ax_e, ax_d, t4, "D–D′  western corridor")
    ax_e.legend(loc="lower left", ncol=2, handlelength=1.4, columnspacing=1.0)

    out_pdf = FIGURES / "fig_supplementary_dd.pdf"
    out_png = FIGURES / "fig_supplementary_dd.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf} and .png")

    print(f"\nCaption values: D-D' (western corridor): length={stats['length_m']:.1f} m, "
          f"relief={stats['relief_m']:.1f} m, max lowering={stats['max_lowering_m']:+.2f} m, "
          f"max raising={stats['max_raising_m']:+.2f} m, "
          f"{stats['pct_above_lod']:.1f}% of {stats['n']} samples exceed the {LOD} m LoD")


if __name__ == "__main__":
    main()
