#!/usr/bin/env python3
"""
make_detection_area_lidar_only.py -- single-panel variant of
fig_detection_area.pdf/png (make_figures.py), lidar-lidar only.

The lidar-SfM analysis is out of scope for this manuscript (companion
study, white2026sfm); the two-panel original still exists for reference/
comparison, but the manuscript figure should not carry a second panel
(48.5% / 90.4% / 3.3%) that no text in this paper discusses. Same data,
same styling constants, imported from make_figures.py so nothing is
recomputed by hand.

Usage:
    /home/pregmi3/mambaforge/envs/helene_dec1/bin/python3 \\
        agent/scripts/make_detection_area_lidar_only.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import C_SFM, C_LIDAR, C_GREY, SINGLE_COL, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


def main():
    df = pd.read_csv(TABLES / "detection_summary.csv")
    sub = df[df["pair"] == "lidar-lidar"].sort_values("scenario")
    xlabels = ["Uniform LoD\n(uncorrected)", "Uniform LoD\n(co-registered)",
              "Canopy-aware LoD\n(co-registered)"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.95))
    x = np.arange(len(sub))
    ero = sub["pct_erosion"].values
    dep = sub["pct_deposition"].values

    ax.bar(x, ero, width=0.62, color=C_SFM, label="Erosion",
          edgecolor="white", linewidth=0.4, zorder=3)
    ax.bar(x, dep, width=0.62, bottom=ero, color=C_LIDAR, label="Deposition",
          edgecolor="white", linewidth=0.4, zorder=3)
    for xi, (e, d) in enumerate(zip(ero, dep)):
        ax.text(xi, e + d + 1.8, f"{e + d:.1f}%", ha="center", fontsize=7,
               color=C_GREY)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Basin area above\nlevel of detection (%)")
    ax.grid(axis="y", color="0.9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=1)

    save(fig, "fig_detection_area_lidar_only", FIGURES)

    print("\nCaption values (lidar-lidar, printed not retyped):")
    for lab, e, d in zip(xlabels, ero, dep):
        print(f"  {lab.replace(chr(10), ' ')}: erosion={e:.1f}%, "
              f"deposition={d:.1f}%, total={e + d:.1f}%")


if __name__ == "__main__":
    main()
