#!/usr/bin/env python3
"""
make_lure_mask_overlay_figure.py
=================================
Diagnostic figure (NOT a manuscript deliverable -- saved under
results/diagnostics/): stable_mask_lure overlaid on the 2024 orthophoto and
hillshade, for visual inspection -- does the mask include anything it
shouldn't (docks, roads under construction, Morse Park works, dredged
areas)?

Reads the GeoTIFFs 05_lure_stable_mask_validation.py exported:
  results/diagnostics/_lure_ortho_2024.tif
  results/diagnostics/_lure_stable_mask.tif
  results/diagnostics/_lure_water_mask_export.tif

Usage (needs the project's conda env):
    /home/pregmi3/mambaforge/envs/helene_dec1/bin/python3 \\
        agent/scripts/make_lure_mask_overlay_figure.py
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIAGNOSTICS = Path(__file__).resolve().parents[2] / "results" / "diagnostics"


def read(path):
    with rasterio.open(path) as src:
        arr = src.read()
        ext = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    return arr, ext


def main():
    ortho, ortho_ext = read(DIAGNOSTICS / "_lure_ortho_2024.tif")
    mask, mask_ext = read(DIAGNOSTICS / "_lure_stable_mask.tif")
    water, water_ext = read(DIAGNOSTICS / "_lure_water_mask_export.tif")
    assert ortho_ext == mask_ext == water_ext, "diagnostic rasters not co-registered"

    ortho_img = np.transpose(ortho, (1, 2, 0)).astype(float) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax in axes:
        ax.imshow(ortho_img, extent=ortho_ext, origin="upper")
        ax.set_aspect("equal")

    # left: full reach
    m = np.ma.masked_where(mask[0] == 0, mask[0])
    axes[0].imshow(m, extent=mask_ext, origin="upper", cmap="autumn_r",
                  alpha=0.55, vmin=0, vmax=1)
    w = np.ma.masked_where(water[0] == 0, water[0])
    axes[0].imshow(w, extent=water_ext, origin="upper", cmap="Blues",
                  alpha=0.5, vmin=0, vmax=1)
    axes[0].set_title("stable_mask_lure (red/orange) + lure_water_mask (blue)\n"
                      "over 2024 orthophoto -- full reach")

    # right: zoom on the developed area near the dam / Morse Park end
    # (south end of the reach, where docks/roads/park infrastructure would be)
    left, right, bottom, top = mask_ext
    axes[1].imshow(m, extent=mask_ext, origin="upper", cmap="autumn_r",
                  alpha=0.55, vmin=0, vmax=1)
    axes[1].imshow(w, extent=water_ext, origin="upper", cmap="Blues",
                  alpha=0.5, vmin=0, vmax=1)
    axes[1].set_xlim(left, left + (right - left) * 0.45)
    axes[1].set_ylim(bottom, bottom + (top - bottom) * 0.35)
    axes[1].set_title("zoom: south end (dam / developed shoreline)")

    fig.suptitle("Lake Lure stable-mask visual QA -- diagnostic, not a "
                 "manuscript figure", fontsize=10)
    out = DIAGNOSTICS / "lure_stable_mask_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
