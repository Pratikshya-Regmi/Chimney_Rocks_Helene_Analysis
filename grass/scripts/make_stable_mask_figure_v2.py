#!/usr/bin/env python3
"""
make_stable_mask_figure_v2.py -- three candidate layouts for the manuscript's
Figure 4 (stable-terrain mask), addressing the watershed/Lake Lure scale
mismatch in the first version (make_stable_mask_figure.py): the watershed is
~0.4 km^2 and Lake Lure ~2.6 km^2, so a side-by-side layout with equal-size
panel boxes renders Lake Lure at roughly half the watershed's true map
scale (more ground per mm), and its mask texture reads as sparse/illegible.

Reuses the GeoTIFFs make_stable_mask_figure.py already exported (no new
GRASS work): cat34_cap_ortho_2024.tif, cat34_stable_mask.tif (results/figures/),
_lure_ortho_2024.tif, _lure_stable_mask.tif (results/diagnostics/).

Three variants, each kept to the map_style.py treatment (graticule, north
arrow + scale bar PER PANEL, no shared/forced frame, no internal place
names beyond what a reviewer needs):

  (a) fig_stable_mask_v2a_stacked.pdf/png
      Vertical stack. Both panels rendered at the SAME ground scale
      (m/mm) -- Lake Lure's width sets the scale (fills the full 174 mm
      column on its long E-W axis), and the watershed panel above it is
      sized to that same scale rather than to a separate, larger one.

  (b) fig_stable_mask_v2b_sidebyside_ownaspect.pdf/png
      Side by side (as before), but panel WIDTHS follow each panel's own
      true aspect ratio at a shared panel HEIGHT, instead of two forced
      equal-width boxes. Not scale-matched (that's variant c) -- just no
      longer artificially squeezed.

  (c) fig_stable_mask_v2c_subarea_matched.pdf/png
      Watershed panel (full reach, as before) beside a Lake Lure SUB-AREA
      cropped to the SAME map scale as the watershed panel, so mask
      density is directly, visually comparable. The sub-area (950 m,
      centred on the reach, E316500-317450) was chosen because its mask
      fraction (23.0%) is closest to the whole-reach average (25.7%) among
      three candidate slices checked (west/dredge 22.1%, central 23.0%,
      east/forest 26.3%) -- i.e. it is not the most/least "stable"-looking
      part of the reach. A small locator inset shows where it sits within
      the full reach.

Usage (needs the project's conda env):
    /home/pregmi3/mambaforge/envs/helene_dec1/bin/python3 \\
        agent/scripts/make_stable_mask_figure_v2.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_style import style_map_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "figures"
DIAGNOSTICS = ROOT / "results" / "diagnostics"

WSHED_ORTHO = FIGURES / "cat34_cap_ortho_2024.tif"
WSHED_MASK = FIGURES / "cat34_stable_mask.tif"
LURE_ORTHO = DIAGNOSTICS / "_lure_ortho_2024.tif"
LURE_MASK = DIAGNOSTICS / "_lure_stable_mask.tif"

MM = 1 / 25.4
FIG_W_MM = 174  # double column, Springer/Natural Hazards spec

MASK_COLOR = (0.0, 0.55, 0.35)
MASK_ALPHA = 0.55


def read_raster(path, band=1):
    with rasterio.open(path) as src:
        arr = src.read(band, masked=True)
        b = src.bounds
        extent = (b.left, b.right, b.bottom, b.top)
    return arr, extent


def draw_ortho_mask(ax, ortho_path, mask_path, extent=None, mask_extent=None):
    ortho, oext = read_raster(ortho_path, band=None) if False else (None, None)
    with rasterio.open(ortho_path) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).astype(float) / 255.0
        oext = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    ax.imshow(rgb, extent=oext, origin="upper", zorder=1)

    mask, mext = read_raster(mask_path)
    on = (~mask.mask) & (mask.data > 0) if hasattr(mask, "mask") else mask > 0
    overlay = np.zeros((*mask.shape, 4))
    overlay[..., 0] = MASK_COLOR[0]
    overlay[..., 1] = MASK_COLOR[1]
    overlay[..., 2] = MASK_COLOR[2]
    overlay[..., 3] = np.where(on, MASK_ALPHA, 0.0)
    ax.imshow(overlay, extent=mext, origin="upper", zorder=3)
    return oext


def panel_label(ax, text):
    ax.text(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top",
           fontsize=8, fontweight="bold", color="white", zorder=10,
           bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55,
                     edgecolor="none"))


# ============================================================ variant (a)
def make_variant_a():
    with rasterio.open(WSHED_ORTHO) as s:
        w_w = s.bounds.right - s.bounds.left
        w_h = s.bounds.top - s.bounds.bottom
    with rasterio.open(LURE_ORTHO) as s:
        l_w = s.bounds.right - s.bounds.left
        l_h = s.bounds.top - s.bounds.bottom

    # Lake Lure sets the scale: full column width for its long (E-W) axis.
    scale_m_per_mm = l_w / FIG_W_MM
    lure_h_mm = l_h / scale_m_per_mm
    wshed_w_mm = w_w / scale_m_per_mm
    wshed_h_mm = w_h / scale_m_per_mm

    fig_h_mm = wshed_h_mm + lure_h_mm + 14  # vertical gap allowance
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h_mm * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[wshed_h_mm, lure_h_mm], hspace=0.30)

    ax_w = fig.add_subplot(gs[0, 0])
    ext_w = draw_ortho_mask(ax_w, WSHED_ORTHO, WSHED_MASK)
    panel_label(ax_w, "(a) Watershed")
    style_map_panel(ax_w, ext_w, basin_label=None)

    ax_l = fig.add_subplot(gs[1, 0])
    ext_l = draw_ortho_mask(ax_l, LURE_ORTHO, LURE_MASK)
    panel_label(ax_l, "(b) Lake Lure")
    style_map_panel(ax_l, ext_l, basin_label=None)

    out = FIGURES / "fig_stable_mask_v2a_stacked"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"(a) wrote {out}.pdf/.png -- common scale {scale_m_per_mm:.2f} m/mm, "
          f"watershed panel {wshed_w_mm:.0f}x{wshed_h_mm:.0f} mm, "
          f"Lake Lure panel {FIG_W_MM}x{lure_h_mm:.0f} mm, "
          f"total figure {FIG_W_MM}x{fig_h_mm:.0f} mm")


# ============================================================ variant (b)
def make_variant_b():
    with rasterio.open(WSHED_ORTHO) as s:
        w_w = s.bounds.right - s.bounds.left
        w_h = s.bounds.top - s.bounds.bottom
    with rasterio.open(LURE_ORTHO) as s:
        l_w = s.bounds.right - s.bounds.left
        l_h = s.bounds.top - s.bounds.bottom

    # shared panel HEIGHT (mm); solve so total width (+ gap) fits FIG_W_MM
    gap_mm = 6
    # width_w = H / (w_h/w_w); width_l = H / (l_h/l_w)
    inv_aspect_w = w_w / w_h
    inv_aspect_l = l_w / l_h
    panel_h_mm = (FIG_W_MM - gap_mm) / (1 / inv_aspect_w + 1 / inv_aspect_l)
    width_w_mm = panel_h_mm * inv_aspect_w
    width_l_mm = panel_h_mm * inv_aspect_l

    fig = plt.figure(figsize=(FIG_W_MM * MM, panel_h_mm * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[width_w_mm, width_l_mm], wspace=0.12)

    ax_w = fig.add_subplot(gs[0, 0])
    ext_w = draw_ortho_mask(ax_w, WSHED_ORTHO, WSHED_MASK)
    panel_label(ax_w, "(a) Watershed")
    # this panel is narrow (~50 mm) -- the default graticule step (~5
    # lines across) packs tick labels too tight to read at that width;
    # widen the step so fewer, legible labels are drawn instead.
    style_map_panel(ax_w, ext_w, basin_label=None, graticule_step=500,
                    scale_color="white")

    ax_l = fig.add_subplot(gs[0, 1])
    ext_l = draw_ortho_mask(ax_l, LURE_ORTHO, LURE_MASK)
    panel_label(ax_l, "(b) Lake Lure")
    style_map_panel(ax_l, ext_l, basin_label=None, scale_color="white")

    out = FIGURES / "fig_stable_mask_v2b_sidebyside_ownaspect"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"(b) wrote {out}.pdf/.png -- shared height {panel_h_mm:.0f} mm, "
          f"watershed panel {width_w_mm:.0f}x{panel_h_mm:.0f} mm "
          f"({w_h/panel_h_mm:.2f} m/mm), Lake Lure panel "
          f"{width_l_mm:.0f}x{panel_h_mm:.0f} mm ({l_h/panel_h_mm:.2f} m/mm)")


# ============================================================ variant (c)
SUBAREA = (316500, 317450)  # E0, E1 -- central slice, mask fraction 23.0%,
                            # closest of 3 candidates checked to the whole-
                            # reach average (25.7%); see module docstring.


def make_variant_c():
    with rasterio.open(WSHED_ORTHO) as s:
        w_w = s.bounds.right - s.bounds.left
        w_h = s.bounds.top - s.bounds.bottom
        w_left, w_bottom = s.bounds.left, s.bounds.bottom
    with rasterio.open(LURE_ORTHO) as s:
        l_full = s.bounds
        l_h = l_full.top - l_full.bottom

    scale_m_per_mm = w_h / 130  # target watershed panel height 130 mm (tall,
    # legible); Lake Lure sub-area panel below is built at this SAME scale.
    wshed_w_mm = w_w / scale_m_per_mm
    wshed_h_mm = w_h / scale_m_per_mm

    sub_w = SUBAREA[1] - SUBAREA[0]
    sub_h_mm = l_h / scale_m_per_mm  # full N-S extent of the reach, at the
    # watershed's own scale
    sub_w_mm = sub_w / scale_m_per_mm

    gap_mm = 6
    fig_w_mm = wshed_w_mm + gap_mm + sub_w_mm
    fig_h_mm = max(wshed_h_mm, sub_h_mm) + 4
    fig = plt.figure(figsize=(fig_w_mm * MM, fig_h_mm * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[wshed_w_mm, sub_w_mm], wspace=0.10)

    ax_w = fig.add_subplot(gs[0, 0])
    ext_w = draw_ortho_mask(ax_w, WSHED_ORTHO, WSHED_MASK)
    panel_label(ax_w, "(a) Watershed")
    style_map_panel(ax_w, ext_w, basin_label=None)

    ax_l = fig.add_subplot(gs[0, 1])
    ext_l = draw_ortho_mask(ax_l, LURE_ORTHO, LURE_MASK)
    ax_l.set_xlim(SUBAREA[0], SUBAREA[1])
    ax_l.set_ylim(l_full.bottom, l_full.top)
    # kept short -- a two-line label collided with the locator inset,
    # which also has to sit upper-right (upper-left is the label, lower-
    # right is the north arrow/scale bar); the "representative segment"
    # detail belongs in the caption, not the panel.
    panel_label(ax_l, "(b) Lake Lure")
    style_map_panel(ax_l, (SUBAREA[0], SUBAREA[1], l_full.bottom, l_full.top),
                    basin_label=None)

    # ---- locator inset: full reach, sub-area boxed ----
    # placed UPPER-right, not lower-right: style_map_panel's north arrow +
    # scale bar always anchor lower-right (map_style.py's one implemented
    # corner), so lower-right would overlap them. Upper-left already holds
    # the "(b) Lake Lure..." panel label, so upper-right is the only corner
    # clear of both.
    inset_w_mm = wshed_w_mm * 0.62
    inset_h_mm = inset_w_mm * (l_h / (l_full.right - l_full.left))
    ax_loc = ax_l.inset_axes(
        [1.0 - inset_w_mm / sub_w_mm - 0.03, 1.0 - inset_h_mm / sub_h_mm - 0.03,
         inset_w_mm / sub_w_mm, inset_h_mm / sub_h_mm], zorder=15)
    with rasterio.open(LURE_ORTHO) as s:
        rgb = np.transpose(s.read([1, 2, 3]), (1, 2, 0)).astype(float) / 255.0
    ax_loc.imshow(rgb, extent=(l_full.left, l_full.right, l_full.bottom, l_full.top),
                 origin="upper")
    ax_loc.add_patch(plt.Rectangle((SUBAREA[0], l_full.bottom),
                                   sub_w, l_h, fill=False, edgecolor="red",
                                   linewidth=1.2, zorder=16))
    ax_loc.set_xlim(l_full.left, l_full.right)
    ax_loc.set_ylim(l_full.bottom, l_full.top)
    ax_loc.set_xticks([])
    ax_loc.set_yticks([])
    for spine in ax_loc.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")
    # label INSIDE the inset, not a title above it -- the inset now sits at
    # the panel's top edge, and a title there would be clipped.
    ax_loc.text(0.03, 0.05, "full reach", transform=ax_loc.transAxes,
               fontsize=5.5, color="white", ha="left", va="bottom",
               path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    out = FIGURES / "fig_stable_mask_v2c_subarea_matched"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"(c) wrote {out}.pdf/.png -- common scale {scale_m_per_mm:.2f} m/mm, "
          f"watershed panel {wshed_w_mm:.0f}x{wshed_h_mm:.0f} mm, "
          f"Lake Lure sub-area panel {sub_w_mm:.0f}x{sub_h_mm:.0f} mm "
          f"(E{SUBAREA[0]}-{SUBAREA[1]}, {sub_w:.0f} m of the reach's "
          f"{l_full.right-l_full.left:.0f} m full length), "
          f"total figure {fig_w_mm:.0f}x{fig_h_mm:.0f} mm")


if __name__ == "__main__":
    make_variant_a()
    make_variant_b()
    make_variant_c()
