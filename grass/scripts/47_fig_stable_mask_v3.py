#!/usr/bin/env python3
"""
47_fig_stable_mask_v3.py
=========================
Alternatives to fig_stable_mask_v2b_sidebyside_ownaspect.

    a  the current side-by-side, refined: BOTH panels at the SAME scale so a
       single shared scale bar is legitimate, one north arrow, titles moved
       out of the map, lighter graticule
    b  single panel -- 2020 hillshade with the stable mask as a
       semi-transparent overlay
    c  the mask as an OUTLINE over the thresholded DoD, so the reader sees
       which areas the mask excludes and what change lies underneath
    d  as (c) plus an inset histogram of dh inside the mask, annotated with
       NMAD, SD and the manuscript LoD -- the mask exists to estimate that
       number, so the figure shows the estimate it produces

WHY (a) HAD TO CHANGE SCALE, NOT JUST TIDY UP
    v2b solved for a shared panel HEIGHT, which leaves the two sites at
    different scales (the watershed panel spans 948 m, Lake Lure 2273 m,
    across similar panel widths). A single shared scale bar is only
    meaningful if both panels are at one scale, so (a) sizes both panels
    from a common metres-per-millimetre instead. The watershed panel gets
    smaller as a result -- that is the honest cost of a shared bar.

MASK AREA (for the caption; computed by this script, printed and logged)
    The mask raster covers the BUFFERED export extent (948 x 1509 m), not
    the basin: 76% of its cells lie outside the watershed. Quoting the raw
    cell count against the basin area would overstate coverage by ~4x
    (49% instead of 12%), so the figure is intersected with the cat 34
    polygon before any percentage is quoted.

Outputs: results/diagnostics/fig_stable_mask_v3_{a,b,c,d}.png (300 dpi) + .pdf
         results/tables/stable_mask_area_summary.csv
         results/logs/stable_mask_v3_<date>.log
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log                                     # noqa: E402
from map_style import (finalize_frame, add_graticule,          # noqa: E402
                       add_scale_north_inline)
from make_transect_figure_v2 import C_EROSION, C_DEPOSITION, LOD  # noqa: E402

RUN = "stable_mask_v3"
ROOT = HERE.parents[1]
FIGS = ROOT / "results" / "figures"
DIAG = ROOT / "results" / "diagnostics"
TABLES = ROOT / "results" / "tables"
DIAG.mkdir(parents=True, exist_ok=True)

W_MASK = FIGS / "cat34_stable_mask.tif"
W_ORTHO = FIGS / "cat34_cap_ortho_2024.tif"
W_HILL = FIGS / "cat34_hillshade_2020.tif"
W_DOD_T = FIGS / "cat34_dod_lidar_lidar_thresholded.tif"
W_DOD_F = FIGS / "cat34_dod_full_untresholded.tif"
L_MASK = DIAG / "_lure_stable_mask.tif"
L_ORTHO = DIAG / "_lure_ortho_2024.tif"

MM = 1 / 25.4
FIG_W_MM = 174
FIG_H_MAX = 205        # page limit; panels are sized to fit inside it
MASK_GREEN = "#00A651"



def fit_scale(mm_per_m, axes_w_mm, candidates=(1000, 500, 250, 100, 50)):
    """Longest round scale bar that still leaves room for the north arrow.

    add_scale_north_inline draws the arrow just right of the bar, so a bar
    wider than ~half its axes pushes the arrow past the edge, where it is
    clipped and silently disappears.
    """
    return next((c for c in candidates if c * mm_per_m <= 0.52 * axes_w_mm),
                candidates[-1])


def read(path, band=None):
    with rasterio.open(path) as s:
        arr = s.read(masked=True) if band is None else s.read(band, masked=True)
        b = s.bounds
        tr, shape = s.transform, (s.height, s.width)
    return arr, (b.left, b.right, b.bottom, b.top), tr, shape


def basin_poly():
    b = pd.read_csv(TABLES / "cat34_boundary.csv").sort_values("vertex_order")
    return (Polygon(zip(b["easting"], b["northing"])),
            np.column_stack([b["easting"].to_numpy(), b["northing"].to_numpy()]))


def mask_stats():
    poly, ring = basin_poly()
    m, ext, tr, shape = read(W_MASK, band=1)
    cell = abs(tr.a * tr.e)
    basin = rasterize([(poly, 1)], out_shape=shape, transform=tr, fill=0,
                      dtype="uint8").astype(bool)
    mask = (~np.ma.getmaskarray(m)) & (np.asarray(m.filled(0)) > 0)
    inb = int((mask & basin).sum())
    nb = int(basin.sum())
    dh, _, _, _ = read(W_DOD_F, band=1)
    v = np.asarray(dh.filled(np.nan))[mask & basin]
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    nmad = 1.4826 * float(np.median(np.abs(v - med)))
    st = dict(
        mask_cells_export=int(mask.sum()),
        mask_area_export_m2=int(mask.sum()) * cell,
        watershed_area_m2=nb * cell,
        mask_area_in_watershed_m2=inb * cell,
        pct_of_watershed=100.0 * inb / nb,
        pct_of_mask_outside_basin=100.0 * (int(mask.sum()) - inb) / int(mask.sum()),
        dh_n=int(v.size), dh_mean=float(v.mean()), dh_median=med,
        dh_sd=float(v.std(ddof=1)), dh_nmad=nmad,
        lod_1p96_nmad=1.96 * nmad, lod_1p96_sd=1.96 * float(v.std(ddof=1)),
        manuscript_lod=LOD)
    return st, mask, basin, ext, ring, v


def mask_overlay(ax, mask, extent, alpha=0.45, color=MASK_GREEN):
    rgba = np.zeros(mask.shape + (4,), float)
    rgba[..., :3] = matplotlib.colors.to_rgb(color)
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    return ax.imshow(rgba, extent=extent, origin="upper", zorder=4,
                     interpolation="nearest")


def mask_outline(ax, mask, extent, color=MASK_GREEN, lw=0.7):
    ny, nx = mask.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[3], extent[2], ny)
    ax.contour(xs, ys, mask.astype(float), levels=[0.5], colors=[color],
               linewidths=lw, zorder=6)


def dod_layer(ax, extent):
    hs, hext, _, _ = read(W_HILL, band=1)
    dod, dext, _, _ = read(W_DOD_T, band=1)
    ax.imshow(hs, cmap="gray", vmin=0, vmax=255, extent=hext, origin="upper",
              zorder=1)
    cmap = LinearSegmentedColormap.from_list(
        "erosion_deposition", [C_EROSION, "#f7f7f7", C_DEPOSITION])
    cmap.set_bad(color=(0, 0, 0, 0))
    return ax.imshow(dod, extent=dext, origin="upper", cmap=cmap,
                     norm=Normalize(vmin=-3, vmax=3), zorder=3, alpha=0.92)


def title(ax, text, fs=7.4):
    ax.set_title(text, fontsize=fs, fontweight="bold", pad=3)


def save(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ (a)
def variant_a(st, ring):
    wm, wext, _, _ = read(W_MASK, band=1)
    wo, _, _, _ = read(W_ORTHO)
    lm, lext, _, _ = read(L_MASK, band=1)
    lo, _, _, _ = read(L_ORTHO)
    wmask = (~np.ma.getmaskarray(wm)) & (np.asarray(wm.filled(0)) > 0)
    lmask = (~np.ma.getmaskarray(lm)) & (np.asarray(lm.filled(0)) > 0)

    left, right, gap, top, xlab, legend = 11.0, 3.0, 6.0, 6.0, 12.0, 17.0
    ww, lw = wext[1] - wext[0], lext[1] - lext[0]
    # ONE scale for both panels -- prerequisite for a shared scale bar
    m_per_mm = (ww + lw) / (FIG_W_MM - left - right - gap)
    w_mm, l_mm = ww / m_per_mm, lw / m_per_mm
    wh = (wext[3] - wext[2]) / m_per_mm
    lh = (lext[3] - lext[2]) / m_per_mm
    fig_h = top + max(wh, lh) + xlab + legend
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    axw = fig.add_axes([left / FIG_W_MM, 1 - (top + wh) / fig_h,
                        w_mm / FIG_W_MM, wh / fig_h])
    axl = fig.add_axes([(left + w_mm + gap) / FIG_W_MM, 1 - (top + lh) / fig_h,
                        l_mm / FIG_W_MM, lh / fig_h])
    for ax, img, ext, msk, t, step in (
            (axw, wo, wext, wmask, "(a) Watershed", 500),
            (axl, lo, lext, lmask, "(b) Lake Lure", 1000)):
        ax.imshow(np.transpose(img, (1, 2, 0)).astype(float) / 255.0,
                  extent=ext, origin="upper", zorder=2)
        mask_overlay(ax, msk, ext)
        finalize_frame(ax, ext)
        add_graticule(ax, ext, step=step, fontsize=5.0)
        title(ax, t)
    axw.plot(ring[:, 0], ring[:, 1], color="black", linewidth=0.9, zorder=7)

    sw = 34.0
    sax = fig.add_axes([(FIG_W_MM - right - sw) / FIG_W_MM, 2.5 / fig_h,
                        sw / FIG_W_MM, 12.0 / fig_h])
    add_scale_north_inline(sax, scale_m=fit_scale(1.0 / m_per_mm, sw),
                           mm_per_m=1.0 / m_per_mm, axes_w_mm=sw,
                           color="black")
    fig.text(left / FIG_W_MM, 6.0 / fig_h,
             f"Stable-terrain mask (green): {st['mask_area_in_watershed_m2']:,.0f}"
             f" m$^2$, {st['pct_of_watershed']:.1f}% of the watershed. "
             "Both panels at one scale.", fontsize=6.4, va="center")
    save(fig, DIAG / "fig_stable_mask_v3_a")
    return fig_h


# ------------------------------------------------------------------ (b)
def variant_b(st, mask, ring):
    hs, ext, _, _ = read(W_HILL, band=1)
    left, right, top, xlab, legend = 11.0, 3.0, 6.0, 12.0, 20.0
    # The watershed export is 948 x 1509 m (aspect 1.59): at full column
    # width the panel would be 254 mm tall, past any page. Size it from the
    # HEIGHT budget instead and centre the narrower panel.
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    h = min(FIG_H_MAX - top - xlab - legend,
            (FIG_W_MM - left - right) * aspect)
    w = h / aspect
    left = (FIG_W_MM - w) / 2.0
    fig_h = top + h + xlab + legend
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
    ax = fig.add_axes([left / FIG_W_MM, 1 - (top + h) / fig_h,
                       w / FIG_W_MM, h / fig_h])
    ax.imshow(hs, cmap="gray", vmin=0, vmax=255, extent=ext, origin="upper",
              zorder=1)
    mask_overlay(ax, mask, ext, alpha=0.50)
    ax.plot(ring[:, 0], ring[:, 1], color="black", linewidth=1.0, zorder=7)
    finalize_frame(ax, ext)
    add_graticule(ax, ext, step=250, fontsize=5.0)
    title(ax, "Stable-terrain mask over 2020 hillshade")
    sw = 34.0
    mpm = w / (ext[1] - ext[0])
    sax = fig.add_axes([(FIG_W_MM - right - sw) / FIG_W_MM, 5.0 / fig_h,
                        sw / FIG_W_MM, 11.0 / fig_h])
    add_scale_north_inline(sax, scale_m=fit_scale(mpm, sw), mm_per_m=mpm,
                           axes_w_mm=sw)
    fig.text(left / FIG_W_MM, 1.8 / fig_h,
             f"Green = stable terrain used for the level of detection: "
             f"{st['mask_area_in_watershed_m2']:,.0f} m$^2$ inside the "
             f"watershed ({st['pct_of_watershed']:.1f}%). Black = cat 34 "
             "boundary.", fontsize=6.4, va="center")
    save(fig, DIAG / "fig_stable_mask_v3_b")
    return fig_h


# ------------------------------------------------------------------ (c)/(d)
def variant_cd(st, mask, ring, v, with_hist):
    _, ext, _, _ = read(W_HILL, band=1)
    left, right, top, xlab, legend = 11.0, 3.0, 6.0, 12.0, 22.0
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    h = min(FIG_H_MAX - top - xlab - legend,
            (FIG_W_MM - left - right) * aspect)
    w = h / aspect
    left = (FIG_W_MM - w) / 2.0
    fig_h = top + h + xlab + legend
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))
    ax = fig.add_axes([left / FIG_W_MM, 1 - (top + h) / fig_h,
                       w / FIG_W_MM, h / fig_h])
    im = dod_layer(ax, ext)
    mask_outline(ax, mask, ext, lw=0.8)
    ax.plot(ring[:, 0], ring[:, 1], color="black", linewidth=1.0, zorder=7)
    finalize_frame(ax, ext)
    add_graticule(ax, ext, step=250, fontsize=5.0)
    title(ax, "Stable-mask outline over the thresholded DoD"
              + (" , with residuals inside the mask" if with_hist else ""))

    if with_hist:
        # inset_axes default to the same zorder as the parent, so the map's
        # contours and boundary (zorder 6-7) drew straight through the
        # histogram; raise it and make the patch opaque. Positioned well
        # inside the panel so its tick labels do not fall onto the
        # graticule labels below the frame.
        ins = ax.inset_axes([0.585, 0.105, 0.385, 0.215], zorder=20)
        ins.hist(v, bins=80, range=(-1, 1), color="0.35", edgecolor="none")
        for x, c, ls in ((st["dh_nmad"], MASK_GREEN, "-"),
                         (-st["dh_nmad"], MASK_GREEN, "-"),
                         (LOD, "black", "--"), (-LOD, "black", "--")):
            ins.axvline(x, color=c, linewidth=0.8, linestyle=ls)
        ins.set_xlabel("$\\Delta z$ in mask [m]", fontsize=5.4, labelpad=1)
        ins.tick_params(labelsize=5.0, length=2, width=0.4)
        ins.set_yticks([])
        ins.set_facecolor("white")
        ins.patch.set_alpha(0.97)
        ins.set_zorder(20)
        for sp in ins.spines.values():
            sp.set_linewidth(0.5)
        # the inset's tick labels and axis label sit on the hillshade, so
        # give them the same white stroke the map labels use
        halo = [pe.withStroke(linewidth=1.8, foreground="white")]
        ins.xaxis.label.set_path_effects(halo)
        for t in ins.get_xticklabels():
            t.set_path_effects(halo)
        ins.text(0.03, 0.93,
                 f"NMAD {st['dh_nmad']:.3f} m\nSD {st['dh_sd']:.3f} m\n"
                 f"LoD {LOD:.2f} m (dashed)",
                 transform=ins.transAxes, fontsize=5.0, va="top",
                 path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])

    cw = 62.0
    cax = fig.add_axes([left / FIG_W_MM, 10.5 / fig_h, cw / FIG_W_MM,
                        3.4 / fig_h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-3, -1.5, 0, 1.5, 3])
    cb.ax.tick_params(labelsize=5.6, length=2, width=0.4)
    cb.set_label("Elevation change [m] (erosion red, deposition blue)",
                 fontsize=6.0)
    cb.outline.set_linewidth(0.5)
    sw = 34.0
    mpm = w / (ext[1] - ext[0])
    sax = fig.add_axes([(FIG_W_MM - right - sw) / FIG_W_MM, 5.5 / fig_h,
                        sw / FIG_W_MM, 11.0 / fig_h])
    add_scale_north_inline(sax, scale_m=fit_scale(mpm, sw), mm_per_m=mpm,
                           axes_w_mm=sw)
    fig.text(left / FIG_W_MM, 1.6 / fig_h,
             f"Green outline = stable mask ({st['pct_of_watershed']:.1f}% of "
             "the watershed); change inside it is the error estimate, not "
             "signal.", fontsize=6.0, va="center")
    save(fig, DIAG / ("fig_stable_mask_v3_d" if with_hist
                      else "fig_stable_mask_v3_c"))
    return fig_h


def main():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    st, mask, basin, ext, ring, v = mask_stats()

    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / "stable_mask_area_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value", "units"])
        for k, val in st.items():
            w.writerow([k, f"{val:.4f}" if isinstance(val, float) else val,
                        "m2" if k.endswith("_m2") else
                        ("%" if k.startswith("pct") else
                         ("m" if k.startswith(("dh_", "lod", "manuscript")) else ""))])
    print(f"wrote {TABLES/'stable_mask_area_summary.csv'}")

    print(f"\nMASK AREA -- watershed {st['watershed_area_m2']:,.0f} m2; "
          f"mask inside watershed {st['mask_area_in_watershed_m2']:,.0f} m2 "
          f"= {st['pct_of_watershed']:.2f}%")
    print(f"  (raster covers the buffered export area; "
          f"{st['pct_of_mask_outside_basin']:.0f}% of mask cells fall outside "
          f"the basin and are excluded from that percentage)")
    print(f"  dh in mask: n={st['dh_n']:,} median={st['dh_median']:+.4f} "
          f"SD={st['dh_sd']:.4f} NMAD={st['dh_nmad']:.4f} m; "
          f"1.96*NMAD={st['lod_1p96_nmad']:.3f} m vs manuscript LoD {LOD} m")
    log(f"mask area in watershed={st['mask_area_in_watershed_m2']:.0f} m2 "
        f"({st['pct_of_watershed']:.2f}% of {st['watershed_area_m2']:.0f} m2); "
        f"{st['pct_of_mask_outside_basin']:.1f}% of mask cells outside basin; "
        f"dh NMAD={st['dh_nmad']:.4f} SD={st['dh_sd']:.4f} m", run=RUN)

    for name, fn in (("a", lambda: variant_a(st, ring)),
                     ("b", lambda: variant_b(st, mask, ring)),
                     ("c", lambda: variant_cd(st, mask, ring, v, False)),
                     ("d", lambda: variant_cd(st, mask, ring, v, True))):
        h = fn()
        print(f"({name}) fig_stable_mask_v3_{name}.png  {FIG_W_MM} x {h:.0f} mm")
        log(f"variant ({name}) -> fig_stable_mask_v3_{name}.png 300 dpi, "
            f"{FIG_W_MM}x{h:.0f} mm", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
