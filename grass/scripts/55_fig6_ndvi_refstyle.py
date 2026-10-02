#!/usr/bin/env python3
"""
55_fig6_ndvi_refstyle.py
=========================
Figure 6 rebuilt in the style of the supplied reference
(results/figures/"NDVI_difference (1).png"), with the three requested
changes:

  1. Delta-NDVI drawn on a RED -> GREEN diverging ramp (red = vegetation
     loss, green = gain) in place of the reference's red/grey/blue.
  2. A rewritten legend: a continuous bar with symmetric limits about zero
     and a proper label, rather than the reference's five uneven ticks
     (-1.36, -0.79, -0.23, 0.34, 0.91).
  3. Scale bar and north arrow placed on the SAME LINE as the legend.

Everything else follows the reference: two full-width rows (aerial imagery
above, Delta-NDVI below), the two analysis areas boxed in both rows and
joined by arrows, and near-zero Delta-NDVI left transparent so the grey
terrain relief reads through it.

BASE LAYER
    ndvi_change_relief (helene_chimney_11_25/test_data_extent) is the grey
    hillshade the reference used; it is exported by the companion probe and
    read here, so the background is the real relief rather than a
    desaturated orthophoto.

RAMPS PRODUCED
    rdylgn  ColorBrewer RdYlGn -- red/amber through to green
    rwg     plain red -> white -> green, higher-contrast limbs

Outputs -> results/figures/fig6_ndvi_refstyle/fig6_refstyle_<ramp>.png/.pdf
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402

RUN = "fig6_refstyle"
ROOT = HERE.parents[1]
FIG = ROOT / "results" / "figures"
OUT = FIG / "fig6_ndvi_refstyle"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM = 174.0
VMAX = 0.70
FADE_LO, FADE_HI = 0.08, 0.22      # |dNDVI| transparent below LO, opaque above HI
FS = dict(panel=8.5, tick=7.0, cbar=7.5, scale=7.0, north=8.5)

NDVI_TIF = FIG / "_ndvi_change.tif"
RELIEF_TIF = FIG / "_ndvi_relief.tif"
CTX_PNG = FIG / "_ndvi_context.png"
CTX_FULL = (312923.0, 318290.0, 190698.0, 193937.0)
BND = {"watershed": FIG / "_ndvi_bnd_watershed.geojson",
       "lure": FIG / "_ndvi_bnd_lure.geojson"}

RAMPS = {"rdylgn": ("RdYlGn", "ColorBrewer RdYlGn"),
         "rwg": (None, "red -> white -> green")}


def red_white_green(n=256):
    return LinearSegmentedColormap.from_list(
        "rwg", [(0.0, "#9E1B32"), (0.25, "#E4663F"), (0.5, "#FFFFFF"),
                (0.75, "#5CB85C"), (1.0, "#11692B")], N=n)


def pale_grey(n=256):
    """Light grey ramp for the shaded-relief base: keeps terrain visible but
    well below the dNDVI layer in contrast."""
    return LinearSegmentedColormap.from_list(
        "palegrey", ["#6E6E6E", "#F2F2F2"], N=n)


def ring_of(p):
    gj = json.loads(Path(p).read_text())
    best = None
    for f in gj.get("features", []):
        g = f.get("geometry") or {}
        polys = ([g["coordinates"]] if g.get("type") == "Polygon"
                 else g.get("coordinates", []))
        for poly in polys:
            for r in poly:
                a = np.asarray(r, float)
                if a.ndim == 2 and len(a) > 2 and (best is None or len(a) > len(best)):
                    best = a
    return best


def box_of(r, pad=30.0):
    return (r[:, 0].min() - pad, r[:, 0].max() + pad,
            r[:, 1].min() - pad, r[:, 1].max() + pad)


_C = {}


def context_image():
    if "img" in _C:
        return _C["img"], _C["ext"]
    img = np.asarray(Image.open(CTX_PNG).convert("RGB"))
    d = img.sum(axis=2) < 30
    rows = np.where(d.mean(axis=1) < 0.98)[0]
    cols = np.where(d.mean(axis=0) < 0.98)[0]
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    H, W = img.shape[:2]
    w0, e0, s0, n0 = CTX_FULL
    _C["img"] = img[r0:r1, c0:c1]
    _C["ext"] = (w0 + (e0 - w0) * c0 / W, w0 + (e0 - w0) * c1 / W,
                 n0 - (n0 - s0) * r1 / H, n0 - (n0 - s0) * r0 / H)
    return _C["img"], _C["ext"]


def read(path):
    with rasterio.open(path) as s:
        a = s.read(1, masked=True)
        b = s.bounds
    return a, (b.left, b.right, b.bottom, b.top)


def bare(ax, ext):
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def tag_corner(ext, boxes, frac_w=0.30, frac_h=0.10):
    """Pick the top corner whose tag footprint is clear of the boxed analysis
    areas, so the grey plate never sits under a box outline or its arrow."""
    x0, x1, y0, y1 = ext
    w, h = (x1 - x0) * frac_w, (y1 - y0) * frac_h
    cands = {"left": (x0, x0 + w, y1 - h, y1),
             "right": (x1 - w, x1, y1 - h, y1)}
    def overlap(a, b):
        return (max(0.0, min(a[1], b[1]) - max(a[0], b[0])) *
                max(0.0, min(a[3], b[3]) - max(a[2], b[2])))
    scored = {k: sum(overlap(v, b) for b in boxes) for k, v in cands.items()}
    return min(scored, key=scored.get)


def panel_tag(ax, text, corner="left"):
    """Reference-style tag on a grey plate, so the text never sits directly
    on imagery."""
    x, ha = (0.012, "left") if corner == "left" else (0.988, "right")
    ax.text(x, 0.965, text, transform=ax.transAxes, fontsize=FS["panel"],
            fontweight="bold", color="white", ha=ha, va="top", zorder=20,
            bbox=dict(boxstyle="square,pad=0.42", facecolor="0.42",
                      edgecolor="0.75", linewidth=0.8, alpha=0.92))


def build(key):
    name, desc = RAMPS[key]
    cmap = red_white_green() if name is None else plt.get_cmap(name)
    img, iext = context_image()
    ndvi, next_ = read(NDVI_TIF)
    relief, rext = read(RELIEF_TIF)
    rw, rl = ring_of(BND["watershed"]), ring_of(BND["lure"])
    boxes = {"watershed": box_of(rw), "lure": box_of(rl)}

    left = right = 3.0
    top, gap, legend_h, bot = 3.0, 2.0, 14.0, 3.0
    usable = FIG_W_MM - left - right
    row_h = usable * (iext[3] - iext[2]) / (iext[1] - iext[0])
    fig_h = top + row_h + gap + row_h + legend_h + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def rect(x, ytop, w, h):
        return [x / FIG_W_MM, 1 - (ytop + h) / fig_h, w / FIG_W_MM, h / fig_h]

    # ---------- top: aerial imagery ----------
    axt = fig.add_axes(rect(left, top, usable, row_h))
    axt.imshow(img, extent=iext, origin="upper", interpolation="bilinear",
               zorder=1)
    for b in boxes.values():
        axt.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2],
                                fill=False, edgecolor="black", lw=1.8,
                                zorder=6))
    bare(axt, iext)
    tcorner = tag_corner(iext, list(boxes.values()))
    panel_tag(axt, "Aerial imagery", tcorner)

    # ---------- bottom: dNDVI over grey relief ----------
    axb = fig.add_axes(rect(left, top + row_h + gap, usable, row_h))
    # The relief raster spans only ~116-140 DN, so a 0-255 stretch renders it
    # as flat mid-grey. Stretch to its own 2nd-98th percentile and map into a
    # pale grey range so terrain is legible without competing with dNDVI.
    rv = np.asarray(relief.filled(np.nan), dtype=float)
    lo, hi = np.nanpercentile(rv, [2, 98])
    axb.imshow(rv, extent=rext, origin="upper", cmap=pale_grey(),
               vmin=lo, vmax=hi, zorder=1, interpolation="bilinear")
    # near-zero left transparent so the relief reads through, as in the
    # reference; alpha ramps in over FADE_LO..FADE_HI
    z = np.asarray(ndvi.filled(np.nan))
    rgba = cmap(Normalize(-VMAX, VMAX)(z))
    a = np.clip((np.abs(z) - FADE_LO) / (FADE_HI - FADE_LO), 0, 1)
    rgba[..., 3] = np.where(np.isfinite(z), a, 0.0)
    im = axb.imshow(rgba, extent=next_, origin="upper", zorder=3,
                    interpolation="nearest")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(-VMAX, VMAX))
    for b in boxes.values():
        axb.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2],
                                fill=False, edgecolor="black", lw=1.8,
                                zorder=6))
    bare(axb, iext)
    panel_tag(axb, "NDVI difference", tcorner)

    # ---------- arrows joining the boxed areas ----------
    for b in boxes.values():
        cx = (b[0] + b[1]) / 2
        fx = (left + usable * (cx - iext[0]) / (iext[1] - iext[0])) / FIG_W_MM
        y_top = 1 - (top + row_h * (1 - (b[2] - iext[2]) /
                                    (iext[3] - iext[2]))) / fig_h
        y_bot = 1 - (top + row_h + gap + row_h *
                     (1 - (b[3] - iext[2]) / (iext[3] - iext[2]))) / fig_h
        fig.add_artist(FancyArrowPatch(
            (fx, y_top), (fx, y_bot), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=11, lw=1.2, color="black",
            shrinkA=1, shrinkB=1, zorder=30))

    # ---------- legend + scale bar + north arrow, one line ----------
    y_leg = top + row_h + gap + row_h + 2.0
    cb_w = 62.0
    cax = fig.add_axes(rect(left + 2.0, y_leg + 2.6, cb_w, 3.4))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-VMAX, -VMAX / 2, 0, VMAX / 2, VMAX])
    cb.ax.tick_params(labelsize=FS["tick"], length=2.5, width=0.5)
    cb.set_label("$\\Delta$NDVI    vegetation loss $\\leftarrow$   "
                 "$\\rightarrow$ gain", fontsize=FS["cbar"])
    cb.outline.set_linewidth(0.5)

    sx = left + 2.0 + cb_w + 14.0
    sw = 42.0
    sax = fig.add_axes(rect(sx, y_leg + 1.0, sw, 9.0))
    sax.set_xlim(0, sw)
    sax.set_ylim(0, 9)
    sax.set_axis_off()
    mm_per_m = usable / (iext[1] - iext[0])
    bar_m = next((c for c in (2000, 1000, 500, 250)
                  if c * mm_per_m <= 0.62 * sw), 250)
    bw = bar_m * mm_per_m
    x0, yy = 2.0, 4.2
    sax.plot([x0, x0 + bw], [yy, yy], color="black", lw=2.0,
             solid_capstyle="butt")
    for xt in (x0, x0 + bw):
        sax.plot([xt, xt], [yy - 0.9, yy + 0.9], color="black", lw=1.3)
    sax.text(x0 + bw / 2, yy + 1.3, f"{bar_m:,.0f} m", ha="center",
             va="bottom", fontsize=FS["scale"])

    nax = fig.add_axes(rect(sx + sw + 4.0, y_leg + 0.5, 12.0, 11.0))
    nax.set_xlim(0, 1)
    nax.set_ylim(0, 1)
    nax.set_axis_off()
    nax.annotate("", xy=(0.5, 0.62), xytext=(0.5, 0.04),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6,
                                 mutation_scale=14))
    nax.text(0.5, 0.66, "N", ha="center", va="bottom", fontsize=FS["north"],
             fontweight="bold")

    stem = OUT / f"fig6_refstyle_{key}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return stem, fig_h, bar_m


def main():
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                                "font.family": "sans-serif",
                                "font.sans-serif": ["DejaVu Sans"]})
    print(f"smallest label {min(FS.values()):.1f} pt at {FIG_W_MM:.0f} mm")
    for k, (n, d) in RAMPS.items():
        stem, h, bar = build(k)
        print(f"  {stem.name}.png/.pdf  {FIG_W_MM:.0f} x {h:.0f} mm  "
              f"scale bar {bar:,} m   ({d})")
        log(f"{stem.name}: {FIG_W_MM:.0f}x{h:.0f} mm, ramp={d}, "
            f"scale bar {bar} m, limits +/-{VMAX}", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
