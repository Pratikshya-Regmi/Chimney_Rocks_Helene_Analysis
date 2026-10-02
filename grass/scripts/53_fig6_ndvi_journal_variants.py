#!/usr/bin/env python3
"""
53_fig6_ndvi_journal_variants.py
=================================
Figure 6 (NDVI change) prepared to Natural Hazards / Springer Nature figure
standards. This script touches Figure 6 only.

FORMAT TARGETS
    174 mm double-column width; 300 dpi PNG + vector PDF; every label >= 7 pt
    AT PRINTED SIZE (the figure is saved without bbox_inches at an exact
    174 mm width, so the point sizes set here are the printed point sizes);
    one sans-serif family throughout; diverging ramp centred on zero with
    symmetric limits; colourblind-safe.

MAP ELEMENTS -- these and nothing else
    a separate scale bar per Delta-NDVI panel (the two areas are at different
    map scales), one north arrow, one shared Delta-NDVI legend, and the panel
    letters. No in-panel titles, no annotation, no coordinate labels: the CRS
    and the two map scales belong in the caption.

TWO CONSTRAINTS THAT NARROWED THE MATRIX, both reported rather than worked
around:

  * RdYlGn was requested but is dropped. ColorBrewer classifies it as NOT
    colourblind-safe (red/green is the deuteranopia failure case) and the
    brief requires colourblind-safety. BrBG and PiYG are both classified
    safe; `lab` below is included because it is the only one of the four
    whose sign survives a greyscale conversion.

  * Only ONE scale-bar arrangement satisfies "no text may overlap data".
    The Lake Lure panel is a filled rectangle with data to every edge, so it
    contains no clear area for an in-panel bar; an inside placement would
    necessarily sit on data. Scale bars are therefore drawn below each panel,
    outside the frame, and the north arrow outside as well. An in-panel
    variant is not produced because it could not meet the stated rule.

VARIANT MATRIX (all saved; none is chosen here)
    ramp    : brbg | piyg | lab
    legend  : below (horizontal, beneath the bottom row)
              right (vertical, right of the two panels)

Outputs -> results/figures/fig6_ndvi_variants/fig6_ndvi_<ramp>_<legend>.png/.pdf
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from PIL import Image
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402

RUN = "fig6_variants"
ROOT = HERE.parents[1]
FIG = ROOT / "results" / "figures"
OUT = FIG / "fig6_ndvi_variants"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM = 174.0
VMAX = 0.70

# every point size used in the figure; all must be >= 7.0
FS = dict(letter=8.0, cbar_tick=7.0, cbar_label=7.5, scale=7.0, north=8.5)

NDVI_TIF = FIG / "_ndvi_change.tif"
CTX_PNG = FIG / "_ndvi_context.png"
CTX_FULL = (312923.0, 318290.0, 190698.0, 193937.0)
BND = {"watershed": FIG / "_ndvi_bnd_watershed.geojson",
       "lure": FIG / "_ndvi_bnd_lure.geojson"}


# ------------------------------------------------------------------ colour
def _srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_M = np.array([[0.4124, 0.3576, 0.1805],
               [0.2126, 0.7152, 0.0722],
               [0.0193, 0.1192, 0.9505]])
_WP = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb):
    xyz = _srgb_to_lin(rgb) @ _M.T / _WP
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])]).T


def lab2rgb(lab):
    lab = np.atleast_2d(lab)
    fy = (lab[:, 0] + 16) / 116
    f = np.stack([fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200], axis=-1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WP
    return np.clip(_lin_to_srgb(xyz @ np.linalg.inv(_M).T), 0, 1)


def lab_diverging(n=256):
    loss = rgb2lab(np.array([[0.32, 0.17, 0.02]]))[0]
    mid = rgb2lab(np.array([[0.97, 0.97, 0.95]]))[0]
    gain = rgb2lab(np.array([[0.30, 0.62, 0.27]]))[0]
    a = np.linspace(0, 1, n // 2)[:, None]
    b = np.linspace(0, 1, n - n // 2)[:, None]
    return LinearSegmentedColormap.from_list(
        "lab_brown_green", np.vstack([lab2rgb(loss + (mid - loss) * a),
                                      lab2rgb(mid + (gain - mid) * b)]))


def limb_sep(cm):
    L = np.array([0.2126 * cm(x)[0] + 0.7152 * cm(x)[1] + 0.0722 * cm(x)[2]
                  for x in np.linspace(0, 1, 101)])
    return float(np.mean(np.abs(L[:50] - L[51:][::-1])))


RAMPS = {"brbg": ("BrBG", "ColorBrewer BrBG, colourblind-safe"),
         "piyg": ("PiYG", "ColorBrewer PiYG, colourblind-safe"),
         "lab": (None, "CIELAB lightness-linear brown-green, greyscale-safe")}


# ------------------------------------------------------------------ data
def ring_of(path):
    gj = json.loads(Path(path).read_text())
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


def bbox(r, pad=40.0):
    return (r[:, 0].min() - pad, r[:, 0].max() + pad,
            r[:, 1].min() - pad, r[:, 1].max() + pad)


_CTX = {}


def context_image():
    if "img" in _CTX:
        return _CTX["img"], _CTX["ext"]
    img = np.asarray(Image.open(CTX_PNG).convert("RGB"))
    dark = img.sum(axis=2) < 30
    rows = np.where(dark.mean(axis=1) < 0.98)[0]
    cols = np.where(dark.mean(axis=0) < 0.98)[0]
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    H, W = img.shape[:2]
    w0, e0, s0, n0 = CTX_FULL
    _CTX["img"] = img[r0:r1, c0:c1]
    _CTX["ext"] = (w0 + (e0 - w0) * c0 / W, w0 + (e0 - w0) * c1 / W,
                   n0 - (n0 - s0) * r1 / H, n0 - (n0 - s0) * r0 / H)
    return _CTX["img"], _CTX["ext"]


def thin_frame(ax, ext):
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("black")


def scalebar(fig, rect, box_w, map_w_mm, ground_m):
    """`rect` comes from the figure's top-down helper. fig.add_axes is
    bottom-up, so mixing the two conventions silently threw the scale bars to
    the top of the page, over the orthophoto."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, box_w)
    ax.set_ylim(0, 7.5)
    ax.set_axis_off()
    mm_per_m = map_w_mm / ground_m
    bar = next((c for c in (1000, 500, 250, 200, 100, 50)
                if c * mm_per_m <= 0.55 * map_w_mm), 50)
    bw = bar * mm_per_m
    x0, y = box_w / 2 - bw / 2, 2.6
    ax.plot([x0, x0 + bw], [y, y], color="black", lw=2.0, solid_capstyle="butt")
    for xt in (x0, x0 + bw):
        ax.plot([xt, xt], [y - 0.85, y + 0.85], color="black", lw=1.3)
    ax.text(box_w / 2, y + 1.3, f"{bar:,.0f} m", ha="center", va="bottom",
            fontsize=FS["scale"])
    return bar


def north(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.annotate("", xy=(0.5, 0.66), xytext=(0.5, 0.08),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6,
                                mutation_scale=14))
    ax.text(0.5, 0.70, "N", ha="center", va="bottom",
            fontsize=FS["north"], fontweight="bold")


def build(ramp_key, legend, arr, ext, rw, rl):
    name, _ = RAMPS[ramp_key]
    cmap = lab_diverging() if name is None else plt.get_cmap(name)
    img, iext = context_image()

    left = right = 4.0
    gap, top, letter_h, sb_h, bot = 7.0, 3.0, 5.5, 7.5, 3.0
    legend_h = 12.5
    usable = FIG_W_MM - left - right
    cbar_col = 15.0 if legend == "right" else 0.0
    panels_w = usable - cbar_col - (4.0 if legend == "right" else 0.0)

    ctx_h = usable * (iext[3] - iext[2]) / (iext[1] - iext[0])
    box_w = (panels_w - gap) / 2.0
    box_h = box_w * 0.92
    vw, vl = bbox(rw), bbox(rl)
    dims = {}
    for key, view in (("watershed", vw), ("lure", vl)):
        asp = (view[3] - view[2]) / (view[1] - view[0])
        if asp >= box_h / box_w:
            h = box_h; w = h / asp
        else:
            w = box_w; h = w * asp
        dims[key] = (w, h, view)

    fig_h = (top + letter_h + ctx_h + letter_h + box_h + sb_h
             + (legend_h if legend == "below" else 0.0) + bot)
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def rect(x, ytop, w, h):
        return [x / FIG_W_MM, 1 - (ytop + h) / fig_h, w / FIG_W_MM, h / fig_h]

    # (a) orthophoto
    y = top + letter_h
    axc = fig.add_axes(rect(left, y, usable, ctx_h))
    axc.imshow(img, extent=iext, origin="upper", zorder=1,
               interpolation="bilinear")
    for r in (rw, rl):
        axc.plot(r[:, 0], r[:, 1], color="black", lw=1.2, zorder=6)
    thin_frame(axc, iext)
    fig.text(left / FIG_W_MM, 1 - (y - 1.0) / fig_h, "(a)",
             fontsize=FS["letter"], fontweight="bold", va="bottom")

    # (b),(c) dNDVI
    y2 = y + ctx_h + letter_h
    im = None
    bars = {}
    for i, key in enumerate(("watershed", "lure")):
        w, h, view = dims[key]
        bx = left + i * (box_w + gap)
        ax = fig.add_axes(rect(bx + (box_w - w) / 2, y2 + (box_h - h) / 2, w, h))
        im = ax.imshow(arr, extent=ext, origin="upper", cmap=cmap,
                       norm=Normalize(-VMAX, VMAX), zorder=3,
                       interpolation="nearest")
        ring = rw if key == "watershed" else rl
        im.set_clip_path(PathPatch(MplPath(ring), transform=ax.transData))
        ax.plot(ring[:, 0], ring[:, 1], color="black", lw=0.9, zorder=6)
        thin_frame(ax, view)
        fig.text((bx + (box_w - w) / 2) / FIG_W_MM, 1 - (y2 - 1.0) / fig_h,
                 f"({chr(98+i)})", fontsize=FS["letter"], fontweight="bold",
                 va="bottom")
        bars[key] = scalebar(fig, rect(bx, y2 + box_h, box_w, 7.5),
                             box_w, w, view[1] - view[0])

    # legend + north arrow
    y_leg = y2 + box_h + sb_h
    if legend == "below":
        cw = usable * 0.58
        cax = fig.add_axes(rect(left, y_leg + 2.0, cw, 3.6))
        orient = "horizontal"
        north(fig, rect(left + cw + 14, y_leg + 0.5, 14.0, 10.5))
    else:
        cax = fig.add_axes(rect(left + panels_w + 5.0, y2 + box_h * 0.10,
                                4.5, box_h * 0.72))
        orient = "vertical"
        north(fig, rect(left + panels_w + 1.0, y2 + box_h * 0.86, 14.0, 10.5))
    cb = fig.colorbar(im, cax=cax, orientation=orient)
    cb.set_ticks(np.linspace(-VMAX, VMAX, 5))
    cb.ax.tick_params(labelsize=FS["cbar_tick"], length=2.5, width=0.5)
    cb.set_label("$\\Delta$NDVI", fontsize=FS["cbar_label"])
    cb.outline.set_linewidth(0.5)

    stem = OUT / f"fig6_ndvi_{ramp_key}_{legend}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return stem, fig_h, dims, bars


def main():
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })
    with rasterio.open(NDVI_TIF) as s:
        arr = s.read(1, masked=True)
        b = s.bounds
    ext = (b.left, b.right, b.bottom, b.top)
    rw, rl = ring_of(BND["watershed"]), ring_of(BND["lure"])

    smallest = min(FS.values())
    print(f"smallest label in the figure: {smallest:.1f} pt at "
          f"{FIG_W_MM:.0f} mm  ->  {'OK' if smallest >= 7 else 'BELOW 7 pt'}")
    print("greyscale limb separation (higher = sign survives greyscale):")
    for k, (n, d) in RAMPS.items():
        cm = lab_diverging() if n is None else plt.get_cmap(n)
        print(f"   {k:5s} {limb_sep(cm):.3f}   {d}")

    rows = []
    for k in RAMPS:
        for legend in ("below", "right"):
            stem, h, dims, bars = build(k, legend, arr, ext, rw, rl)
            rows.append((stem.name, h, bars))
            print(f"  {stem.name}.png/.pdf  {FIG_W_MM:.0f} x {h:.0f} mm  "
                  f"bars: watershed {bars['watershed']} m, "
                  f"Lake Lure {bars['lure']} m")
            log(f"{stem.name}: {FIG_W_MM:.0f}x{h:.0f} mm, legend={legend}, "
                f"scale bars {bars['watershed']}/{bars['lure']} m", run=RUN)
    ww = dims["watershed"][2]
    ll = dims["lure"][2]
    print(f"\nground widths: watershed {ww[1]-ww[0]:.0f} m, "
          f"Lake Lure {ll[1]-ll[0]:.0f} m  (map scales differ by "
          f"{(ll[1]-ll[0])/(ww[1]-ww[0]):.1f}x)")
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
