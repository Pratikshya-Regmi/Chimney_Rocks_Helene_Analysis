#!/usr/bin/env python3
"""
52_fig_ndvi_v3.py
==================
fig_ndvi_v3 -- the v2_a layout refined on three points:

 1. several diverging colour ramps to choose between;
 2. no text straddling a map frame: every label is either fully outside the
    frame, or inside with a white halo behind it;
 3. SEPARATE scale bars. The two analysis areas differ in extent by about
    2x, so one shared bar was misleading. Each panel is drawn in an equally
    sized box (so the composition stays balanced) with its own scale bar and
    its own ground-width note, which is what makes the differing scales
    visible rather than hidden.

The Delta-NDVI colour scale is still shared between the two panels -- it is
the map scales that differ, not the data scale.

COLOUR RAMPS
    brbg     ColorBrewer BrBG, the conventional vegetation-change choice
    rdylgn   ColorBrewer RdYlGn, red = loss (NOT reversed: reversing it would
             put green on vegetation loss)
    piyg     ColorBrewer PiYG, the best greyscale separation of the three
    lab      constructed here: CIELAB lightness-linear brown -> near-white ->
             green with ASYMMETRIC limb lightness, so the sign survives a
             greyscale conversion

    Crameri's vik/broc were requested. cmcrameri is not installed, and the
    Crameri maps that matplotlib 3.11 ships (berlin, managua, vanimo) are all
    DARK-CENTRED: on this data, where most of the corridor is near zero, they
    render the unchanged majority as a dark mass. They were tested and
    rejected rather than shipped; `lab` is offered as the perceptually
    motivated option instead.

GREYSCALE
    Every symmetric diverging ramp is ambiguous in greyscale because both
    limbs darken away from the centre. Measured mean limb-luminance
    separation: BrBG 0.025, RdYlGn 0.082, PiYG 0.094, lab (asymmetric) is
    printed at run time. Only `lab` is designed to be unambiguous.

Outputs: results/diagnostics/fig_ndvi_v3_{brbg,rdylgn,piyg,lab}.png + .pdf
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from PIL import Image
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log                                    # noqa: E402
from map_style import finalize_frame                          # noqa: E402

RUN = "fig_ndvi_v3"
ROOT = HERE.parents[1]
FIG = ROOT / "results" / "figures"
DIAG = ROOT / "results" / "diagnostics"

MM = 1 / 25.4
FIG_W_MM = 174
NDVI_TIF = FIG / "_ndvi_change.tif"
CTX_PNG = FIG / "_ndvi_context.png"
CTX_FULL = (312923.0, 318290.0, 190698.0, 193937.0)
BND = {"watershed": FIG / "_ndvi_bnd_watershed.geojson",
       "lure": FIG / "_ndvi_bnd_lure.geojson"}
TITLES = {"watershed": "tributary watershed", "lure": "Lake Lure reach"}
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]


# ---------------------------------------------------------------- colour
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
    return np.array([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])]).T


def lab2rgb(lab):
    lab = np.atleast_2d(lab)
    fy = (lab[:, 0] + 16) / 116
    fx, fz = fy + lab[:, 1] / 500, fy - lab[:, 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WP
    return np.clip(_lin_to_srgb(xyz @ np.linalg.inv(_M).T), 0, 1)


def lab_diverging(n=256):
    """Brown -> near-white -> green, linear in CIELAB lightness, with the two
    limbs given DIFFERENT end lightness so the sign survives greyscale."""
    loss = rgb2lab(np.array([[0.32, 0.17, 0.02]]))[0]     # dark brown
    mid = rgb2lab(np.array([[0.97, 0.97, 0.95]]))[0]
    gain = rgb2lab(np.array([[0.30, 0.62, 0.27]]))[0]     # mid green, lighter
    half = n // 2
    a = np.linspace(0, 1, half)[:, None]
    b = np.linspace(0, 1, n - half)[:, None]
    cols = np.vstack([lab2rgb(loss + (mid - loss) * a),
                      lab2rgb(mid + (gain - mid) * b)])
    return LinearSegmentedColormap.from_list("lab_brown_green", cols)


def lum(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def limb_separation(cm):
    L = np.array([lum(cm(x)) for x in np.linspace(0, 1, 101)])
    return float(np.mean(np.abs(L[:50] - L[51:][::-1])))


RAMPS = {"brbg": ("BrBG", "ColorBrewer BrBG"),
         "rdylgn": ("RdYlGn", "ColorBrewer RdYlGn"),
         "piyg": ("PiYG", "ColorBrewer PiYG"),
         "lab": (None, "CIELAB lightness-linear brown-green")}


# ---------------------------------------------------------------- data
def rings(path):
    gj = json.loads(Path(path).read_text())
    out = []
    for f in gj.get("features", []):
        g = f.get("geometry") or {}
        polys = ([g["coordinates"]] if g.get("type") == "Polygon"
                 else g.get("coordinates", []))
        for poly in polys:
            for r in poly:
                a = np.asarray(r, float)
                if a.ndim == 2 and len(a) > 2:
                    out.append(a)
    return max(out, key=len)


def bbox(ring, pad=40.0):
    return (ring[:, 0].min() - pad, ring[:, 0].max() + pad,
            ring[:, 1].min() - pad, ring[:, 1].max() + pad)


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


def load_ndvi():
    with rasterio.open(NDVI_TIF) as s:
        a = s.read(1, masked=True)
        b = s.bounds
    return a, (b.left, b.right, b.bottom, b.top)


def frame(ax, ext):
    finalize_frame(ax, ext)
    ax.set_xticks([])
    ax.set_yticks([])


def scale_bar_below(fig, x_mm, y_mm, box_w_mm, map_w_mm, ground_w_m,
                    fig_w, fig_h, fontsize=6.6):
    """Scale bar drawn OUTSIDE the map frame, centred under the panel."""
    ax = fig.add_axes([x_mm / fig_w, y_mm / fig_h, box_w_mm / fig_w,
                       9.0 / fig_h])
    ax.set_xlim(0, box_w_mm)
    ax.set_ylim(0, 9)
    ax.set_axis_off()
    mm_per_m = map_w_mm / ground_w_m
    bar_m = next((c for c in (1000, 500, 250, 200, 100, 50)
                  if c * mm_per_m <= 0.55 * map_w_mm), 50)
    bar_mm = bar_m * mm_per_m
    x0 = box_w_mm / 2 - bar_mm / 2
    y = 5.4
    ax.plot([x0, x0 + bar_mm], [y, y], color="black", lw=2.2,
            solid_capstyle="butt")
    for xt in (x0, x0 + bar_mm):
        ax.plot([xt, xt], [y - 0.9, y + 0.9], color="black", lw=1.4)
    ax.text(box_w_mm / 2, y + 1.4, f"{bar_m:,.0f} m", ha="center",
            va="bottom", fontsize=fontsize)
    ax.text(box_w_mm / 2, y - 2.0, f"panel spans {ground_w_m:,.0f} m",
            ha="center", va="top", fontsize=fontsize - 0.8, style="italic")
    return bar_m, mm_per_m


def build(key, cmap, arr, ext, rw, rl, vmax):
    img, iext = context_image()
    left, right, gap = 3.0, 3.0, 8.0
    top, cap, sb_h, legend, bot = 3.0, 6.5, 9.5, 13.0, 3.0
    usable = FIG_W_MM - left - right
    ctx_h = usable * (iext[3] - iext[2]) / (iext[1] - iext[0])

    # equal boxes for the two analysis panels
    box_w = (usable - gap) / 2.0
    vw, vl = bbox(rw), bbox(rl)
    aw = (vw[3] - vw[2]) / (vw[1] - vw[0])
    al = (vl[3] - vl[2]) / (vl[1] - vl[0])
    box_h = box_w * 0.92                     # common box, maps fit inside
    dims = {}
    for name, view, asp in (("watershed", vw, aw), ("lure", vl, al)):
        if asp >= box_h / box_w:             # height-limited
            h = box_h
            w = h / asp
        else:                                # width-limited
            w = box_w
            h = w * asp
        dims[name] = (w, h, view)

    fig_h = top + ctx_h + cap + cap + box_h + sb_h + legend + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def rect(x, ytop, w, h):
        return [x / FIG_W_MM, 1 - (ytop + h) / fig_h, w / FIG_W_MM, h / fig_h]

    # ---- (a) context orthophoto, boundaries outlined -------------------
    axc = fig.add_axes(rect(left, top + cap, usable, ctx_h))
    axc.imshow(img, extent=iext, origin="upper", zorder=1,
               interpolation="bilinear")
    for ring, lab in ((rw, "(b)"), (rl, "(c)")):
        axc.plot(ring[:, 0], ring[:, 1], color="black", lw=1.2, zorder=6)
        axc.text(ring[:, 0].mean(), ring[:, 1].max() + 55, lab, fontsize=7.5,
                 fontweight="bold", ha="center", va="bottom", zorder=9,
                 path_effects=HALO)          # halo: sits over imagery
    axc.set_xlim(iext[0], iext[1])
    axc.set_ylim(iext[2], iext[3])
    frame(axc, iext)
    fig.text(left / FIG_W_MM, 1 - (top + cap - 1.2) / fig_h,
             "(a) Post-event orthophoto, both analysis areas outlined",
             fontsize=7.4, fontweight="bold", va="bottom")

    # ---- (b),(c) dNDVI, equal boxes, own scale bars --------------------
    y2 = top + ctx_h + cap + cap
    im = None
    for i, name in enumerate(("watershed", "lure")):
        w, h, view = dims[name]
        bx = left + i * (box_w + gap)
        ax = fig.add_axes(rect(bx + (box_w - w) / 2, y2 + (box_h - h) / 2, w, h))
        im = ax.imshow(arr, extent=ext, origin="upper", cmap=cmap,
                       norm=Normalize(-vmax, vmax), zorder=3,
                       interpolation="nearest")
        ring = rw if name == "watershed" else rl
        im.set_clip_path(PathPatch(MplPath(ring), transform=ax.transData))
        ax.plot(ring[:, 0], ring[:, 1], color="black", lw=0.9, zorder=6)
        ax.set_xlim(view[0], view[1])
        ax.set_ylim(view[2], view[3])
        frame(ax, view)
        fig.text((bx + box_w / 2) / FIG_W_MM, 1 - (y2 - 1.4) / fig_h,
                 f"({chr(98+i)}) $\\Delta$NDVI, {TITLES[name]}",
                 fontsize=7.4, fontweight="bold", ha="center", va="bottom")
        scale_bar_below(fig, bx, bot + legend, box_w, w, view[1] - view[0],
                        FIG_W_MM, fig_h)

    # ---- shared dNDVI legend + one north arrow -------------------------
    cb_w = usable * 0.60
    cax = fig.add_axes(rect(left, fig_h - bot - 8.0, cb_w, 3.6))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks(np.linspace(-vmax, vmax, 5))
    cb.ax.tick_params(labelsize=6.2, length=2.5, width=0.5)
    cb.set_label("$\\Delta$NDVI (post $-$ pre)   loss $\\leftarrow$   "
                 "$\\rightarrow$ gain", fontsize=6.8)
    cb.outline.set_linewidth(0.5)
    nax = fig.add_axes(rect(left + cb_w + 10, fig_h - bot - 12.0, 16.0, 12.0))
    nax.set_xlim(0, 1)
    nax.set_ylim(0, 1)
    nax.set_axis_off()
    nax.annotate("", xy=(0.5, 0.72), xytext=(0.5, 0.12),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6,
                                 mutation_scale=15))
    nax.text(0.5, 0.76, "N", ha="center", va="bottom", fontsize=8.5,
             fontweight="bold")

    stem = DIAG / f"fig_ndvi_v3_{key}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return stem, fig_h, dims


def main():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    arr, ext = load_ndvi()
    rw, rl = rings(BND["watershed"]), rings(BND["lure"])
    vmax = 0.70

    print("greyscale limb-luminance separation (higher = sign survives "
          "greyscale better):")
    for key, (name, desc) in RAMPS.items():
        cm = lab_diverging() if name is None else plt.get_cmap(name)
        print(f"   {key:7s} {desc:42s} {limb_separation(cm):.3f}")

    for key, (name, desc) in RAMPS.items():
        cm = lab_diverging() if name is None else plt.get_cmap(name)
        stem, h, dims = build(key, cm, arr, ext, rw, rl, vmax)
        ww = dims["watershed"][2][1] - dims["watershed"][2][0]
        lw_ = dims["lure"][2][1] - dims["lure"][2][0]
        print(f"  {stem.name}.png  {FIG_W_MM} x {h:.0f} mm   "
              f"watershed spans {ww:.0f} m, Lake Lure {lw_:.0f} m")
        log(f"{key}: {desc}; {FIG_W_MM}x{h:.0f} mm; panel ground widths "
            f"{ww:.0f} m and {lw_:.0f} m (separate scale bars)", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
