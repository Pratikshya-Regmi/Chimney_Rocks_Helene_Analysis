#!/usr/bin/env python3
"""
54_fig6_ndvi_final.py
======================
Figure 6 (NDVI change) as a standard publication map for Natural Hazards.
Figure 6 only; writes to its own folder and overwrites nothing.

LAYOUTS
  L1 "stack"  (a) full-width post-event orthophoto, areas outlined+labelled
              (b) full-width Delta-NDVI on the SAME extent and scale, the two
                  analysis areas at full opacity, the rest under a white scrim
              -> two panels, letters (a)/(b), and because both rows share one
                 extent a SINGLE scale bar is correct.

  L2 "split"  (a) full-width orthophoto
              (b),(c) Delta-NDVI per site at their own map scales, equal boxes
              -> three panels, and two scale bars are then required.

RAMPS
  brbg    ColorBrewer BrBG. Colourblind-safe.
  rdylgn  ColorBrewer RdYlGn. Produced because it was named, but ColorBrewer
          classifies it as NOT colourblind-safe (red/green is the deuteranopia
          failure case). Flagged, not silently shipped.
  lab     CIELAB lightness-linear brown->green built here, asymmetric limb
          lightness so the sign survives greyscale.

Area labels sit OUTSIDE the map frame with leader lines: the orthophoto
carries imagery to every edge, so there is no clear area inside it and an
in-panel label would sit on data.

Outputs -> results/figures/fig6_ndvi_final/fig6_<ramp>_<layout>.png/.pdf
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

RUN = "fig6_final"
ROOT = HERE.parents[1]
FIG = ROOT / "results" / "figures"
OUT = FIG / "fig6_ndvi_final"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM = 174.0
VMAX = 0.70
FS = dict(letter=8.0, area=7.5, tick=7.0, cbar=7.5, scale=7.0, north=8.5)

NDVI_TIF = FIG / "_ndvi_change.tif"
CTX_PNG = FIG / "_ndvi_context.png"
CTX_FULL = (312923.0, 318290.0, 190698.0, 193937.0)
BND = {"watershed": FIG / "_ndvi_bnd_watershed.geojson",
       "lure": FIG / "_ndvi_bnd_lure.geojson"}
AREA_LABEL = {"watershed": "tributary watershed", "lure": "Lake Lure reach"}


# ------------------------------------------------------------- colour
def _s2l(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _l2s(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
               [0.0193, 0.1192, 0.9505]])
_WP = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb):
    xyz = _s2l(rgb) @ _M.T / _WP
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])]).T


def lab2rgb(lab):
    lab = np.atleast_2d(lab)
    fy = (lab[:, 0] + 16) / 116
    f = np.stack([fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200], axis=-1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WP
    return np.clip(_l2s(xyz @ np.linalg.inv(_M).T), 0, 1)


def lab_diverging(n=256):
    lo = rgb2lab(np.array([[0.32, 0.17, 0.02]]))[0]
    mid = rgb2lab(np.array([[0.97, 0.97, 0.95]]))[0]
    hi = rgb2lab(np.array([[0.30, 0.62, 0.27]]))[0]
    a = np.linspace(0, 1, n // 2)[:, None]
    b = np.linspace(0, 1, n - n // 2)[:, None]
    return LinearSegmentedColormap.from_list(
        "lab_brown_green", np.vstack([lab2rgb(lo + (mid - lo) * a),
                                      lab2rgb(mid + (hi - mid) * b)]))


def limb_sep(cm):
    L = np.array([0.2126 * cm(x)[0] + 0.7152 * cm(x)[1] + 0.0722 * cm(x)[2]
                  for x in np.linspace(0, 1, 101)])
    return float(np.mean(np.abs(L[:50] - L[51:][::-1])))


RAMPS = {"brbg": ("BrBG", "ColorBrewer BrBG (colourblind-safe)"),
         "rdylgn": ("RdYlGn", "ColorBrewer RdYlGn (NOT colourblind-safe)"),
         "lab": (None, "CIELAB brown-green (greyscale-safe)")}


# ------------------------------------------------------------- data
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


def bbox(r, pad=40.0):
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


def signed_area(r):
    return float(np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1]))


def thin_frame(ax, ext):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.7); s.set_color("black")


def scalebar(fig, rect, box_w, map_w_mm, ground_m):
    ax = fig.add_axes(rect); ax.set_xlim(0, box_w); ax.set_ylim(0, 7.5)
    ax.set_axis_off()
    mpm = map_w_mm / ground_m
    bar = next((c for c in (2000, 1000, 500, 250, 200, 100)
                if c * mpm <= 0.5 * map_w_mm), 100)
    bw = bar * mpm
    x0, y = box_w / 2 - bw / 2, 2.6
    ax.plot([x0, x0 + bw], [y, y], color="black", lw=2.0, solid_capstyle="butt")
    for xt in (x0, x0 + bw):
        ax.plot([xt, xt], [y - 0.85, y + 0.85], color="black", lw=1.3)
    ax.text(box_w / 2, y + 1.3, f"{bar:,.0f} m", ha="center", va="bottom",
            fontsize=FS["scale"])
    return bar


def north(fig, rect):
    ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.annotate("", xy=(0.5, 0.62), xytext=(0.5, 0.05),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6,
                                mutation_scale=14))
    ax.text(0.5, 0.66, "N", ha="center", va="bottom", fontsize=FS["north"],
            fontweight="bold")


def area_labels(fig, ax, rings_, iext, y_label_mm, fig_h, x0_mm, w_mm):
    """Labels ABOVE the frame with leader lines: the orthophoto has imagery to
    every edge, so an in-panel label would sit on data."""
    for key, r in rings_.items():
        cx = r[:, 0].mean()
        fx = x0_mm + w_mm * (cx - iext[0]) / (iext[1] - iext[0])
        fig.text(fx / FIG_W_MM, 1 - (y_label_mm - 0.8) / fig_h,
                 AREA_LABEL[key], fontsize=FS["area"], ha="center",
                 va="bottom")
        ax.annotate("", xy=(cx, r[:, 1].max()),
                    xytext=(cx, iext[3] + (iext[3] - iext[2]) * 0.055),
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.7))


def build(ramp_key, layout, arr, ext, rw, rl):
    name, _ = RAMPS[ramp_key]
    cmap = lab_diverging() if name is None else plt.get_cmap(name)
    img, iext = context_image()
    rings_ = {"watershed": rw, "lure": rl}

    left = right = 4.0
    top, lab_h, letter_h, sb_h, legend_h, bot = 3.0, 5.5, 5.0, 7.5, 12.5, 3.0
    usable = FIG_W_MM - left - right
    ctx_h = usable * (iext[3] - iext[2]) / (iext[1] - iext[0])

    if layout == "stack":
        fig_h = (top + lab_h + letter_h + ctx_h + letter_h + ctx_h + sb_h
                 + legend_h + bot)
    else:
        gap = 7.0
        box_w = (usable - gap) / 2.0
        box_h = box_w * 0.92
        dims = {}
        for k, v in (("watershed", bbox(rw)), ("lure", bbox(rl))):
            asp = (v[3] - v[2]) / (v[1] - v[0])
            if asp >= box_h / box_w:
                h = box_h; w = h / asp
            else:
                w = box_w; h = w * asp
            dims[k] = (w, h, v)
        fig_h = (top + lab_h + letter_h + ctx_h + letter_h + box_h + sb_h
                 + legend_h + bot)

    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def rect(x, ytop, w, h):
        return [x / FIG_W_MM, 1 - (ytop + h) / fig_h, w / FIG_W_MM, h / fig_h]

    # ---- (a) orthophoto -------------------------------------------------
    y_a = top + lab_h + letter_h
    axa = fig.add_axes(rect(left, y_a, usable, ctx_h))
    axa.imshow(img, extent=iext, origin="upper", zorder=1,
               interpolation="bilinear")
    for r in (rw, rl):
        axa.plot(r[:, 0], r[:, 1], color="black", lw=1.2, zorder=6)
    thin_frame(axa, iext)
    fig.text(left / FIG_W_MM, 1 - (y_a - 1.0) / fig_h, "(a)",
             fontsize=FS["letter"], fontweight="bold", va="bottom")
    area_labels(fig, axa, rings_, iext, top + lab_h, fig_h, left, usable)

    im = None
    if layout == "stack":
        y_b = y_a + ctx_h + letter_h
        axb = fig.add_axes(rect(left, y_b, usable, ctx_h))
        im = axb.imshow(arr, extent=ext, origin="upper", cmap=cmap,
                        norm=Normalize(-VMAX, VMAX), zorder=3,
                        interpolation="nearest")
        outer = np.array([[iext[0], iext[2]], [iext[1], iext[2]],
                          [iext[1], iext[3]], [iext[0], iext[3]],
                          [iext[0], iext[2]]])
        verts = [outer]
        codes = [[MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]]
        for r in (rw, rl):
            rr = r if signed_area(r) < 0 else r[::-1]
            verts.append(rr)
            codes.append([MplPath.MOVETO] + [MplPath.LINETO] * (len(rr) - 2)
                         + [MplPath.CLOSEPOLY])
        axb.add_patch(PathPatch(MplPath(np.concatenate(verts),
                                        np.concatenate(codes)),
                                facecolor="white", alpha=0.68,
                                edgecolor="none", zorder=5))
        for r in (rw, rl):
            axb.plot(r[:, 0], r[:, 1], color="black", lw=1.2, zorder=7)
        thin_frame(axb, iext)
        fig.text(left / FIG_W_MM, 1 - (y_b - 1.0) / fig_h, "(b)",
                 fontsize=FS["letter"], fontweight="bold", va="bottom")
        bars = {"shared": scalebar(fig, rect(left, y_b + ctx_h, usable, 7.5),
                                   usable, usable, iext[1] - iext[0])}
        y_leg = y_b + ctx_h + sb_h
    else:
        y_b = y_a + ctx_h + letter_h
        bars = {}
        for i, k in enumerate(("watershed", "lure")):
            w, h, view = dims[k]
            bx = left + i * (box_w + gap)
            ax = fig.add_axes(rect(bx + (box_w - w) / 2,
                                   y_b + (box_h - h) / 2, w, h))
            im = ax.imshow(arr, extent=ext, origin="upper", cmap=cmap,
                           norm=Normalize(-VMAX, VMAX), zorder=3,
                           interpolation="nearest")
            r = rings_[k]
            im.set_clip_path(PathPatch(MplPath(r), transform=ax.transData))
            ax.plot(r[:, 0], r[:, 1], color="black", lw=0.9, zorder=6)
            thin_frame(ax, view)
            fig.text((bx + (box_w - w) / 2) / FIG_W_MM,
                     1 - (y_b - 1.0) / fig_h, f"({chr(98+i)})",
                     fontsize=FS["letter"], fontweight="bold", va="bottom")
            bars[k] = scalebar(fig, rect(bx, y_b + box_h, box_w, 7.5),
                               box_w, w, view[1] - view[0])
        y_leg = y_b + box_h + sb_h

    # ---- shared legend + north arrow ------------------------------------
    cw = usable * 0.56
    cax = fig.add_axes(rect(left, y_leg + 2.5, cw, 3.6))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks(np.linspace(-VMAX, VMAX, 5))
    cb.ax.tick_params(labelsize=FS["tick"], length=2.5, width=0.5)
    cb.set_label("$\\Delta$NDVI", fontsize=FS["cbar"])
    cb.outline.set_linewidth(0.5)
    north(fig, rect(left + cw + 16, y_leg + 0.5, 14.0, 11.0))

    stem = OUT / f"fig6_{ramp_key}_{layout}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return stem, fig_h, bars


def main():
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                                "font.family": "sans-serif",
                                "font.sans-serif": ["DejaVu Sans"]})
    with rasterio.open(NDVI_TIF) as s:
        arr = s.read(1, masked=True); b = s.bounds
    ext = (b.left, b.right, b.bottom, b.top)
    rw, rl = ring_of(BND["watershed"]), ring_of(BND["lure"])

    print(f"smallest label {min(FS.values()):.1f} pt at {FIG_W_MM:.0f} mm -> "
          f"{'OK' if min(FS.values()) >= 7 else 'TOO SMALL'}")
    for k, (n, d) in RAMPS.items():
        cm = lab_diverging() if n is None else plt.get_cmap(n)
        print(f"   {k:7s} greyscale separation {limb_sep(cm):.3f}   {d}")
    for k in RAMPS:
        for layout in ("stack", "split"):
            stem, h, bars = build(k, layout, arr, ext, rw, rl)
            bs = ", ".join(f"{a}={v} m" for a, v in bars.items())
            print(f"  {stem.name}  {FIG_W_MM:.0f} x {h:.0f} mm   bars: {bs}")
            log(f"{stem.name}: {FIG_W_MM:.0f}x{h:.0f} mm, layout={layout}, "
                f"bars {bs}", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
