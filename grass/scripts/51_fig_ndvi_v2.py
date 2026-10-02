#!/usr/bin/env python3
"""
51_fig_ndvi_v2.py
==================
Rebuilt NDVI-change figure (manuscript Figure 6). The published version shows
Delta-NDVI over the whole corridor with no imagery context and no distinction
between the two analysis areas and the surrounding terrain.

Three layout variants, all 300 dpi + vector PDF:

  a  post-event orthophoto spanning the full width above (both analysis-area
     boundaries outlined), the two Delta-NDVI panels below at a common scale
  b  two columns of two -- orthophoto above Delta-NDVI for each site, all four
     panels at one common scale
  c  two full-width rows on ONE extent and scale: orthophoto above,
     Delta-NDVI below with terrain outside the two analysis areas faded and
     both boundaries outlined

A NOTE ON THE SINGLE SCALE BAR
    In (a) the context strip spans 5.4 km and cannot be drawn at the same
    scale as the analysis panels below it in a 174 mm column, so its scale
    differs from theirs. The scale bar there therefore belongs to the bottom
    row and the top strip is a locator. (b) and (c) hold every panel at one
    scale, so a single scale bar is unambiguous -- which is why (c) is the
    variant recommended if the reviewer's "one scale bar" is read strictly.

Colour scale: diverging, symmetric about zero, brown = vegetation loss,
green = gain, with limits set from the robust spread of Delta-NDVI inside
the two analysis areas (printed and logged, and quoted in the caption).

Outputs: results/diagnostics/fig_ndvi_v2_{a,b,c}.png / .pdf
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from PIL import Image
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log                                   # noqa: E402
from map_style import finalize_frame, add_scale_north_inline  # noqa: E402

RUN = "fig_ndvi_v2"
ROOT = HERE.parents[1]
FIG = ROOT / "results" / "figures"
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM = 174
CMAP = "BrBG"          # brown (loss) -> white -> green (gain)

NDVI_TIF = FIG / "_ndvi_change.tif"
CTX_PNG = FIG / "_ndvi_context.png"
BND = {"watershed": FIG / "_ndvi_bnd_watershed.geojson",
       "lure": FIG / "_ndvi_bnd_lure.geojson"}
TITLES = {"watershed": "tributary watershed", "lure": "Lake Lure reach"}
# context strip extent, as exported by script 50. The 2024 orthophoto does
# not reach the southern edge of that box, leaving a black no-data band; the
# extent is trimmed to the imagery's actual footprint at load time.
CTX_FULL = (312923.0, 318290.0, 190698.0, 193937.0)
CTX = CTX_FULL


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
    return out


def biggest(path):
    rs = rings(path)
    return max(rs, key=lambda r: len(r))


def bbox(ring, pad=40.0):
    return (ring[:, 0].min() - pad, ring[:, 0].max() + pad,
            ring[:, 1].min() - pad, ring[:, 1].max() + pad)


def load_ndvi():
    with rasterio.open(NDVI_TIF) as s:
        a = s.read(1, masked=True)
        b = s.bounds
    return a, (b.left, b.right, b.bottom, b.top)


def clip(artist, ax, ring):
    artist.set_clip_path(PathPatch(MplPath(ring), transform=ax.transData))


def frame(ax, ext):
    finalize_frame(ax, ext)
    ax.set_xticks([])
    ax.set_yticks([])


def panel_ndvi(ax, arr, ext, ring, vmax, view, faded=False):
    if faded:
        ax.imshow(arr, extent=ext, origin="upper", cmap=CMAP,
                  norm=Normalize(-vmax, vmax), alpha=0.30, zorder=2,
                  interpolation="nearest")
    im = ax.imshow(arr, extent=ext, origin="upper", cmap=CMAP,
                   norm=Normalize(-vmax, vmax), zorder=3,
                   interpolation="nearest")
    if ring is not None:
        clip(im, ax, ring)
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    return im


_CTX_CACHE = {}


def context_image():
    """Orthophoto strip with its black no-data margin trimmed off."""
    if "img" in _CTX_CACHE:
        return _CTX_CACHE["img"], _CTX_CACHE["ext"]
    img = np.asarray(Image.open(CTX_PNG).convert("RGB"))
    dark = img.sum(axis=2) < 30
    rows = np.where(dark.mean(axis=1) < 0.98)[0]
    cols = np.where(dark.mean(axis=0) < 0.98)[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    H, W = img.shape[:2]
    w0, e0, s0, n0 = CTX_FULL
    ext = (w0 + (e0 - w0) * c0 / W, w0 + (e0 - w0) * c1 / W,
           n0 - (n0 - s0) * r1 / H, n0 - (n0 - s0) * r0 / H)
    _CTX_CACHE["img"], _CTX_CACHE["ext"] = img[r0:r1, c0:c1], ext
    return _CTX_CACHE["img"], _CTX_CACHE["ext"]


def panel_ctx(ax, view, outlines=(), lw=1.0):
    img, iext = context_image()
    ax.imshow(img, extent=iext, origin="upper", zorder=1,
              interpolation="bilinear")
    for r in outlines:
        ax.plot(r[:, 0], r[:, 1], color="black", lw=lw, zorder=6)
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])


def colourbar(fig, rect, im, vmax, fig_w, fig_h):
    x, y, w, h = rect
    cax = fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks(np.linspace(-vmax, vmax, 5))
    cb.ax.tick_params(labelsize=6.2, length=2.5, width=0.5)
    cb.set_label("$\\Delta$NDVI (post $-$ pre)   brown = vegetation loss, "
                 "green = gain", fontsize=6.8)
    cb.outline.set_linewidth(0.5)
    return cax


def scale_group(fig, rect, mm_per_m, fig_w, fig_h):
    x, y, w, h = rect
    sax = fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
    scale_m = next((c for c in (2000, 1000, 500, 200, 100)
                    if c * mm_per_m <= 0.52 * w), 100)
    add_scale_north_inline(sax, scale_m=scale_m, mm_per_m=mm_per_m,
                           axes_w_mm=w, color="black", fontsize=7.5,
                           lw_scale=1.5)
    return scale_m


def save(fig, name):
    stem = DIAG / f"fig_ndvi_v2_{name}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return stem


# ------------------------------------------------------------------ (a)
def variant_a(arr, ext, rw, rl, vmax):
    vw, vl = bbox(rw), bbox(rl)
    left, right, gap, top, cap, legend, bot = 3.0, 3.0, 6.0, 4.0, 7.0, 13.0, 4.0
    usable = FIG_W_MM - left - right
    _, _iext = context_image()
    ctx_h = usable * (_iext[3] - _iext[2]) / (_iext[1] - _iext[0])
    # bottom row: one common scale for both analysis panels
    s = ((vw[1] - vw[0]) + (vl[1] - vl[0])) / (usable - gap)   # m per mm
    ww, lw_ = (vw[1] - vw[0]) / s, (vl[1] - vl[0]) / s
    wh, lh = (vw[3] - vw[2]) / s, (vl[3] - vl[2]) / s
    row_h = max(wh, lh)
    fig_h = top + ctx_h + cap + row_h + cap + legend + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def rect(x, ytop, w, h):
        return [x / FIG_W_MM, 1 - (ytop + h) / fig_h, w / FIG_W_MM, h / fig_h]

    _, iext = context_image()
    axc = fig.add_axes(rect(left, top, usable, ctx_h))
    panel_ctx(axc, iext, outlines=[rw, rl])
    frame(axc, iext)
    axc.set_title("Post-event orthophoto, both analysis areas outlined "
                  "(locator; scale differs from the panels below)",
                  fontsize=7.0, fontweight="bold", pad=2)

    y2 = top + ctx_h + cap
    im = None
    for i, (key, view, w_, h_) in enumerate(
            (("watershed", vw, ww, wh), ("lure", vl, lw_, lh))):
        x = left + (0 if i == 0 else ww + gap)
        ax = fig.add_axes(rect(x, y2, w_, h_))
        im = panel_ndvi(ax, arr, ext, {"watershed": rw, "lure": rl}[key],
                        vmax, view)
        ring = rw if key == "watershed" else rl
        ax.plot(ring[:, 0], ring[:, 1], color="black", lw=0.8, zorder=6)
        frame(ax, view)
        ax.set_title(f"({chr(97+i)}) $\\Delta$NDVI, {TITLES[key]}",
                     fontsize=7.2, fontweight="bold", pad=2,
                     loc="left" if i == 0 else "center")
    cb_w = usable * 0.62
    colourbar(fig, (left, bot + 4.0, cb_w, 3.6), im, vmax, FIG_W_MM, fig_h)
    sm = scale_group(fig, (left + cb_w + 8, bot, usable - cb_w - 8, 12.0),
                     1.0 / s, FIG_W_MM, fig_h)
    return save(fig, "a"), fig_h, sm


# ------------------------------------------------------------------ (b)
def variant_b(arr, ext, rw, rl, vmax):
    vw, vl = bbox(rw), bbox(rl)
    left, right, gap, top, cap, legend, bot = 3.0, 3.0, 6.0, 4.0, 7.0, 13.0, 4.0
    usable = FIG_W_MM - left - right
    s = ((vw[1] - vw[0]) + (vl[1] - vl[0])) / (usable - gap)
    ww, lw_ = (vw[1] - vw[0]) / s, (vl[1] - vl[0]) / s
    wh, lh = (vw[3] - vw[2]) / s, (vl[3] - vl[2]) / s
    row_h = max(wh, lh)
    fig_h = top + cap + row_h + cap + row_h + legend + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def rect(x, ytop, w, h):
        return [x / FIG_W_MM, 1 - (ytop + h) / fig_h, w / FIG_W_MM, h / fig_h]

    im = None
    for i, (key, view, w_, h_) in enumerate(
            (("watershed", vw, ww, wh), ("lure", vl, lw_, lh))):
        x = left + (0 if i == 0 else ww + gap)
        ring = rw if key == "watershed" else rl
        axo = fig.add_axes(rect(x, top + cap, w_, h_))
        panel_ctx(axo, view, outlines=[ring], lw=0.8)
        frame(axo, view)
        axo.set_title(f"({chr(97+i)}) {TITLES[key]}", fontsize=7.2,
                      fontweight="bold", pad=2)
        axn = fig.add_axes(rect(x, top + cap + row_h + cap, w_, h_))
        im = panel_ndvi(axn, arr, ext, ring, vmax, view)
        axn.plot(ring[:, 0], ring[:, 1], color="black", lw=0.8, zorder=6)
        frame(axn, view)
    fig.text(left / FIG_W_MM, 1 - (top + 2.0) / fig_h,
             "Top: post-event orthophoto    Bottom: $\\Delta$NDVI",
             fontsize=6.8, va="bottom")
    cb_w = usable * 0.62
    colourbar(fig, (left, bot + 4.0, cb_w, 3.6), im, vmax, FIG_W_MM, fig_h)
    sm = scale_group(fig, (left + cb_w + 8, bot, usable - cb_w - 8, 12.0),
                     1.0 / s, FIG_W_MM, fig_h)
    return save(fig, "b"), fig_h, sm


# ------------------------------------------------------------------ (c)
def _signed_area(r):
    return float(np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1]))


def variant_c(arr, ext, rw, rl, vmax):
    left, right, top, cap, legend, bot = 3.0, 3.0, 4.0, 7.0, 13.0, 4.0
    usable = FIG_W_MM - left - right
    _, iext = context_image()
    s = (iext[1] - iext[0]) / usable               # one scale for both rows
    row_h = (iext[3] - iext[2]) / s
    fig_h = top + cap + row_h + cap + row_h + legend + bot
    fig = plt.figure(figsize=(FIG_W_MM * MM, fig_h * MM))

    def rect(ytop, h):
        return [left / FIG_W_MM, 1 - (ytop + h) / fig_h, usable / FIG_W_MM,
                h / fig_h]

    axo = fig.add_axes(rect(top + cap, row_h))
    panel_ctx(axo, iext, outlines=[rw, rl])
    frame(axo, iext)
    axo.set_title("(a) Post-event orthophoto", fontsize=7.2,
                  fontweight="bold", pad=2)

    axn = fig.add_axes(rect(top + cap + row_h + cap, row_h))
    im = panel_ndvi(axn, arr, ext, None, vmax, iext)
    # A white scrim over everything OUTSIDE the two analysis areas, built as
    # one path with the two boundaries as holes. An alpha fade on the data
    # layer was tried first and read as barely different, because most of the
    # corridor is near zero and already pale.
    outer = np.array([[iext[0], iext[2]], [iext[1], iext[2]],
                      [iext[1], iext[3]], [iext[0], iext[3]],
                      [iext[0], iext[2]]])
    verts, codes = [outer], [[MplPath.MOVETO] + [MplPath.LINETO] * 3
                             + [MplPath.CLOSEPOLY]]
    for ring in (rw, rl):
        # holes must wind OPPOSITE to the outer rectangle (which is
        # counter-clockwise, positive area) or they fill instead of cut
        r = ring if _signed_area(ring) < 0 else ring[::-1]
        verts.append(r)
        codes.append([MplPath.MOVETO] + [MplPath.LINETO] * (len(r) - 2)
                     + [MplPath.CLOSEPOLY])
    axn.add_patch(PathPatch(MplPath(np.concatenate(verts),
                                    np.concatenate(codes)),
                            facecolor="white", alpha=0.66, edgecolor="none",
                            zorder=5))
    for ring in (rw, rl):
        axn.plot(ring[:, 0], ring[:, 1], color="black", lw=1.1, zorder=7)
    axn.set_xlim(iext[0], iext[1])
    axn.set_ylim(iext[2], iext[3])
    frame(axn, iext)
    axn.set_title("(b) $\\Delta$NDVI  --  analysis areas at full opacity, "
                  "surrounding corridor faded", fontsize=7.2,
                  fontweight="bold", pad=2)
    for ring, lab, dy in ((rw, "(i) tributary watershed", 60),
                          (rl, "(ii) Lake Lure reach", 60)):
        axn.text(ring[:, 0].mean(), ring[:, 1].max() + dy, lab, fontsize=6.4,
                 ha="center", va="bottom", fontweight="bold", zorder=9)
    cb_w = usable * 0.62
    colourbar(fig, (left, bot + 4.0, cb_w, 3.6), im, vmax, FIG_W_MM, fig_h)
    sm = scale_group(fig, (left + cb_w + 8, bot, usable - cb_w - 8, 12.0),
                     1.0 / s, FIG_W_MM, fig_h)
    return save(fig, "c"), fig_h, sm


def main():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    arr, ext = load_ndvi()
    rw, rl = biggest(BND["watershed"]), biggest(BND["lure"])

    # symmetric limits from the robust spread INSIDE the two analysis areas
    ny, nx = arr.shape
    xs = np.linspace(ext[0], ext[1], nx)
    ys = np.linspace(ext[3], ext[2], ny)
    X, Y = np.meshgrid(xs, ys)
    inside = np.zeros(arr.shape, bool)
    for r in (rw, rl):
        inside |= MplPath(r).contains_points(
            np.column_stack([X.ravel(), Y.ravel()])).reshape(arr.shape)
    v = np.asarray(arr.filled(np.nan))[inside]
    v = v[np.isfinite(v)]
    p98 = float(np.percentile(np.abs(v), 98))
    vmax = float(np.ceil(p98 * 10) / 10)
    print(f"Delta-NDVI inside the two analysis areas: n={v.size:,} "
          f"min={v.min():+.3f} max={v.max():+.3f} "
          f"98th pct |value|={p98:.3f} -> symmetric limits +/-{vmax:.1f}")
    log(f"colour limits +/-{vmax:.2f} (98th pct |dNDVI| inside areas "
        f"= {p98:.3f}; full range {v.min():+.3f} to {v.max():+.3f}, n={v.size})",
        run=RUN)

    for fn, name in ((variant_a, "a"), (variant_b, "b"), (variant_c, "c")):
        stem, h, sm = fn(arr, ext, rw, rl, vmax)
        print(f"  ({name}) {stem.name}.png  {FIG_W_MM} x {h:.0f} mm  "
              f"scale bar {sm} m")
        log(f"variant {name} -> {stem.name}.png, {FIG_W_MM}x{h:.0f} mm, "
            f"scale bar {sm} m", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
