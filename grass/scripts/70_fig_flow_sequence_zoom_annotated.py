#!/usr/bin/env python3
"""
70_fig_flow_sequence_zoom_annotated.py
======================================
Annotated rebuild of the standalone zoom figure (fig_flow_sequence_zoom,
32_fig_flow_sequence_zoom.py). Everything 32 draws is drawn identically --
same zones and extents, the three epoch columns (2020 lidar / 2024 CAP SfM /
2024 lidar), the common log discharge scale (vmin=1e-4, vmax=0.48, saturated
above), graticule, per-row scale bar, one north arrow -- and one overlay
layer is added on top of every panel, with a single shared key:

  old    dashed vermillion outline  2020 lidar flow course with no 2024
                                    lidar channel within 2 m
  new    solid black outline        2024 lidar flow course with no 2020
                                    lidar channel within 2 m
  split  open ring                  Z1 only: where the single 2020 channel
                                    divides into the new braided pair

Course definitions are those of 56_fig_flow_consolidated.py (r.thin
skeletons of the two lidar epochs at the discharge > 0.01 threshold used for
the Jaccard metric, 2 m test), so the annotation matches the analysis. The
SfM epoch takes no part in defining them; the same outlines are drawn over
it so it can be compared against the lidar courses.

DISPLAY THRESHOLD: only connected stretches of at least MIN_ANNOT_CELLS
(30 m of course) are drawn. 56 keeps stretches of >= 20 cells; at 20 the
zoom panels also carry eight 20-27 m stubs in Z1 and Z2 that are not part
of the change described in the text. At 30 the surviving stretches are
exactly: Z1 one old channel + the new braided pair; Z2 the abandoned 2020
strand and tributary + the new concentrated course; Z3 the old corridor +
the new corridor position.

OUTLINES, NOT LINES: each course is drawn as the contour BUFFER_M from its
centreline, so the channel stays visible inside the outline instead of being
painted over. Every mark carries a thin white casing so it reads on both the
pale and the dark end of the discharge ramp.

COLOUR: vermillion vs black, checked with the Machado et al. (2009)
severity-1.0 CVD simulation: old-vs-new OKLab dE x100 = 56-66 under protan,
deutan and tritan (target >= 8). Dashed vs solid carries the distinction a
second time, so it survives greyscale print.

Requires 30_flow_sequence_setup.py's GeoTIFF exports and the
__ovl_<epoch>_thin.tif skeletons used by 56. Run from the repository root so
results/ resolves to the canonical tree.

Outputs:
  results/figures/fig_flow_sequence_zoom_annotated.pdf / .png
  results/tables/flow_zoom_annotation_courses.csv   every old/new stretch
      >= 10 cells per zone, its size and whether it is drawn; the split point
  results/tables/flow_zoom_annotation_cvd.csv       old-vs-new colour
      separation under normal, protan, deutan and tritan vision
  results/diagnostics/flow_zoom_annotation_components.png   the stretches
      behind the display threshold, labelled by size
  results/logs/flow_sequence_zoom_annotated_<date>.log
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402
from map_style import style_map_panel  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 32 supplies the figure as it stands; 56 supplies the course definitions.
# 56 sets rcParams at import -- keep them out of this figure so it renders
# exactly as 32 does.
z32 = _load("fz32", "32_fig_flow_sequence_zoom.py")
with matplotlib.rc_context():
    m = _load("fc56", "56_fig_flow_consolidated.py")

MM = z32.MM
FIG_W_MM = z32.FIG_W_MM
VMIN, VMAX = z32.VMIN, z32.VMAX
COLUMNS = z32.COLUMNS
read_window = z32.read_window

C_OLD = "#D55E00"         # vermillion -- 2020 course gone by 2024
C_NEW = "#000000"         # black -- 2024 course absent in 2020
C_SPLIT = "#000000"
CASING = "white"

MIN_ANNOT_CELLS = 30      # shortest course stretch drawn, 1 m cells
BUFFER_M = 2.5            # outline offset from the course centreline
UPSAMPLE = 5              # sub-cells per 1 m cell for a smooth outline
SPLIT_ZONES = {"Z1"}      # the one bifurcation the text describes
LW = 0.8                  # outline width, pt
LW_CASE = 2.0             # white casing width, pt
DASH = (0, (3.2, 1.6))
SPLIT_MS = 8.5


def zone_changes(ext, box):
    """Old and new course masks, and the split point, for one zone.

    Same tests as 56.zone_annotation -- old = 2020 channel with no 2024
    channel within 2 m, new = the mirror -- but returned as cell masks so
    they can be outlined, and kept only in stretches of >= MIN_ANNOT_CELLS.

    The split point is 56's: the cell carrying channel in both epochs, inside
    the zone box, that is simultaneously closest to an old and a new stretch,
    accepted only if both distances are <= 3 m."""
    m20, e = m._mask_window(m.TIF["thin2020"], ext)
    m24, _ = m._mask_window(m.TIF["thin2024"], ext)
    d24 = ndi.distance_transform_edt(~m24)
    d20 = ndi.distance_transform_edt(~m20)
    st = np.ones((3, 3), bool)

    def keep(mask):
        lab, n = ndi.label(mask, structure=st)
        sizes = np.bincount(lab.ravel())
        big = np.where(sizes >= MIN_ANNOT_CELLS)[0]
        big = big[big > 0]
        return np.isin(lab, big), sizes[big].tolist()

    old_all = m20 & (d24 > 2.0)
    new_all = m24 & (d20 > 2.0)
    old, old_sizes = keep(old_all)
    new, new_sizes = keep(new_all)

    # every stretch of >= 10 cells, drawn or not, for the record
    res = (e[1] - e[0]) / m20.shape[1]
    stretches = []
    for kind, mask in (("old", old_all), ("new", new_all)):
        lab, n = ndi.label(mask, structure=st)
        sizes = np.bincount(lab.ravel())
        for k in range(1, n + 1):
            if sizes[k] < 10:
                continue
            r, c = np.nonzero(lab == k)
            stretches.append(dict(
                kind=kind, n_cells=int(sizes[k]),
                centroid_e=round(e[0] + (c.mean() + 0.5) * res, 1),
                centroid_n=round(e[3] - (r.mean() + 0.5) * res, 1),
                drawn=bool(sizes[k] >= MIN_ANNOT_CELLS)))

    split = None
    shared = m20 & (d24 <= 1.5)
    if old.any() and new.any():
        do = ndi.distance_transform_edt(~old)
        dn = ndi.distance_transform_edt(~new)
        rr, cc = np.mgrid[0:m20.shape[0], 0:m20.shape[1]]
        xx = e[0] + (cc + 0.5) * res
        yy = e[3] - (rr + 0.5) * res
        inbox = (shared & (xx >= box[0]) & (xx <= box[1]) &
                 (yy >= box[2]) & (yy <= box[3]))
        if inbox.any():
            score = np.where(inbox, np.maximum(do, dn), np.inf)
            r, c = np.unravel_index(int(np.argmin(score)), score.shape)
            if score[r, c] <= 3.0:
                split = (float(xx[r, c]), float(yy[r, c]))
    return dict(old=old, new=new, old_all=old_all, new_all=new_all, e=e,
                split=split, old_sizes=old_sizes, new_sizes=new_sizes,
                stretches=stretches)


# ---- colour check: Machado, Oliveira & Fernandes (2009), severity 1.0,
# OKLab delta E x100 -- the same model and scale as the dataviz validator
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def _lin(hex_color):
    c = np.array(matplotlib.colors.to_rgb(hex_color))
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _oklab(lin):
    lms = np.cbrt(np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                            [0.2119034982, 0.6806995451, 0.1073969566],
                            [0.0883024619, 0.2817188376, 0.6299787005]]) @ lin)
    return np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                     [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]]) @ lms


def delta_e(c1, c2, vision="normal"):
    a, b = _lin(c1), _lin(c2)
    if vision != "normal":
        a = np.clip(np.array(MACHADO[vision]) @ a, 0, 1)
        b = np.clip(np.array(MACHADO[vision]) @ b, 0, 1)
    return float(100 * np.linalg.norm(_oklab(a) - _oklab(b)))


def write_diagnostic(zones, changes, path):
    """Every old/new stretch >= 10 cells on the 2020 and 2024 lidar panels,
    labelled by size, faint where it falls below MIN_ANNOT_CELLS -- the
    evidence behind the display threshold."""
    fig, axs = plt.subplots(len(zones), 2, figsize=(12, 5.4 * len(zones)))
    for row, (_, z) in enumerate(zones.iterrows()):
        ch = changes[z["zone_id"]]
        e = ch["e"]
        ext = (e[0], e[1], e[2], e[3])
        for col, key in enumerate(("2020", "2024")):
            ax = axs[row, col]
            m.draw_discharge(ax, key, ext)
            for kind, mask, color in (("old", ch["old_all"], C_OLD),
                                      ("new", ch["new_all"], C_NEW)):
                lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
                sizes = np.bincount(lab.ravel())
                for k in range(1, n + 1):
                    if sizes[k] < 10:
                        continue
                    r, c = np.nonzero(lab == k)
                    xs, ys = e[0] + c + 0.5, e[3] - r - 0.5
                    drawn = sizes[k] >= MIN_ANNOT_CELLS
                    ax.plot(xs, ys, "s", ms=2.2, color=color,
                            alpha=0.9 if drawn else 0.35)
                    ax.text(xs.mean(), ys.mean(), f"{kind} {sizes[k]}",
                            fontsize=8, color=color, weight="bold",
                            bbox=dict(fc="white", ec="none", alpha=0.7,
                                      pad=0.5))
            ax.add_patch(plt.Rectangle((z["e_min"], z["n_min"]), z["width_m"],
                                       z["height_m"], fill=False, ec="red",
                                       lw=1))
            m.bare(ax, ext)
            ax.set_title(f"{z['zone_id']} {key} lidar -- stretches >= 10 "
                         f"cells; drawn if >= {MIN_ANNOT_CELLS} (solid)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def draw_outline(ax, mask, e, color, dashed):
    """Contour BUFFER_M from the course centreline. Distances are measured
    from the centre of each skeleton cell on a UPSAMPLE-times finer grid, so
    the outline is a smooth offset curve rather than a staircase."""
    if not mask.any():
        return
    u = UPSAMPLE
    fine = np.zeros((mask.shape[0] * u, mask.shape[1] * u), bool)
    r, c = np.nonzero(mask)
    fine[r * u + u // 2, c * u + u // 2] = True
    d = ndi.distance_transform_edt(~fine, sampling=1.0 / u)
    res = (e[1] - e[0]) / fine.shape[1]
    xs = e[0] + (np.arange(fine.shape[1]) + 0.5) * res
    ys = e[3] - (np.arange(fine.shape[0]) + 0.5) * res
    cs = ax.contour(xs, ys[::-1], d[::-1], levels=[BUFFER_M], colors=[color],
                    linewidths=LW, zorder=7)
    if dashed:
        cs.set_linestyle(DASH)
    cs.set_path_effects([pe.withStroke(linewidth=LW_CASE, foreground=CASING)])


def draw_split(ax, pt):
    ax.plot([pt[0]], [pt[1]], ls="none", marker="o", ms=SPLIT_MS, mfc="none",
            mec=C_SPLIT, mew=1.0, zorder=9,
            path_effects=[pe.withStroke(linewidth=2.6, foreground=CASING)])


def key_handles():
    case = [pe.withStroke(linewidth=LW_CASE, foreground=CASING)]
    return [
        Patch(facecolor="none", edgecolor=C_OLD, linewidth=LW, linestyle="--",
              path_effects=case, label="old flow path (2020, gone by 2024)"),
        Patch(facecolor="none", edgecolor=C_NEW, linewidth=LW,
              path_effects=case, label="new flow path (absent in 2020)"),
        Line2D([], [], ls="none", marker="o", ms=SPLIT_MS - 2, mfc="none",
               mec=C_SPLIT, mew=1.0, label="split point (Z1)",
               path_effects=[pe.withStroke(linewidth=2.6,
                                           foreground=CASING)]),
    ]


def main():
    if not z32.DISCHARGE_2020_TIF.exists():
        sys.exit("Run 30_flow_sequence_setup.py first.")
    zones = pd.read_csv(z32.TABLES / "flow_change_zones.csv")

    n_zones = len(zones)
    fig = plt.figure(figsize=(FIG_W_MM * MM, (n_zones * 62 + 14) * MM))
    gs_fig = fig.add_gridspec(n_zones + 1, 3,
                              height_ratios=[1] * n_zones + [0.06],
                              hspace=0.38, wspace=0.10)

    im = None
    summary = []
    changes = {}
    rows_out = []
    for row, (_, z) in enumerate(zones.iterrows()):
        e_min, e_max = z["e_min"], z["e_max"]
        n_min, n_max = z["n_min"], z["n_max"]
        hs, hs_extent = read_window(z32.HILLSHADE_TIF, e_min, e_max,
                                    n_min, n_max)
        # courses are found over the padded window so the 2 m test is not
        # truncated at the panel edge; the panel itself clips them
        pad_ext = (e_min - m.PAD_M, e_max + m.PAD_M,
                   n_min - m.PAD_M, n_max + m.PAD_M)
        ch = zone_changes(pad_ext, (e_min, e_max, n_min, n_max))
        split = ch["split"] if z["zone_id"] in SPLIT_ZONES else None
        changes[z["zone_id"]] = ch
        rows_out += [dict(zone_id=z["zone_id"], **s) for s in ch["stretches"]]
        if split is not None:
            rows_out.append(dict(zone_id=z["zone_id"], kind="split", n_cells=1,
                                 centroid_e=round(split[0], 1),
                                 centroid_n=round(split[1], 1), drawn=True))
        summary.append(f"{z['zone_id']}: old {ch['old_sizes']} new "
                       f"{ch['new_sizes']} cells, split "
                       + (f"E{split[0]:.1f} N{split[1]:.1f}" if split
                          else "not drawn"))

        for col, (col_label, tif) in enumerate(COLUMNS):
            ax = fig.add_subplot(gs_fig[row, col])
            arr, extent = read_window(tif, e_min, e_max, n_min, n_max)

            ax.imshow(hs, cmap="gray", extent=hs_extent, origin="upper",
                      zorder=1)
            arr_clipped = np.clip(arr, VMIN, VMAX)
            im = ax.imshow(arr_clipped, cmap="YlGnBu",
                           norm=LogNorm(vmin=VMIN, vmax=VMAX), extent=extent,
                           origin="upper", alpha=0.88, zorder=2)

            # overlay only: identical on every panel of the row
            draw_outline(ax, ch["old"], ch["e"], C_OLD, dashed=True)
            draw_outline(ax, ch["new"], ch["e"], C_NEW, dashed=False)
            if split is not None:
                draw_split(ax, split)

            style_map_panel(ax, extent, graticule_step=50,
                            scale_color="white",
                            draw_scale=(col == len(COLUMNS) - 1),
                            draw_north=(row == 0 and col == len(COLUMNS) - 1))

            if row == 0:
                ax.text(0.5, 1.15, col_label, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=6.8,
                        fontweight="bold", clip_on=False, zorder=21)
            if col == 0:
                ax.text(-0.38, 0.5, z["zone_id"], transform=ax.transAxes,
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="red", clip_on=False,
                        zorder=21, rotation=90)

    fig.subplots_adjust(top=0.90)
    cax = fig.add_subplot(gs_fig[n_zones, :])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Unit discharge [log scale, m$^3$ s$^{-1}$ "
                   "m$^{-1}$-equivalent] -- values above the scale maximum "
                   "are saturated", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2.5, width=0.5)

    # one shared key, once, directly under the colour bar's label
    fig.canvas.draw()
    lab_bb = cbar.ax.xaxis.label.get_window_extent().transformed(
        fig.transFigure.inverted())
    cb_pos = cax.get_position()
    fig.legend(handles=key_handles(), loc="upper center", ncol=3,
               bbox_to_anchor=((cb_pos.x0 + cb_pos.x1) / 2, lab_bb.y0 - 0.006),
               frameon=False, fontsize=7, handlelength=2.4, handleheight=0.9,
               columnspacing=1.8, handletextpad=0.6, borderaxespad=0.0)

    out_pdf = z32.FIGURES / "fig_flow_sequence_zoom_annotated.pdf"
    out_png = z32.FIGURES / "fig_flow_sequence_zoom_annotated.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf} and .png")
    for s in summary:
        print("  " + s)

    courses_csv = z32.TABLES / "flow_zoom_annotation_courses.csv"
    pd.DataFrame(rows_out).to_csv(courses_csv, index=False)
    cvd = pd.DataFrame([dict(pair=f"old {C_OLD} vs new {C_NEW}", vision=v,
                             delta_e_oklab_x100=round(delta_e(C_OLD, C_NEW, v),
                                                      1))
                        for v in ("normal", "protan", "deutan", "tritan")])
    cvd_csv = z32.TABLES / "flow_zoom_annotation_cvd.csv"
    cvd.to_csv(cvd_csv, index=False)
    diag_png = (z32.FIGURES.parent / "diagnostics" /
                "flow_zoom_annotation_components.png")
    write_diagnostic(zones, changes, diag_png)
    print(f"wrote {courses_csv.name}, {cvd_csv.name}, {diag_png.name}")
    summary.append("old-vs-new dE (normal/protan/deutan/tritan) "
                   + "/".join(f"{x:.1f}" for x in cvd.delta_e_oklab_x100))

    log("70_fig_flow_sequence_zoom_annotated.py run. Rebuild of 32's zoom "
        "grid with a change overlay on every panel: old (2020 course, no 2024 "
        "channel within 2 m) dashed vermillion outline, new (mirror) solid "
        f"black outline, split ring in {sorted(SPLIT_ZONES)} only; stretches "
        f">= {MIN_ANNOT_CELLS} cells, outline {BUFFER_M} m from centreline. "
        f"Discharge scale unchanged (vmin={VMIN}, vmax={VMAX}). "
        + "; ".join(summary) + f". Written to {out_pdf.name}/.png, "
        f"{courses_csv.name}, {cvd_csv.name}, {diag_png.name}.",
        run="flow_sequence_zoom_annotated")


if __name__ == "__main__":
    main()
