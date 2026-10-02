#!/usr/bin/env python3
"""
49_fig_profile_zoom_options.py
===============================
Replacement options for the A--B profile figures (before / after vertical
bias correction). The published pair plots the whole 1,317 m transect over a
75 m elevation range, at which scale a 3--4 m vertical bias is invisible.
These options zoom to a short window chosen so the correction is obvious.

WINDOW CHOICE (not by eye)
    Every 250 m window along the transect was scored on three criteria:
    ground stable between the lidar epochs (median |2024-2020| < 0.25 m),
    modest relief so a tight y-axis can resolve a few metres, and the size of
    the correction actually applied. The window 125--375 m wins: relief
    5.9 m, median |2024-2020| = 0.08 m, and the SfM surface moves from
    +3.83 m to +0.11 m relative to the 2020 lidar. Full scoring is printed
    when this script runs.

THE 2024 LIDAR "BEFORE" SURFACE
    dtm_2024_filled is delivered on a different vertical datum: it sits
    31.11 m below the 2020 surface (the NAVD88 / NAD83-ellipsoid geoid
    separation). Plotted raw it would leave the axes entirely. It is shown
    here with that single constant removed and is labelled as such, so what
    remains on the "before" panel is the co-registration and bias residual
    rather than the datum.

SURFACES
    before : dtm_2017, dtm_2020_filled, dtm_2024_filled (datum-shifted),
             cap_dtm_2024_filled
    after  : dtm_2017, lidar_2020_regis, lidar_2024_regis, cap_sfm_regis

OPTIONS WRITTEN
    A  side-by-side before | after, shared y-axis
    B  stacked before / after, shared x-axis
    C  elevation on top, residual against the 2020 reference below
       (before dashed, after solid) -- puts the correction in metres
    D  residual-only, before vs after, the most compact statement

Outputs: results/diagnostics/fig_profile_zoom_{A,B,C,D}.png (300 dpi) + .pdf
         results/tables/profile_AB_window_scores.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402

RUN = "profile_zoom"
ROOT = HERE.parents[1]
TABLES = ROOT / "results" / "tables"
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FIG_W_MM = 174
W0, W1 = 125.0, 375.0          # chosen window (see docstring)

# Okabe-Ito, colour-blind safe, one colour per epoch across every option
C = {"2017": "#0072B2", "2020": "#E69F00", "2024": "#009E73", "sfm": "#D55E00"}
LBL = {"2017": "2017 lidar", "2020": "2020 lidar (reference)",
       "2024": "2024 lidar", "sfm": "2024 CAP SfM"}


def load():
    df = pd.read_csv(TABLES / "profile_AB_surfaces.csv")
    datum = float((df.e20 - df.e24).median())
    df["e24s"] = df.e24 + datum          # constant datum shift only
    before = {"2017": "e17", "2020": "e20", "2024": "e24s", "sfm": "esfm"}
    after = {"2017": "e17", "2020": "r20", "2024": "r24", "sfm": "rsfm"}
    return df, before, after, datum


def score_windows(df, width=250.0, step=25.0):
    rows = []
    for s in np.arange(0, df.dist.max() - width, step):
        w = df[(df.dist >= s) & (df.dist < s + width)]
        if len(w) < width * 0.8:
            continue
        rows.append(dict(start=s, end=s + width,
                         relief_m=float(w.e20.max() - w.e20.min()),
                         median_abs_2024_2020=float((w.r24 - w.r20).abs().median()),
                         sfm_before_m=float((w.esfm - w.e20).median()),
                         sfm_after_m=float((w.rsfm - w.r20).median())))
    out = pd.DataFrame(rows)
    out["improvement_m"] = out.sfm_before_m.abs() - out.sfm_after_m.abs()
    out.to_csv(TABLES / "profile_AB_window_scores.csv", index=False)
    return out


def sub(df, cols):
    w = df[(df.dist >= W0) & (df.dist <= W1)]
    return w, {k: w[v].to_numpy() for k, v in cols.items()}, w.dist.to_numpy()


def style(ax, fs=7):
    ax.tick_params(labelsize=fs - 0.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def save(fig, name):
    stem = DIAG / f"fig_profile_zoom_{name}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return stem


def opt_A(df, before, after):
    w, B, d = sub(df, before)
    _, A, _ = sub(df, after)
    lo = min(min(v.min() for v in B.values()), min(v.min() for v in A.values()))
    hi = max(max(v.max() for v in B.values()), max(v.max() for v in A.values()))
    pad = 0.06 * (hi - lo)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_MM * MM, 68 * MM), sharey=True)
    for ax, dat, t in ((axes[0], B, "(a) Before vertical bias correction"),
                       (axes[1], A, "(b) After co-registration and bias correction")):
        for k in ("2017", "2020", "2024", "sfm"):
            ax.plot(d, dat[k], color=C[k], lw=1.1, label=LBL[k])
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Distance along A--B (m)", fontsize=7)
        ax.set_title(t, fontsize=7.5, fontweight="bold")
        style(ax)
    axes[0].set_ylabel("Elevation (m)", fontsize=7)
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    return save(fig, "A")


def opt_B(df, before, after):
    w, B, d = sub(df, before)
    _, A, _ = sub(df, after)
    lo = min(min(v.min() for v in B.values()), min(v.min() for v in A.values()))
    hi = max(max(v.max() for v in B.values()), max(v.max() for v in A.values()))
    pad = 0.06 * (hi - lo)
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W_MM * MM, 105 * MM), sharex=True)
    for ax, dat, t in ((axes[0], B, "(a) Before vertical bias correction"),
                       (axes[1], A, "(b) After co-registration and bias correction")):
        for k in ("2017", "2020", "2024", "sfm"):
            ax.plot(d, dat[k], color=C[k], lw=1.1, label=LBL[k])
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel("Elevation (m)", fontsize=7)
        ax.set_title(t, fontsize=7.5, fontweight="bold", loc="left")
        style(ax)
    axes[1].set_xlabel("Distance along A--B (m)", fontsize=7)
    axes[0].legend(fontsize=6, frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    return save(fig, "B")


def opt_C(df, before, after):
    w, B, d = sub(df, before)
    _, A, _ = sub(df, after)
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W_MM * MM, 105 * MM), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.15, 1]))
    for k in ("2017", "2020", "2024", "sfm"):
        axes[0].plot(d, A[k], color=C[k], lw=1.1, label=LBL[k])
    axes[0].set_ylabel("Elevation (m)", fontsize=7)
    axes[0].set_title("(a) Corrected surfaces along A--B", fontsize=7.5,
                      fontweight="bold", loc="left")
    axes[0].legend(fontsize=6, frameon=False, ncol=4, loc="upper left")
    style(axes[0])
    for k in ("2017", "2024", "sfm"):
        axes[1].plot(d, B[k] - B["2020"], color=C[k], lw=0.9, ls="--", alpha=0.75)
        axes[1].plot(d, A[k] - A["2020"], color=C[k], lw=1.2)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("Difference from 2020 (m)", fontsize=7)
    axes[1].set_xlabel("Distance along A--B (m)", fontsize=7)
    axes[1].set_title("(b) Residual against the 2020 reference: "
                      "dashed = before, solid = after", fontsize=7.5,
                      fontweight="bold", loc="left")
    style(axes[1])
    fig.tight_layout()
    return save(fig, "C")


def opt_D(df, before, after):
    w, B, d = sub(df, before)
    _, A, _ = sub(df, after)
    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM, 62 * MM))
    for k in ("2017", "2024", "sfm"):
        ax.plot(d, B[k] - B["2020"], color=C[k], lw=1.0, ls="--", alpha=0.8,
                label=f"{LBL[k]} — before")
        ax.plot(d, A[k] - A["2020"], color=C[k], lw=1.4,
                label=f"{LBL[k]} — after")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(-0.32, 0.32, color="0.6", alpha=0.28, lw=0,
               label="$\\pm$0.32 m detection limit")
    ax.set_ylabel("Difference from 2020 reference (m)", fontsize=7)
    ax.set_xlabel("Distance along A--B (m)", fontsize=7)
    ax.legend(fontsize=5.8, frameon=False, ncol=2, loc="upper left")
    style(ax)
    fig.tight_layout()
    return save(fig, "D")


def main():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    df, before, after, datum = load()
    sc = score_windows(df)
    sel = sc[(sc.median_abs_2024_2020 < 0.25) & (sc.relief_m < 25)] \
        .sort_values("improvement_m", ascending=False)
    print(f"2024 lidar datum offset removed for the 'before' panel: "
          f"{datum:.2f} m\n")
    print("top-scoring 250 m windows (stable ground, modest relief):")
    for r in sel.head(5).itertuples():
        print(f"   {r.start:6.0f}-{r.end:<6.0f} relief {r.relief_m:5.1f} m  "
              f"|2024-2020| {r.median_abs_2024_2020:.3f} m  "
              f"SfM {r.sfm_before_m:+5.2f} -> {r.sfm_after_m:+5.2f} m")
    print(f"\nchosen window: {W0:.0f}-{W1:.0f} m")
    log(f"window {W0:.0f}-{W1:.0f} m; 2024 datum offset {datum:.2f} m removed "
        f"for the before panel", run=RUN)
    for fn, name in ((opt_A, "A"), (opt_B, "B"), (opt_C, "C"), (opt_D, "D")):
        stem = fn(df, before, after)
        print(f"  wrote {stem.name}.png/.pdf")
        log(f"option {name} -> {stem.name}.png", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
