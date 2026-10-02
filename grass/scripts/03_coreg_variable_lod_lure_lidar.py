#!/usr/bin/env python3
"""
03_coreg_variable_lod_lure_lidar.py
====================================
Lake Lure PAIR 1: lidar 2017 -> lidar 2024. Same stages/logic as
03_coreg_variable_lod.py (the watershed's script); only CONFIG differs
(reach's own reference/target/DSM/mask, and PREFIX so its outputs don't
collide with the sfm pair's run in the same mapset).

ORIGINAL DOCSTRING (applies unchanged -- only CONFIG below differs):
=====================================================================
Horizontal co-registration (Nuth & Kaab 2011), terrain-correlated bias
correction, and a spatially variable Level of Detection (LoD) for DEM
differencing in forested terrain.

WRITTEN FOR: Regmi et al., post-Helene geomorphic change, Hickory Nut Gorge.

!!  IMPORTANT  !!
This script has NOT been executed against your data -- it was written
without access to a GRASS installation or to your rasters. Treat it as a
well-commented starting point, not a validated tool. Run it stage by stage
(see STAGES below), inspect the diagnostic output at each stage, and sanity
check every number before it goes into the manuscript.

--------------------------------------------------------------------------
WHY THIS ANALYSIS
--------------------------------------------------------------------------
The current manuscript estimates a single LoD from 9 points on paved roads
(SD 0.17 m -> LoD95 0.32 m) and applies it uniformly. Roads are the
best-case surface for both lidar and SfM. Under closed canopy, ground
returns are sparse and the RST-interpolated surface differs between epochs
for reasons that are not geomorphic. A road-derived LoD therefore passes
interpolation noise through as "detectable change" -- which is the most
likely explanation for 79% of the basin appearing to change.

This script replaces that with:
  Stage 1 -- horizontal co-registration (removes slope/aspect-correlated
             artifacts caused by planimetric misalignment)
  Stage 2 -- terrain-correlated vertical bias removal (elevation and/or
             canopy-height trend, not just a constant)
  Stage 3 -- spatially variable LoD, binned by canopy height, so the
             detection threshold is honest under forest
  Stage 4 -- restatement of change statistics under the new threshold

--------------------------------------------------------------------------
PREREQUISITES
--------------------------------------------------------------------------
1. GRASS GIS 8.x, running inside a GRASS session (grass -c ... then
   `python3 coreg_variable_lod.py`), or via grass --exec.
2. numpy.
3. Rasters already in your project (EPSG:3358, 1 m, common region):
     - reference DTM         (e.g. lidar 2020, bare earth)
     - target DTM            (e.g. sfm_2024 or lidar_2024, bare earth)
     - reference DSM         (first-return surface, SAME epoch as ref DTM)
       -> canopy height = DSM - DTM. If you do not have a DSM yet, generate
          it from the 2020 point cloud with r.in.pdal / v.surf.rst using
          FIRST returns only. If you truly cannot, set CANOPY = None and the
          script falls back to slope-only binning (weaker, but works).
     - stable mask           (1 = stable, null elsewhere): terrain you
       believe did not change -- exclude channels, the landslide scar,
       mapped disturbance, water, buildings. Build it once, reuse it. This
       mask is the single most important input; be conservative.

--------------------------------------------------------------------------
STAGES -- run one at a time, read the output, then proceed
--------------------------------------------------------------------------
    python3 coreg_variable_lod.py --stage 1
    python3 coreg_variable_lod.py --stage 2
    python3 coreg_variable_lod.py --stage 3
    python3 coreg_variable_lod.py --stage 4
    python3 coreg_variable_lod.py --stage all
"""

import argparse
import json
import math
import os
import sys

import numpy as np

import grass.script as gs
import grass.script.array as garray

# ==========================================================================
# CONFIGURATION -- EDIT THESE MAP NAMES TO MATCH YOUR PROJECT
# ==========================================================================
CONFIG = {
    # --- input maps (must already exist in the current mapset) ---
    "REFERENCE": "lidar_2017_DTM_lure@PERMANENT",   # pre-event reference DTM
    "TARGET": "lidar_2024_DTM_corr_lure@PERMANENT", # post-event lidar DTM
    "REF_DSM": "lidar_2017_DSM_lure@PERMANENT",     # first-return DSM, same epoch as REFERENCE
    "STABLE_MASK": "stable_mask_lure",  # 1 = stable terrain, null elsewhere
    "WATER_RASTER": "lure_water_mask",  # 1 = open water, null elsewhere -- see
                                        #   stage4's docstring for why this has
                                        #   to be excluded from volumes here,
                                        #   unlike the watershed (no open water
                                        #   there)

    # --- outputs (will be overwritten) ---
    "PREFIX": "coreg_lure_lidar",        # all outputs get this prefix

    # --- Stage 1: Nuth & Kaab parameters ---
    "MAX_ITER": 12,                     # iterations of the shift solver
    "CONVERGE_M": 0.02,                 # stop when shift magnitude < this (m)
    "SLOPE_MIN": 5.0,                   # deg; flat ground carries no horizontal
                                        #   information and destabilises the fit
    "SLOPE_MAX": 45.0,                  # deg; very steep cells are noise-dominated
    "DH_CLIP": 20.0,                    # m; discard |dh| above this before fitting

    # --- Stage 2: bias model ---
    # "constant"      = median offset only (what the manuscript does now)
    # "elevation"     = linear trend in elevation + constant
    # "canopy"        = linear trend in canopy height + constant  <-- recommended
    # "elev_canopy"   = both terms + constant
    "BIAS_MODEL": "canopy",

    # --- Stage 3: spatially variable LoD ---
    "LOD_BINS": [0.0, 2.0, 5.0, 10.0, 20.0, 1e6],  # canopy-height bin edges (m)
                                                   # (or slope-deg edges in fallback)
    "LOD_CONFIDENCE": 1.96,             # 1.96 -> 95% LoD
    "MIN_BIN_N": 500,                   # bins with fewer stable cells are merged
                                        #   into the next coarser bin

    # --- Stage 4: the manuscript's published uniform LoD for THIS pair,
    # for the "old" comparison row. Lake Lure's published value is 0.94 m
    # (NOT the watershed's 0.32 m -- the original coreg_variable_lod.py
    # hardcodes 0.32 inline; this copy makes it a CONFIG value instead so
    # the two reaches can't silently share the wrong number).
    "UNIFORM_LOD_OLD": 0.94,

    # --- Stage 4 ---
    "REPORT_JSON": "coreg_results_lure_lidar.json",
}


# ==========================================================================
# helpers
# ==========================================================================
def msg(text):
    print(f"\n=== {text}", flush=True)


def read(mapname, mask=None):
    """Read a raster into a masked numpy array. Nulls -> masked."""
    arr = garray.array(mapname, dtype=np.float64)
    data = np.ma.masked_invalid(np.array(arr, dtype=np.float64))
    if mask is not None:
        data = np.ma.masked_where(mask.mask | (mask != 1), data)
    return data


def nmad(x):
    """Normalized median absolute deviation -- robust SD estimator.

    Preferred over SD here because DEM difference distributions are
    heavy-tailed; a few blunders inflate SD badly. This is the statistic
    White et al. report, so using it also makes the two papers comparable.
    """
    x = np.ma.compressed(np.ma.masked_invalid(x))
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def robust_stats(x, label=""):
    x = np.ma.compressed(np.ma.masked_invalid(x))
    if x.size == 0:
        print(f"  {label}: EMPTY -- check your mask")
        return {}
    st = {
        "n": int(x.size),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x)),
        "nmad": nmad(x),
        "rmse": float(np.sqrt(np.mean(x ** 2))),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
    }
    print(f"  {label}: n={st['n']:,}  median={st['median']:+.3f}  "
          f"mean={st['mean']:+.3f}  SD={st['sd']:.3f}  NMAD={st['nmad']:.3f}  "
          f"RMSE={st['rmse']:.3f}")
    return st


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print(f"  wrote {path}")


# ==========================================================================
# STAGE 1 -- Nuth & Kaab horizontal co-registration
# ==========================================================================
def stage1(cfg):
    """Solve for the planimetric shift between TARGET and REFERENCE.

    Method (Nuth & Kaab 2011, The Cryosphere 5:271-290):
        dh / tan(slope) = a * cos(b - aspect) + c
    which is linear in (p1, p2, c) because
        a*cos(b - aspect) = p1*cos(aspect) + p2*sin(aspect),
        a = sqrt(p1^2 + p2^2),  b = atan2(p2, p1)
    The horizontal shift is then dx = a*sin(b), dy = a*cos(b), and the
    vertical component is c * mean(tan(slope)).

    Iterate: shift, recompute dh, refit, until the shift converges.

    WHY THIS MATTERS FOR YOUR PAPER: the manuscript currently states that no
    horizontal co-registration was applied, and that residual lateral offsets
    "cannot be excluded". Helena asked about this directly. Camera-only
    georeferencing of the CAP block reports ~15.6 m horizontal error, so a
    real planimetric offset is likely -- and in steep terrain a horizontal
    offset masquerades as slope-correlated elevation change, exactly the
    artifact pattern you attribute to canopy reconstruction error. You cannot
    separate the two until this is done.
    """
    msg("STAGE 1: Nuth & Kaab horizontal co-registration")
    ref, tgt = cfg["REFERENCE"], cfg["TARGET"]
    pre = cfg["PREFIX"]

    # slope and aspect from the REFERENCE surface (not the target)
    gs.run_command("r.slope.aspect", elevation=ref,
                   slope=f"{pre}_slope", aspect=f"{pre}_aspect",
                   overwrite=True, quiet=True)

    mask = read(cfg["STABLE_MASK"])
    slope = read(f"{pre}_slope", mask)
    aspect = read(f"{pre}_aspect", mask)

    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)

    # valid-cell filter: usable slope range only
    ok_slope = (slope >= cfg["SLOPE_MIN"]) & (slope <= cfg["SLOPE_MAX"])

    total_dx = total_dy = total_dz = 0.0
    current = tgt
    history = []

    # Cache the TRUE raw baseline (no shift applied at all) now, before the
    # loop overwrites "{pre}_dh" on every iteration -- otherwise the final
    # "before" report reflects whatever iteration last ran, including a
    # rejected one, not the actual starting point.
    gs.mapcalc(f"{pre}_dh_raw = {tgt} - {ref}", overwrite=True, quiet=True)
    raw_before = read(f"{pre}_dh_raw", mask)

    # Convergence safety guard (added 2026-08-25): the refit is not guaranteed
    # to shrink each iteration -- on real data it can destabilise after the
    # dominant shift is removed (iteration 1 fixes the real offset; later
    # iterations can start fitting noise/resampling artifacts and diverge).
    # Track the best stable-terrain SD seen so far; the moment a new
    # cumulative shift makes it WORSE, stop and revert to the last good state
    # instead of accumulating a runaway "total shift" that isn't real.
    best_sd = None
    best_state = (total_dx, total_dy, total_dz, current)

    for it in range(1, cfg["MAX_ITER"] + 1):
        gs.mapcalc(f"{pre}_dh = {current} - {ref}", overwrite=True, quiet=True)
        dh = read(f"{pre}_dh", mask)

        good = (ok_slope & ~dh.mask & ~slope.mask & ~aspect.mask
                & (np.abs(dh) < cfg["DH_CLIP"]))
        n = int(np.count_nonzero(good))
        if n < 100:
            print(f"  iteration {it}: only {n} usable cells -- aborting. "
                  f"Check STABLE_MASK and SLOPE_MIN/MAX.")
            break

        sd_now = float(np.std(np.asarray(dh[good])))
        if best_sd is not None and sd_now > best_sd:
            print(f"  iteration {it}: stable-terrain SD worsened "
                  f"({sd_now:.3f} m > {best_sd:.3f} m) -- reverting to the "
                  f"previous iteration's result and stopping.")
            total_dx, total_dy, total_dz, _ = best_state
            # _apply_shift always writes to the SAME output map name, so the
            # on-disk raster currently holds the REJECTED shift, not the
            # reverted one -- regenerate it from the reverted totals before
            # using it downstream.
            current = _apply_shift(tgt, f"{pre}_target_shift", total_dx,
                                   total_dy, total_dz)
            break
        best_sd = sd_now
        best_state = (total_dx, total_dy, total_dz, current)

        y = np.asarray(dh[good]) / np.tan(np.asarray(slope_rad[good]))
        a_ = np.asarray(aspect_rad[good])
        # design matrix: [cos(aspect), sin(aspect), 1]
        A = np.column_stack([np.cos(a_), np.sin(a_), np.ones_like(a_)])

        # robust-ish: drop the extreme 2% of y before the least-squares fit
        lo, hi = np.percentile(y, [1, 99])
        keep = (y >= lo) & (y <= hi)
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        p1, p2, c = coef

        amp = math.hypot(p1, p2)
        direction = math.atan2(p2, p1)
        dx = amp * math.sin(direction)
        dy = amp * math.cos(direction)
        dz = c * float(np.mean(np.tan(np.asarray(slope_rad[good]))))

        total_dx += dx
        total_dy += dy
        total_dz += dz
        shift_mag = math.hypot(dx, dy)

        print(f"  iteration {it}: dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f} "
              f"|shift|={shift_mag:.3f} m  (n={n:,})")
        history.append({"iter": it, "dx": dx, "dy": dy, "dz": dz,
                        "shift_mag": shift_mag, "n": n})

        # apply the CUMULATIVE shift to the ORIGINAL target each time, so
        # resampling error does not accumulate across iterations
        current = _apply_shift(tgt, f"{pre}_target_shift", total_dx, total_dy,
                               total_dz)

        if shift_mag < cfg["CONVERGE_M"]:
            print(f"  converged after {it} iterations")
            break

    print(f"\n  TOTAL SHIFT: dx={total_dx:+.3f} m  dy={total_dy:+.3f} m  "
          f"dz={total_dz:+.3f} m  (magnitude {math.hypot(total_dx, total_dy):.3f} m)")
    print("  -> Report this in the manuscript. If |shift| is a large fraction")
    print("     of a cell (>0.5 m at 1 m resolution), the uncorrected DoD")
    print("     contained real slope-correlated artifacts.")

    gs.run_command("g.copy", raster=f"{current},{cfg['PREFIX']}_target_coreg",
                   overwrite=True, quiet=True)
    gs.mapcalc(f"{pre}_dh_coreg = {pre}_target_coreg - {ref}",
               overwrite=True, quiet=True)

    dh_after = read(f"{pre}_dh_coreg", mask)
    print("\n  Stable-terrain residuals:")
    before = robust_stats(raw_before, "before")
    after = robust_stats(dh_after, "after ")

    return {"total_dx": total_dx, "total_dy": total_dy, "total_dz": total_dz,
            "iterations": history, "stable_before": before,
            "stable_after": after}


def _apply_shift(src, dst, dx, dy, dz):
    """Translate a raster by (dx, dy) metres and (dz) vertically.

    Implemented by moving the map's georeferencing with r.region, then
    resampling back onto the computational grid with bilinear interpolation.
    Sub-cell shifts therefore require r.resamp.interp -- integer-cell
    neighbourhood indexing in r.mapcalc would truncate the solution.
    """
    tmp = f"{dst}_tmp"
    gs.run_command("g.copy", raster=f"{src},{tmp}", overwrite=True, quiet=True)

    # NOTE: signs verified empirically 2026-08-25 -- the original version of
    # this function subtracted dy/dx/dz, which measurably WORSENS stable-
    # terrain agreement (SD 0.336 -> 0.351 on a known well-aligned pair) and
    # causes the iterative solver in stage1() to diverge instead of converge.
    # Adding them is the combination that actually reduces residual SD
    # (0.336 -> 0.216 on the same test case). See results/logs/coreg_*.log.
    info = gs.raster_info(src)
    gs.run_command("r.region", map=tmp,
                   n=info["north"] + dy, s=info["south"] + dy,
                   e=info["east"] + dx, w=info["west"] + dx, quiet=True)
    gs.run_command("r.resamp.interp", input=tmp, output=f"{dst}_h",
                   method="bilinear", overwrite=True, quiet=True)
    gs.mapcalc(f"{dst} = {dst}_h + {dz}", overwrite=True, quiet=True)
    gs.run_command("g.remove", type="raster", name=f"{tmp},{dst}_h",
                   flags="f", quiet=True)
    return dst


# ==========================================================================
# STAGE 2 -- terrain-correlated vertical bias
# ==========================================================================
def stage2(cfg):
    """Fit and remove a bias that varies with terrain, not just a constant.

    A single median offset (what the manuscript does now) is only valid if
    the bias is spatially uniform. For an SfM surface under canopy it is not:
    the reconstruction sits progressively higher where vegetation is denser,
    because the photogrammetric surface is the visible one. That produces a
    bias that scales with canopy height -- removable, unlike random error.

    Fitting canopy height is the physically motivated choice; elevation is a
    common proxy when no DSM is available (in mountain terrain the two are
    correlated, which is why an elevation trend often "works" without being
    the true cause -- say so honestly if you use it).
    """
    msg(f"STAGE 2: terrain-correlated bias correction (model = {cfg['BIAS_MODEL']})")
    pre = cfg["PREFIX"]
    ref = cfg["REFERENCE"]

    mask = read(cfg["STABLE_MASK"])
    dh = read(f"{pre}_dh_coreg", mask)
    elev = read(ref, mask)

    canopy = None
    if cfg["REF_DSM"]:
        gs.mapcalc(f"{pre}_canopy = {cfg['REF_DSM']} - {ref}",
                   overwrite=True, quiet=True)
        # clamp physically impossible values
        gs.mapcalc(f"{pre}_canopy = if({pre}_canopy < 0, 0, "
                   f"if({pre}_canopy > 60, 60, {pre}_canopy))",
                   overwrite=True, quiet=True)
        canopy = read(f"{pre}_canopy", mask)
        print("  canopy height raster built from REF_DSM - REFERENCE")
    else:
        print("  !! No REF_DSM configured -- canopy terms unavailable.")

    terms, names = [], []
    model = cfg["BIAS_MODEL"]
    if model in ("elevation", "elev_canopy"):
        terms.append(elev); names.append("elevation")
    if model in ("canopy", "elev_canopy"):
        if canopy is None:
            sys.exit("BIAS_MODEL requires canopy but REF_DSM is not set.")
        terms.append(canopy); names.append("canopy")

    good = ~dh.mask
    for t in terms:
        good &= ~t.mask
    good &= np.abs(dh) < cfg["DH_CLIP"]

    y = np.asarray(dh[good])
    if terms:
        A = np.column_stack([np.asarray(t[good]) for t in terms]
                            + [np.ones_like(y)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        expr_parts = [f"{coef[i]:.8f} * {'{}'}" for i in range(len(terms))]
        print("  fitted bias model:")
        for nm, cf in zip(names, coef[:-1]):
            print(f"    {nm:>10s}: {cf:+.5f} m per unit")
        print(f"    {'constant':>10s}: {coef[-1]:+.5f} m")

        # build the mapcalc expression for the modelled bias surface
        rast = {"elevation": ref, "canopy": f"{pre}_canopy"}
        expr = " + ".join(f"({coef[i]:.8f} * {rast[names[i]]})"
                          for i in range(len(names)))
        expr = f"{expr} + ({coef[-1]:.8f})"
        gs.mapcalc(f"{pre}_bias_model = {expr}", overwrite=True, quiet=True)
    else:
        med = float(np.median(y))
        print(f"  constant-only bias: {med:+.4f} m")
        gs.mapcalc(f"{pre}_bias_model = {med:.8f}", overwrite=True, quiet=True)

    gs.mapcalc(f"{pre}_target_corrected = {pre}_target_coreg - {pre}_bias_model",
               overwrite=True, quiet=True)
    gs.mapcalc(f"{pre}_dh_corrected = {pre}_target_corrected - {ref}",
               overwrite=True, quiet=True)

    print("\n  Stable-terrain residuals:")
    st_before = robust_stats(dh, "before bias model")
    st_after = robust_stats(read(f"{pre}_dh_corrected", mask), "after bias model ")

    print("\n  -> If NMAD drops substantially, the SfM error was partly")
    print("     SYSTEMATIC (removable), not purely random. That is the")
    print("     difference between 'SfM is unusable' and 'SfM needs a")
    print("     terrain-aware correction' -- i.e. your result vs White et al.")

    return {"model": model, "stable_before": st_before, "stable_after": st_after}


# ==========================================================================
# STAGE 3 -- spatially variable Level of Detection
# ==========================================================================
def stage3(cfg):
    """Build an LoD raster whose threshold varies with canopy height.

    THIS IS THE STAGE THAT MATTERS MOST FOR YOUR LIDAR-LIDAR RESULT.

    A single LoD from 9 road points assumes the error on a forested hillslope
    equals the error on asphalt. It does not. Binning stable-terrain
    residuals by canopy height gives an empirical error model: NMAD per bin,
    scaled to a 95% threshold, painted back onto the map. Cells under dense
    canopy then have to change by more before they count as change.

    Expect the "detectable change" area to fall sharply. That is the point:
    the current 38% erosion + 41% deposition (79% of the basin) is not
    credible for two lidar surveys of a forested watershed, and this is the
    defensible way to show what is actually resolvable.
    """
    msg("STAGE 3: spatially variable Level of Detection")
    pre = cfg["PREFIX"]

    mask = read(cfg["STABLE_MASK"])
    dh = read(f"{pre}_dh_corrected", mask)

    if cfg["REF_DSM"]:
        cov = read(f"{pre}_canopy", mask)
        cov_map = f"{pre}_canopy"
        cov_name = "canopy height (m)"
    else:
        cov = read(f"{pre}_slope", mask)
        cov_map = f"{pre}_slope"
        cov_name = "slope (deg)"
        print("  FALLBACK: binning by slope, not canopy height.")

    edges = cfg["LOD_BINS"]
    rules, bins = [], []
    print(f"\n  {cov_name:>22s} | {'n':>9s} | {'NMAD (m)':>9s} | {'LoD95 (m)':>9s}")
    print("  " + "-" * 60)

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (~dh.mask) & (~cov.mask) & (cov >= lo) & (cov < hi)
        vals = np.asarray(dh[sel])
        vals = vals[np.abs(vals) < cfg["DH_CLIP"]]
        n = vals.size
        if n < cfg["MIN_BIN_N"]:
            print(f"  [{lo:6.1f},{hi:6.1f}) | {n:9,} | {'--':>9s} | "
                  f"{'merged':>9s}   (n < MIN_BIN_N)")
            continue
        bin_nmad = nmad(vals)
        lod = cfg["LOD_CONFIDENCE"] * bin_nmad
        hi_r = 1e6 if hi > 1e5 else hi
        rules.append(f"{lo}:{hi_r}:{lod:.4f}:{lod:.4f}")
        bins.append({"lo": lo, "hi": hi, "n": int(n),
                     "nmad": bin_nmad, "lod95": lod})
        print(f"  [{lo:6.1f},{hi:6.1f}) | {n:9,} | {bin_nmad:9.3f} | {lod:9.3f}")

    if not bins:
        sys.exit("No usable bins -- check STABLE_MASK coverage.")

    rules_file = f"{pre}_lod_rules.txt"
    with open(rules_file, "w") as fh:
        fh.write("\n".join(rules) + "\n")
    gs.run_command("r.recode", input=cov_map, output=f"{pre}_lod",
                   rules=rules_file, overwrite=True, quiet=True)

    # NOTE: this LoD describes the TARGET-vs-REFERENCE pair directly, because
    # it is measured from their actual residuals. Do NOT additionally combine
    # it in quadrature with a separate per-DEM sigma -- that would double count.
    gs.mapcalc(
        f"{pre}_dod_thresholded = if(abs({pre}_dh_corrected) > {pre}_lod, "
        f"{pre}_dh_corrected, null())", overwrite=True, quiet=True)

    print(f"\n  wrote LoD raster: {pre}_lod")
    print(f"  wrote thresholded DoD: {pre}_dod_thresholded")
    print("\n  -> Compare the LoD in the open bin against the manuscript's "
          f"published uniform LoD ({cfg['UNIFORM_LOD_OLD']} m). They should")
    print("     be similar. The forested bins are the finding: report that "
          "ratio in the paper.")

    return {"covariate": cov_name, "bins": bins}


# ==========================================================================
# STAGE 4 -- restate change statistics
# ==========================================================================
def stage4(cfg):
    """Recompute the numbers that go into the manuscript.

    LAKE LURE DIFFERS FROM THE WATERSHED HERE: this reach has open water: the
    reservoir surface itself. Checked directly (r.univar over just the
    WATER_RASTER footprint of {pre}_dh_corrected): mean dh = +1.71 m,
    n=441,396, sum = +755,235 m3 -- almost the ENTIRE inflated "deposition"
    volume this stage originally produced (810,516-951,364 m3, vs. ~99,471 m3
    of real signal over land) comes from the lake surface itself sitting
    ~1.7 m higher in one epoch than the other. That is a water-level /
    lidar-water-return difference between two survey dates, not sediment --
    the ORIGINAL coreg_variable_lod.py never has to worry about this (no open
    water in the watershed), so its stage4 computes stats over the WHOLE
    region unmasked. Here that silently mixes reservoir-level change into
    "deposition." Fixed by excluding WATER_RASTER before computing any
    statistic, and by using the LAND-only cell count as the percentage
    denominator (not the full reference DTM footprint, which still counts
    the lake as basin area to compare against).
    """
    msg("STAGE 4: change statistics under the variable LoD")
    pre = cfg["PREFIX"]

    total = int(gs.parse_command(
        "r.univar", map=cfg["REFERENCE"], flags="g")["n"])
    water_n = 0
    if cfg.get("WATER_RASTER"):
        water_n = int(gs.parse_command(
            "r.univar", map=cfg["WATER_RASTER"], flags="g").get("n", 0))
        total -= water_n
        print(f"  excluding {water_n:,} open-water cells ({cfg['WATER_RASTER']}) "
              f"from both the numerator and the denominator -- {total:,} land "
              f"cells remain")

    out = {}
    for label, dod in (("uniform LoD (old)", f"{pre}_dh_corrected"),
                       ("variable LoD (new)", f"{pre}_dod_thresholded")):
        if label.startswith("uniform"):
            gs.mapcalc(f"{pre}_tmp_old = if(abs({dod}) > {cfg['UNIFORM_LOD_OLD']}, "
                      f"{dod}, null())", overwrite=True, quiet=True)
            dod = f"{pre}_tmp_old"

        if cfg.get("WATER_RASTER"):
            gs.mapcalc(f"{pre}_tmp_land = if(isnull({cfg['WATER_RASTER']}), "
                      f"{dod}, null())", overwrite=True, quiet=True)
            dod = f"{pre}_tmp_land"

        gs.mapcalc(f"{pre}_tmp_ero = if({dod} < 0, {dod}, null())",
                   overwrite=True, quiet=True)
        gs.mapcalc(f"{pre}_tmp_dep = if({dod} > 0, {dod}, null())",
                   overwrite=True, quiet=True)

        ero = gs.parse_command("r.univar", map=f"{pre}_tmp_ero",
                               flags="ge", quiet=True)
        dep = gs.parse_command("r.univar", map=f"{pre}_tmp_dep",
                               flags="ge", quiet=True)

        n_ero = int(ero.get("n", 0))
        n_dep = int(dep.get("n", 0))
        rec = {
            "erosion_pct": 100.0 * n_ero / total if total else 0,
            "deposition_pct": 100.0 * n_dep / total if total else 0,
            "erosion_median_m": float(ero.get("median", "nan")),
            "deposition_median_m": float(dep.get("median", "nan")),
            # volumes: cell area x depth. Confirm your region resolution is 1 m
            # before quoting these -- g.region -p
            "erosion_volume_m3": float(ero.get("sum", 0)),
            "deposition_volume_m3": float(dep.get("sum", 0)),
        }
        out[label] = rec
        print(f"\n  {label}")
        print(f"    erosion    : {rec['erosion_pct']:5.1f}% of basin, "
              f"median {rec['erosion_median_m']:+.2f} m")
        print(f"    deposition : {rec['deposition_pct']:5.1f}% of basin, "
              f"median {rec['deposition_median_m']:+.2f} m")

    print("\n  -> The difference between these two rows IS the paper's new")
    print("     methodological result. If the 'old' row reproduces 38%/41%")
    print("     and the 'new' row is far smaller, you have demonstrated that")
    print("     open-terrain LoDs overstate detectable change under canopy.")

    save_json(cfg["REPORT_JSON"], out)
    return out


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["1", "2", "3", "4", "all"])
    args = ap.parse_args()

    if not os.environ.get("GISRC"):
        sys.exit("Not inside a GRASS session. Start GRASS first, e.g.\n"
                 "  grass /path/to/project/mapset --exec python3 "
                 "coreg_variable_lod.py --stage 1")

    cfg = CONFIG
    print("Region in use:")
    print(gs.read_command("g.region", flags="p"))

    results = {}
    if args.stage in ("1", "all"):
        results["stage1"] = stage1(cfg)
    if args.stage in ("2", "all"):
        results["stage2"] = stage2(cfg)
    if args.stage in ("3", "all"):
        results["stage3"] = stage3(cfg)
    if args.stage in ("4", "all"):
        results["stage4"] = stage4(cfg)

    if args.stage == "all":
        save_json("coreg_all_stages_lure_lidar.json", results)
    msg("done")


if __name__ == "__main__":
    main()
