#!/usr/bin/env python3
"""
48_volumetric_uncertainty.py
=============================
Volumetric uncertainty for every reported DoD volume, propagated with
spatially correlated DEM error.

WHY NOT sigma/sqrt(N)
    Treating each 1 m cell as an independent sample of the elevation error
    would divide by sqrt(N) with N in the tens of thousands and return
    uncertainties of a few cubic metres -- physically meaningless. DEM error
    is spatially correlated over tens of metres, so neighbouring cells carry
    largely the same error. The standard treatment (Rolstad et al. 2009;
    Anderson 2019; Hugonnet et al. 2022) replaces N with an effective sample
    size set by the decorrelation length of the error field.

METHOD
    1. Take the post-co-registration residuals on stable terrain at each
       site -- the same cells the level of detection is estimated from.
    2. Fit a semivariogram to those residuals and take the practical range
       as the decorrelation length L.
    3. For a reported volume covering area A:
           n_eff   = A / (pi * L^2)
           sigma_V = sigma * A / sqrt(n_eff)
       sigma is the NMAD of the stable-terrain residuals (robust to the
       heavy tails of DEM differences, per the project's rule 4).
    4. Net volumes combine erosion and deposition in quadrature:
           sigma_net = sqrt(sigma_ero^2 + sigma_dep^2)
       The two are measured over disjoint areas from the same error field;
       quadrature is the standard treatment and is conservative relative to
       assuming the errors cancel.

    The semivariogram is computed from a random subsample of stable cells
    (all pairs within the subsample) rather than the full field, which would
    be O(n^2) on ~10^5 cells.

OUTPUTS
    results/tables/volumetric_uncertainty.csv        per-volume sigma_V
    results/tables/volumetric_uncertainty_variogram.csv  fitted models
    results/figures/fig_volumetric_uncertainty_variograms.png
    results/logs/vol_uncertainty_<date>.log
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import grass_session, log  # noqa: E402

RUN = "vol_uncertainty"
ROOT = HERE.parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

SITES = {
    "watershed": dict(project="DEM_generation", mapset="for_codem",
                      dh="coreg_dh_corrected", mask="stable_mask",
                      label="Watershed (cat 34)", max_lag=300.0, n_bins=30),
    "lure": dict(project="DEM_generation_lure", mapset="PERMANENT",
                 dh="coreg_lure_lidar_dh_corrected", mask="stable_mask_lure",
                 # wider window: at 300 m the Lure variogram had not reached
                 # a sill and the fitted range ran into its upper bound
                 label="Lake Lure", max_lag=1500.0, n_bins=50),
}

# Reported volumes (results/MANUSCRIPT_FINAL_NUMBERS.md, 50 m^2 min feature
# area, exclusions applied) with the area each is integrated over.
VOLUMES = [
    # key,                site,        volume_m3,  area_m2, kind
    ("watershed_erosion",   "watershed", -84016.86,  73827, "gross"),
    ("watershed_deposition","watershed",  24563.05,  31366, "gross"),
    ("lure_erosion",        "lure",      -83451.35, 104460, "gross"),
    ("lure_deposition",     "lure",      181629.20, 209888, "gross"),
    ("corridor_main_BB",    "watershed",  -3517.0,    7767, "gross"),
    ("corridor_west_CC",    "watershed",  -2917.0,    7028, "gross"),
]
NETS = [
    ("watershed_net", "watershed", -59453.81,
     "watershed_erosion", "watershed_deposition"),
    ("lure_net", "lure", 98177.85, "lure_erosion", "lure_deposition"),
]

SUBSAMPLE = 16000         # stable cells used for the variogram
MAX_LAG_M = 300.0
N_BINS = 30
RNG_SEED = 20260910


# ------------------------------------------------------------------ models
def spherical(h, nugget, sill, rng):
    h = np.asarray(h, float)
    out = np.where(h < rng,
                   nugget + sill * (1.5 * h / rng - 0.5 * (h / rng) ** 3),
                   nugget + sill)
    return out


def exponential(h, nugget, sill, rng):
    # `rng` is the PRACTICAL range: the lag at which ~95% of the sill is
    # reached, which is what a decorrelation length means here.
    return nugget + sill * (1.0 - np.exp(-3.0 * np.asarray(h, float) / rng))


# ------------------------------------------------------------------ data
def stable_residuals(site_key):
    cfg = SITES[site_key]
    gs, _ = grass_session(project=cfg["project"], mapset=cfg["mapset"])
    import grass.script as g
    import grass.script.array as garray
    ms = g.read_command("g.mapsets", flags="l").split()
    g.run_command("g.mapsets", mapset=",".join(ms), operation="set")

    for name in (cfg["dh"], cfg["mask"]):
        if not g.find_file(name=name, element="cell")["name"]:
            sys.exit(f"{site_key}: raster {name} not found")
    g.run_command("g.region", raster=cfg["dh"], flags="a")
    reg = g.parse_command("g.region", flags="g")

    g.mapcalc(f"_vu_res = if(!isnull({cfg['mask']}), {cfg['dh']}, null())",
              overwrite=True, quiet=True)
    # null=np.nan is essential: without it grass.script.array returns GRASS
    # nulls as 0.0, so every masked-out cell enters the statistics as a
    # perfect zero. That silently turned 68,248 stable cells into 607,732
    # "observations" and drove the NMAD to exactly 0.
    arr = np.asarray(garray.array("_vu_res", null=np.nan), dtype=float)
    g.run_command("g.remove", type="raster", name="_vu_res", flags="f",
                  quiet=True)

    ns, ew = float(reg["nsres"]), float(reg["ewres"])
    north, west = float(reg["n"]), float(reg["w"])
    rows, cols = arr.shape
    ok = np.isfinite(arr)
    ri, ci = np.nonzero(ok)
    x = west + (ci + 0.5) * ew
    y = north - (ri + 0.5) * ns
    z = arr[ri, ci]
    log(f"{site_key}: {z.size:,} stable cells, res {ew:.2f} x {ns:.2f} m",
        run=RUN)
    return x, y, z


def robust_sigma(z):
    med = float(np.median(z))
    nmad = 1.4826 * float(np.median(np.abs(z - med)))
    return dict(n=int(z.size), mean=float(z.mean()), median=med,
                sd=float(z.std(ddof=1)), nmad=nmad)


def empirical_variogram(x, y, z, rng, max_lag, n_bins):
    """Binned semivariogram using ALL pairs within MAX_LAG_M of each other.

    A plain random subsample fails here: the stable mask is a scatter of
    patches over ~1.4 km, so random pairs are nearly all long-range and the
    short-lag bins -- the ones that set the decorrelation length -- end up
    almost empty. The first attempt did exactly that and returned a flat
    variogram for the watershed (R^2 = 0.03). Querying a KD-tree for
    neighbours within MAX_LAG_M instead concentrates every pair where the
    structure actually is.
    """
    from scipy.spatial import cKDTree
    idx = rng.choice(z.size, size=min(SUBSAMPLE, z.size), replace=False)
    xs, ys, zs = x[idx], y[idx], z[idx]
    zs = zs - np.median(zs)

    tree = cKDTree(np.column_stack([xs, ys]))
    pairs = tree.query_pairs(r=max_lag, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
    dz2 = (zs[i] - zs[j]) ** 2

    edges = np.linspace(0.0, max_lag, n_bins + 1)
    b = np.digitize(d, edges) - 1
    keep = (b >= 0) & (b < n_bins)
    sums = np.bincount(b[keep], weights=dz2[keep], minlength=n_bins)
    cnts = np.bincount(b[keep], minlength=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    good = cnts > 500
    gamma = np.zeros(n_bins)
    gamma[good] = 0.5 * sums[good] / cnts[good]
    return centres[good], gamma[good], cnts[good], len(idx)


def fit_models(h, gamma, max_lag):
    out = {}
    sill0 = float(np.nanmax(gamma))
    for name, fn in (("spherical", spherical), ("exponential", exponential)):
        try:
            popt, _ = curve_fit(
                fn, h, gamma, p0=[gamma[0], sill0, 100.0],
                bounds=([0.0, 1e-6, 5.0],
                        [sill0 * 2 + 1e-9, sill0 * 5, 3.0 * max_lag]),
                maxfev=20000)
            pred = fn(h, *popt)
            ss_res = float(np.sum((gamma - pred) ** 2))
            ss_tot = float(np.sum((gamma - gamma.mean()) ** 2))
            out[name] = dict(nugget=popt[0], sill=popt[1], range_m=popt[2],
                             r2=1 - ss_res / ss_tot if ss_tot else float("nan"))
        except Exception as exc:                       # noqa: BLE001
            log(f"fit failed ({name}): {exc}", run=RUN)
    return out


def main():
    rng = np.random.default_rng(RNG_SEED)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    site_stats, vg_rows = {}, []
    fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 70 / 25.4))
    for ax, (key, cfg) in zip(axes, SITES.items()):
        x, y, z = stable_residuals(key)
        st = robust_sigma(z)
        h, gamma, cnts, n_used = empirical_variogram(
            x, y, z, rng, cfg["max_lag"], cfg["n_bins"])
        fits = fit_models(h, gamma, cfg["max_lag"])
        best = min(fits, key=lambda k: -fits[k]["r2"]) if fits else None
        L = fits[best]["range_m"] if best else float("nan")
        site_stats[key] = dict(stats=st, L=L, fits=fits, model=best,
                               n_variogram=n_used)
        log(f"{key}: n={st['n']:,} NMAD={st['nmad']:.4f} m SD={st['sd']:.4f} m; "
            f"variogram on {n_used:,} cells; "
            + "; ".join(f"{m}: range={v['range_m']:.1f} m nugget={v['nugget']:.4f} "
                        f"sill={v['sill']:.4f} R2={v['r2']:.3f}"
                        for m, v in fits.items())
            + f"; adopted {best} L={L:.1f} m", run=RUN)
        for m, v in fits.items():
            vg_rows.append(dict(site=key, label=cfg["label"], model=m,
                                nugget_m2=v["nugget"], sill_m2=v["sill"],
                                range_m=v["range_m"], r2=v["r2"],
                                adopted=(m == best), nmad_m=st["nmad"],
                                sd_m=st["sd"], n_stable_cells=st["n"],
                                n_variogram_cells=n_used))
        ax.plot(h, gamma, "o", ms=2.5, color="0.25", label="empirical")
        hh = np.linspace(0, cfg["max_lag"], 300)
        for m, v in fits.items():
            fn = spherical if m == "spherical" else exponential
            ax.plot(hh, fn(hh, v["nugget"], v["sill"], v["range_m"]),
                    lw=1.2, label=f"{m} (L={v['range_m']:.0f} m)")
        ax.axvline(L, color="red", lw=0.8, ls="--")
        ax.set_title(cfg["label"], fontsize=8, fontweight="bold")
        ax.set_xlabel("lag distance h (m)", fontsize=7)
        ax.set_ylabel(r"$\gamma(h)$ (m$^2$)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_volumetric_uncertainty_variograms.png", dpi=300)
    fig.savefig(FIGURES / "fig_volumetric_uncertainty_variograms.pdf")
    plt.close(fig)

    with open(TABLES / "volumetric_uncertainty_variogram.csv", "w",
              newline="") as fh:
        f = ["site", "label", "model", "nugget_m2", "sill_m2", "range_m", "r2",
             "adopted", "nmad_m", "sd_m", "n_stable_cells", "n_variogram_cells"]
        w = csv.DictWriter(fh, fieldnames=f)
        w.writeheader()
        w.writerows(vg_rows)

    # ---------------- per-volume sigma_V --------------------------------
    rows, by_key = [], {}
    for key, site, vol, area, kind in VOLUMES:
        st = site_stats[site]
        sigma, L = st["stats"]["nmad"], st["L"]
        n_eff = area / (math.pi * L ** 2)
        sigma_V = sigma * area / math.sqrt(n_eff)
        # Cap at the fully correlated limit. When the integration area is
        # smaller than one correlation patch (n_eff < 1) the formula returns
        # more than sigma * A, which is impossible: sigma * A is what you get
        # if every cell in the area shares one common error. Rolstad et al.
        # (2009) treat that regime as fully correlated.
        fully_corr = sigma * area
        capped = sigma_V > fully_corr
        if capped:
            sigma_V = fully_corr
        by_key[key] = sigma_V
        rows.append(dict(quantity=key, site=site, kind=kind,
                         volume_m3=vol, area_m2=area, sigma_m=sigma,
                         L_m=L, n_eff=n_eff, sigma_V_m3=sigma_V,
                         fully_correlated_cap=capped,
                         pct_of_volume=100 * sigma_V / abs(vol)))
        if capped:
            log(f"{key}: n_eff={n_eff:.2f} < 1 -> capped at the fully "
                f"correlated limit sigma*A = {fully_corr:,.0f} m3", run=RUN)
        log(f"{key}: A={area:,} m2 L={L:.1f} m n_eff={n_eff:.1f} "
            f"sigma={sigma:.4f} -> sigma_V={sigma_V:,.0f} m3 "
            f"({100*sigma_V/abs(vol):.1f}% of {vol:,.0f})", run=RUN)

    for key, site, vol, k_ero, k_dep in NETS:
        sigma_V = math.hypot(by_key[k_ero], by_key[k_dep])
        by_key[key] = sigma_V
        rows.append(dict(quantity=key, site=site, kind="net",
                         volume_m3=vol, area_m2="", sigma_m=site_stats[site]["stats"]["nmad"],
                         L_m=site_stats[site]["L"], n_eff="",
                         sigma_V_m3=sigma_V, fully_correlated_cap="",
                         pct_of_volume=100 * sigma_V / abs(vol)))
        log(f"{key}: quadrature of {by_key[k_ero]:,.0f} and {by_key[k_dep]:,.0f} "
            f"-> sigma_V={sigma_V:,.0f} m3 ({100*sigma_V/abs(vol):.1f}%)", run=RUN)

    with open(TABLES / "volumetric_uncertainty.csv", "w", newline="") as fh:
        f = ["quantity", "site", "kind", "volume_m3", "area_m2", "sigma_m",
             "L_m", "n_eff", "sigma_V_m3", "fully_correlated_cap",
             "pct_of_volume"]
        w = csv.DictWriter(fh, fieldnames=f)
        w.writeheader()
        w.writerows(rows)

    print("\n=== decorrelation length and stable-terrain sigma ===")
    for k, v in site_stats.items():
        print(f"  {SITES[k]['label']}: NMAD={v['stats']['nmad']:.3f} m, "
              f"SD={v['stats']['sd']:.3f} m, L={v['L']:.0f} m "
              f"({v['model']}), n={v['stats']['n']:,}")
    print("\n=== sigma_V ===")
    for r in rows:
        print(f"  {r['quantity']:<22} {r['volume_m3']:>12,.0f} +/- "
              f"{r['sigma_V_m3']:>9,.0f} m3   ({r['pct_of_volume']:.1f}%)")

    # ---------------- do the comparative claims survive? -----------------
    print("\n=== comparative claims ===")
    lure_dep, s_ld = 181629.20, by_key["lure_deposition"]
    w_dep, s_wd = 24563.05, by_key["watershed_deposition"]
    ratio = lure_dep / w_dep
    s_ratio = ratio * math.hypot(s_ld / lure_dep, s_wd / w_dep)
    print(f"  Lure deposition / watershed deposition = {ratio:.2f} "
          f"+/- {s_ratio:.2f}  (claim: 'more than seven times')")
    log(f"CLAIM 'more than seven times': ratio={ratio:.3f}+/-{s_ratio:.3f}, "
        f"lower bound {ratio - s_ratio:.3f}", run=RUN)

    w_net, s_wn = 59453.81, by_key["watershed_net"]
    l_net, s_ln = 98177.85, by_key["lure_net"]
    r2 = l_net / w_net
    s_r2 = r2 * math.hypot(s_ln / l_net, s_wn / w_net)
    print(f"  |Lure net| / |watershed net|            = {r2:.2f} "
          f"+/- {s_r2:.2f}  (claim: 'approximately 1.65 times')")
    log(f"CLAIM 'approximately 1.65 times': ratio={r2:.3f}+/-{s_r2:.3f}, "
        f"range {r2-s_r2:.3f}-{r2+s_r2:.3f}", run=RUN)
    log("DONE", run=RUN)


if __name__ == "__main__":
    main()
