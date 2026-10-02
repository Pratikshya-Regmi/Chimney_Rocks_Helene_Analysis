#!/usr/bin/env python3
"""
69_scheip_number_comparison.py
==============================
Side-by-side comparison of every quantitative value that overlaps between
this study and Scheip et al. (2026), the repeat-lidar case study of the same
gorge.

  Scheip, C., Lang, K., Prince, P., Wegmann, K., Wooten, R., Reynolds, A. and
  Harris, D. (2026) "A storm-sculpted landscape---Observations from
  post-Helene lidar in the Hickory Nut Gorge, North Carolina."
  Earth Surf. Process. Landforms 51(1), e70228. doi:10.1002/esp.70228
  Read from: agent/references /scheip2026_gorge.pdf

EVERY value in the `theirs` column is quoted from that PDF, with the page or
figure it came from in `their_source`. Nothing is inferred, rescaled or
converted. Where they do not report a quantity, the cell says
"not reported" -- it is NOT left blank and NOT filled by estimation.

THE HEADLINE OF THIS TABLE, stated plainly because it is easy to miss:
Scheip et al. report NO erosion, deposition or net volumes anywhere in the
paper. Their channel analysis is cross-sectional geometry, boulder-size
distributions and bed-roughness metrics, not volumetric sediment accounting.
So the volume rows have nothing to compare against, and the only directly
comparable quantities are the change-detection threshold, the input lidar
datasets, and the colour-scale ranges on the change figures.

Our values come from results/MANUSCRIPT_FINAL_NUMBERS.md and the manuscript
Results, not from re-computation here.

Output: results/tables/scheip_comparison.csv
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grass_env import log  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "results" / "tables" / "scheip_comparison.csv"

PDF = "agent/references /scheip2026_gorge.pdf"
MFN = "results/MANUSCRIPT_FINAL_NUMBERS.md"

# quantity, ours, theirs, their_source, comparable, note
ROWS = [
    # ---------------------------------------------------- input datasets
    ("Pre-event lidar, tributary/channel analysis",
     "2020 (14 Nov 2020)", "2020 (November 2020)",
     "Table 1, p.4", "yes",
     "Same acquisition. Their channel change detection also uses 2020 vs 2024."),
    ("Pre-event lidar, downstream reach",
     "2017 (Feb-Apr 2017)", "2017 (February 2017)",
     "Table 1, p.4", "yes",
     "Same acquisition; they use 2017 for hillslope pairings, we use it as the "
     "Lake Lure baseline because 2020 does not cover the reach."),
    ("Post-event lidar",
     "15-16 Nov 2024", "November 2024",
     "Table 1, p.4", "yes", "Same NCALM acquisition."),
    ("2017 point density (pts/m2)",
     "8.0", "8", "Table 1, p.4", "yes", "Agree."),
    ("2020 point density (pts/m2)",
     "17.5", "18", "Table 1, p.4", "yes",
     "Agree; they round to integer."),
    ("2024 point density (pts/m2)",
     "25.0", "25", "Table 1, p.4", "yes", "Agree."),
    ("Analysis extent",
     "tributary watershed (cat34) + Lake Lure reach",
     "Hickory Nut Gorge region; 7.5 km of Rocky Broad River channel",
     "Table 1 p.4; Fig. 4 caption p.6", "partly",
     "Theirs is the gorge-wide corridor and the trunk channel; ours is one "
     "tributary basin plus the reservoir inflow. Nested, not coincident."),

    # ---------------------------------------------------- detection limit
    ("Change-detection threshold applied (m)",
     "0.32 uniform at both sites", "0.25-0.35 ('typically')",
     "Sec. 3.2, p.4", "yes",
     "DIRECTLY COMPARABLE AND IN AGREEMENT. Our 0.32 m sits inside their "
     "0.25-0.35 m band."),
    ("Detection limit, measured range across canopy bins (m)",
     "0.24-0.34 (LoD95, ten bins, two sites)", "not reported",
     "-", "no",
     "They state an applied threshold, not a measured limit; no canopy "
     "stratification and no per-bin values are given."),
    ("Basis of the threshold",
     "1.96 x NMAD of residuals over distributed stable terrain, "
     "stratified by canopy height",
     "'applied to reduce noise and clarify meaningful surface change'",
     "Sec. 3.2, p.4", "no",
     "Ours is measured from the data; theirs is an applied noise filter with "
     "no stated derivation. The numbers agree even though the basis differs."),
    ("Independent check on the limit",
     "NMAD 0.034 m over verified impervious surfaces, n=1,201",
     "not reported", "-", "no", ""),

    # ---------------------------------------------------- alignment
    ("Horizontal co-registration method",
     "Nuth & Kaab (2011) slope/aspect solution on stable terrain",
     "Iterative Closest Point (ICP) on classified point clouds",
     "Sec. 3.2, p.4", "partly",
     "Both align before differencing, by different methods. They report no "
     "residual shift magnitude, so the 0.80 m we solved cannot be compared."),
    ("Planimetric offset solved (m)",
     "0.80 (dx +0.38, dy -0.70)", "not reported",
     "-", "no",
     "They do not quantify the misalignment ICP removed."),
    ("Differencing operator",
     "DEM of Difference on a shared 1 m grid",
     "M3C2 point-cloud distances along surface normals; "
     "DEM of Difference for channel work",
     "Sec. 3.2, p.4", "partly",
     "Surface-normal differencing is less slope-sensitive than vertical "
     "DoD, which is one reason their threshold needs no canopy stratification "
     "to behave."),

    # ---------------------------------------------------- volumes
    ("Gross erosion (m3)",
     "-84,017 watershed; -83,451 Lake Lure", "not reported",
     "-", "no", "No volumetric sediment accounting in their paper."),
    ("Gross deposition (m3)",
     "+24,563 watershed; +181,629 Lake Lure", "not reported",
     "-", "no", "As above."),
    ("Net volume (m3)",
     "-59,454 watershed; +98,178 Lake Lure", "not reported",
     "-", "no", "As above."),
    ("Feature counts above threshold",
     "191/161 erosion/deposition features (watershed); "
     "513/654 (Lake Lure)",
     "not reported as counts; 2,217 landslides region-wide cited from "
     "Burgi et al. (2025)",
     "Sec. 1, p.3", "no",
     "Their count is a regional landslide inventory, not features above a "
     "differencing threshold in our sense."),

    # ---------------------------------------------------- magnitudes
    ("Change magnitudes reported in text (m)",
     "incision up to 2.69; lowering 1.08; up to 2.86",
     "boulder transport threshold ~4 m diameter; "
     "3 m boulders moved across channel",
     "Sec. 4.1-4.2, pp.5-7", "no",
     "Ours are elevation differences; theirs are clast dimensions. "
     "Different quantities."),
    ("Colour-scale range, hillslope change figure",
     "not stated numerically in caption (GRASS colour rules)",
     "+2 to -2 m", "Fig. 2 legend, p.3", "partly",
     "Their +/-2 m bracket is consistent with our reported magnitudes; our "
     "own captions should state the range explicitly."),
    ("Colour-scale range, channel/regional change figure",
     "not stated numerically in caption",
     ">3 m erosion to >3 m deposition", "Fig. 4a legend, p.6", "partly",
     "As above."),

    # ---------------------------------------------------- channel change
    ("Channel change metric",
     "simulated flow-path agreement: Jaccard 0.50-0.68, "
     "16-32% of post-event network new",
     "cross-sectional geometry at 22 sections: area change -100 to +200 m2, "
     "width and depth change -10 to +50 m",
     "Fig. 4a, p.6", "no",
     "Ours is modelled drainage-network overlap; theirs is measured channel "
     "cross-section geometry. Both describe channel change; neither "
     "quantity converts to the other."),
    ("Channel direction of change",
     "net erosional in the tributary; net depositional at the reservoir "
     "inflow",
     "net erosional along the majority of the channel, increasing "
     "cross-sectional area everywhere except near the mouth",
     "Sec. 4.2, p.5", "yes",
     "AGREE IN SIGN AND IN PATTERN. Their 'except near the mouth' is the "
     "reservoir inflow, where we measure net deposition."),
    ("Bed roughness / boulder response",
     "not measured",
     "boulder abundance up by a factor of 7.8; median intermediate axis "
     "4.1 -> 2.4 m; bed roughness increased",
     "Sec. 4.2, p.7; Fig. 4b,c", "no",
     "No equivalent in our analysis."),

    # ---------------------------------------------------- storm context
    ("Storm rainfall context",
     "peak rates 50-75 mm/hr (NHC report)",
     ">500 mm over 72 h; >750 mm near Busick, NC; "
     "75 mm precursor storm 25 Sept",
     "Sec. 1, pp.2-3", "partly",
     "Both trace to the same NHC source; different aggregations."),
]

HEADER = ["quantity", "this_study", "scheip_et_al_2026", "their_source",
          "directly_comparable", "note"]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in ROWS:
            w.writerow(r)

    n_yes = sum(1 for r in ROWS if r[4] == "yes")
    n_partly = sum(1 for r in ROWS if r[4] == "partly")
    n_no = sum(1 for r in ROWS if r[4] == "no")
    n_nr = sum(1 for r in ROWS if r[2] == "not reported")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(ROWS)} rows)")
    print(f"  directly comparable : {n_yes}")
    print(f"  partly comparable   : {n_partly}")
    print(f"  not comparable      : {n_no}")
    print(f"  they do not report  : {n_nr}")
    print(f"\n  their values read from : {PDF}")
    print(f"  our values read from   : {MFN} and manuscript Results")
    log(f"scheip_comparison.csv: {len(ROWS)} rows, {n_yes} directly "
        f"comparable, {n_nr} not reported by Scheip et al.",
        run="scheip_comparison")


if __name__ == "__main__":
    main()
