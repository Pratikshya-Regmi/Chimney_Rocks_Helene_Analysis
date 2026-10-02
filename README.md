# Storm-driven terrain and drainage change after Hurricane Helene: processing code

Processing workflows (Google Earth Engine, PDAL and GRASS) for:

> Regmi, P., White, C. T., and Mitasova, H. *Assessment of storm-driven terrain
> and drainage re-organization in a forested mountain landscape.* Submitted to
> *Natural Hazards*.

The analysis compares pre-event airborne lidar (2017, 2020) with post-event
airborne lidar (November 2024) and a rapid-response Civil Air Patrol (CAP)
structure-from-motion (SfM) surface (October 2024). There are two study areas
in Hickory Nut Gorge, North Carolina: a landslide-affected tributary watershed
and the Lake Lure reach. The code builds the DTMs, co-registers them, estimates
canopy-binned detection limits from a stable-terrain mask, and computes DEMs
of Difference (DoD), feature volumes with spatially correlated uncertainty,
and changes in simulated overland-flow networks.

## Repository layout

```
gee/                 Sentinel-2 NDVI composites (Earth Engine) + study-area polygon
pdal/                SMRF ground classification of the 2024 lidar
grass/
  notebooks/         DEM generation, vertical alignment, basin delineation, flow simulation
  commands/          GRASS commands run interactively (NDVI change, mask inputs)
  scripts/           Analysis pipeline: stable mask, co-registration, LoD, volumes,
                     uncertainty, flow-network agreement, figures
  color_rules/       Colour tables used for the DoD and NDVI maps
environment.yml      Software versions
```

Scripts write their tables, logs and figures to `results/` at the repository
root (created on first run; not tracked).

## Requirements

| Software | Version used |
|---|---|
| GRASS | 8.5.0dev (built with PDAL support, for `v.in.pdal`) |
| PDAL | 2.6.2 |
| Python | 3.12, with numpy 2.2, scipy 1.16, pandas 2.3, matplotlib 3.10, rasterio 1.4, pyproj 3.7, shapely 2.1, geopandas 1.1 |
| Google Earth Engine | Code Editor (JavaScript API) |

```bash
conda env create -f environment.yml
conda activate helene-geomorphic-change
```

## Data

The input data are not redistributed here.

| Dataset | Source |
|---|---|
| 2017 lidar (NC Phase 5, Rutherford Co.) | OpenTopography |
| 2020 lidar | OpenTopography, https://doi.org/10.5069/G9RF5S7S |
| 2024 post-event lidar (15–16 Nov 2024) | Zenodo, https://doi.org/10.5281/zenodo.15600354 |
| CAP post-event imagery and SfM point cloud | Civil Air Patrol imagery; SfM reconstruction by White et al. (2026) in Agisoft Metashape |
| Sentinel-2 L2A (harmonized) | Copernicus, via Google Earth Engine |
| NAIP orthoimagery | USDA |

Derived products (DEMs, DoDs) are available from the corresponding author on
reasonable request.

## GRASS projects

All analysis runs in NAD83(HARN) / North Carolina (EPSG:3358) at 1 m
resolution. Three GRASS projects are used; the first two reference each
other's mapsets.

| Project | Mapsets | Contents |
|---|---|---|
| `DEM_generation` | `PERMANENT`, `DTM_DSM`, `for_codem` | Tributary watershed: 2017/2020/2024 lidar and CAP SfM DTMs, 2020 DSM, NAIP orthophotos, basin `basins_90_v_cat34`, flow simulations, co-registration outputs |
| `DEM_generation_lure` | `PERMANENT`, `DTM_DSM_lure`, `Analysis_lure` | Lake Lure: 2017 lidar DTM/DSM, 2024 lidar and SfM DTM/DSM, reach `boundary` |
| `helene_chimney_11_25` | `test_data_extent` | Sentinel-2 NDVI composites and `ndvi_change` |

Paths are set in two places:

* **Notebooks:** edit `GISDBASE`, the GRASS executable (`grass85`) and the
  data folders (`os.chdir(...)`) in the first cells.
* **Scripts:** set environment variables. `grass/scripts/grass_env.py` reads
  them.

  ```bash
  export GRASSBIN=/path/to/grass          # GRASS executable
  export GISDBASE=/path/to/grassdata      # folder holding DEM_generation and DEM_generation_lure
  ```

  A few scripts also contain an absolute path to a point-cloud folder or to
  the `helene_chimney_11_25` database (`16`, `36`, `38`, `50`); edit those
  constants before running.

## Workflow

Steps are listed in run order. Section names refer to the manuscript.

### 1. Disturbance screening with NDVI (*Screening for disturbance using NDVI*)

| File | What it does |
|---|---|
| `gee/sentinel2_ndvi_composites.js` | Median NDVI composites, 20–25 Sep and 5–10 Oct 2024, scenes < 20 % cloud, B8/B4, 10 m, over `gee/chimney_boundary.kml` |
| `grass/commands/ndvi_change.sh` | Imports both composites into `helene_chimney_11_25` and computes `ndvi_change = NDVI_post − NDVI_pre` with `r.mapcalc` |

### 2. Ground classification of the 2024 lidar (*Point cloud processing and DEM generation*)

| File | What it does |
|---|---|
| `pdal/smrf_ground_classification_2024.json` | SMRF in PDAL 2.6.2: window 16 m, slope 0.15, threshold 0.5 m, scalar 1.25 |

The 2017 and 2020 lidar keep the providers' ground classes. The SfM cloud was
classified in Metashape (maximum angle 30°, cell size 50 m). The 2024 lidar
was converted from NAD83(2011) ellipsoidal heights to NAVD88 (GEOID18) with
NOAA VDatum, outside this code.

### 3. DEM generation

All DTMs are interpolated from ground and road returns (`v.in.pdal
class_filter="2,11"`, reprojected on import) with

```
v.surf.rst  tension=20 smooth=1 npmin=300 dmin=2 segmax=40     (1 m)
r.fillnulls method=rst tension=10 smooth=1 edge=15 npmin=100 segmax=80
```

| Notebook | Area | Produces |
|---|---|---|
| `DTM_DSM_Generation_Helene_dec4_ver_corre.ipynb` | Watershed (`DEM_generation/DTM_DSM`, `PERMANENT`) | 2020 lidar DTM/DSM and canopy height (imported OpenTopography rasters; the DTM is kept in `PERMANENT` as `lidar_2020_DTM_ref` and used for basin delineation), NAIP orthophoto composites, data hull mask `hull_mask` |
| `raster_creation_for_codem.ipynb` | Watershed (`DEM_generation/for_codem`) | `dtm_2017` (imported), `dtm_2020_filled`, `dtm_2024_filled`, `cap_dtm_2024_filled` |
| `DTM_DSM_Generation_lure_jan8.ipynb` | Lake Lure (`DEM_generation_lure`) | `lidar_2017_DTM_lure`/`_DSM_lure` (patched from provider tiles), 2024 lidar and SfM DTM/DSM, vertically aligned surfaces `lidar_2024_DTM_corr_lure`, `sfm_DTM_corr_lure`, initial DoD |

### 4. Initial vertical alignment (watershed)

`multitemporal_dem_coregistration.ipynb` shifts each watershed surface
vertically to `dtm_2017`. The shift is the signed root-median-square
difference at the stable control points (`more_stable_pts`). The outputs are
renamed `lidar_2020_regis`, `lidar_2024_regis` and `cap_sfm_regis`, and
profiles are extracted before and after the shift. The Lake Lure equivalent
is in `DTM_DSM_Generation_lure_jan8.ipynb`.

### 5. Basin delineation and overland flow (*Watershed delineation and flow pattern analysis*)

`Multitemporal_hillshade_watershed_DOD.ipynb` runs the following steps:

* `r.watershed threshold=90000` (MFD) on the 2020 lidar DTM, giving
  `basins_90`; basin 34 is extracted as `basins_90_v_cat34`.
* `r.sim.water rain_value=50 man_value=0.4 niterations=48 output_step=4`
  (zero infiltration) on `lidar_2020_regis`, `lidar_2024_regis` and
  `cap_sfm_regis`. This gives `discharge_2020_regis`,
  `discharge_may_2024_lidar` and `discharge_2024_cap`.
* Preliminary DoD maps (`Change_in_DTM_lidar_may`, `Change_in_DTM_lidar_cap`).

### 6. Stable-terrain mask (*Stable-terrain definition*)

| File | What it does |
|---|---|
| `grass/commands/prepare_mask_inputs.sh` | Projects `ndvi_change` onto each analysis grid; derives the Lake Lure water mask from the 2017 lidar |
| `02_build_stable_mask.py` | Watershed mask `stable_mask`: \|ΔNDVI\| < 0.10, > 30 m from the pre-event channel network, slope < 45°, on the 2020 DTM |
| `02_build_stable_mask_lure.py` | Lake Lure mask `stable_mask_lure`: same criteria on the 2017 DTM, plus open-water exclusion |
| `05_lure_stable_mask_validation.py` | Shoreline-buffer sensitivity of the LoD (20 m buffer) and mask composition |

### 7. Co-registration and detection limits (*DEM co-registration*, *DEM differencing and level of detection*)

`03_coreg_variable_lod*.py` work in four stages:

1. A Nuth & Kääb (2011) horizontal shift, solved iteratively over stable
   cells with 5–45° slope (at most 12 iterations, converging at 0.02 m,
   |Δh| clipped at 20 m). The shift is applied with `r.region` and
   `r.resamp.interp`.
2. Removal of the residual vertical bias, modelled as a constant plus a
   linear term in canopy height (DSM − DTM).
3. LoD95 = 1.96 × NMAD of stable-terrain residuals, computed per canopy-height
   bin (0–2, 2–5, 5–10, 10–20, ≥ 20 m).
4. Change statistics (area above LoD, volumes) restated under the new LoD.

Each stage is run separately (`--stage 1` … `--stage 4`, or `--stage all`).

| Script | Pair | Main output |
|---|---|---|
| `03_coreg_variable_lod.py` | Watershed: 2020 lidar → 2024 lidar | `coreg_dh_corrected` |
| `03_coreg_variable_lod.py` with `TARGET = cap_sfm_regis@for_codem` | Watershed: 2020 lidar → 2024 CAP SfM | SfM comparison |
| `03_coreg_variable_lod_lure_lidar.py` | Lake Lure: 2017 lidar → 2024 lidar | `coreg_lure_lidar_dh_corrected` |
| `03_coreg_variable_lod_lure_sfm.py` | Lake Lure: 2017 lidar → 2024 CAP SfM | SfM comparison |
| `10_consolidated_lod_table.py` | Both sites | LoD by canopy bin (detection-limit table) |
| `06_lure_known_stable_ground_test.py` | Lake Lure | Independent check on verified impervious surfaces (n = 1,201) |

### 8. Change features and volumes

The DoD is thresholded at the uniform 0.32 m LoD. Features are delineated
with `r.clump` (4-connectivity) and retained if ≥ 50 m²; volume = cell area ×
Δh.

| Script | What it does |
|---|---|
| `17_watershed_lod_min_feature_area_d1_excluded.py` | Watershed gross erosion, deposition and net at 0/50/100 m² minimum area |
| `18_dominant_features_d1_excluded.py` | Watershed features ranked by volume |
| `25_lure_lod_min_feature_area_flagged_excluded.py` | Lake Lure equivalent of 17 |
| `26_dominant_features_lure_flagged_excluded.py` | Lake Lure equivalent of 18 |
| `19_dominant_features_inspection.py` | Screening of each feature > 1 % of gross volume (canopy, slope, flow context, orthophoto crops) |
| `16_resolve_d1_deposition.py`, `22_resolve_lure_flagged_features.py` | Full checks of features excluded as artifacts (*Features excluded as artifacts*) |
| `23_lure_volumes_with_without_flagged.py` | Lake Lure volumes with and without the excluded features |
| `24_diffuse_deposition_characterization.py` | Character of deposition outside the dominant features |
| `14_watershed_corridor_comparison.py`, `21_watershed_corridor_comparison_d1_checked.py` | Main-scar vs. western corridor budgets |
| `27_sediment_budget_comparison_final.py` | Watershed vs. Lake Lure magnitude comparison |
| `28_manuscript_final_summary.py` | Consolidated volume table for both sites |

`11`, `12`, `13`, `15` and `20` are the same calculations before the artifact
exclusions. They are kept because the with/without comparison is reported.

### 9. Volumetric uncertainty (*Volumetric uncertainty*)

`48_volumetric_uncertainty.py` fits a spherical semivariogram to the
stable-terrain residuals at each site and propagates σ_V = σA/√n_eff, with
n_eff = A/(πL²). The result is capped at σA in the fully correlated limit,
and erosion and deposition terms are combined in quadrature for net volumes.

### 10. Drainage-network agreement (*Watershed delineation and flow pattern analysis*)

| Script | What it does |
|---|---|
| `08_watershed_flow_agreement.py` | Jaccard index and retained/new channel fractions, 2020 vs. 2024 lidar |
| `09_watershed_flow_agreement_sensitivity.py` | Same at discharge thresholds 0.005, 0.01, 0.02, 0.05 |
| `33_flow_lidar_sfm_agreement.py` | Same for 2024 lidar vs. 2024 SfM |
| `35_flow_lidar_sfm_diagnostic.py` | Diagnostics on the lidar–SfM agreement |
| `30_flow_sequence_setup.py` | Flow-change zones and exports for the flow figures |

### 11. Supporting analyses

| Script | What it does |
|---|---|
| `00_inventory.py` | Lists mapsets, maps and regions in both projects |
| `01_ground_density_check.py` | Ground-return density by epoch and canopy bin (classification-consistency check) |
| `07_lure_lod_sensitivity.py` | Lake Lure volumes across LoD thresholds |
| `29_sfm_spatial_bias_variation.py` | Corridor-wide SfM vertical residual (Supplementary Material) |
| `36_vsurfrst_cross_validation.py` | Leave-one-out cross-validation of `v.surf.rst` on sample windows |
| `37_`, `39_` | Sensitivity of the surfaces to RST tension, smoothing and `dmin` |
| `38_dod_volume_tension_sensitivity.py` | Sensitivity of watershed volumes to RST tension |
| `40_grid_alignment_diagnostic.py` | Effect of epoch grid alignment on the DoD |
| `41_bias_normalised_volume_comparison.py` | Volumes of DoD variants after identical bias correction |
| `60_`, `61_`, `63_`, `65_` | A–B profile extraction before and after correction |
| `50_export_ndvi_figure_data.py` | Exports the NDVI-change layers for plotting |
| `69_scheip_number_comparison.py` | Comparison with values reported by Scheip et al. (2026); needs that paper's PDF, which is not distributed |

## Manuscript figures

| Figure | Script |
|---|---|
| A–B profile before/after correction (`fig_pl_combined_datumfixed`) | `66_fig_profile_short_line_datum_fixed.py` (data from `65_`, `61_`) |
| LoD vs. canopy height (`fig_lod_vs_canopy`) | `make_figures.py` |
| Detectable change area (`fig_detection_area_lidar_only`) | `make_detection_area_lidar_only.py` |
| Watershed DEMs (`fig_watershed_dem_three_panel`) | `42_fig_watershed_dem_three_panel.py` |
| Transects (`fig_transects_v4_labels_c`) | `43_transect_label_variants.py`, variant c |
| Overland flow, three epochs (`fig_flow_consolidated_v2_e`) | `67_fig_flow_consolidated_v2_legends.py`, variant e |
| Flow change map (`fig_flow_change_map`) | `34_flow_alternative_figures.py` |
| Stable-terrain mask, Supplementary (`fig_stable_mask_v2b_sidebyside_ownaspect`) | `make_stable_mask_figure_v2.py`, variant b |
| SfM residual along the corridor, Supplementary (`fig_sfm_bias_corridor_map`) | `29_sfm_spatial_bias_variation.py` |

The study-area map, the NDVI-change map, the pre/post orthophoto change map,
the Lake Lure maps and the debris profile were composed outside the scripted
workflow from the GRASS maps above.

Other figure scripts in `grass/scripts/` are layout alternatives prepared
during revision. `map_style.py` and `make_transect_figure_v2.py`/`_v3.py` are
shared modules that other figure scripts import.

## Running the scripts

Run from the repository root.

* `02_*` and `03_*` must be launched inside the GRASS project/mapset they
  work in:

  ```bash
  grass $GISDBASE/DEM_generation/for_codem --exec python grass/scripts/02_build_stable_mask.py
  grass $GISDBASE/DEM_generation/for_codem --exec python grass/scripts/03_coreg_variable_lod.py --stage all
  grass $GISDBASE/DEM_generation_lure/PERMANENT --exec python grass/scripts/02_build_stable_mask_lure.py
  grass $GISDBASE/DEM_generation_lure/PERMANENT --exec python grass/scripts/03_coreg_variable_lod_lure_lidar.py --stage all
  ```

* Most other analysis scripts open their own session through `grass_env.py`:

  ```bash
  python grass/scripts/17_watershed_lod_min_feature_area_d1_excluded.py
  ```

* The figure scripts read the tables and GeoTIFFs exported to `results/` by
  the analysis scripts.

Each script's docstring states its inputs, method and outputs.

## Notes

* `gee/sentinel2_ndvi_composites.js` and
  `pdal/smrf_ground_classification_2024.json` were reconstructed from the
  methods and parameters reported in the manuscript; the original files were
  not archived. Everything else is the code that was run.
* Some steps are not scripted: SfM reconstruction and classification
  (Metashape), vertical datum conversion (VDatum), and visual inspection of
  candidate artifacts against orthophotos.
* `r.sim.water` is a Monte Carlo solver, so re-running the simulations
  reproduces the flow patterns, but not bit-identical rasters.

## Citation

Please cite the article above if you use this code.

## Contact

Pratikshya Regmi, Center for Geospatial Analytics, North Carolina State
University, pregmi3@ncsu.edu
