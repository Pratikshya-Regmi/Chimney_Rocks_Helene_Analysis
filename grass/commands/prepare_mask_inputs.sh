#!/usr/bin/env bash
# Inputs for the stable-terrain masks (grass/scripts/02_build_stable_mask*.py).
#
# These steps were run interactively. The r.proj commands and the final
# r.mapcalc are transcribed from the GRASS raster history of the maps they
# created; the two water-candidate steps follow the procedure described in
# 02_build_stable_mask_lure.py (threshold the 2017 lidar DTM at <= 301 m and
# keep the largest r.clump component).
set -euo pipefail

NDVI_DBASE=${NDVI_DBASE:?set NDVI_DBASE to the GISDBASE holding helene_chimney_11_25}

# --- Tributary watershed: project DEM_generation, mapset for_codem ----------
# Delta-NDVI (from grass/commands/ndvi_change.sh) onto the 1 m analysis grid.
#   grass $GISDBASE/DEM_generation/for_codem --exec bash prepare_mask_inputs.sh watershed
if [[ "${1:-}" == "watershed" ]]; then
  r.proj --overwrite project=helene_chimney_11_25 mapset=test_data_extent \
         dbase="$NDVI_DBASE" input=ndvi_change output=ndvi_diff \
         method=bilinear resolution=1
fi

# --- Lake Lure: project DEM_generation_lure, mapset PERMANENT ---------------
#   grass $GISDBASE/DEM_generation_lure/PERMANENT --exec bash prepare_mask_inputs.sh lure
if [[ "${1:-}" == "lure" ]]; then
  g.region raster=boundary res=1
  r.proj --overwrite project=helene_chimney_11_25 mapset=test_data_extent \
         dbase="$NDVI_DBASE" input=ndvi_change output=ndvi_diff_lure \
         method=bilinear

  # Open water from the 2017 lidar: reservoir surface at 297-301 m.
  r.mapcalc --overwrite expression="_water_cand = if(lidar_2017_DTM_lure <= 301, 1, null())"
  r.clump --overwrite input=_water_cand output=_water_clump
  # Category 16 was the single largest component (441,400 cells) in the
  # original run; check with `r.stats -c _water_clump` before reusing.
  r.mapcalc --overwrite expression="lure_water_mask = if(_water_clump == 16, 1, null())"
fi
