#!/usr/bin/env bash
# Delta-NDVI screening layer (manuscript Sect. "Screening for disturbance using NDVI").
#
# These are the GRASS commands that were run interactively in the GRASS GUI,
# transcribed from the command history of project helene_chimney_11_25,
# mapset test_data_extent (EPSG:3358). Display-only r.colors trials are omitted.
#
# Inputs are the two Sentinel-2 NDVI median composites exported from Google
# Earth Engine (see gee/sentinel2_ndvi_composites.js).
#
# Run inside a GRASS session, e.g.
#   grass /path/to/grassdata/helene_chimney_11_25/test_data_extent --exec bash ndvi_change.sh
set -euo pipefail

NDVI_DIR=${NDVI_DIR:-.}   # folder holding the two GeoTIFFs exported from GEE
RULES_DIR=$(cd "$(dirname "$0")/../color_rules" && pwd)

# Import the pre- and post-event NDVI composites (reprojected to the project CRS).
r.import input="$NDVI_DIR/sentinel2_ndvi_2024_sep_20_25.tif" output=sentinel2_ndvi_2024_sep_20_25
r.import input="$NDVI_DIR/sentinel2_ndvi_2024_oct_5_10.tif"  output=sentinel2_ndvi_2024_oct_5_10

r.colors map=sentinel2_ndvi_2024_sep_20_25 color=ndvi
r.colors map=sentinel2_ndvi_2024_oct_5_10 color=ndvi

# Delta NDVI = NDVI_post - NDVI_pre (negative = vegetation loss).
r.mapcalc --overwrite expression="ndvi_change = sentinel2_ndvi_2024_oct_5_10 - sentinel2_ndvi_2024_sep_20_25"
r.colors map=ndvi_change rules="$RULES_DIR/ndvi_change_colors.txt"

# Shaded version used for display.
r.relief input=ndvi_change output=ndvi_change_relief
r.shade shade=ndvi_change_relief color=ndvi_change output=ndvi_change_relief_shaded --overwrite
