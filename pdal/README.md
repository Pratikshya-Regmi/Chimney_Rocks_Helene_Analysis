# 2024 lidar ground classification (PDAL SMRF)

`smrf_ground_classification_2024.json` classifies ground returns in the
unclassified November 2024 post-event lidar with the Simple Morphological
Filter (Pingel et al., 2013) in PDAL 2.6.2. Parameters are as reported in the
manuscript (Sect. "Point cloud processing and DEM generation"):

| parameter | value |
|---|---|
| `window` | 16 m |
| `slope` | 0.15 |
| `threshold` | 0.5 m |
| `scalar` | 1.25 |

**Note:** this pipeline was reconstructed from the parameters reported in the
manuscript. The original pipeline file was not archived with the analysis
files. Replace it with the original if it is recovered.

Run:

```bash
pdal pipeline smrf_ground_classification_2024.json \
  --readers.las.filename=<2024_tile>.laz \
  --writers.las.filename=<2024_tile>_smrf.las
```

The classified output is what the GRASS notebooks import with
`v.in.pdal class_filter="2,11"`: `NEWAREA.las` for the tributary watershed
(`raster_creation_for_codem.ipynb`) and `after_lidar_grass.las` for Lake Lure
(`DTM_DSM_Generation_lure_jan8.ipynb`). The 2017 and 2020 lidar keep the
providers' ground classification, and the CAP SfM cloud was classified in
Agisoft Metashape. Neither is reclassified here.
