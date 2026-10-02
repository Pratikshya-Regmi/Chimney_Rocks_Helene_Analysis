// Sentinel-2 NDVI median composites, pre- and post-Hurricane Helene
// (manuscript Sect. "Screening for disturbance using NDVI").
//
// NOTE: this script was RECONSTRUCTED from the Methods text and from the names
// of the exported files that were imported into GRASS
// (sentinel2_ndvi_2024_sep_20_25.tif, sentinel2_ndvi_2024_oct_5_10.tif).
// The original Earth Engine Code Editor script was not archived with the
// analysis files. Replace this file with the original if it is recovered.
//
// Method (as stated in the manuscript):
//   - Harmonized Sentinel-2 Level-2A surface reflectance
//   - pre-event 20-25 Sep 2024, post-event 5-10 Oct 2024
//   - scenes with < 20 % cloud cover, no per-pixel cloud mask
//   - median composite, NDVI = (B8 - B4) / (B8 + B4), 10 m
//
// Paste into https://code.earthengine.google.com and run. The area of interest
// is gee/chimney_boundary.kml; upload it as a table asset and set AOI_ASSET.

var AOI_ASSET = 'users/YOUR_USERNAME/chimney_boundary';
var aoi = ee.FeatureCollection(AOI_ASSET).geometry();

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

// filterDate's end date is exclusive, so the end is the day after the window.
function ndviComposite(start, end) {
  return s2.filterDate(start, end)
    .median()
    .normalizedDifference(['B8', 'B4'])
    .rename('NDVI')
    .clip(aoi);
}

var ndviPre = ndviComposite('2024-09-20', '2024-09-26');
var ndviPost = ndviComposite('2024-10-05', '2024-10-11');

Map.centerObject(aoi, 12);
var vis = {min: -0.2, max: 0.9, palette: ['brown', 'yellow', 'green']};
Map.addLayer(ndviPre, vis, 'NDVI pre (20-25 Sep 2024)');
Map.addLayer(ndviPost, vis, 'NDVI post (5-10 Oct 2024)');
Map.addLayer(ndviPost.subtract(ndviPre),
             {min: -0.5, max: 0.5, palette: ['red', 'white', 'green']},
             'Delta NDVI (post - pre)');

// Export both composites; the differencing is done in GRASS
// (grass/commands/ndvi_change.sh).
Export.image.toDrive({
  image: ndviPre,
  description: 'sentinel2_ndvi_2024_sep_20_25',
  region: aoi,
  scale: 10,
  maxPixels: 1e10
});
Export.image.toDrive({
  image: ndviPost,
  description: 'sentinel2_ndvi_2024_oct_5_10',
  region: aoi,
  scale: 10,
  maxPixels: 1e10
});
