We work with 2 pipelines:
    a). VECTOR PIPELINE(Shapes)
    b). RASTER PIPELINE(Images)

a) The Vector pipeline available at src/spatial_preprocessing as section (A) uses QGIS/Geopandas as its tool.
Draws a 60m legal Boundary polygon within Kasarani.
Outputs a .geojson file for training.

b) The Raster pipeline also available at src/spatial_preprocessing as section (B) uses the Google Earth Engine as its tool together with python libs like rasterio.
Produces a clean satelite image tile.
Outputs a (.tif/.png) image for training which represents two distinct seasons:
 - dry_season.
 - wet_season.
 The machine learning models from the Random Forest, read the color and texture of these pixels, while the YOLOv8-seg scans the imagery shapes to trace and count.
