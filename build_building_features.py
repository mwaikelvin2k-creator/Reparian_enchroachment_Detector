"""
Build a real building-level feature table from the uploaded Sentinel-2 tile
and buildings footprint layer.

Inputs (real, uploaded):
  - data/raw/kasarani_sentinel2_composite.tif  (3-band RGB, EPSG:32737)
  - data/vectors/kasarani_buildings.geojson    (57k+ building footprints, EPSG:4326)

Output:
  - data/processed/kasarani_building_features.csv
    columns: id, R_mean, R_std, G_mean, G_std, B_mean, B_std, area_m2

NOTE ON LABELS: the source raster has only B4/B3/B2 (true-color RGB) — no
NIR/SWIR bands, so NDVI/NDWI cannot be computed from it. The buildings file
has no attribute columns beyond id/FID — no roof-material, land-cover, or
encroachment label. This script produces real, unlabeled features only.
The RF classifier trained earlier (train_riparian_rf.py) needs a
'class_label' or 'encroachment' column added before this table can be used
to train anything — this data isn't self-labeling.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

RASTER_PATH = "data/raw/kasarani_sentinel2_composite.tif"
BUILDINGS_PATH = "data/vectors/kasarani_buildings.geojson"
OUTPUT_CSV = "data/processed/kasarani_building_features.csv"

BAND_NAMES = ["R", "G", "B"]  # B4, B3, B2 per the tile's band descriptions


def main():
    with rasterio.open(RASTER_PATH) as src:
        raster_crs = src.crs
        affine = src.transform
        bands = [src.read(i + 1) for i in range(src.count)]

    buildings = gpd.read_file(BUILDINGS_PATH)
    n_before = len(buildings)
    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    dropped = n_before - len(buildings)
    if dropped:
        print(f"Dropped {dropped} non-polygon geometries "
              f"(e.g. GeometryCollection) before zonal stats.")

    buildings = buildings.to_crs(raster_crs)

    feature_frames = []
    for band_arr, name in zip(bands, BAND_NAMES):
        stats = zonal_stats(
            buildings, band_arr, affine=affine,
            stats=["mean", "std"], nodata=np.nan, all_touched=True,
        )
        df = pd.DataFrame(stats).rename(columns={"mean": f"{name}_mean", "std": f"{name}_std"})
        feature_frames.append(df)

    features = pd.concat(feature_frames, axis=1)
    features["id"] = buildings["id"].values
    features["area_m2"] = buildings.geometry.area.values

    before_dropna = len(features)
    features = features.dropna()
    dropped_na = before_dropna - len(features)
    if dropped_na:
        print(f"Dropped {dropped_na} buildings with no valid pixels under their footprint "
              "(likely off the edge of the tile or in a nodata gap).")

    cols = ["id"] + [f"{n}_mean" for n in BAND_NAMES] + [f"{n}_std" for n in BAND_NAMES] + ["area_m2"]
    features = features[cols]

    features.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(features)} building feature rows to {OUTPUT_CSV}")
    print(features.head())
    print(f"\nNo label column present. This table is ready for labeling "
          f"(e.g. add a 'roof_material' or 'class_label' column) before it "
          f"can be used to train train_riparian_rf.py.")


if __name__ == "__main__":
    main()
