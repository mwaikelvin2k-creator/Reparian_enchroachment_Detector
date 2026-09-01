"""
Auto-label buildings for riparian encroachment based on real distance to river.

Inputs (real, uploaded):
  - data/vectors/kasarani_buildings.geojson       (57k+ building footprints, EPSG:4326)
  - data/vectors/kasarani_river.gpkg              (21 river features, EPSG:32737)
  - data/vectors/kasarani_aoi.gpkg                (1 AOI polygon, EPSG:32737)
  - data/processed/kasarani_building_features.csv (R/G/B zonal stats, no label yet)

Output:
  - data/processed/kasarani_building_features_labeled.csv
    same columns as before, plus:
      dist_to_river_m : real distance in metres from building centroid to nearest river
      encroachment    : 1 if dist_to_river_m <= BUFFER_METERS else 0
"""

import geopandas as gpd
import pandas as pd

BUILDINGS_PATH = "data/vectors/kasarani_buildings.geojson"
RIVER_PATH = "data/vectors/kasarani_river.gpkg"
AOI_PATH = "data/vectors/kasarani_aoi.gpkg"
FEATURES_CSV = "data/processed/kasarani_building_features.csv"
OUTPUT_CSV = "data/processed/kasarani_building_features_labeled.csv"

BUFFER_METERS = 60  # matches the legal riparian buffer used elsewhere in this project
TARGET_CRS = "EPSG:32737"  # UTM 37S — matches river and AOI layers, metric units


def main():
    buildings = gpd.read_file(BUILDINGS_PATH)
    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings = buildings.to_crs(TARGET_CRS)

    river = gpd.read_file(RIVER_PATH).to_crs(TARGET_CRS)
    aoi = gpd.read_file(AOI_PATH).to_crs(TARGET_CRS)

    # Clip buildings to the AOI boundary
    aoi_geom = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
    before = len(buildings)
    buildings = buildings[buildings.geometry.centroid.within(aoi_geom)].copy()
    print(f"Clipped to AOI: {before} -> {len(buildings)} buildings.")

    # Merge all river features into a single geometry for nearest-distance calc
    river_union = river.union_all() if hasattr(river, "union_all") else river.unary_union

    centroids = buildings.geometry.centroid
    buildings["dist_to_river_m"] = centroids.distance(river_union)
    buildings["encroachment"] = (buildings["dist_to_river_m"] <= BUFFER_METERS).astype(int)

    print(f"\nEncroachment label distribution:")
    print(buildings["encroachment"].value_counts())
    print(f"\ndist_to_river_m stats:\n{buildings['dist_to_river_m'].describe()}")

    labels = buildings[["id", "dist_to_river_m", "encroachment"]]

    features = pd.read_csv(FEATURES_CSV)
    merged = features.merge(labels, on="id", how="inner")
    dropped = len(features) - len(merged)
    if dropped:
        print(f"\nDropped {dropped} rows from feature table with no matching AOI-clipped building "
              "(outside AOI or id mismatch).")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(merged)} labeled rows to {OUTPUT_CSV}")
    print(merged.head())


if __name__ == "__main__":
    main()
