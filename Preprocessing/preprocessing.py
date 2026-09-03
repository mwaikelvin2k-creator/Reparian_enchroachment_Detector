"""
Preprocessing module for land-cover and structure candidate screening.
Computes multispectral indices (NDVI, NDBI, MNDWI) from Sentinel-2
and exports sanitized training tables free of spatial distance leakage.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import rasterio
from rasterio.mask import mask
import ee
import geemap

WGS84 = "EPSG:4326"


def utm_epsg_for(lon: float, lat: float) -> str:
    zone = math.floor((lon + 180) / 6) + 1
    epsg_code = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg_code}"


def get_aoi(place_name: str, metric_crs: str, cache_path: Path | None = None) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if cache_path and cache_path.exists():
        aoi_wgs84 = gpd.read_file(cache_path).to_crs(WGS84)
    else:
        aoi_wgs84 = ox.geocode_to_gdf(place_name).to_crs(WGS84)
        if cache_path:
            aoi_wgs84.to_file(cache_path, driver="GPKG")
    return aoi_wgs84.to_crs(metric_crs), aoi_wgs84


def get_river(place_name: str, aoi_metric: gpd.GeoDataFrame, metric_crs: str) -> gpd.GeoDataFrame:
    try:
        waterways = ox.features_from_place(place_name, tags={"waterway": ["river", "stream"]})
    except Exception:
        aoi_wgs84 = aoi_metric.to_crs(WGS84)
        waterways = ox.features_from_polygon(aoi_wgs84.geometry.iloc[0], tags={"waterway": ["river", "stream"]})
        
    waterways = waterways[waterways.geometry.type.isin(["LineString", "MultiLineString"])]
    return gpd.clip(waterways.to_crs(metric_crs), aoi_metric)


def get_buildings_gee(gee_asset: str, aoi_wgs84: gpd.GeoDataFrame, metric_crs: str, local_path: str | None = None) -> gpd.GeoDataFrame:
    local_file = Path(local_path) if local_path else Path("gee_buildings.geojson")
    if local_file.exists():
        buildings = gpd.read_file(local_file)
    else:
        geom = aoi_wgs84.geometry.iloc[0]
        ee_geom = ee.Geometry.Polygon(list(geom.exterior.coords))
        fc = ee.FeatureCollection(gee_asset).filterBounds(ee_geom)
        buildings = geemap.ee_to_gdf(fc)
        buildings.to_file(local_file, driver="GeoJSON")
    return gpd.clip(buildings.to_crs(metric_crs), aoi_wgs84.to_crs(metric_crs))


def validate_and_clean_geometries(gdf: gpd.GeoDataFrame, min_area_m2: float = 10.0) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].make_valid()
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf.drop_duplicates(subset=["geometry"])
    gdf = gdf[gdf.geometry.area >= min_area_m2].reset_index(drop=True)
    gdf["id"] = gdf.index.astype(str)
    return gdf


def build_multispectral_composite(bounds: tuple, start_date: str, end_date: str, cloud_pct: float = 10):
    """Builds a composite with Red (B4), Green (B3), Blue (B2), NIR (B8), and SWIR (B11) bands."""
    region = ee.Geometry.BBox(*bounds)
    
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
          .select(["B4", "B3", "B2", "B8", "B11"]))
          
    composite = s2.median().clip(region)
    return composite, region


def compute_spectral_features(raster_path: str, polygons: gpd.GeoDataFrame) -> pd.DataFrame:
    """Calculates mean spectral band reflection and computes NDVI, NDBI, and MNDWI."""
    band_names = ["B4", "B3", "B2", "B8", "B11"]
    records = []
    
    with rasterio.open(raster_path) as src:
        for _, row in polygons.iterrows():
            feat = {"id": row["id"]}
            try:
                out_img, _ = mask(src, [row.geometry], crop=True)
                
                # Mean Band Values
                means = {}
                for idx, bname in enumerate(band_names):
                    valid = out_img[idx][out_img[idx] > 0]
                    means[bname] = float(np.mean(valid)) if len(valid) > 0 else 0.0
                    feat[f"mean_{bname}"] = means[bname]
                
                # Compute Normalized Indices
                # NDVI: (NIR - Red) / (NIR + Red)
                b8, b4 = means["B8"], means["B4"]
                feat["ndvi"] = (b8 - b4) / (b8 + b4) if (b8 + b4) != 0 else 0.0
                
                # NDBI: (SWIR - NIR) / (SWIR + NIR)
                b11 = means["B11"]
                feat["ndbi"] = (b11 - b8) / (b11 + b8) if (b11 + b8) != 0 else 0.0
                
                # MNDWI: (Green - SWIR) / (Green + SWIR)
                b3 = means["B3"]
                feat["mndwi"] = (b3 - b11) / (b3 + b11) if (b3 + b11) != 0 else 0.0
                
            except Exception:
                for bname in band_names:
                    feat[f"mean_{bname}"] = 0.0
                feat["ndvi"] = feat["ndbi"] = feat["mndwi"] = 0.0
                
            records.append(feat)
            
    return pd.DataFrame(records)


def run_preprocessing(
    place_name: str,
    gee_project: str,
    buffer_m: float = 30,
    gee_buildings_asset: str | None = None,
    gee_buildings_local_path: str | None = None,
    raster_local_path: str | None = None,  
    output_dir: str = "./output",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ee.Initialize(project=gee_project)

    aoi_cache_path = output_dir / "aoi_pinned.gpkg"
    if aoi_cache_path.exists():
        aoi_wgs84_probe = gpd.read_file(aoi_cache_path).to_crs(WGS84)
    else:
        aoi_wgs84_probe = ox.geocode_to_gdf(place_name)
        
    lon, lat = aoi_wgs84_probe.geometry.iloc[0].centroid.coords[0]
    metric_crs = utm_epsg_for(lon, lat)

    aoi, aoi_wgs84 = get_aoi(place_name, metric_crs, cache_path=aoi_cache_path)
    river = get_river(place_name, aoi, metric_crs)

    buildings = get_buildings_gee(gee_buildings_asset, aoi_wgs84, metric_crs, local_path=gee_buildings_local_path)
    buildings = validate_and_clean_geometries(buildings)

    # Label Encroachment solely as target label (NOT added to feature set)
    river_union = river.geometry.unary_union
    buildings["encroachment"] = buildings.geometry.intersects(river_union.buffer(buffer_m)).astype(int)

    raster_path = Path(raster_local_path) if raster_local_path else output_dir / "sentinel_composite.tif"
    if not raster_path.exists():
        raise FileNotFoundError(f"Multispectral raster not found at {raster_path}")

    # Compute Spectral Features & Indices
    features = compute_spectral_features(str(raster_path), buildings)
    
    # Merge Features (Excluding spatial distance columns from feature_cols)
    training_table = buildings[["id", "encroachment"]].merge(features, on="id")
    feature_cols = [c for c in features.columns if c != "id"]

    return {
        "aoi": aoi, "river": river, "buildings": buildings,
        "training_table": training_table, "feature_cols": feature_cols,
        "output_dir": output_dir, "buffer_m": buffer_m
    }