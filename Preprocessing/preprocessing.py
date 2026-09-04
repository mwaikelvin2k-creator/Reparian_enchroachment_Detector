"""
Preprocessing module for land-cover and structure candidate screening.
Computes multispectral indices (NDVI, NDBI, MNDWI) from Sentinel-2 and
exports sanitized training tables for three separate tasks: encroachment
(building-only), building-vs-not-building detection, and four-class
land-cover classification.

FIXES applied (vs original):
- label_land_cover now emits a loud warning that labels are heuristic
  (threshold-based) and NOT validated ground truth.
- Added optional external_landcover_path parameter to run_land_cover_preprocessing
  so users can supply real land-cover labels (e.g. ESA WorldCover) instead
  of the circular index-threshold approach.
- compute_spectral_features now also computes polygon centroids (x_centroid,
  y_centroid) and stores them in the returned DataFrame, enabling spatial
  block train/test splits in modeling.py.
"""

from pathlib import Path
import math
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import rasterio
from rasterio.features import geometry_mask
import ee
import geemap

WGS84 = "EPSG:4326"
BAND_NAMES = ["B4", "B3", "B2", "B8", "B11"]


def utm_epsg_for(lon: float, lat: float) -> str:
    zone = math.floor((lon + 180) / 6) + 1
    epsg_code = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg_code}"


def get_aoi(place_name: str, metric_crs: str,
            cache_path: Path | None = None) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
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


def get_buildings_gee(gee_asset: str, aoi_wgs84: gpd.GeoDataFrame, metric_crs: str,
                       local_path: str | None = None) -> gpd.GeoDataFrame:
    """Building footprints for the AOI. Tries a direct in-memory pull first;
    falls back to an export-to-Drive task for AOIs too large for that path
    (a manual download is then required before re-running with local_path
    pointing at the downloaded file)."""
    local_file = Path(local_path) if local_path else Path("gee_buildings.geojson")
    if local_file.exists():
        buildings = gpd.read_file(local_file)
    else:
        geom = aoi_wgs84.geometry.iloc[0]
        ee_geom = ee.Geometry.Polygon(list(geom.exterior.coords))
        fc = ee.FeatureCollection(gee_asset).filterBounds(ee_geom)
        try:
            buildings = geemap.ee_to_gdf(fc)
        except Exception:
            task = ee.batch.Export.table.toDrive(
                collection=fc, description="gee_buildings_export", fileFormat="GeoJSON"
            )
            task.start()
            wait_for_task(task)
            raise FileNotFoundError(
                "Building footprints were too large to pull directly and have been "
                "queued as a Drive export named gee_buildings_export.geojson. "
                "Download it, then re-run with local_path pointing at that file."
            )
        buildings.to_file(local_file, driver="GeoJSON")
    if "fid" in buildings.columns:
        buildings = buildings.drop(columns=["fid"])
    buildings = buildings.reset_index(drop=True)
    return gpd.clip(buildings.to_crs(metric_crs), aoi_wgs84.to_crs(metric_crs))


def wait_for_task(task, poll_seconds: int = 15) -> None:
    while task.active():
        print("Export running...", task.status()["state"])
        time.sleep(poll_seconds)
    state = task.status()["state"]
    print("Export finished:", state)
    if state == "FAILED":
        raise RuntimeError(task.status().get("error_message", "GEE export failed"))


def validate_and_clean_geometries(gdf: gpd.GeoDataFrame, min_area_m2: float = 10.0) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].make_valid()
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf.drop_duplicates(subset=["geometry"])
    gdf = gdf[gdf.geometry.area >= min_area_m2].reset_index(drop=True)
    gdf["id"] = gdf.index.astype(str)
    return gdf


def build_multispectral_composite(bounds: tuple, start_date: str, end_date: str, cloud_pct: float = 10):
    """Cloud-filtered median composite over the given bounds and date range.
    Bands: Red (B4), Green (B3), Blue (B2), NIR (B8), SWIR (B11)."""
    region = ee.Geometry.BBox(*bounds)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
          .select(BAND_NAMES))

    composite = s2.median().clip(region).toFloat()
    return composite, region


def export_composite_to_drive(image, region, filename: str, metric_crs: str,
                               scale: int = 10, folder: str = "EarthEngine") -> None:
    task = ee.batch.Export.image.toDrive(
        image=image, description=filename, folder=folder,
        fileNamePrefix=filename, region=region, scale=scale, crs=metric_crs,
    )
    task.start()
    wait_for_task(task)


def _sample_raster_at_points(raster_path: str, points: list) -> np.ndarray:
    """Pixel values at each (x, y) coordinate, one row per point, one column
    per band. NaN rows mark points with no data at that location."""
    with rasterio.open(raster_path) as src:
        samples = np.array(list(src.sample(points)), dtype=float)
    samples[samples == 0] = np.nan
    all_zero_rows = np.all(np.isnan(samples) | (samples == 0), axis=1)
    samples[all_zero_rows] = np.nan
    return samples


def _spectral_indices(means: dict) -> dict:
    b8, b4, b11, b3 = means["B8"], means["B4"], means["B11"], means["B3"]
    return {
        "ndvi": (b8 - b4) / (b8 + b4) if (b8 + b4) else np.nan,
        "ndbi": (b11 - b8) / (b11 + b8) if (b11 + b8) else np.nan,
        "mndwi": (b3 - b11) / (b3 + b11) if (b3 + b11) else np.nan,
    }


def compute_spectral_features(raster_path: str, polygons: gpd.GeoDataFrame) -> pd.DataFrame:
    """Mean spectral reflectance and NDVI/NDBI/MNDWI under each polygon.
    Uses all_touched=True since footprints are frequently smaller than a
    single Sentinel-2 pixel; a polygon with no overlapping valid pixels
    gets NaN features rather than a fabricated all-zero reading, so it can
    be dropped downstream instead of silently entering training as real data.

    FIX: Also stores polygon centroids as x_centroid, y_centroid to enable
    spatial block train/test splits in modeling.py.
    """
    records = []
    with rasterio.open(raster_path) as src:
        band_stack = src.read().astype(float)
        band_stack[band_stack == 0] = np.nan
        transform = src.transform
        out_shape = (src.height, src.width)

        for _, row in polygons.iterrows():
            feat = {"id": row["id"]}
            # Store centroid for spatial splits
            centroid = row.geometry.centroid
            feat["x_centroid"] = centroid.x
            feat["y_centroid"] = centroid.y

            inside = geometry_mask(
                [row.geometry.__geo_interface__], out_shape=out_shape,
                transform=transform, invert=True, all_touched=True,
            )
            means = {}
            valid = inside.any()
            if valid:
                for idx, bname in enumerate(BAND_NAMES):
                    vals = band_stack[idx][inside]
                    vals = vals[~np.isnan(vals)]
                    if vals.size == 0:
                        valid = False
                        break
                    means[bname] = float(vals.mean())
                    feat[f"mean_{bname}"] = means[bname]

            if valid:
                feat.update(_spectral_indices(means))
            else:
                for bname in BAND_NAMES:
                    feat[f"mean_{bname}"] = np.nan
                feat["ndvi"] = feat["ndbi"] = feat["mndwi"] = np.nan

            records.append(feat)

    return pd.DataFrame(records)


def extract_point_features(raster_path: str, points: gpd.GeoDataFrame) -> pd.DataFrame:
    """Same feature schema as compute_spectral_features, for point geometries
    (single-pixel lookups rather than polygon zonal statistics).
    For point samples we store the point coordinates as x_centroid/y_centroid."""
    coords = [(geom.x, geom.y) for geom in points.geometry]
    samples = _sample_raster_at_points(raster_path, coords)

    records = []
    for pid, (geom, vals) in enumerate(zip(points.geometry, samples)):
        feat = {"id": pid if "id" not in points.columns else points.iloc[pid]["id"]}
        feat["x_centroid"] = geom.x
        feat["y_centroid"] = geom.y
        if np.any(np.isnan(vals)):
            for bname in BAND_NAMES:
                feat[f"mean_{bname}"] = np.nan
            feat["ndvi"] = feat["ndbi"] = feat["mndwi"] = np.nan
        else:
            means = dict(zip(BAND_NAMES, vals))
            for bname, v in means.items():
                feat[f"mean_{bname}"] = float(v)
            feat.update(_spectral_indices(means))
        records.append(feat)

    return pd.DataFrame(records)


def sample_negative_points(aoi_metric: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame,
                            n_samples: int, min_dist_m: float = 15,
                            random_state: int = 42) -> gpd.GeoDataFrame:
    """Random points inside the AOI that fall outside every building
    footprint (plus a small buffer), used as negative examples for the
    building-vs-not-building classifier."""
    rng = np.random.default_rng(random_state)
    minx, miny, maxx, maxy = aoi_metric.total_bounds
    buildings_union = buildings.geometry.union_all().buffer(min_dist_m)
    aoi_union = aoi_metric.geometry.union_all()

    points, attempts, max_attempts = [], 0, n_samples * 50
    while len(points) < n_samples and attempts < max_attempts:
        candidate = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        attempts += 1
        if aoi_union.contains(candidate) and not buildings_union.contains(candidate):
            points.append(candidate)

    ids = [f"neg_{i}" for i in range(len(points))]
    return gpd.GeoDataFrame({"id": ids}, geometry=points, crs=aoi_metric.crs)


def sample_grid_points(aoi_metric: gpd.GeoDataFrame, spacing_m: float = 200) -> gpd.GeoDataFrame:
    """Regularly spaced points across the AOI, used to build a land-cover
    training set independent of building locations."""
    minx, miny, maxx, maxy = aoi_metric.total_bounds
    aoi_union = aoi_metric.geometry.union_all()
    xs = np.arange(minx, maxx, spacing_m)
    ys = np.arange(miny, maxy, spacing_m)

    points = [Point(x, y) for x in xs for y in ys if aoi_union.contains(Point(x, y))]
    ids = [f"grid_{i}" for i in range(len(points))]
    return gpd.GeoDataFrame({"id": ids}, geometry=points, crs=aoi_metric.crs)


def label_land_cover(features: pd.DataFrame) -> pd.Series:
    """Assigns one of four land-cover classes from spectral indices:
    water, vegetation, built_up, bare_soil.

    WARNING: These are HEURISTIC, threshold-based pseudo-labels derived
    from the SAME indices used as model features. This creates a circular
    dependency: the model learns to replicate index thresholds rather than
    generalizable spectral patterns. For any publication or operational
    deployment, replace this with validated ground-truth labels
    (e.g. ESA WorldCover, manual photo-interpretation).
    """
    warnings.warn(
        "label_land_cover() uses heuristic index thresholds (MNDWI>0 -> water, "
        "NDVI>0.3 -> vegetation, NDBI>0 -> built_up). These are NOT validated "
        "ground-truth labels. The resulting model learns to replicate index "
        "thresholds, which is circular. For production use, supply real labels "
        "via external_landcover_path in run_land_cover_preprocessing().",
        category=UserWarning,
        stacklevel=2,
    )

    def classify(row):
        if row["mndwi"] > 0:
            return "water"
        if row["ndvi"] > 0.3:
            return "vegetation"
        if row["ndbi"] > 0:
            return "built_up"
        return "bare_soil"
    return features.apply(classify, axis=1)


# --------------------------------------------------------------------------
# Orchestrators
# --------------------------------------------------------------------------

def run_preprocessing(
    place_name: str,
    gee_project: str,
    buffer_m: float = 30,
    gee_buildings_asset: str | None = None,
    gee_buildings_local_path: str | None = None,
    sentinel_dates: tuple = ("2024-01-01", "2024-12-31"),
    cloud_pct: float = 10,
    raster_local_path: str | None = None,
    output_dir: str = "./output",
) -> dict:
    """Acquisition through feature extraction for the encroachment task
    (building-only). Also returns the raw building features, raster path,
    and AOI so the building-detection and land-cover tasks can reuse them
    without re-acquiring anything."""
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

    buildings = get_buildings_gee(gee_buildings_asset, aoi_wgs84, metric_crs,
                                   local_path=gee_buildings_local_path)
    buildings = validate_and_clean_geometries(buildings)

    river_union = river.geometry.union_all()
    buildings["encroachment"] = buildings.geometry.intersects(river_union.buffer(buffer_m)).astype(int)

    raster_path = Path(raster_local_path) if raster_local_path else output_dir / "sentinel_composite.tif"
    if not raster_path.exists():
        image, region = build_multispectral_composite(aoi_wgs84.total_bounds, *sentinel_dates,
                                                        cloud_pct=cloud_pct)
        export_composite_to_drive(image, region, raster_path.stem, metric_crs)
        raise FileNotFoundError(
            f"Composite exported as {raster_path.stem} to the EarthEngine Drive folder. "
            f"Download it to {raster_path}, then re-run with raster_local_path set."
        )

    features = compute_spectral_features(str(raster_path), buildings)
    training_table = buildings[["id", "encroachment"]].merge(features, on="id")
    training_table = training_table.dropna(subset=[c for c in features.columns if c != "id"])
    feature_cols = [c for c in features.columns if c not in ("id", "x_centroid", "y_centroid")]

    return {
        "aoi": aoi, "aoi_wgs84": aoi_wgs84, "river": river, "buildings": buildings,
        "features": features, "training_table": training_table, "feature_cols": feature_cols,
        "raster_path": raster_path, "metric_crs": metric_crs,
        "output_dir": output_dir, "buffer_m": buffer_m,
    }


def run_building_detection_preprocessing(prep: dict, n_negative_samples: int | None = None,
                                          random_state: int = 42) -> dict:
    """Building-vs-not-building training table, reusing the AOI/buildings/
    raster already acquired by run_preprocessing. Positive examples are the
    known building footprints; negative examples are random non-building
    locations across the AOI."""
    buildings, aoi, raster_path = prep["buildings"], prep["aoi"], prep["raster_path"]
    n_negative_samples = n_negative_samples or len(buildings)

    positive_features = prep["features"].copy()
    positive_features["is_building"] = 1

    negatives = sample_negative_points(aoi, buildings, n_negative_samples, random_state=random_state)
    negative_features = extract_point_features(str(raster_path), negatives)
    negative_features["is_building"] = 0

    feature_cols = [c for c in positive_features.columns if c not in ("id", "is_building", "x_centroid", "y_centroid")]
    combined = pd.concat([positive_features, negative_features], ignore_index=True)
    combined = combined.dropna(subset=feature_cols).reset_index(drop=True)

    return {"training_table": combined, "feature_cols": feature_cols, "output_dir": prep["output_dir"]}


def run_land_cover_preprocessing(prep: dict, grid_spacing_m: float = 200,
                                  external_landcover_path: str | None = None) -> dict:
    """Four-class land-cover training table, reusing the AOI/raster already
    acquired by run_preprocessing. Sampled on a regular grid across the AOI,
    independent of building locations.

    NEW: If external_landcover_path is provided, it must be a GeoDataFrame
    (or path to one) with a 'land_cover' column and polygon geometry. Grid
    points are spatially joined to these polygons to obtain real labels
    instead of the heuristic index-threshold labels.
    """
    aoi, raster_path = prep["aoi"], prep["raster_path"]

    grid_points = sample_grid_points(aoi, spacing_m=grid_spacing_m)
    features = extract_point_features(str(raster_path), grid_points)

    feature_cols = [c for c in features.columns if c not in ("id", "x_centroid", "y_centroid")]
    features = features.dropna(subset=feature_cols).reset_index(drop=True)

    if external_landcover_path is not None:
        # Use real external labels
        lc_gdf = gpd.read_file(external_landcover_path) if isinstance(external_landcover_path, str) else external_landcover_path
        if "land_cover" not in lc_gdf.columns:
            raise ValueError("External land-cover data must contain a 'land_cover' column.")
        lc_gdf = lc_gdf.to_crs(aoi.crs)
        # Spatial join: each grid point gets the land_cover of the polygon it falls inside
        joined = gpd.sjoin(
            gpd.GeoDataFrame(features, geometry=gpd.points_from_xy(features.x_centroid, features.y_centroid), crs=aoi.crs),
            lc_gdf[["geometry", "land_cover"]],
            how="left", predicate="within"
        )
        features["land_cover"] = joined["land_cover"]
        # Drop points that didn't fall inside any land-cover polygon
        features = features.dropna(subset=["land_cover"]).reset_index(drop=True)
        print(f"Using external land-cover labels: {features['land_cover'].value_counts().to_dict()}")
    else:
        features["land_cover"] = label_land_cover(features)

    return {"training_table": features, "feature_cols": feature_cols, "output_dir": prep["output_dir"]}