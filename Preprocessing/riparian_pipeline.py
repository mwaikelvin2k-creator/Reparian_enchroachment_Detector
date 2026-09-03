"""
Reusable riparian-encroachment preprocessing pipeline.

Every function takes a place/parameter as an argument instead of a hardcoded
name — swapping locations means changing arguments, not code. Run the whole
thing end-to-end with run_full_pipeline(), or call the functions individually
when only one stage needs to be redone.

Requires: geopandas, osmnx, shapely, rasterio, scipy, scikit-learn, joblib,
          earthengine-api, geemap
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.validation import make_valid
import rasterio
from rasterio.features import geometry_mask
from scipy.ndimage import uniform_filter
import ee
import geemap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

WGS84 = "EPSG:4326"


# --------------------------------------------------------------------------
# CRS
# --------------------------------------------------------------------------

def utm_epsg_for(lon: float, lat: float) -> str:
    """Metric UTM CRS for any point on Earth — replaces a hardcoded EPSG."""
    zone = int((lon + 180) // 6) + 1
    hemisphere = 326 if lat >= 0 else 327   # 326xx = northern, 327xx = southern
    return f"EPSG:{hemisphere}{zone:02d}"


# --------------------------------------------------------------------------
# Vector acquisition
# --------------------------------------------------------------------------

def get_aoi(place_name: str, metric_crs: str,
            cache_path: str | Path | None = None) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Study area boundary, geocoded by name. Returns (metric_crs copy, WGS84 copy).

    Geocoding the same place_name twice can return a different boundary
    between calls (OSM/Nominatim isn't perfectly stable), which silently
    breaks anything clipped against it later. If cache_path is given, the
    boundary is geocoded once and reused from disk on every subsequent call.
    """
    if cache_path and Path(cache_path).exists():
        aoi_wgs84 = gpd.read_file(cache_path).to_crs(WGS84)
    else:
        aoi_wgs84 = ox.geocode_to_gdf(place_name)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            aoi_wgs84.to_file(cache_path, driver="GPKG")

    aoi_metric = aoi_wgs84.to_crs(metric_crs)
    return aoi_metric, aoi_wgs84


def get_river(place_name: str, aoi_metric: gpd.GeoDataFrame, metric_crs: str,
              waterway_tag: str = "river") -> gpd.GeoDataFrame:
    """River (or other waterway) centerline from OSM, clipped to the AOI."""
    river = ox.features_from_place(place_name, tags={"waterway": waterway_tag})
    river = river[river.geometry.type.isin(["LineString", "MultiLineString"])]
    river = river.to_crs(metric_crs)
    return gpd.clip(river, aoi_metric)


def get_buildings_gee(gee_asset_path: str, aoi_wgs84: gpd.GeoDataFrame,
                       metric_crs: str, export_filename: str = "buildings_export",
                       local_path: str | None = None) -> gpd.GeoDataFrame:
    """Building footprints from a GEE FeatureCollection, filtered to the AOI
    on Earth Engine's servers, exported, then clipped and reprojected locally.

    gee_asset_path examples: "projects/sat-io/open-datasets/MSBuildings/Kenya"
    Call ee.Initialize(project=...) before this.

    local_path: where the file actually landed after downloading from Drive.
    Defaults to "<export_filename>.geojson" — pass the real filename if you
    renamed it or Drive changed it on download (e.g. added a prefix).
    """
    fc = ee.FeatureCollection(gee_asset_path)
    aoi_geom = geemap.geopandas_to_ee(aoi_wgs84)
    filtered = fc.filterBounds(aoi_geom)

    task = ee.batch.Export.table.toDrive(
        collection=filtered, description=export_filename, fileFormat="GeoJSON"
    )
    task.start()
    wait_for_task(task)

    path = local_path or f"{export_filename}.geojson"
    gdf = gpd.read_file(path).to_crs(metric_crs)
    return gpd.clip(gdf, aoi_wgs84.to_crs(metric_crs))


def get_buildings_osm(place_name: str, aoi_metric: gpd.GeoDataFrame,
                       metric_crs: str) -> gpd.GeoDataFrame:
    """Fallback building source when no GEE asset exists for the country."""
    gdf = ox.features_from_place(place_name, tags={"building": True})
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].to_crs(metric_crs)
    return gpd.clip(gdf, aoi_metric)


def wait_for_task(task, poll_seconds: int = 15) -> None:
    """Block until a GEE export task finishes; raises if it fails."""
    while task.active():
        print("Export running...", task.status()["state"])
        time.sleep(poll_seconds)
    state = task.status()["state"]
    print("Export finished:", state)
    if state == "FAILED":
        raise RuntimeError(task.status().get("error_message", "GEE export failed"))


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def validate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop empty geometries, repair invalid ones, explode multipart features."""
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(lambda g: make_valid(g) if not g.is_valid else g)
    return gdf.explode(index_parts=False).reset_index(drop=True)


def remove_duplicates_and_slivers(gdf: gpd.GeoDataFrame, min_area_m2: float = 4.0) -> gpd.GeoDataFrame:
    """Drop exact-duplicate geometries and slivers below a minimum area."""
    gdf = gdf.drop_duplicates(subset="geometry")
    return gdf[gdf.geometry.area > min_area_m2].reset_index(drop=True)


def clean_attributes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Lowercase column names, fill missing building-type tags."""
    gdf = gdf.rename(columns=str.lower)
    if "building" in gdf.columns:
        gdf["building_type"] = gdf["building"].fillna("unknown")
    return gdf


# --------------------------------------------------------------------------
# Buffer + labeling
# --------------------------------------------------------------------------

def label_encroachment(buildings: gpd.GeoDataFrame, river: gpd.GeoDataFrame,
                        buffer_m: float) -> gpd.GeoDataFrame:
    """Adds distance-to-river, buffer-intersection, encroached area, and a
    risk tier scaled to whatever buffer_m is passed."""
    river_union = river.geometry.union_all()
    buffer_geom = river_union.buffer(buffer_m)

    out = buildings.copy()
    out["dist_to_river_m"] = out.geometry.distance(river_union)
    out["inside_buffer"] = out.geometry.intersects(buffer_geom)

    intersections = out.geometry.intersection(buffer_geom)
    out["total_area_m2"] = out.geometry.area
    out["encroached_area_m2"] = intersections.area
    out["pct_encroached"] = (out["encroached_area_m2"] / out["total_area_m2"]) * 100

    third = buffer_m / 3
    def risk(dist):
        if dist <= third:
            return f"High Risk (<{third:.0f}m)"
        if dist <= 2 * third:
            return f"Medium Risk ({third:.0f}m-{2*third:.0f}m)"
        if dist <= buffer_m:
            return f"Low Risk ({2*third:.0f}m-{buffer_m:.0f}m)"
        return f"Safe Zone (>{buffer_m:.0f}m)"
    out["risk_category"] = out["dist_to_river_m"].apply(risk)

    return out, buffer_geom


# --------------------------------------------------------------------------
# Raster acquisition
# --------------------------------------------------------------------------

def build_sentinel_composite(bbox_wgs84, start_date: str, end_date: str,
                              bands=("B4", "B3", "B2", "B8", "B11"),
                              cloud_pct: float = 10, add_indices: bool = True):
    """Cloud-filtered median composite over the given date range and bbox."""
    region = ee.Geometry.Rectangle(list(bbox_wgs84))
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    image = coll.median().clip(region).select(list(bands))

    if add_indices and "B8" in bands and "B4" in bands:
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        image = image.addBands(ndvi)
    if add_indices and "B11" in bands and "B8" in bands:
        ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
        image = image.addBands(ndbi)

    return image.toFloat(), region   # toFloat unifies dtype across raw + derived bands


def preview_thumbnail_url(image, region, bands=("B4", "B3", "B2"),
                           vis_min=0, vis_max=3000, dimensions=512) -> str:
    """Cheap sanity check before spending minutes on a full export."""
    return image.select(list(bands)).getThumbURL({
        "region": region, "dimensions": dimensions,
        "min": vis_min, "max": vis_max, "bands": list(bands),
    })


def export_image_to_drive(image, region, description: str, filename: str,
                           metric_crs: str, scale: int = 10, folder: str = "EarthEngine") -> None:
    task = ee.batch.Export.image.toDrive(
        image=image, description=description, folder=folder,
        fileNamePrefix=filename, region=region, scale=scale, crs=metric_crs,
    )
    task.start()
    wait_for_task(task)


def confirm_raster_alignment(raster_path: str, aoi_metric: gpd.GeoDataFrame) -> bool:
    """Bounds-overlap and nodata-fraction check. Returns False on a likely bad tile."""
    with rasterio.open(raster_path) as src:
        raster_bounds = src.bounds
        raster_crs = src.crs
        arr = src.read(1).astype(float)

    aoi_bounds = aoi_metric.to_crs(raster_crs).total_bounds
    overlap_x = min(raster_bounds.right, aoi_bounds[2]) - max(raster_bounds.left, aoi_bounds[0])
    overlap_y = min(raster_bounds.top, aoi_bounds[3]) - max(raster_bounds.bottom, aoi_bounds[1])
    nodata_fraction = (np.isnan(arr) | (arr == 0)).mean()

    print("Raster bounds:", raster_bounds, " AOI bounds:", aoi_bounds)
    print(f"Nodata/zero fraction: {nodata_fraction:.2%}")

    return overlap_x > 0 and overlap_y > 0 and nodata_fraction < 0.5


# --------------------------------------------------------------------------
# Zonal + texture feature extraction
# --------------------------------------------------------------------------

def _local_std(band: np.ndarray, window: int) -> np.ndarray:
    mean = uniform_filter(band, size=window)
    sq_mean = uniform_filter(band**2, size=window)
    return np.sqrt(np.maximum(sq_mean - mean**2, 0))


def compute_building_features(raster_path: str, buildings: gpd.GeoDataFrame,
                               band_names, texture_window: int = 5) -> pd.DataFrame:
    """Mean/std/texture per band under each footprint. Uses all_touched=True
    since footprints are frequently smaller than one pixel."""
    with rasterio.open(raster_path) as src:
        band_stack = src.read().astype(float)
        transform = src.transform
        out_shape = (src.height, src.width)

    texture_stack = np.stack([_local_std(band_stack[b], texture_window)
                               for b in range(len(band_names))])

    col_names = [f"{b}_{stat}" for b in band_names for stat in ("mean", "std")]
    col_names += [f"{b}_texture" for b in band_names]

    rows = []
    for geom in buildings.geometry:
        inside = geometry_mask(
            [geom.__geo_interface__], out_shape=out_shape, transform=transform,
            invert=True, all_touched=True,
        )
        row = []
        for b in range(len(band_names)):
            vals = band_stack[b][inside]
            row.extend([vals.mean(), vals.std()] if vals.size else [np.nan, np.nan])
        for b in range(len(band_names)):
            tex = texture_stack[b][inside]
            row.append(tex.mean() if tex.size else np.nan)
        rows.append(row)

    return pd.DataFrame(rows, columns=col_names)


# --------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------

def prepare_training_table(buildings: gpd.GeoDataFrame, feature_df: pd.DataFrame,
                            river: gpd.GeoDataFrame, buffer_m: float) -> tuple[pd.DataFrame, list]:
    """Joins zonal features onto buildings, drops unscoreable rows, and labels
    encroachment by centroid distance — the label the model is trained on."""
    df = buildings.reset_index(drop=True).copy()
    df = pd.concat([df, feature_df], axis=1)
    df["id"] = df.index.astype(str)

    feature_cols = list(feature_df.columns) + ["total_area_m2"]
    df = df.dropna(subset=list(feature_df.columns)).reset_index(drop=True)

    river_union = river.geometry.union_all()
    df["centroid_dist_to_river_m"] = df.geometry.centroid.distance(river_union)
    df["encroachment"] = (df["centroid_dist_to_river_m"] <= buffer_m).astype(int)

    return df, feature_cols


def train_random_forest(df: pd.DataFrame, feature_cols: list, label_col: str = "encroachment",
                         test_size: float = 0.2, random_state: int = 42,
                         rf_params: dict | None = None):
    """Stratified split + fit. Falls back to a plain split if the minority
    class is too small to stratify safely."""
    rf_params = rf_params or dict(n_estimators=300, max_depth=10,
                                   class_weight="balanced", n_jobs=-1)
    X, y = df[feature_cols], df[label_col]

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df["id"], test_size=test_size, stratify=stratify, random_state=random_state,
    )

    df["split"] = "train"
    df.loc[df["id"].isin(id_test), "split"] = "test"

    model = RandomForestClassifier(random_state=random_state, **rf_params)
    model.fit(X_train, y_train)

    return model, df, X_train, X_test, y_train, y_test, rf_params


def evaluate_model(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def export_model_artifacts(model, df: pd.DataFrame, feature_cols: list, metrics: dict,
                            rf_params: dict, buffer_m: float,
                            models_dir: Path, data_dir: Path) -> None:
    """Writes the model + metadata + predictions in the schema a dashboard reads."""
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "riparian_rf_model.joblib")

    X = df[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    predictions = pd.DataFrame({
        "id": df["id"], "split": df["split"], "y_true": df["encroachment"],
        "rf_pred": (proba >= 0.5).astype(int), "rf_proba": proba,
    })
    predictions.to_csv(data_dir / "rf_predictions.csv", index=False)

    labeled_cols = ["id", "split"] + feature_cols + ["centroid_dist_to_river_m", "encroachment"]
    df[labeled_cols].to_csv(data_dir / "building_features_labeled.csv", index=False)

    class_balance = {str(k): int(v) for k, v in df["encroachment"].value_counts().items()}
    metadata = {
        "model_type": "RandomForestClassifier",
        "params": rf_params,
        "feature_cols": feature_cols,
        "buffer_meters": buffer_m,
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "n_total": int(len(df)),
        "class_balance": class_balance,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {"encroachment": {k: metrics[k] for k in ("precision", "recall", "f1")},
                    "roc_auc": metrics["roc_auc"]},
        "feature_importances": dict(zip(feature_cols, map(float, model.feature_importances_))),
    }
    with open(models_dir / "rf_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def run_full_pipeline(
    place_name: str,
    gee_project: str,
    buffer_m: float = 30,
    building_source: str = "gee",          # "gee" or "osm"
    gee_buildings_asset: str | None = None,  # required if building_source == "gee"
    gee_buildings_local_path: str | None = None,  # actual downloaded filename, if it differs
    sentinel_dates: tuple = ("2024-01-01", "2024-12-31"),
    cloud_pct: float = 10,
    raster_local_path: str | None = None,  # actual downloaded raster filename, if it differs
    output_dir: str = "./output",
) -> dict:
    """Runs acquisition through model export for one location. Returns the
    key outputs/paths so the caller can inspect or chain further steps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ee.Initialize(project=gee_project)

    aoi_cache_path = output_dir / "aoi_pinned.gpkg"
    aoi_wgs84_probe = ox.geocode_to_gdf(place_name) if not aoi_cache_path.exists() \
        else gpd.read_file(aoi_cache_path).to_crs(WGS84)
    lon, lat = aoi_wgs84_probe.geometry.iloc[0].centroid.coords[0]
    metric_crs = utm_epsg_for(lon, lat)

    aoi, aoi_wgs84 = get_aoi(place_name, metric_crs, cache_path=aoi_cache_path)
    river = get_river(place_name, aoi, metric_crs)

    if building_source == "gee":
        if not gee_buildings_asset:
            raise ValueError("gee_buildings_asset is required when building_source='gee'")
        buildings = get_buildings_gee(gee_buildings_asset, aoi_wgs84, metric_crs,
                                       local_path=gee_buildings_local_path)
    else:
        buildings = get_buildings_osm(place_name, aoi, metric_crs)

    buildings = validate_geometries(buildings)
    buildings = remove_duplicates_and_slivers(buildings)
    buildings = clean_attributes(buildings)
    buildings, buffer_geom = label_encroachment(buildings, river, buffer_m)

    image, region = build_sentinel_composite(aoi_wgs84.total_bounds, *sentinel_dates,
                                              cloud_pct=cloud_pct)
    raster_path = Path(raster_local_path) if raster_local_path else output_dir / "sentinel_composite.tif"
    export_image_to_drive(image, region, "sentinel_composite", raster_path.stem, metric_crs)
    # NOTE: download the exported file from Drive to raster_path before continuing
    # (or pass raster_local_path pointing at wherever it actually landed).

    if not confirm_raster_alignment(str(raster_path), aoi):
        raise RuntimeError("Raster tile failed alignment/nodata check — inspect before proceeding")

    band_names = ["B4", "B3", "B2"]
    features = compute_building_features(str(raster_path), buildings, band_names)
    training_table, feature_cols = prepare_training_table(buildings, features, river, buffer_m)

    model, training_table, X_train, X_test, y_train, y_test, rf_params = train_random_forest(
        training_table, feature_cols
    )
    metrics = evaluate_model(model, X_test, y_test)

    export_model_artifacts(
        model, training_table, feature_cols, metrics, rf_params, buffer_m,
        models_dir=output_dir / "models", data_dir=output_dir / "data",
    )

    buildings.to_file(output_dir / "buildings_classified.gpkg", driver="GPKG")
    river.to_file(output_dir / "river.gpkg", driver="GPKG")
    aoi.to_file(output_dir / "aoi.gpkg", driver="GPKG")

    return {
        "aoi": aoi, "river": river, "buildings": buildings,
        "model": model, "metrics": metrics, "output_dir": output_dir,
    }


def run_multiple_locations(locations: list[dict], gee_project: str,
                            stop_on_error: bool = False) -> dict:
    """Runs run_full_pipeline once per location. Each dict in `locations`
    supplies that location's arguments (place_name, gee_buildings_asset,
    output_dir, etc.) — anything accepted by run_full_pipeline except
    gee_project, which is shared across all locations.

    A failure in one location is recorded and skipped rather than stopping
    the batch, unless stop_on_error=True.

    Returns {place_name: {"status": "ok"/"failed", "results"/"error": ...}}
    """
    ee.Initialize(project=gee_project)

    summary = {}
    for cfg in locations:
        place_name = cfg["place_name"]
        print(f"\n=== {place_name} ===")
        try:
            results = run_full_pipeline(gee_project=gee_project, **cfg)
            summary[place_name] = {"status": "ok", "results": results}
        except Exception as exc:
            print(f"FAILED: {place_name} — {exc}")
            summary[place_name] = {"status": "failed", "error": str(exc)}
            if stop_on_error:
                raise

    ok = sum(1 for v in summary.values() if v["status"] == "ok")
    print(f"\n{ok}/{len(locations)} locations completed successfully")
    return summary


if __name__ == "__main__":
    run_full_pipeline(
        place_name="Kasarani, Nairobi, Kenya",
        gee_project="causal-bus-404912",
        buffer_m=30,
        gee_buildings_asset="projects/sat-io/open-datasets/MSBuildings/Kenya",
        output_dir="./output/kasarani",
    )

    # Multiple locations in one call — each dict supplies that location's
    # own arguments; gee_project is shared.
    # run_multiple_locations(
    #     locations=[
    #         dict(place_name="Kasarani, Nairobi, Kenya", buffer_m=30,
    #              gee_buildings_asset="projects/sat-io/open-datasets/MSBuildings/Kenya",
    #              output_dir="./output/kasarani"),
    #         dict(place_name="Ibadan, Nigeria", buffer_m=30,
    #              gee_buildings_asset="projects/sat-io/open-datasets/MSBuildings/Nigeria",
    #              output_dir="./output/ibadan"),
    #     ],
    #     gee_project="causal-bus-404912",
    # )