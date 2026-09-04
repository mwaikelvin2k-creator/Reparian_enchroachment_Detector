# File: Preprocessing/preprocessing_utils.py
"""
Preprocessing pipeline for the Nairobi River riparian encroachment project.
Acquisition -> cleaning -> vector-raster fusion -> training CSV, tiled.
"""

import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import geopandas as gpd
from shapely.geometry import box
from shapely.validation import make_valid
import rasterio
from rasterio.transform import xy
from rasterio.features import rasterize
from scipy.spatial import cKDTree

import ee


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_CRS = "EPSG:32737"              # metric CRS for Kenya (UTM 37S)
WGS84 = "EPSG:4326"
MIN_BUILDING_AREA_M2 = 4
BAND_ORDER = ["B4", "B3", "B2", "B8"]
GEE_PROJECT_ID = "riparian-encroachment"
TILE_SIZE_M = 5000                     # 5km tiles keep each GEE export small
SENTINEL_SCALE_M = 10 

RAW_DIR = Path("../data/raw")
TILE_DIR = RAW_DIR / "tiles" 
OUTPUT_DIR = Path("../data/output")
for d in (RAW_DIR, TILE_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

AOI_PATH = RAW_DIR / "nairobi_aoi.gpkg"
RIVER_SHAPEFILE_PATH = RAW_DIR / "gis_osm_waterways_free_1.shp"  # local vector input
RIVER_PATH = RAW_DIR / "nairobi_river.gpkg"                      # cleaned/clipped output
TRAINING_CSV_PATH = OUTPUT_DIR / "nairobi_training_pixels.csv"


# ---------------------------------------------------------------------------
# Step 1 — AOI acquisition
# ---------------------------------------------------------------------------

class AOIAcquirer:
    """Pulls the Nairobi administrative boundary via OSM (osmnx)."""

    def __init__(self, place_name: str = "Nairobi, Kenya"):
        self.place_name = place_name

    def acquire(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        import osmnx as ox
        aoi_wgs84 = ox.geocode_to_gdf(self.place_name)
        aoi_metric = aoi_wgs84.to_crs(TARGET_CRS)
        return aoi_wgs84, aoi_metric

    def save(self, aoi_metric: gpd.GeoDataFrame, path: str):
        aoi_metric.to_file(path, driver="GPKG")

# ---------------------------------------------------------------------------
# Step 2 — River vector: load from the local shapefile, not a live pull
# ---------------------------------------------------------------------------

class LocalRiverLoader:
    """Loads river/stream geometries from a local Geofabrik-style shapefile."""

    def __init__(self, shapefile_path: str):
        self.shapefile_path = shapefile_path

    def load(self, aoi_metric: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        raw = gpd.read_file(self.shapefile_path)

        if "fclass" in raw.columns:
            raw = raw[raw["fclass"].isin(["river", "stream"])].copy()

        raw = raw[raw.geometry.type.isin(["LineString", "MultiLineString"])]
        river = raw.to_crs(TARGET_CRS)
        river = gpd.clip(river, aoi_metric)
        return river.reset_index(drop=True)

    def save(self, river: gpd.GeoDataFrame, path: str):
        river.to_file(path, driver="GPKG")

# ---------------------------------------------------------------------------
# Step 3 — AOI tiling
# ---------------------------------------------------------------------------

class AOITiler:
    """Splits the AOI bounding box into a grid of tiles for per-tile processing."""

    def __init__(self, tile_size_m: float = TILE_SIZE_M):
        self.tile_size_m = tile_size_m

    def generate_tiles(self, aoi_metric: gpd.GeoDataFrame) -> list:
        aoi_geom = aoi_metric.geometry.iloc[0]
        minx, miny, maxx, maxy = aoi_metric.total_bounds

        tiles = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                tile_box = box(x, y, x + self.tile_size_m, y + self.tile_size_m)
                if aoi_geom.intersects(tile_box):
                    tiles.append(tile_box)
                y += self.tile_size_m
            x += self.tile_size_m
        return tiles

    @staticmethod
    def to_ee_geometry(tile_box):
        """Converts a metric-CRS tile box to an Earth Engine geometry (WGS84)."""
        tile_gdf = gpd.GeoDataFrame(geometry=[tile_box], crs=TARGET_CRS).to_crs(WGS84)
        bounds = tile_gdf.total_bounds
        return ee.Geometry.Rectangle(list(bounds))

# ---------------------------------------------------------------------------
# Vector cleaning
# ---------------------------------------------------------------------------

class VectorCleaner:
    def __init__(self, min_area_m2: float = MIN_BUILDING_AREA_M2):
        self.min_area_m2 = min_area_m2

    def clean(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        buildings = buildings.to_crs(TARGET_CRS)
        buildings = buildings[~buildings.geometry.is_empty & buildings.geometry.notna()]
        buildings["geometry"] = buildings.geometry.apply(
            lambda g: make_valid(g) if not g.is_valid else g
        )
        buildings = buildings.explode(index_parts=False).reset_index(drop=True)
        buildings = buildings.drop_duplicates(subset="geometry")
        buildings = buildings[buildings.geometry.area > self.min_area_m2]
        return buildings.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Raster wrapper
# ---------------------------------------------------------------------------

class RasterTile:
    def __init__(self, raster_path: str, band_names: list[str] = BAND_ORDER):
        self.raster_path = raster_path
        self.band_names = band_names
        self._bands = None
        self.transform = None
        self.shape = None

    def load(self):
        with rasterio.open(self.raster_path) as src:
            self._bands = src.read()
            self.transform = src.transform
            self.shape = (src.height, src.width)
        return self

    @property
    def bands(self) -> dict:
        if self._bands is None:
            self.load()
        return {name: self._bands[i].ravel() for i, name in enumerate(self.band_names)}

    def pixel_centroids(self):
        height, width = self.shape
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs, ys = xy(self.transform, rows.ravel(), cols.ravel())
        return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Distance to river (vector-derived)
# ---------------------------------------------------------------------------

class DistanceToRiver:
    def __init__(self, river_geometry, densify_spacing_m: float = 5.0):
        self.river_geometry = river_geometry
        self.densify_spacing_m = densify_spacing_m
        self._tree = None

    def _build_tree(self):
        densified = self.river_geometry.segmentize(self.densify_spacing_m)
        coords = shapely.get_coordinates(densified)   # works for any geometry type
        self._tree = cKDTree(coords)

    def compute(self, x_array, y_array) -> np.ndarray:
        if self._tree is None:
            self._build_tree()
        points = np.column_stack([x_array, y_array])
        distances, _ = self._tree.query(points)
        return distances


# ---------------------------------------------------------------------------
# Spectral indices (raster-derived)
# ---------------------------------------------------------------------------

def _safe_ratio(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def compute_ndvi(nir, red):
    return _safe_ratio(nir - red, nir + red)

def compute_ndbi(swir, nir):
    return _safe_ratio(swir - nir, swir + nir)

def compute_ndwi(green, nir):
    return _safe_ratio(green - nir, green + nir)

# ---------------------------------------------------------------------------
# Earth Engine session
# ---------------------------------------------------------------------------

class EarthEngineSession:
    _initialized = False

    @classmethod
    def init(cls, project_id: str = GEE_PROJECT_ID):
        if not cls._initialized:
            ee.Initialize(project=project_id)
            cls._initialized = True

    @staticmethod
    def wait_for_task(task, poll_seconds: int = 15):
        task.start()
        while task.active():
            print("Export running:", task.status()["state"])
            time.sleep(poll_seconds)
        state = task.status()["state"]
        print("Export finished:", state)
        return state


# ---------------------------------------------------------------------------
# Submits GEE exports only — download is manual, into data/raw/tiles/
# ---------------------------------------------------------------------------

class TileExporter:
    BUILDINGS_DATASET = "projects/sat-io/open-datasets/MSBuildings/Kenya"

    def __init__(self, gdrive_folder: str = "EarthEngine"):
        EarthEngineSession.init()
        self.gdrive_folder = gdrive_folder

    def export_tile(self, tile_index: int, tile_box) -> dict:
        ee_geom = AOITiler.to_ee_geometry(tile_box)

        buildings_desc = f"nairobi_buildings_tile_{tile_index}"
        buildings_fc = ee.FeatureCollection(self.BUILDINGS_DATASET).filterBounds(ee_geom)
        buildings_task = ee.batch.Export.table.toDrive(
            collection=buildings_fc, description=buildings_desc,
            folder=self.gdrive_folder, fileFormat="GeoJSON",
        )

        raster_desc = f"nairobi_sentinel_tile_{tile_index}"
        image = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(ee_geom)
            .filterDate("2025-01-01", "2025-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .median()
            .clip(ee_geom)
            .select(BAND_ORDER)
        )
        raster_task = ee.batch.Export.image.toDrive(
            image=image, description=raster_desc, folder=self.gdrive_folder,
            fileNamePrefix=raster_desc, region=ee_geom,
            scale=SENTINEL_SCALE_M, crs=TARGET_CRS,
        )

        EarthEngineSession.wait_for_task(buildings_task)
        EarthEngineSession.wait_for_task(raster_task)

        return {
            "tile_index": tile_index,
            "buildings_filename": f"{buildings_desc}.geojson",
            "raster_filename": f"{raster_desc}.tif",
        }

# ---------------------------------------------------------------------------
# Fuses one tile's local files into a DataFrame — no downloading, no deleting
# ---------------------------------------------------------------------------

class TileFuser:
    def __init__(self, river_union, cleaner: VectorCleaner, tile_dir: Path = TILE_DIR):
        self.river_union = river_union
        self.cleaner = cleaner
        self.tile_dir = Path(tile_dir)

    def files_ready(self, manifest: dict) -> bool:
        buildings_path = self.tile_dir / manifest["buildings_filename"]
        raster_path = self.tile_dir / manifest["raster_filename"]
        return buildings_path.exists() and raster_path.exists()

    def fuse(self, manifest: dict) -> pd.DataFrame:
        buildings_path = self.tile_dir / manifest["buildings_filename"]
        raster_path = self.tile_dir / manifest["raster_filename"]

        buildings_raw = gpd.read_file(buildings_path)
        buildings_clean = self.cleaner.clean(buildings_raw)

        raster = RasterTile(str(raster_path)).load()
        x, y = raster.pixel_centroids()
        bands = raster.bands

        built_mask = rasterize(
            [(geom, 1) for geom in buildings_clean.geometry],
            out_shape=raster.shape, transform=raster.transform, fill=0, dtype="uint8",
            all_touched=True,   # marks any pixel the building geometry touches, not just center-covers
        ).ravel()


        distance_to_river_m = DistanceToRiver(self.river_union).compute(x, y)

        ndvi = compute_ndvi(bands["B8"], bands["B4"])
        ndbi = compute_ndbi(bands.get("B11", bands["B8"]), bands["B8"])
        ndwi = compute_ndwi(bands["B3"], bands["B8"])

        tile_df = pd.DataFrame({
            "tile_id": manifest["tile_index"],
            "x": x, "y": y,
            **bands,
            "ndvi": ndvi, "ndbi": ndbi, "ndwi": ndwi,
            "distance_to_river_m": distance_to_river_m,
            "built_up": built_mask,
        })

        band_cols = list(raster.band_names)
        tile_df = tile_df[(tile_df[band_cols] != 0).any(axis=1)].reset_index(drop=True)
        return tile_df


def processed_tile_ids(csv_path: Path) -> set:
    """Which tile_ids are already written to the output CSV — for resuming."""
    if not Path(csv_path).exists():
        return set()
    existing = pd.read_csv(csv_path, usecols=["tile_id"])
    return set(existing["tile_id"].unique())

# ---------------------------------------------------------------------------
# Buffer filter — applied at query/UI time, never re-baked into the CSV
# ---------------------------------------------------------------------------

def apply_buffer_filter(df: pd.DataFrame, buffer_meters: float) -> pd.DataFrame:
    result = df.copy()
    result["inside_buffer"] = result["distance_to_river_m"] <= buffer_meters
    return result