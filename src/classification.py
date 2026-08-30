"""
Random Forest built-up classifier — trained once on a diverse, multi-region
sample to avoid settlement-type bias, then reused (not retrained) for any
user-specified AOI.
"""
import ee

BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
FEATURE_NAMES = BANDS + ['NDVI', 'NDBI']


def get_region_boundary(place_name, fallback_radius_m=3000):
    """
    Geocode a place name to an Earth Engine geometry.

    Tries a real administrative boundary polygon first. Falls back to a
    point + radius buffer if the place has no boundary in OSM (common for
    informal settlements, which usually aren't administratively bounded).
    """
    import osmnx as ox
    import ee

    try:
        gdf = ox.geocode_to_gdf(place_name)
        minx, miny, maxx, maxy = gdf.total_bounds
        return ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    except TypeError:
        # No polygon available — fall back to point + buffer
        lat, lon = ox.geocode(place_name)
        print(f"  (no boundary polygon for '{place_name}' — using "
              f"{fallback_radius_m}m point buffer instead)")
        return ee.Geometry.Point([lon, lat]).buffer(fallback_radius_m)


def get_sentinel2_composite(region, start_date, end_date, cloud_threshold=20):
    """Cloud-masked median composite for one region/date range."""
    def mask_clouds(img):
        scl = img.select('SCL')
        # SCL classes 3=shadow, 8/9/10=cloud (medium/high/cirrus)
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(mask)

    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
        .map(mask_clouds)
    )
    scene_count = collection.size().getInfo()
    composite = collection.median().clip(region)
    return composite, scene_count


def build_feature_image(composite):
    """6 raw bands + NDVI + NDBI."""
    ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndbi = composite.normalizedDifference(['B11', 'B8']).rename('NDBI')
    return composite.select(BANDS).addBands(ndvi).addBands(ndbi)


def get_worldcover_builtup(region):
    """ESA WorldCover class 50 = built-up, our reference labels."""
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').clip(region)
    return wc.eq(50).rename('builtup')


def sample_region_points(feature_image, worldcover_builtup, region, num_points=500, seed=42):
    """
    Stratified sample from ONE region. Called once per training region, then
    the results get merged — this is what actually prevents settlement-type
    bias, not anything in the training step itself.
    """
    training_image = feature_image.addBands(worldcover_builtup)
    return training_image.stratifiedSample(
        numPoints=num_points,
        classBand='builtup',
        region=region,
        scale=10,
        seed=seed,
        geometries=True,
    )


def train_random_forest(train_samples, num_trees=100, seed=42):
    return ee.Classifier.smileRandomForest(numberOfTrees=num_trees, seed=seed).train(
        features=train_samples,
        classProperty='builtup',
        inputProperties=FEATURE_NAMES,
    )


def classify_builtup(feature_image, classifier):
    return feature_image.classify(classifier).rename('builtup')