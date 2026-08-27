### The Goal
Build a way to identify individual buildings encroaching on Nairobi's riparian (riverside) buffer zones — specifically in Kasarani — accurately enough to approximate what a manual ground survey would find, without needing someone to physically walk every riverbank.

### What this notebook does
Works with real vector geometry locally, using geopandas and shapely, rather than raster/pixel statistics. It identifies which individual buildings sit within a riparian buffer, how far each one is from the river, what fraction of each building's footprint actually falls inside the buffer, and assigns each a risk tier.

### Step by Step
1. Study area boundary — geocodes "Kasarani, Nairobi, Kenya" via OpenStreetMap (osmnx).
2. River line — pulled from OpenStreetMap (waterway=river tag), clipped to the Kasarani boundary.
3. Building footprints — pulled from Microsoft's Building Footprints dataset (via the community "sat-io" Earth Engine catalog).
4. Geometry cleanup — real-world spatial data is messy:
        drop empty/null geometries
        repair invalid polygons (self-intersections) via make_valid
        "explode" multi-part building shapes into individual single polygons
5. Remove duplicates and slivers — drop exact duplicate polygons and tiny fragments under 4 m² (too small to be a real structure — usually clipping artifacts).
6. Clean attributes — standardize column names, fill missing building-type labels.
7. Build the riparian buffer — merge all river segments into one shape, buffer it by a fixed 30m.
8. Per-building metrics — the real value-add of this notebook. For every single building:
        exact distance to the river, in meters (not pixel-approximate)
        whether it intersects the buffer at all
        what percentage of its footprint actually falls inside the buffer (90%-inside vs. just-clips-the-edge is a meaningful difference)
        a risk tier: High (<10m), Medium (10-20m), Low (20-30m), Safe (>30m) 
9. Satellite image backdrop — pulls a 2024 Sentinel-2 tile over Kasarani for visual context under the vector layers.
10. Unresolved bug — the notebook's own sanity check ("Confirm the tile is correct") fails: the satellite tile's bounds don't overlap the AOI's bounds at all, and the image comes back 100% empty/zero pixels.
11. Export — writes a GeoPackage (.gpkg) with two layers: classified buildings (with all distance/risk columns) and the buffer polygon.
12. Visualization — two matplotlib maps: a vector-only view (buildings colored by inside/outside buffer) and a satellite-backed view.