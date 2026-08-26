#Handles loading, clipping,projecting and saving spatial data.

import geopandas as gpd
from shapely.geometry import box, LineString
import os

def process_riparian_pipeline(input_shapefile, output_geojson, buffer_distance=60):
	"""
	Loads waterways shapefile in data folder, clips it down to Kasarani, projects it into meters(EPSG:32737), draws a legal buffer zone, and exports it back as clean global coordinates (EPSG:4326).
	"""
	
	print("Step 1: Initializing production spatial pipeline...")

	# fallback verification to check if file exists
	if not os.path.exists(input_shapefile):
		raise FileNotFoundError(
			f" Missing raw vector file! please ensure you have downloaded "
			f" and placed the shapefile at: {inpuy_shapefile}"
		)

	# 1. OPTIMIZATION: define bounding box strictly over Kasarani
	# Coordinates format: (Min Longitude, Min Latitude, Max Longitude, Max Latitude)
	# Kasarani's precise coordinates

	kasarani_box = box(36.80, -1.32, 36.95, -1.20)

	print("Data Engineering: Streaming and clipping a massive spatial layer...")
	# Load only features that intersect our box to protect machine RAM
	kasarani_rivers = gpd.read_file(input_shapefile, bbox=kasarani_box)
	print(f" Loaded {len(kasarani_rivers)} raw water features within the local coordinate matric.")



	# 2. FILTERING: Keep only active flowing rivers, drop minor drainage ditches
	# Geofabrik uses 'fclass' column to sort waterways
	if 'fclass' in kasarani_rivers.columns:
		cleaned_rivers = kasarani_rivers[kasarani_rivers['fclass'].isin(['river', 'stream'])].copy()
	else:
		#Fallback if using the HDX dataset which might use a column name like 'TYPE'
		cleaned_rivers = kasarani_rivers.copy()

	print(f" Data Cleaning: Isolated {len(cleaned_rivers)} legal river paths. Dropped dry ditches.")


