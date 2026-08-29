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
			f" and placed the shapefile at: {input_shapefile}"
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


	# 3. GIS TRANSFORMATION:  Project degrees to Meters
	# EPSG: 32737- official mathematical coordinate system for Nairobi
	print(" Converting coordinates from Degrees to metric system(EPSG:32737)...")
	rivers_metric = cleaned_rivers.to_crs(epsg=32737)


	# 4. DRAW 60M BOUNDARY
	print(f"Buffer Zone: Drawing the legal mandatory{buffer_distance} - meter protection zone...")
	buffer_metric = rivers_metric.buffer(buffer_distance)


	# 5. REPROJECTION:
	# Convert back to EPSG:4326(Degrees) so it perfectly overlays on standard satellite images
	print(f" Re-aligning vector layer back to standard GPS Degrees(EPSG:4326)...")
	buffer_gdf_global = buffer_gdf_metric.to_crs(epsg=4326)

	# 6. Exporting The Results
	# create the output directory folder if not exists
	os.makedirs(os.path.dirname(output_geojson), exist_ok=True)

	print(f "File I/O: Saving the final riparian zone map to file...")
	buffer_gdf_global.to_file(output_geojson, driver="GeoJSON")

	print(f" Success!! Clean evidence file saved at: {output_geojson}\n")
	return buffer_gdf_global

#TEST THE SCRIPTS LOCALLY- prevent bugs

if __name__ == "__main__":
	#Define project file paths
	RAW_SHAPEFILE_PATH = "data/vectors/gis_osm_waterways_free_1.shp"
	OUTPUT_GEOJSON_PATH = "data/processed/kasarani_60m_riparian_zone.geojson"

	try:
		final_buffer = process_riparian_pipeline(
			input_shapefile=RAW_SHAPEFILE_PATH,
			output_geojson=OUTPUT_GEOJSON_PATH,
			buffer_distance=60
		)
		print(final_buffer.head()) # final Buffer layer overview

	except:
		print(f"\n Pipeline stopped with an error: \n {str(e)}")


