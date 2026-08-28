What is a satellite image to a computer?

To a human, a satellite images shows rivers, trees, and tin roofs. To a computer, a satellite image is nothing but a massive grid of numbers arranged in rows and columns.

For a standard color image, every single pixel contains three numbers:
1. How much RED is in a dot(0-255)
2. How much GREEN is in a dot(0-255)
3. How much BLUE is in a dot (0-255)

RGB Data.

The core problem:
In informal settlements e.g. Kasarani houses are built so close together that their corrugated iron roofs touch.
- a  basic computer model looks at the pixel, sees a massive, continous block of numbers.
- the computer clumps the houses together into one giant blob.
- a basic government model counts the blobs resulting to a count of 118.

Our solution innovation.

We break this problem down into a two step process to solve the blob limitation.
1. The Material Highlighter(Random Forest):
- we look at pixels one by one to isolate the exact color and texture of corrugated iron.

2. Structure Boundary Tracer (YOLOv8-seg)
- we train a neural network to look at patterns, scanning for the tiny dark shadow lines and angular corners where one roof ends and another begins.


STEP 1: SPATIAL PREPROCESSING

Goal: Load a geographic line representing the Nairobi River and draw a mathematically perfect 60-meter protection zone around it.

Challenge:
- Earth is a round 3D Sphere, but maps are 2D.
- Standard GPS coordinates are measured in Degrees, not meters.

Fix:
- Reproject our map into a coordinate system that uses meters. Wher for Nairobi, the official standard metric projection is EPSG: 32737(A.K.A UTM Zone 37S)



DATASETS:
- The official hyrology vector(shapefile(shp) / GeoJSON) for the Nairobi and Kasarani river channels is available either from:

1. Geofabrik Africa Repo:
- Its shp file maps every river and stream in Nairobi with high precision.

2. Humanitarian Data Exchange(HDX) kenya Hydrography:
- it provides pre-cleaned, standardized (.geojson) lines already vetted by humanitarian agencies.

3. ENERGYDATA.INFO(World Bank Group):
- offers a specific file polygon excellent for tracking wider riparian buffers.

If the Geofabrik repo way, the data can be large because it tracks every stream in Kenya.
- to bypass this massive dataset bottleneck, we use SPATIAL CLIPPING & ATTRIBUTES FILTERING data engineering practices to filter the dataset dynamically using a localized bounding box wrapper.


Optimal Data Range:

June 2025 - May 2026 because:
1. Track 2026 Post-Flood Reconstruction Era
- Our model should capture the structural baseline before the clearing events and track the rapid encroachment patterns that occurred directly during the heavy long rains of April-May 2026.

2. Train the Model on Extreme Seasonal Variance
Dry Peak(July-Aug 2025):
- Riverbed narrows, vegetation dries up, tree canopies thin out offering our YOLOv8-seg deep learning model an unobstructed view.

Wet Peak(April-May 2026):
- River line widens significantly, seasonal mud structures emerge providing our Random Forest model with exact training pixels it needs to map seasonal water fluctuations and turbid water pixels.
