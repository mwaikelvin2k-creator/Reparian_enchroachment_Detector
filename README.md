# Reparian_enchroachment_Detector
Traditional government remote-sensing models often rely on coarse or medium-resolution imagery. This leads to severe underestimations of informal structures inside protected riparian zones because clustered homes blend together into a single pixel grid (e.g., detecting only 118 structures when 700 actually exist).
The goal is to accurately isolate and approximate the real-world count of 700 structures within a 60-meter riparian buffer zone.

## Module 1 — Preprocessing

I'm building `notebooks/01_preprocessing.ipynb`, which turns the raw OSM waterways shapefile
into a clean riparian buffer and river-line layer per region — Kasarani (the calibration case
study), Gatharaini, and Motoine.

**My part covers the first two steps:**

1. Define the three regions using one consistent method (a fixed-radius circle per region, not
   a hand-picked bounding box — more on why in the notebook itself).
2. Load and clip the OSM waterways shapefile, build the 60m riparian buffer, and save both the
   buffer polygon and the underlying river lines per region.

**Still open — the next two steps, for whoever picks this up after me:**

3. Pull the Sentinel-2 composite for each region (full band set, not just RGB).
4. Build the per-pixel feature table, labelled against ESA WorldCover ground truth, and save it
   per region.

## Module 2 — Modelling & Structure Detection

`notebooks/02_modelling.ipynb` takes the per-region feature tables from Module 1 and turns them
into a scored building layer — the input Module 4 (fusion) depends on.

**Steps covered:**

1. Train a Random Forest baseline on the combined feature tables from all three regions (not
   one model per region), labelled against the same ESA WorldCover ground truth used in
   Module 1.
2. Check whether a custom pixel-based detector (e.g. YOLO) is even viable on Sentinel-2's 10m
   resolution, by comparing it against the real median building size in the data — not an
   assumed number.
3. Detect candidate structures. Two strategies share one interface
   (`detect_structures(aoi_ee, resolution_m) -> DataFrame`), so picking one never means deleting
   the other's code:
   - **Open Buildings** (Google's dataset) — the current default, since Step 2 found buildings
     are smaller than a Sentinel-2 pixel.
   - **YOLO** — stubbed behind the same interface, to be wired up once imagery finer than the
     building size is available.
4. Score every detected building with the trained RF: sample the same spectral/NDVI/NDBI bands
   at each building's centroid and run them through the model to get a built-up probability.
5. Run all of the above for each region and save `{region}_buildings.csv`.

**Ground rule:** every step's output file name is the next step's input — don't rename an
output without checking who reads it downstream.

## Module 4 — Fusion & Calibration Results

Module 4 fuses candidate building detections (Google Open Buildings) with the Random Forest material classification model and calculates exact metric distances to river vector baselines[cite: 1, 2].

### Pipeline Workflow
1. **Distance Measurement:** Calculates straight-line distance from each detected structure centroid to the nearest river geometry in EPSG:32737[cite: 2].
2. **Multi-Filter Screening:** Applies confidence screening (`>= 0.70`) and Random Forest material-overlap checks (`rf_builtup_prob >= 0.05`)[cite: 2].
3. **Ground Truth Distance Calibration:** Sweeps metric distance thresholds against Kasarani's independent field baseline (Pamoja Trust manual survey, ~700 buildings)[cite: 2]. The optimal calibrated cutoff was identified at **16 meters**[cite: 2].

### Summary Results

| Region | Total Screened | Passed Filters | Encroaching Count (<= 16m) | Ground Truth Calibrated |
| :--- | :--- | :--- | :--- | :--- |
| **Kasarani** | 56,678 | 56,415 | **731** | Yes (vs. ~700 ground truth)[cite: 2] |
| **Gatharaini** | 27,123 | 27,046 | **87** | No (16m cutoff extrapolation)[cite: 2] |
| **Motoine** | 28,554 | 28,104 | **348** | No (16m cutoff extrapolation)[cite: 2] |

*Note: Gatharaini and Motoine apply the 16m threshold as an untested extrapolation until field validation data becomes available[cite: 2].*

### Primary Output Artifacts
* `{region}_encroaching_buildings.csv` — Tabular triage data including coordinates, confidence scores, and metric river distances[cite: 2].
* `{region}_encroaching_buildings.geojson` — Spatial vector layers ready for GIS analysis and mapping.
* `pipeline_summary.json` — Consolidated execution stats and calibration metadata[cite: 2].
* `kasarani_encroachment_map.html` — Interactive Folium web visualization map.
