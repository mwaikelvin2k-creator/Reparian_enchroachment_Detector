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

**Ground rule:** every step's output file name is the next step's input — don't rename an
output without checking who reads it downstream.

Setup:

```
pip install -r requirements.txt
```
