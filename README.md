# Riparian Encroachment Detector

Kenya's rule for how close a building can legally stand to a river isn't as settled as it
sounds. Thirty metres has been the long-enforced standard — a newer, stricter 60-metre line
is now on the table, and some of the structures already caught in enforcement were approved
by the very government now demolishing them. Legal compliance here is a moving target, set
by policy, not by physics.

Satellite imagery can't resolve that ambiguity — it was never built to. Coarse government
remote-sensing surveys have found as few as 118 structures in an area a Pamoja Trust ground
survey later confirmed holds roughly 700: dense, clustered informal homes blend into a
single pixel and simply vanish from the count. A satellite can tell you where a rooftop
probably is. It can't tell you where a wall ends, when a building went up, who owns it, or
which version of the rule applies to it.

This project sits in that gap. It pairs high-resolution building detections (Google Open
Buildings) with a Random Forest built-up classifier and precise river-distance measurement,
calibrated against that same independent field survey — not to declare what's legal, but to
tell a human being where to look first, and exactly how much to trust what they find once
they get there.

**Live dashboard:** run locally via the steps in [SETUP.md](SETUP.md), or see the team for
the current deployed link.

## Team

Kelvin Mwaniki · Jane Murungi · Miring'u Kamau · Kimberly Wangui · Lewis Ndung'u · Christopher Bosire

*Moringa School — Data Science Capstone*

## How it fits together

| Module | What it does | Notebook / file |
|---|---|---|
| 1. Preprocessing | Defines the three study regions, clips OSM waterways, builds the riparian buffer, pulls the Sentinel-2 composite and labels it against ESA WorldCover | `notebooks/01_preprocessing.ipynb` |
| 2. Modelling | Trains the shared Random Forest, detects candidate structures via Google Open Buildings, scores each one | `notebooks/02_modelling.ipynb` |
| 3. Dashboard | Interactive Streamlit app — explore detections region by region, adjust screening thresholds, compare areas, inspect the model | `app.py` |
| 4. Fusion & calibration | Measures true distance to the nearest river, applies the screening filters, calibrates the flagging distance against ground truth | `notebooks/03_fusion_and_report.ipynb` |

## Module 1 — Preprocessing

`notebooks/01_preprocessing.ipynb` turns the raw OSM waterways shapefile into a clean
riparian buffer and river-line layer per region — Kasarani (the calibration case study),
Gatharaini, and Motoine.

1. Define the three regions using one consistent method — a fixed-radius circle per region,
   not a hand-picked bounding box (more on why in the notebook itself).
2. Load and clip the OSM waterways shapefile, build the 60m riparian buffer, and save both
   the buffer polygon and the underlying river lines per region.
3. Pull the Sentinel-2 composite for each region (full band set, not just RGB).
4. Build the per-pixel feature table, labelled against ESA WorldCover ground truth, and save
   it per region.

## Module 2 — Modelling & Structure Detection

`notebooks/02_modelling.ipynb` takes the per-region feature tables from Module 1 and turns
them into a scored building layer — the input Module 4 (fusion) depends on.

1. Train a Random Forest baseline on the combined feature tables from all three regions (one
   shared model, not one per region), labelled against the same ESA WorldCover ground truth
   used in Module 1.
2. Check whether a custom pixel-based detector (e.g. YOLO) is even viable at Sentinel-2's 10m
   resolution, by comparing it against the real median building size in the data — not an
   assumed number.
3. Detect candidate structures. Two strategies share one interface
   (`detect_structures(aoi_ee, resolution_m) -> DataFrame`), so picking one never means
   deleting the other's code:
   - **Open Buildings** (Google's dataset) — the current default, since Step 2 found
     buildings are smaller than a Sentinel-2 pixel.
   - **YOLO** — stubbed behind the same interface, to be wired up once imagery finer than the
     building size is available.
4. Score every detected building with the trained RF: sample the same spectral/NDVI/NDBI
   bands at each building's centroid and run them through the model to get a built-up
   probability.
5. Run all of the above for each region and save `{region}_buildings.csv`.

**Ground rule:** every step's output file name is the next step's input — don't rename an
output without checking who reads it downstream.

## Module 3 — Dashboard

`app.py` is the interactive Streamlit front end. Pick a region, adjust the screening
thresholds and flagging distance live, browse detections on the map, compare areas, and
inspect what the model is actually doing under the hood. Only Kasarani is calibrated against
an independent field count — Gatharaini and Motoine are labelled as not field-validated
everywhere their numbers appear, not just in a footnote.

See [SETUP.md](SETUP.md) for how to run it locally.

## Module 4 — Fusion & Calibration Results

`notebooks/03_fusion_and_report.ipynb` fuses the candidate building detections with the
Random Forest classification and calculates exact metric distances to the river vector
baseline.

### Pipeline workflow

1. **Distance measurement** — straight-line distance from each detected structure's centroid
   to the nearest river geometry, in EPSG:32737.
2. **Multi-filter screening** — confidence screening (`>= 0.70`) and Random Forest
   built-up-probability screening (`rf_builtup_prob >= 0.05`).
3. **Ground-truth distance calibration** — sweeps candidate distance thresholds against
   Kasarani's independent field baseline (a Pamoja Trust manual survey, ~700 structures). The
   distance that best matches that count was **16 metres**. That's a count match, not a
   verified legal boundary — see the dashboard's Method tab for the full caveat.

### Summary results

| Region | Total screened | Passed filters | Encroaching (≤ 16m) | Ground-truth calibrated |
|---|---|---|---|---|
| **Kasarani** | 56,678 | 56,415 | **731** | Yes — matched against ~700 ground truth |
| **Gatharaini** | 27,123 | 27,046 | **87** | No — 16m cutoff applied untested |
| **Motoine** | 28,554 | 28,104 | **348** | No — 16m cutoff applied untested |

*Gatharaini and Motoine reuse the 16m threshold as an untested extrapolation until field
validation data becomes available for either.*

### Primary output artifacts

- `{region}_encroaching_buildings.csv` — tabular triage data: coordinates, confidence scores,
  and metric river distances.
- `{region}_riparian_buffer.geojson` / `{region}_rivers.geojson` — spatial vector layers for
  GIS analysis and mapping.
- `pipeline_summary.json` — consolidated execution stats and calibration metadata.
- `rf_baseline.joblib` — the trained, shared Random Forest.

## Getting started

Full setup instructions, including how to run just the dashboard versus regenerating the
data end to end, are in [SETUP.md](SETUP.md).
