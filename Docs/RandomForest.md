# Random Forest Built-Up Classification — Reference

## What this is
A multi-region Random Forest classifier that labels Sentinel-2 pixels as
built-up or not built-up. Trained once, on a deliberately diverse set of
Nairobi neighborhoods, then reused (never retrained) for any future
user-specified region — this is what lets the end product accept a place
name from a user and produce results without needing a fresh training run
each time.

Lives across two files:
- `src/classification.py` — reusable functions (the actual logic)
- `Preprocessing/RandomForest.ipynb` — runs the training pipeline once,
  using those functions, and exports the trained model

## Why multiple training regions, not just Kasarani
Training only on Kasarani risks the model learning Kasarani's specific
spectral signature (informal-settlement roofing, density, etc.) and
misclassifying areas that look different — e.g. formal, lower-density
suburbs with more tree cover. Six regions were used instead, chosen to span
distinct settlement types:

| Region | Type |
|---|---|
| Kasarani | informal / mixed |
| Kibera | dense informal |
| Karen | formal, low density |
| Kileleshwa | formal, medium density |
| Nairobi CBD | dense commercial |
| Nairobi National Park | non-built control |

## `src/classification.py` — function reference

| Function | Does |
|---|---|
| `get_region_boundary(place_name, fallback_radius_m=3000)` | Geocodes a place name to an Earth Engine geometry. Tries a real administrative boundary polygon first (via `osmnx`); falls back to a point + radius buffer if none exists (common for informal settlements — Kibera hit this). |
| `get_sentinel2_composite(region, start_date, end_date, cloud_threshold=20)` | Cloud/shadow-masked median composite for one region and date range, via the Sentinel-2 SCL band. |
| `build_feature_image(composite)` | Builds the 8-feature stack used for classification: 6 raw bands (B2, B3, B4, B8, B11, B12) + NDVI + NDBI. |
| `get_worldcover_builtup(region)` | Pulls ESA WorldCover's built-up class (value 50) for a region — the reference labels used for both training and evaluation. |
| `sample_region_points(feature_image, worldcover_builtup, region, num_points=500, seed=42)` | Draws a stratified sample of labeled points from one region. Called once per training region; the diversity comes from merging the results of multiple calls, not from anything in this function itself. |
| `train_random_forest(train_samples, num_trees=100, seed=42)` | Trains `ee.Classifier.smileRandomForest` on the merged, stratified sample. |
| `classify_builtup(feature_image, classifier)` | Applies a trained (or loaded) classifier to a feature image, producing the built-up/not-built-up layer. |

## Notebook flow (`RandomForest.ipynb`)

1. **Define training regions** — the six areas above, each as a
   `(place_name, fallback_radius)` pair.
2. **Sample each region, merge** — build features, pull WorldCover labels,
   draw a stratified sample per region, then merge all six into one pool.
   Result: 5,063 total training points.
3. **Diagnostic check** — CBD returned only 63 points (vs. ~1000 for every
   other region). Confirmed via a direct WorldCover check: CBD is 100%
   built-up, so there were no non-built-up pixels to sample. Not a bug —
   CBD's contribution is built-up-only by geography.
4. **70/30 split, train, evaluate** — held out 30% of the merged pool,
   trained on the rest, evaluated only on the held-out points.
   **Result: 85.9% held-out accuracy.**
5. **Per-region accuracy breakdown** — re-checked accuracy separately for
   each of the six regions, to surface any settlement-type bias the overall
   number could hide. Results:

   | Region | n | Accuracy |
   |---|---|---|
   | Kasarani | 275 | 85.8% |
   | Kibera | 308 | 85.1% |
   | Karen | 323 | 83.0% |
   | Kileleshwa | 275 | 82.5% |
   | CBD | 23 | 100.0% *(not comparable — see below)* |
   | Nairobi National Park | 318 | 92.8% |

6. **Export the trained classifier** — persisted as an Earth Engine asset
   (`projects/riparian-encroachment/assets/nairobi_builtup_rf_v1`) via
   `Export.classifier.toAsset()`, since `ee.Classifier` objects live
   server-side and can't be saved locally. This is the step that makes
   "train once, reuse for any future region" actually possible.

## Key finding: formal/lower-density suburbs are the harder case, not informal ones
The original concern going into this notebook was that training on
informal settlements might bias the model *against* formal areas. The
per-region breakdown suggests the opposite pattern in this run: Karen and
Kileleshwa (formal, lower-density) scored lowest (82-83%), while Kasarani
and Kibera (informal, denser) scored close to the overall average
(85-86%). Nairobi National Park scored highest (92.8%) — expected, since a
park is spectrally the cleanest case (uniform vegetation).

A plausible reason: lower-density formal housing tends to have more tree
cover between buildings, which blurs the spectral boundary between
built-up and non-built-up. Dense settlements (formal or informal) tend to
have a more uniform, easily separable signature.

## Caveat: CBD's 100% accuracy is not a meaningful comparison point
CBD is 100% built-up per WorldCover, so its test set contains no
non-built-up points — there was nothing for the model to get wrong.
Reported for completeness, but should not be read alongside the other five
regions' scores as if it were a comparable result.

## Reusing the trained classifier later (no retraining)
```python
frozen_classifier = ee.Classifier.load(
    'projects/riparian-encroachment/assets/nairobi_builtup_rf_v1'
)

user_region = get_region_boundary("<user-typed place name>")
composite, _ = get_sentinel2_composite(user_region, START_DATE, END_DATE)
features = build_feature_image(composite)
builtup_layer = classify_builtup(features, frozen_classifier)
```
This is the actual entry point for the "user types a region name" product
flow — no part of this re-runs the training pipeline above.
