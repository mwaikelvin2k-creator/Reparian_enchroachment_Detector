# Reparian_enchroachment_Detector
Traditional government remote-sensing models often rely on coarse or medium-resolution imagery. This leads to severe underestimations of informal structures inside protected riparian zones because clustered homes blend together into a single pixel grid (e.g., detecting only 118 structures when 700 actually exist).
The goal is to accurately isolate and approximate the real-world count of 700 structures within a 60-meter riparian buffer zone.

## Running the dashboard

The Streamlit dashboard (`app.py`) reads the preprocessed layers in `Preprocessing/` (run `Cleaning.ipynb` first if they're missing) and lets you query any coordinate, adjust the riparian buffer distance, and see live structure counts by risk tier. For a fuller walkthrough — creating the virtual environment, troubleshooting installs, and regenerating the source data — see [SETUP.md](SETUP.md).

```bash
python -m venv .venv
.venv/Scripts/activate       # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

The dashboard has three tabs: **Detection map** (interactive map, live counts, risk-tier breakdown), **Model performance** (the Random Forest's model card, confusion matrix, precision/recall-vs-threshold curve, feature importances), and **Method & data** (how each layer was produced, plus which build artifacts are present on disk).

## The Phase 1 Random Forest

`train_rf.py` is the trained-model pipeline: per-building zonal statistics over the Sentinel-2 B4/B3/B2 composite, auto-labeling by real centroid distance to the river against the 60m legal buffer, then a `RandomForestClassifier` with `class_weight="balanced"`. `kasarani_rf_pipeline.ipynb` is the same pipeline in notebook form with inline plots.

```bash
pip install -r requirements-notebook.txt
python train_rf.py
```

Its outputs — `models/riparian_rf_model.joblib`, `models/rf_metadata.json`, and `data/processed/kasarani_rf_predictions.csv` — are gitignored, so a fresh clone won't have them. Until you run the script the dashboard falls back to the geometric layer only and tells you what's missing; once they exist, the sidebar's **Detection layer** control gains the Random Forest and rule-vs-model comparison layers, and the threshold slider lets you move the model's operating point with precision and recall updating live.

**What the model is and isn't good for.** Encroaching structures are about 2% of the AOI, so headline accuracy is meaningless here — the encroachment row of the classification report is the number that matters, and it is weak. That's a data limitation, not a tuning problem: the tile carries only true-colour B4/B3/B2, so the classifier sees six RGB statistics plus footprint area, and a roof's colour says little about how far it sits from a river. The geometric distance rule remains the authoritative classifier for enforcement; the Random Forest is the Phase 1 baseline the proposal scoped, and the dashboard reports it honestly rather than flattering it. Adding NIR/SWIR bands (NDVI/NDWI), texture, or neighbourhood-density features is the path to a model that earns a place in the decision. The Phase 2 deep-learning model is not yet trained.
