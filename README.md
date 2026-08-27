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

Model comparison metrics (Random Forest / deep-learning) will populate once Phase 1–2 training (see the project proposal) is complete — the dashboard currently reports the geometric distance-to-river classification from the preprocessing pipeline.
