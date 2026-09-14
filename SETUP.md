# Local Setup Guide

Step-by-step instructions for getting this project running on your own machine, from a bare Python install to a working dashboard. If you just want the short version, see the "Running the dashboard" section in [README.md](README.md) — this file goes into more depth and also covers regenerating the source data.

## What you're setting up

Two things live in this repo:

1. **The Streamlit dashboard** (`app.py`) — reads the *already-generated* data files in `data/processed/` and lets you explore them interactively. This is what you probably want, and it's quick to set up (no external accounts needed). These files are committed to the repo, so a fresh clone has everything the dashboard needs out of the box.
2. **The data + modelling pipeline** (`notebooks/01_preprocessing.ipynb`, `02_modelling.ipynb`, `03_fusion_and_report.ipynb`) — the three notebooks that produced the files in `data/processed/`, in order: preprocessing pulls and cleans the source data, modelling trains the shared Random Forest, and fusion & report applies the distance filter and writes the final artifacts. You only need these if you want to regenerate the data (e.g. a different buffer distance, or retraining the model). Preprocessing needs a free Google Earth Engine account and a heavier geospatial package set.

Sections 1–4 below get the dashboard running. Section 5 covers regenerating the source data.

## Prerequisites

- **Python 3.10 or newer.** This project was built and tested on Python 3.14.6 — all dependencies installed cleanly from prebuilt wheels. If you're on an unusual platform and hit a package-install error, falling back to Python 3.11 or 3.12 is a reasonable first thing to try, since some niche packages publish wheels for established versions first.
- **Git**, to clone the repository.
- **(Notebook path only)** A [Google Earth Engine](https://earthengine.google.com/) account and a Google Cloud project with the Earth Engine API enabled — the preprocessing notebook authenticates against this to pull satellite imagery and building footprints.

Check your Python version:

```bash
python --version
```

On Windows, if `python` isn't recognized, try `py --version` instead (and use `py` in place of `python` in the commands below).

## 1. Clone the repository

```bash
git clone <your-fork-or-repo-url>
cd Reparian_enchroachment_Detector
```

If you already have the project locally, just `cd` into it.

## 2. Create a virtual environment

A virtual environment keeps this project's packages separate from anything else on your machine. Create one called `.venv` in the project root:

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script with an execution-policy error, run this once (in an admin PowerShell) and try again: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll know it worked because your terminal prompt gets a `(.venv)` prefix. Every command below assumes the environment is active — if you close your terminal, re-run the activation line (not the `venv` creation line) before continuing.

## 3. Install dependencies

With the environment active:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs Streamlit plus the geospatial stack the dashboard needs: `geopandas`, `folium`, `streamlit-folium`, `shapely`, `pyproj`, `pyogrio`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `joblib`, `requests`. All of these ship prebuilt wheels on Windows/macOS/Linux, so this step shouldn't need a C compiler or a system GDAL install.

## 4. Run the dashboard

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) and should open it in your browser automatically. If not, open that URL yourself.

The dashboard reads directly from the files already committed under `data/processed/` — there's nothing else to configure. Use the sidebar to switch localities, search a place by name, adjust the screening filters, and change the flagging distance; the map, live counts, and charts update from the real classified building data.

It opens on tabs including:

- **Detection map** — the interactive map and the live screening funnel.
- **Compare areas** — encroaching structures grouped by named locality within the study area.

Model performance (model card, confusion matrix, feature importances) lives under an expander on the detection map tab rather than a separate top-level tab in the current layout.

To stop the app, go back to the terminal and press `Ctrl+C`.

## 5. (Optional) Regenerate the source data

Skip this unless you specifically want to re-run the data pipeline — the dashboard works fine against the data already in the repo.

1. Install the extra notebook dependencies on top of what you installed in step 3:

   ```bash
   pip install -r requirements-notebook.txt
   ```

2. Authenticate with Google Earth Engine (one-time per machine):

   ```bash
   earthengine authenticate
   ```

   This opens a browser window to sign in and grant access. You'll also need to edit the `ee.Initialize(project="...")` call near the top of `notebooks/01_preprocessing.ipynb` to point at your own Google Cloud project ID (the one committed in the notebook belongs to the original author and you won't have access to it).

3. Launch Jupyter and run the notebooks **in order**:

   ```bash
   jupyter lab
   ```

   Then run, top to bottom:

   - `notebooks/01_preprocessing.ipynb` — clips OSM waterways to each region's AOI, builds the 60m riparian buffer, and pulls the Sentinel-2 composite sampled against ESA WorldCover ground truth. Re-downloads OSM/Earth Engine data, so some cells (the Earth Engine export tasks) poll a remote job and can take several minutes.
   - `notebooks/02_modelling.ipynb` — trains the shared Random Forest across all three regions' feature tables and detects candidate structures via Google Open Buildings.
   - `notebooks/03_fusion_and_report.ipynb` — computes true distance to the nearest river per building, applies the confidence/probability filters, and writes the final artifacts (including `rf_baseline.joblib` and `pipeline_summary.json`) into `data/processed/`.

Restart Streamlit (or just reload the page) afterwards to see the regenerated data.

## Project structure

```
app.py                        Streamlit dashboard (entry point — "streamlit run app.py")
requirements.txt              Dependencies for the dashboard
requirements-notebook.txt     Extra dependencies for the data + modelling notebooks
.streamlit/config.toml        Dashboard theme (dark, matches the design mockups)
notebooks/
  01_preprocessing.ipynb      Data acquisition + cleaning (OSM + Earth Engine)
  02_modelling.ipynb          Random Forest training on Open Buildings + Sentinel-2 features
  03_fusion_and_report.ipynb  Distance filtering + final artifact export
data/
  processed/                  Feature tables, predictions, the trained model, and per-region
                               summary stats — committed, so the dashboard runs on a fresh clone
  vectors/                    Source OSM waterways shapefile
design/                       UI mockups used to design the dashboard
```

## Troubleshooting

**`streamlit: command not found` / `No module named streamlit`**
Your virtual environment isn't active, or dependencies weren't installed into it. Re-run the activation command from step 2, confirm your prompt shows `(.venv)`, then re-run `pip install -r requirements.txt`.

**Port 8501 already in use**
Another Streamlit app (or a previous run) is still using that port. Either stop it, or run this one on a different port: `streamlit run app.py --server.port 8502`.

**`geopandas`/`pyogrio` fails to install**
This usually means pip is falling back to building from source because no prebuilt wheel exists for your exact Python version/OS/architecture combination. Try a slightly older Python version (3.11 or 3.12), or install via `conda`/`mamba` instead of `pip`, which bundles the underlying GDAL/GEOS/PROJ libraries for you.

**Notebook: `ee.Initialize` fails with a permissions or project error**
The notebook is hardcoded to a Google Cloud project belonging to the original author. Replace it with your own project ID (create one for free at [console.cloud.google.com](https://console.cloud.google.com), enable the Earth Engine API on it) and re-run `earthengine authenticate`.

**Dashboard loads but the map is blank**
Basemap tiles are fetched live over the network (CartoDB / Esri / OpenStreetMap) — check your internet connection, or switch basemap in the sidebar if one provider is being blocked by a firewall/proxy.
