# Local Setup Guide

Step-by-step instructions for getting this project running on your own machine, from a bare Python install to a working dashboard. If you just want the short version, see the "Running the dashboard" section in [README.md](README.md) — this file goes into more depth and also covers regenerating the source data.

## What you're setting up

Two separate things live in this repo:

1. **The Streamlit dashboard** (`app.py`) — reads the *already-generated* data files in `Preprocessing/` and lets you explore them interactively. This is what most people want, and it's quick to set up (no external accounts needed).
2. **The data-prep notebook** (`Preprocessing/Cleaning.ipynb`) — the pipeline that *produced* those files from OpenStreetMap and Google Earth Engine. You only need this if you want to regenerate the data (e.g. for a different area, or a different buffer distance baked into the source files). It needs a free Google Earth Engine account and a heavier set of geospatial packages.

Section 1–4 below get the dashboard running. Section 5 is optional and covers the notebook.

## Prerequisites

- **Python 3.10 or newer.** This project was built and tested on Python 3.14.6, including the notebook's heavier geospatial stack (`rasterio`, `osmnx`, `geemap`) — all installed cleanly from prebuilt wheels. If you're on an unusual platform and hit a package-install error, falling back to Python 3.11 or 3.12 is a reasonable first thing to try, since some niche packages publish wheels for established versions first.
- **Git**, to clone the repository.
- **(Notebook path only)** A [Google Earth Engine](https://earthengine.google.com/) account and a Google Cloud project with the Earth Engine API enabled — the notebook authenticates against this to pull satellite imagery and building footprints.

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

This installs Streamlit plus the geospatial stack the dashboard needs: `geopandas`, `folium`, `streamlit-folium`, `shapely`, `pyproj`, `pyogrio`, `pandas`, `plotly`. On Windows/macOS/Linux these all ship prebuilt wheels, so this step shouldn't need a C compiler or system GDAL install.

## 4. Run the dashboard

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) and should open it in your browser automatically. If not, open that URL yourself.

The dashboard reads directly from the `.gpkg` files already committed under `Preprocessing/` — there's nothing else to configure. Use the sidebar to enter a coordinate, drag the riparian buffer slider, and switch basemaps; the map, live counter, and charts update from the real classified building data.

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

   This opens a browser window to sign in and grant access. You'll also need to edit the `ee.Initialize(project="...")` calls near the top of the notebook to point at your own Google Cloud project ID (the one committed in the notebook belongs to the original author and you won't have access to it).

3. Launch Jupyter and run the notebook top to bottom:

   ```bash
   jupyter lab
   ```

   Then open `Preprocessing/Cleaning.ipynb` and run all cells. It will re-download OSM/Earth Engine data and regenerate the `.gpkg`, `.geojson`, and `.png` files in `Preprocessing/`. Some cells (the Earth Engine export tasks) poll a remote job and can take several minutes.

## Project structure

```
app.py                       Streamlit dashboard (entry point — "streamlit run app.py")
requirements.txt             Dependencies for the dashboard
requirements-notebook.txt    Extra dependencies for the data-prep notebook
.streamlit/config.toml       Dashboard theme (dark, matches the design mockups)
Preprocessing/
  Cleaning.ipynb             Data acquisition + cleaning pipeline (OSM + Earth Engine)
  kasarani_aoi.gpkg          Study area boundary
  kasarani_river.gpkg        Nairobi River centerline
  kasarani_encroachment_results.gpkg   Classified building footprints + buffer polygon
  kasarani_buildings_export.geojson    Raw building footprint export
  kasarani_*.png / .tif      Static preview images and the (currently mismatched) raster tile
design/                      UI mockups used to design the dashboard
```

## Troubleshooting

**`streamlit: command not found` / `No module named streamlit`**
Your virtual environment isn't active, or dependencies weren't installed into it. Re-run the activation command from step 2, confirm your prompt shows `(.venv)`, then re-run `pip install -r requirements.txt`.

**Port 8501 already in use**
Another Streamlit app (or a previous run) is still using that port. Either stop it, or run this one on a different port: `streamlit run app.py --server.port 8502`.

**`geopandas`/`pyogrio` fails to install**
This usually means pip is falling back to building from source because no prebuilt wheel exists for your exact Python version/OS/architecture combination. Try a slightly older Python version (3.11 or 3.12), or install via `conda`/`mamba` instead of `pip`, which bundles the underlying GDAL/GEOS/PROJ libraries for you.

**Notebook: `ee.Initialize` fails with a permissions or project error**
The notebook is hardcoded to a Google Cloud project (`causal-bus-404912`) that belongs to the original author. Replace it with your own project ID (create one for free at [console.cloud.google.com](https://console.cloud.google.com), enable the Earth Engine API on it) and re-run `earthengine authenticate`.

**Dashboard loads but the map is blank**
Basemap tiles are fetched live over the network (CartoDB / Esri / OpenStreetMap) — check your internet connection, or switch basemap in the sidebar if one provider is being blocked by a firewall/proxy.
