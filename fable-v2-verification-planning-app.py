from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import math

import geopandas as gpd
import folium
from folium.plugins import FastMarkerCluster
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import linemerge, substring
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from streamlit_folium import st_folium

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = DATA_DIR / "rf_baseline.joblib"
SUMMARY_PATH = DATA_DIR / "pipeline_summary.json"

METRIC_CRS = "EPSG:32737"
WGS84 = "EPSG:4326"

REGIONS = ["Kasarani", "Gatharaini", "Motoine"]
# Matches 01_preprocessing.ipynb / 02_modelling.ipynb exactly — the fixed-radius
# circle each region's AOI is built from. Kasarani is the only one calibrated
# against an independent field count (Pamoja Trust, ~700 structures) — Gatharaini
# and Motoine reuse that same flagging distance untested. Flagged in the toolbar
# badge, the map header, the stretch table and every export whenever either is selected.
REGION_CENTERS = {
    "Kasarani": (36.8969, -1.2296),
    "Gatharaini": (36.95952127354356, -1.2252700532171976),
    "Motoine": (36.74237847877641, -1.3113205815273903),
}
CASE_STUDY_RADIUS_KM = 3

# Named localities within the Kasarani AOI, for map navigation — no verified
# locality coordinates exist for Gatharaini or Motoine yet. Best-effort
# approximations — worth spot-checking against OSM before treating as exact.
KASARANI_AREAS = {
    "Kasarani town centre": (-1.2295, 36.8908),
    "Mwiki": (-1.2154, 36.8967),
    "Sunton": (-1.2244, 36.9012),
    "Hunters": (-1.2312, 36.8874),
    "Clay City": (-1.2401, 36.8851),
    "Roysambu": (-1.2190, 36.8890),
    "Zimmerman": (-1.2080, 36.8890),
    "Githurai 44": (-1.1990, 36.9020),
    "Kahawa West": (-1.1890, 36.9150),
}
FIRST_LOCALITY = list(KASARANI_AREAS.keys())[0]
FEATURE_COLS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI"]
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_MATERIAL_OVERLAP_THRESHOLD = 0.05

# The unit a field team actually walks: the river is cut into ~400 m stretches
# and every candidate is assigned to its nearest one.
STRETCH_LENGTH_M = 400
STRETCH_NONE = "__whole_region__"

# Provenance facts checked against the notebooks. The two "not recorded" values
# are real gaps in the pipeline, surfaced deliberately rather than papered over.
S2_WINDOW = "2024-01-01 to 2024-12-31"
NOT_RECORDED = "not recorded in pipeline"

TEAL = "#4dd0c4"
AMBER = "#f2b544"
RED = "#e2604f"
GREEN = "#6ad18a"
GREY = "#4a5860"
BG = "#0f1417"

BASEMAPS = {
    "OpenStreetMap": "OpenStreetMap",
    "Satellite (Esri)": "Esri.WorldImagery",
    "Dark (CartoDB — requires API key)": "CartoDB dark_matter",
}

KNOWN_LIMITS = [
    ("Centroid, not footprint",
     "Distance is measured from each building's centre point. Large footprints whose walls reach "
     "the river but whose centre sits further out are under-flagged; small structures are not. "
     "The amber 'edge-possible' set is an area-based partial correction, not a fix."),
    ("Centreline, not bank",
     "Distance is to the OSM river centreline. On a wide channel the same distance sits much "
     "closer to the water than on a narrow tributary — the threshold means different things "
     "stretch to stretch, and region to region."),
    ("OSM completeness",
     "Unmapped tributaries, culverts and drains produce no flags at all. Gaps concentrate where "
     "mapping is thinnest."),
    ("Imagery vintage",
     f"Open Buildings footprints predate the Sentinel-2 composite ({S2_WINDOW}) by an unrecorded "
     "interval; structures demolished since may still appear, newer ones are absent."),
    ("Structures ≠ households",
     "Every count is an Open Buildings point. One structure may hold several households, or be a "
     "wall or kiosk. No number here is a population figure."),
]

st.set_page_config(
    page_title="Riparian Encroachment Detector",
    page_icon="\U0001f6f0️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------- file paths


def buildings_csv(region: str) -> Path:
    return DATA_DIR / f"{region.lower()}_buildings.csv"


def riparian_buffer_geojson(region: str) -> Path:
    return DATA_DIR / f"{region.lower()}_riparian_buffer.geojson"


def rivers_geojson(region: str) -> Path:
    return DATA_DIR / f"{region.lower()}_rivers.geojson"


def feature_table_csv(region: str) -> Path:
    return DATA_DIR / f"{region.lower()}_feature_table.csv"


def encroaching_csv(region: str) -> Path:
    return DATA_DIR / f"{region.lower()}_encroaching_buildings.csv"


# ---------------------------------------------------------------- data loading


@st.cache_data(show_spinner=False)
def load_pipeline_summary() -> dict | None:
    if not SUMMARY_PATH.exists():
        return None
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_buildings_raw(region: str) -> pd.DataFrame | None:
    path = buildings_csv(region)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_rivers_metric(region: str) -> gpd.GeoDataFrame | None:
    path = rivers_geojson(region)
    if not path.exists():
        return None
    rivers = gpd.read_file(path)
    if rivers.crs is None:
        rivers = rivers.set_crs(METRIC_CRS)  # saved in EPSG:32737 by 01_preprocessing.ipynb
    else:
        rivers = rivers.to_crs(METRIC_CRS)
    return rivers


@st.cache_resource(show_spinner=False)
def load_river_union(region: str):
    rivers = load_rivers_metric(region)
    if rivers is None:
        return None
    return rivers.union_all()


@st.cache_data(show_spinner=False)
def load_buffer_wgs84(region: str) -> gpd.GeoDataFrame | None:
    path = riparian_buffer_geojson(region)
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    return gdf.to_crs(WGS84) if gdf.crs else gdf.set_crs(WGS84)


@st.cache_data(show_spinner=False)
def load_feature_table(region: str) -> pd.DataFrame | None:
    path = feature_table_csv(region)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_frozen_encroaching_count(region: str) -> int | None:
    """Row count of the notebook's frozen encroaching CSV, so the live count can
    be shown next to it rather than silently disagreeing with it."""
    path = encroaching_csv(region)
    if not path.exists():
        return None
    return int(len(pd.read_csv(path)))


def region_aoi_wgs84(region: str) -> gpd.GeoDataFrame:
    """Reconstructs each region's fixed-radius circle exactly as
    01_preprocessing.ipynb / 02_modelling.ipynb define it — no separate AOI
    file is saved by the pipeline, so this is derived rather than read."""
    lon, lat = REGION_CENTERS[region]
    center_metric = gpd.GeoSeries([Point(lon, lat)], crs=WGS84).to_crs(METRIC_CRS)
    circle_metric = center_metric.buffer(CASE_STUDY_RADIUS_KM * 1000)
    return gpd.GeoDataFrame(geometry=circle_metric, crs=METRIC_CRS).to_crs(WGS84)


def leaflet_bounds(gdf: gpd.GeoDataFrame) -> list:
    """[[south, west], [north, east]] — the format Leaflet's fitBounds /
    flyToBounds expect, from any WGS84 GeoDataFrame's total_bounds."""
    minx, miny, maxx, maxy = gdf.total_bounds
    return [[miny, minx], [maxy, maxx]]


@st.cache_data(show_spinner=False)
def build_region_table(region: str) -> pd.DataFrame | None:
    """Buildings with true distance-to-river added — mirrors
    03_fusion_and_report.ipynb's add_river_distance exactly, so the same
    thresholds the notebook calibrated against reproduce the same counts."""
    buildings = load_buildings_raw(region)
    river_union = load_river_union(region)
    if buildings is None or river_union is None:
        return None

    points = gpd.GeoDataFrame(
        buildings.copy(),
        geometry=[Point(xy) for xy in zip(buildings["lon"], buildings["lat"])],
        crs=WGS84,
    ).to_crs(METRIC_CRS)
    points["distance_to_river_m"] = points.geometry.distance(river_union)
    return pd.DataFrame(points.drop(columns="geometry"))


def apply_filters(table: pd.DataFrame, confidence_threshold: float,
                  material_threshold: float, flag_distance_m: float) -> pd.DataFrame:
    """Same three-step filter as 03_fusion_and_report.ipynb's
    filter_candidates() + the final distance cutoff, applied live so the
    Filters sliders actually change what counts as a candidate."""
    candidates = table[
        (table["confidence"] >= confidence_threshold)
        & (table["rf_builtup_prob"] >= material_threshold)
    ]
    return candidates[candidates["distance_to_river_m"] <= flag_distance_m]


# ------------------------------------------------------------------ stretches


@st.cache_data(show_spinner=False)
def build_stretches(region: str, stretch_m: int = STRETCH_LENGTH_M) -> gpd.GeoDataFrame:
    """Cuts the region's river network into ~stretch_m segments. IDs depend only
    on geometry (not on filter settings), so they stay stable while sliders move.
    Returns an empty GeoDataFrame with the expected columns if there are no lines."""
    empty = gpd.GeoDataFrame(
        {"stretch_id": pd.Series(dtype=str), "length_m": pd.Series(dtype=int),
         "mid_lat": pd.Series(dtype=float), "mid_lon": pd.Series(dtype=float)},
        geometry=gpd.GeoSeries([], crs=METRIC_CRS), crs=METRIC_CRS,
    )
    rivers = load_rivers_metric(region)
    if rivers is None or rivers.empty:
        return empty

    merged = linemerge(rivers.union_all())
    lines = [g for g in getattr(merged, "geoms", [merged]) if g.geom_type == "LineString" and g.length > 0]
    segments = []
    for line in lines:
        n = max(1, int(math.ceil(line.length / stretch_m)))
        step = line.length / n
        for i in range(n):
            seg = substring(line, i * step, (i + 1) * step)
            if seg.is_empty or seg.geom_type != "LineString":
                continue
            segments.append(seg)
    if not segments:
        return empty

    gdf = gpd.GeoDataFrame(geometry=segments, crs=METRIC_CRS)
    gdf["length_m"] = gdf.length.round(0).astype(int)
    mids = gdf.geometry.interpolate(0.5, normalized=True).to_crs(WGS84)
    gdf["mid_lat"] = mids.y.values
    gdf["mid_lon"] = mids.x.values
    # Stable ordering north→south, then west→east, so IDs read roughly across the map.
    order = np.lexsort((gdf["mid_lon"].values, -gdf["mid_lat"].values))
    gdf = gdf.iloc[order].reset_index(drop=True)
    gdf["stretch_id"] = [f"{region[0]}-{i + 1:02d}" for i in range(len(gdf))]
    return gdf


def assign_stretches(frame: pd.DataFrame, stretches: gpd.GeoDataFrame) -> pd.Series:
    """Nearest stretch id for every row of `frame` (needs lon/lat columns).
    Returns a Series aligned to frame.index; NaN where no stretch exists."""
    if frame.empty or stretches.empty:
        return pd.Series(np.nan, index=frame.index, dtype=object)
    pts = gpd.GeoDataFrame(
        index=frame.index,
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
        crs=WGS84,
    ).to_crs(METRIC_CRS)
    joined = gpd.sjoin_nearest(pts, stretches[["stretch_id", "geometry"]], how="left")
    joined = joined[~joined.index.duplicated(keep="first")]  # exact-tie duplicates
    return joined["stretch_id"].reindex(frame.index)


@st.cache_data(show_spinner="Searching OpenStreetMap...", ttl=3600)
def geocode_place_osm(query: str, region: str) -> tuple[float, float, str] | None:
    """Free-text place lookup via OSM's Nominatim, biased toward the
    selected region so a search stays relevant to what's on screen."""
    lon, lat = REGION_CENTERS[region]
    pad = (CASE_STUDY_RADIUS_KM / 111.0) * 1.5
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": f"{query}, Nairobi, Kenya", "format": "json", "limit": 1,
                "viewbox": f"{lon - pad},{lat - pad},{lon + pad},{lat + pad}", "bounded": 1,
            },
            headers={"User-Agent": "riparian-encroachment-detector/1.0"},
            timeout=8,
        )
        response.raise_for_status()
        results = response.json()
    except (requests.RequestException, ValueError):
        return None
    if not results:
        return None
    match = results[0]
    return float(match["lat"]), float(match["lon"]), match.get("display_name", query)


# ---------------------------------------------------------------- model evaluation


@st.cache_data(show_spinner=False)
def evaluate_model_heldout() -> dict | None:
    """Reproduces 02_modelling.ipynb's held-out split (stratified 80/20,
    random_state=42, tables concatenated in REGIONS order) and scores the
    saved model on the 20% it never saw. The notebook printed this report but
    never saved it; the fixed random_state lets it be recomputed here. The
    train-split fit is returned as a sanity check: an RF fits its own training
    rows near-perfectly, so a train accuracy well below 1.0 means the
    reproduced split does not match the notebook's."""
    model = load_model()
    if model is None:
        return None
    frames = []
    for r in REGIONS:
        t = load_feature_table(r)
        if t is None:
            return None
        frames.append(t.assign(_region_=r))
    full = pd.concat(frames, ignore_index=True)
    X, y = full[FEATURE_COLS], full["builtup"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pred_te = model.predict(X_te)
    pred_tr = model.predict(X_tr)
    report = classification_report(y_te, pred_te, target_names=["not built-up", "built-up"],
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pred_te).tolist()
    train_acc = float((pred_tr == y_tr.values).mean())
    region_te = full.loc[X_te.index, "_region_"].values
    y_te_vals = y_te.values
    per_region = []
    for r in REGIONS:
        mask = region_te == r
        if mask.sum() == 0:
            continue
        per_region.append({
            "Region": r,
            "Held-out rows": int(mask.sum()),
            "Accuracy": round(float((pred_te[mask] == y_te_vals[mask]).mean()), 3),
        })
    return {"report": report, "confusion_matrix": cm, "n_test": int(len(X_te)),
            "n_train": int(len(X_tr)), "train_accuracy": train_acc, "per_region": per_region}


# ------------------------------------------------------------------- chrome/CSS


def inject_css() -> None:
    st.markdown(
        """
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap">
        <style>
        :root{
            --bg-panel:#161c20; --bg-panel-alt:#1b2226; --bg-raised:#20282d;
            --border:#2a333a; --border-soft:#232b30;
            --text-primary:#eef2f3; --text-secondary:#8a9aa3; --text-tertiary:#5c6b73;
            --accent-teal:#4dd0c4; --accent-amber:#f2b544; --accent-green:#6ad18a; --accent-red:#e2604f;
        }
        html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
        .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
        [data-testid="stAppViewContainer"] { background: #0f1417; }
        [data-testid="stSidebar"] { background: var(--bg-panel); border-right: 1px solid var(--border); }
        [data-testid="stSidebar"] .stMarkdown p { color: var(--text-secondary); }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 0.6rem; }
        .rd-eyebrow{ text-transform:uppercase; letter-spacing:.08em; font-size:11px; color:var(--text-tertiary); font-weight:600; margin-bottom:6px; }
        .rd-card{ background:var(--bg-panel); border:1px solid var(--border); border-radius:12px; padding:16px 18px; height:100%; }
        .rd-card-title{ font-size:12px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:var(--text-primary); }
        .rd-sub{ font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--text-tertiary); margin-top:2px; line-height:1.5; }
        .rd-metric-big{ font-family:'IBM Plex Mono',monospace; font-size:38px; font-weight:700; color:var(--accent-amber); line-height:1; }
        .rd-metric-mid{ font-family:'IBM Plex Mono',monospace; font-size:19px; font-weight:600; color:var(--text-primary); }
        .rd-metric-label{ text-transform:uppercase; letter-spacing:.06em; font-size:10.5px; color:var(--text-tertiary); }
        .rd-badge{ font-family:'IBM Plex Mono',monospace; font-size:11px; padding:6px 12px; border-radius:6px; border:1px solid var(--border); color:var(--text-tertiary); display:inline-block; }
        .rd-badge-ok{ background:rgba(106,209,138,0.12); border:1px solid var(--accent-green); color:var(--accent-green); }
        .rd-badge-warn{ background:rgba(242,181,68,0.10); border:1px solid var(--accent-amber); color:var(--accent-amber); }
        .rd-badge-inline{ font-size:9.5px; padding:3px 8px; vertical-align:middle; margin-left:8px; }
        .rd-legend-row{ display:flex; gap:14px; flex-wrap:wrap; margin-top:10px; }
        .rd-legend-item{ display:flex; align-items:center; gap:6px; font-size:11px; color:var(--text-secondary); }
        .rd-swatch{ width:10px; height:10px; border-radius:50%; display:inline-block; }
        .rd-swatch-line{ width:16px; height:4px; border-radius:2px; display:inline-block; }
        .rd-header{ display:flex; align-items:center; justify-content:space-between; padding:4px 4px 6px; border-bottom:1px solid var(--border); margin-bottom:6px; }
        .rd-title{ font-size:13px; font-weight:600; letter-spacing:.01em; color:var(--text-primary); }
        .rd-subtitle{ font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--text-tertiary); letter-spacing:.03em; }
        .rd-note{ border-left:2px solid var(--accent-amber); background:rgba(242,181,68,0.06); padding:12px 14px; border-radius:0 8px 8px 0; font-size:12.5px; color:var(--text-secondary); line-height:1.65; }
        .rd-note-list{ margin:6px 0 0 0; padding-left:16px; }
        .rd-note-list li{ margin-bottom:4px; }
        .rd-kv{ display:flex; justify-content:space-between; gap:12px; padding:5px 0; border-bottom:1px solid var(--border-soft); font-size:12px; }
        .rd-kv span:first-child{ color:var(--text-tertiary); }
        .rd-kv span:last-child{ color:var(--text-primary); font-family:'IBM Plex Mono',monospace; text-align:right; }
        .stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--border); }
        .stTabs [data-baseweb="tab"]{ font-size:12px; letter-spacing:.04em; text-transform:uppercase; font-weight:600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


_card_marker_n = [0]


@contextmanager
def card(extra_style: str = ""):
    """A real card: st.container() grouped with everything rendered inside it,
    styled via a hidden per-instance marker + a :has() rule targeting that
    container. Plain unsafe_allow_html div/span pairs don't work for this —
    Streamlit renders each st.markdown() call as its own isolated DOM node, so
    an opening <div> and its later closing </div> never actually nest around
    the content in between; the "card" ends up an empty box floating above
    unstyled content. st.container() is a real DOM element multiple calls can
    render into, which is what the marker/:has() rule hooks onto."""
    _card_marker_n[0] += 1
    marker = f"rd-card-{_card_marker_n[0]}"
    with st.container():
        st.markdown(
            f'<div class="{marker}" style="display:none"></div>'
            f'<style>div[data-testid="stVerticalBlock"]:has(> div .{marker}) {{'
            f'background:var(--bg-panel); border:1px solid var(--border); '
            f'border-radius:12px; padding:16px 18px; {extra_style}'
            f'}}</style>',
            unsafe_allow_html=True,
        )
        yield


def metric_card(label: str, value: str, sub: str = "", color: str | None = None) -> None:
    with card():
        st.markdown(f'<div class="rd-metric-label">{label}</div>', unsafe_allow_html=True)
        style = f"margin-top:4px;{'color:' + color + ';' if color else ''}"
        st.markdown(f'<div class="rd-metric-mid" style="{style}">{value}</div>', unsafe_allow_html=True)
        if sub:
            st.markdown(f'<div style="font-size:11px;color:var(--text-tertiary);margin-top:2px;">{sub}</div>',
                        unsafe_allow_html=True)


def plotly_layout(**overrides) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", color="#8a9aa3", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(bgcolor="#1b2226", bordercolor="#2a333a",
                        font=dict(family="IBM Plex Mono, monospace", color="#eef2f3", size=11)),
    )
    base.update(overrides)
    return base


def trust_state(region: str, at_defaults: bool, conf: float, mat: float,
                dist: int, calibrated_dist: int) -> dict:
    """The trust state that travels with every number. 'Calibrated' means the
    Kasarani *total* was matched to a field count — never structure-level
    agreement, never a legal reserve line — and only holds at the settings the
    match was made with."""
    if region == "Kasarani" and at_defaults:
        return dict(
            key="count-matched", css="rd-badge-ok", badge="Calibrated (count-matched)",
            text=(f"Flagging distance tuned so Kasarani's total ≈ Pamoja Trust's ~700 field count "
                  f"at {calibrated_dist} m. The total matched; structure-by-structure agreement "
                  "was never measured. Not a statutory riparian reserve line."),
        )
    if region == "Kasarani":
        return dict(
            key="calibrated-region-off-defaults", css="rd-badge-warn",
            badge="Calibrated region · off defaults",
            text=(f"Kasarani's count was matched at confidence ≥ {DEFAULT_CONFIDENCE_THRESHOLD:.2f}, "
                  f"P(built-up) ≥ {DEFAULT_MATERIAL_OVERLAP_THRESHOLD:.2f}, {calibrated_dist} m. "
                  f"Current settings ({conf:.2f} / {mat:.2f} / {dist} m) differ, so that match no "
                  "longer holds for the number shown."),
        )
    return dict(
        key="not-field-validated", css="rd-badge-warn", badge="Not field-validated",
        text=(f"No independent field count exists for {region}. The Kasarani flagging distance is "
              "reused here untested — the size of the error is unknown."),
    )


# ------------------------------------------------------------------- app state

inject_css()

summary = load_pipeline_summary()
model = load_model()
model_ready = model is not None
calibrated_distance_default = int(summary["calibrated_distance_m"]) if summary else 16

region_peek = st.session_state.get("region_select", REGIONS[0])

st.markdown(
    f"""
    <div class="rd-header">
      <div style="display:flex;align-items:center;gap:7px;">
        <svg width="14" height="14" viewBox="0 0 30 30" fill="none">
          <circle cx="15" cy="15" r="13" stroke="#4dd0c4" stroke-width="2"/>
          <path d="M9 17 L15 11 L21 17 Z" fill="#f2b544"/>
        </svg>
        <span class="rd-title">Riparian Encroachment Detector</span>
        <span class="rd-subtitle">&middot; {region_peek.upper()} &middot; VERIFICATION PLANNING VIEW</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="width:6px;height:6px;border-radius:50%;background:#4dd0c4;display:inline-block;"></span>
        <span class="mono" style="font-size:10px;color:#8a9aa3;text-transform:uppercase;">Live</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

(tb_region, tb_badge, tb_area, tb_filters, tb_stretch, tb_basemap, tb_model,
 tb_spacer, tb_export1, tb_export2) = st.columns(
    [1.6, 1.3, 0.8, 0.8, 0.95, 0.95, 0.75, 0.5, 1.15, 1.05], gap="small"
)

with tb_region:
    with st.popover(f"Region: {region_peek}", width="stretch"):
        region = st.selectbox("Region", REGIONS, key="region_select", label_visibility="collapsed")

# Tracks the previously shown region so the map can animate away from where
# the user was, rather than jump-cutting straight to the newly picked region.
# Computed here — right after the Region control — so the Area/Stretch focus
# logic below can reset itself on a genuine region change.
if "shown_region" not in st.session_state:
    st.session_state["shown_region"] = region
region_changed = st.session_state["shown_region"] != region
prev_region = st.session_state["shown_region"]
st.session_state["shown_region"] = region

# Map focus: either ("bounds",) = fit the region's river extent, or
# ("center", lat, lon, zoom). Last change wins between locality / search /
# stretch; reset on region change so the fly animation's landing view is the
# one later reruns keep. The *_applied trackers stop a persisted widget value
# from re-firing its jump on every unrelated rerun.
if region_changed or "focus" not in st.session_state:
    st.session_state["focus"] = ("bounds",)
    st.session_state["locality_applied"] = st.session_state.get("locality_select", FIRST_LOCALITY)
    st.session_state["search_applied"] = (region, st.session_state.get("area_query", ""))
    st.session_state["stretch_applied"] = st.session_state.get(f"stretch_jump_{region}", STRETCH_NONE)

with tb_badge:
    tb_badge_slot = st.empty()  # filled once the Filters values are known

with tb_area:
    with st.popover("Area", width="stretch"):
        if region == "Kasarani":
            area_mode = st.radio("Area input", ["Named locality", "Search (OpenStreetMap)"],
                                 key="area_mode", label_visibility="collapsed")
        else:
            area_mode = "Search (OpenStreetMap)"
            st.caption("No named localities catalogued for this region yet — search below, "
                       "use the Stretches control, or the map's own pan/zoom.")

        if area_mode == "Named locality":
            area_choice = st.selectbox("Locality", list(KASARANI_AREAS.keys()), key="locality_select")
            if area_choice != st.session_state.get("locality_applied"):
                loc_lat, loc_lon = KASARANI_AREAS[area_choice]
                st.session_state["focus"] = ("center", loc_lat, loc_lon, 15)
                st.session_state["locality_applied"] = area_choice
            st.caption(f"{KASARANI_AREAS[area_choice][0]:.5f}, {KASARANI_AREAS[area_choice][1]:.5f}")
        else:
            query = st.text_input("Search (OpenStreetMap)", placeholder="e.g. Mwiki, Nairobi",
                                  key="area_query", label_visibility="collapsed")
            if query:
                result = geocode_place_osm(query, region)
                if result is None:
                    st.warning("No match found — try a more specific name.")
                else:
                    found_lat, found_lon, display_name = result
                    if st.session_state.get("search_applied") != (region, query):
                        st.session_state["focus"] = ("center", found_lat, found_lon, 15)
                    st.caption(f"Found: {display_name}")
            st.session_state["search_applied"] = (region, query)
            st.caption("Free via OpenStreetMap's Nominatim service — no API key required.")

with tb_filters:
    with st.popover("Filters", width="stretch"):
        st.markdown('<div class="rd-eyebrow">Screening Filters</div>', unsafe_allow_html=True)
        confidence_threshold = st.slider("Open Buildings confidence ≥", 0.5, 1.0,
                                         DEFAULT_CONFIDENCE_THRESHOLD, 0.05)
        material_threshold = st.slider("RF built-up probability ≥", 0.0, 0.5,
                                       DEFAULT_MATERIAL_OVERLAP_THRESHOLD, 0.01)
        st.caption(
            "In the shipped data every row already has confidence ≥ 0.70, so that slider changes "
            "nothing until raised above 0.70. At its default the built-up probability filter "
            "removes under 2% of rows in every region. The count you see is driven almost "
            "entirely by the distance below."
        )

        st.markdown('<div class="rd-eyebrow" style="margin-top:14px;">Flagging Distance</div>',
                    unsafe_allow_html=True)
        flag_distance_m = st.slider("Flag within (m) of river", 5, 60, calibrated_distance_default, 1)
        st.caption(
            f"{calibrated_distance_default} m is the distance at which Kasarani's total matched "
            "Pamoja Trust's field count (~700 structures) — a count match, not a statutory "
            "reserve line. The same cutoff is applied to Gatharaini and Motoine untested; there "
            "is no independent count for either yet. Distance is centroid-to-centreline."
        )

with tb_stretch:
    tb_stretch_slot = st.empty()  # filled once stretches are computed

with tb_basemap:
    with st.popover("Basemap", width="stretch"):
        basemap_label = st.selectbox("Basemap", list(BASEMAPS.keys()), label_visibility="collapsed")

with tb_model:
    with st.popover("Model", width="stretch"):
        if model_ready:
            st.markdown(
                f"""
                <div class="mono" style="font-size:10.5px;color:var(--text-tertiary);line-height:1.6;">
                RandomForestClassifier &middot; {model.n_estimators} trees<br>
                Trained once on all three regions combined<br>
                Label: ESA WorldCover built-up (10 m land cover, not building presence)<br>
                Features: {", ".join(FEATURE_COLS)}<br>
                Role here: secondary filter only — it does not detect buildings
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("Model not available.")

with tb_export1:
    export1_slot = st.empty()
with tb_export2:
    export2_slot = st.empty()

# --------------------------------------------------------------------- compute

at_defaults = (
    math.isclose(confidence_threshold, DEFAULT_CONFIDENCE_THRESHOLD, abs_tol=1e-6)
    and math.isclose(material_threshold, DEFAULT_MATERIAL_OVERLAP_THRESHOLD, abs_tol=1e-6)
    and int(flag_distance_m) == calibrated_distance_default
)
trust = trust_state(region, at_defaults, confidence_threshold, material_threshold,
                    int(flag_distance_m), calibrated_distance_default)

with tb_badge_slot:
    st.markdown(
        f"<div style='padding-top:8px;'>"
        f"<span class='rd-badge {trust['css']}' style='font-size:9.5px;padding:5px 9px;' "
        f"title=\"{trust['text']}\">{trust['badge']}</span></div>",
        unsafe_allow_html=True,
    )

region_table = build_region_table(region)
data_ready = region_table is not None

if not data_ready:
    st.markdown(
        f"""
        <div class="rd-note">
          <b>No data found for {region}.</b> Looking for
          <span class="mono">{buildings_csv(region).relative_to(ROOT)}</span> and
          <span class="mono">{rivers_geojson(region).relative_to(ROOT)}</span>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

encroaching = apply_filters(region_table, confidence_threshold, material_threshold, flag_distance_m)
passed_filter = region_table[
    (region_table["confidence"] >= confidence_threshold)
    & (region_table["rf_builtup_prob"] >= material_threshold)
]
removed_conf = int((region_table["confidence"] < confidence_threshold).sum())
removed_mat = int(((region_table["confidence"] >= confidence_threshold)
                   & (region_table["rf_builtup_prob"] < material_threshold)).sum())

# area_m2 is a real size signal the pipeline computes but never used. Treating
# each footprint as a disc of equal area gives the closest a wall could plausibly
# sit to the river; centroids outside the flag distance whose edge could still
# be inside it are surfaced as "edge-possible" — a partial correction for the
# centroid rule's bias toward small structures, not a substitute for footprints.
HAS_AREA = "area_m2" in region_table.columns
if HAS_AREA:
    eq_radius = np.sqrt(passed_filter["area_m2"].clip(lower=0) / np.pi)
    edge_possible = passed_filter[
        (passed_filter["distance_to_river_m"] > flag_distance_m)
        & (passed_filter["distance_to_river_m"] - eq_radius <= flag_distance_m)
    ]
else:
    edge_possible = passed_filter.iloc[0:0]

not_flagged = passed_filter.loc[
    ~passed_filter.index.isin(encroaching.index) & ~passed_filter.index.isin(edge_possible.index)
]

# Stretches: the unit a field team walks. Counts on the map/table are the
# centroid-flagged candidates (the number this tool has always produced);
# edge-possible is carried alongside, never folded in.
stretches = build_stretches(region)
worksheet = pd.concat([
    encroaching.assign(flag_basis=f"centroid within {flag_distance_m} m of OSM centreline"),
    edge_possible.assign(flag_basis="edge-possible (area-adjusted); centroid outside flag distance"),
])
all_stretch_ids = assign_stretches(worksheet, stretches)
stretch_counts = all_stretch_ids.loc[encroaching.index].value_counts()
edge_counts = all_stretch_ids.loc[edge_possible.index].value_counts()
stretches["candidates"] = stretches["stretch_id"].map(stretch_counts).fillna(0).astype(int)
stretches["edge_possible"] = stretches["stretch_id"].map(edge_counts).fillna(0).astype(int)
stretches["per_100m"] = (stretches["candidates"] / stretches["length_m"].clip(lower=1) * 100).round(1)
ranked = stretches.sort_values(["candidates", "per_100m"], ascending=False).reset_index(drop=True)
ranked["rank"] = ranked.index + 1
top_ids = ranked.loc[ranked["candidates"] > 0, "stretch_id"].head(3).tolist()
stretches_with_candidates = int((stretches["candidates"] > 0).sum())
top_row = ranked.iloc[0] if (not ranked.empty and int(ranked.iloc[0]["candidates"]) > 0) else None

stretch_labels = {STRETCH_NONE: "Whole region (river extent)"}
for row in ranked.itertuples():
    stretch_labels[row.stretch_id] = (
        f"{row.stretch_id} · {int(row.candidates)} candidates · {int(row.length_m)} m"
    )
stretch_key = f"stretch_jump_{region}"
with tb_stretch_slot:
    with st.popover("Stretches", width="stretch"):
        st.markdown('<div class="rd-eyebrow">Centre map on a stretch</div>', unsafe_allow_html=True)
        stretch_choice = st.selectbox(
            "Stretch", [STRETCH_NONE] + ranked["stretch_id"].tolist(), key=stretch_key,
            format_func=lambda s: stretch_labels.get(s, s), label_visibility="collapsed",
        )
        st.caption(
            f"River cut into ~{STRETCH_LENGTH_M} m stretches, ranked by candidate count. "
            "On the map: top three red, any candidates amber, none grey."
        )

if stretch_choice != st.session_state.get("stretch_applied", STRETCH_NONE):
    if stretch_choice == STRETCH_NONE:
        st.session_state["focus"] = ("bounds",)
    else:
        sel = stretches.loc[stretches["stretch_id"] == stretch_choice].iloc[0]
        st.session_state["focus"] = ("center", float(sel["mid_lat"]), float(sel["mid_lon"]), 16)
    st.session_state["stretch_applied"] = stretch_choice
focused_stretch_id = None if stretch_choice == STRETCH_NONE else stretch_choice

# Verification worksheet: candidates + blank field columns + provenance on every
# row, so any copy of this file can be reproduced and contested on its merits.
rank_map = dict(zip(ranked["stretch_id"], ranked["rank"]))
worksheet.insert(0, "stretch_id", all_stretch_ids.reindex(worksheet.index).values)
worksheet.insert(1, "stretch_rank", worksheet["stretch_id"].map(rank_map))
worksheet["field_status"] = ""   # to be filled by the team: confirmed / not a structure / demolished / outside reserve
worksheet["field_notes"] = ""
worksheet["trust_state"] = trust["key"]
worksheet["region"] = region
worksheet["flag_distance_m"] = int(flag_distance_m)
worksheet["confidence_threshold"] = round(float(confidence_threshold), 2)
worksheet["material_threshold"] = round(float(material_threshold), 2)
worksheet["distance_basis"] = "building centroid to OSM river centreline"
worksheet["sentinel2_window"] = S2_WINDOW
worksheet["open_buildings_version"] = NOT_RECORDED
worksheet["osm_extract_date"] = NOT_RECORDED
worksheet["counts_are"] = "structures (Open Buildings points), not households"
worksheet["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
worksheet = worksheet.sort_values(["stretch_rank", "distance_to_river_m"], na_position="last")

# ---------------------------------------------------------------- metric cards

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns([1, 1, 1, 1.4], gap="small")
with metric_col1:
    with card("padding:8px 14px;"):
        st.markdown(f'<div class="rd-metric-label">Candidates within {flag_distance_m}m</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="rd-metric-mid">{len(encroaching):,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="rd-sub">structures, not households</div>', unsafe_allow_html=True)
with metric_col2:
    with card("padding:8px 14px;"):
        st.markdown('<div class="rd-metric-label">Priority stretch</div>', unsafe_allow_html=True)
        if top_row is not None:
            st.markdown(f'<div class="rd-metric-mid">{top_row["stretch_id"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="rd-sub">{int(top_row["candidates"])} candidates over '
                        f'{int(top_row["length_m"])} m</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rd-metric-mid">—</div>', unsafe_allow_html=True)
            st.markdown('<div class="rd-sub">no candidates at these settings</div>', unsafe_allow_html=True)
with metric_col3:
    with card("padding:8px 14px;"):
        st.markdown('<div class="rd-metric-label">Flagging distance</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rd-metric-mid">{flag_distance_m}m</div>', unsafe_allow_html=True)
        st.markdown('<div class="rd-sub">centroid to OSM centreline</div>', unsafe_allow_html=True)
with metric_col4:
    with st.popover("Screening funnel & provenance", width="stretch"):
        st.markdown('<div class="rd-card-title">Screening Funnel</div>', unsafe_allow_html=True)
        funnel = [
            ("Detected (Open Buildings)", f"{len(region_table):,}"),
            (f"Removed: confidence < {confidence_threshold:.2f}", f"{removed_conf:,}"),
            (f"Removed: P(built-up) < {material_threshold:.2f}",
             f"{removed_mat:,} ({removed_mat / max(len(region_table), 1):.1%})"),
            ("Passed both filters", f"{len(passed_filter):,}"),
            (f"Within {flag_distance_m}m of river (centroid)", f"{len(encroaching):,}"),
            ("Edge-possible, not flagged (area-adjusted)",
             f"{len(edge_possible):,}" if HAS_AREA else "n/a — no area_m2 column"),
        ]
        for label, value in funnel:
            st.markdown(
                f'<div class="rd-kv"><span>{label}</span><span>{value}</span></div>',
                unsafe_allow_html=True,
            )
        frozen_count = load_frozen_encroaching_count(region)
        if frozen_count is not None or (summary and region in summary):
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="rd-card-title">As Last Reported (notebook, frozen)</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="rd-sub">From the last full notebook run, at its original '
                        'screening settings. The live count above may legitimately differ — '
                        'if it does, the settings differ.</div>', unsafe_allow_html=True)
            if frozen_count is not None:
                st.markdown(f'<div class="rd-kv"><span>{encroaching_csv(region).name} rows</span>'
                            f'<span>{frozen_count:,}</span></div>', unsafe_allow_html=True)
            if summary and region in summary:
                for k, v in summary[region].items():
                    st.markdown(f'<div class="rd-kv"><span>{k}</span><span>{v}</span></div>',
                                unsafe_allow_html=True)
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="rd-card-title">Provenance</div>', unsafe_allow_html=True)
        for k, v in [("Sentinel-2 composite", S2_WINDOW),
                     ("Open Buildings version", NOT_RECORDED),
                     ("OSM waterways extract", NOT_RECORDED),
                     ("Trust state", trust["badge"])]:
            st.markdown(f'<div class="rd-kv"><span>{k}</span><span>{v}</span></div>',
                        unsafe_allow_html=True)

with export1_slot:
    st.download_button(
        "⬇ Worksheet",
        data=worksheet.to_csv(index=False).encode("utf-8"),
        file_name=f"{region.lower()}_verification_worksheet_{flag_distance_m}m_{trust['key']}.csv",
        mime="text/csv",
        width="stretch",
        help=(f"{len(encroaching):,} centroid-flagged + {len(edge_possible):,} edge-possible "
              "candidates with stretch id, blank field_status/field_notes, and provenance on "
              "every row. A verification list, not an enforcement list."),
    )
with export2_slot:
    st.download_button(
        "⬇ Full table",
        data=region_table.to_csv(index=False).encode("utf-8"),
        file_name=f"{region.lower()}_screened_structures.csv",
        mime="text/csv",
        width="stretch",
        help="Every Open Buildings point in the AOI with its distance to river — no filters applied.",
    )

map_tab, compare_tab, method_tab = st.tabs(["Detection map", "Compare areas", "Method & data"])

# ------------------------------------------------------------- tab 1: the map

with map_tab:
    with card("padding:0;overflow:hidden;"):
        st.markdown(
            f"""
            <div style="padding:14px 18px;border-bottom:1px solid var(--border-soft);">
              <div class="rd-card-title">{region} &middot; {flag_distance_m}m flagging distance
                <span class="rd-badge {trust['css']} rd-badge-inline">{trust['badge']}</span>
              </div>
              <div class="rd-sub">{len(encroaching):,} candidate structures on {stretches_with_candidates} of
                {len(stretches)} stretches &middot; {len(edge_possible):,} edge-possible not flagged &middot;
                {len(region_table):,} screened &middot; structures, not households</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        aoi = region_aoi_wgs84(region)
        aoi_bounds = leaflet_bounds(aoi)

        rivers_metric = load_rivers_metric(region)
        rivers_display = rivers_metric.to_crs(WGS84) if rivers_metric is not None else None
        river_bounds = leaflet_bounds(rivers_display) if rivers_display is not None else aoi_bounds

        if region_changed:
            # Start the map where the user just was (the previous region's
            # boundary), so switching regions reads as a pan/zoom away from
            # it rather than a jump-cut straight to the new one.
            prev_bounds = leaflet_bounds(region_aoi_wgs84(prev_region))
            (prev_s, prev_w), (prev_n, prev_e) = prev_bounds
            fmap = folium.Map(location=[(prev_s + prev_n) / 2, (prev_w + prev_e) / 2], zoom_start=13,
                              tiles=BASEMAPS[basemap_label], control_scale=True,
                              scrollWheelZoom=False)
            fmap.fit_bounds(prev_bounds)
        else:
            focus = st.session_state.get("focus", ("bounds",))
            if focus[0] == "center":
                _, f_lat, f_lon, f_zoom = focus
                fmap = folium.Map(location=[f_lat, f_lon], zoom_start=int(f_zoom),
                                  tiles=BASEMAPS[basemap_label], control_scale=True,
                                  scrollWheelZoom=False)
            else:
                (rb_s, rb_w), (rb_n, rb_e) = river_bounds
                fmap = folium.Map(location=[(rb_s + rb_n) / 2, (rb_w + rb_e) / 2], zoom_start=13,
                                  tiles=BASEMAPS[basemap_label], control_scale=True,
                                  scrollWheelZoom=False)
                fmap.fit_bounds(river_bounds, padding=(30, 30), max_zoom=16)

        folium.GeoJson(
            json.loads(aoi.to_json()),
            style_function=lambda _f: {"fillOpacity": 0, "color": TEAL, "weight": 1.2,
                                       "dashArray": "8 6", "opacity": 0.7},
            name="Region boundary",
        ).add_to(fmap)

        buffer_gdf = load_buffer_wgs84(region)
        if buffer_gdf is not None:
            folium.GeoJson(
                json.loads(buffer_gdf.to_json()),
                style_function=lambda _f: {"fillColor": TEAL, "color": AMBER, "weight": 1.1,
                                           "dashArray": "5 4", "fillOpacity": 0.10},
                name="60m riparian buffer",
            ).add_to(fmap)

        # Stretches, coloured by candidate count, drawn under the centreline.
        # Built as plain dicts (not GeoDataFrame.to_json) so every property is
        # a native Python type folium can serialise.
        if not stretches.empty:
            stretches_wgs = stretches.to_crs(WGS84)
            stretch_features = []
            for row in stretches_wgs.itertuples():
                if int(row.candidates) == 0:
                    color = GREY
                elif row.stretch_id in top_ids:
                    color = RED
                else:
                    color = AMBER
                stretch_features.append({
                    "type": "Feature",
                    "geometry": row.geometry.__geo_interface__,
                    "properties": {
                        "stretch_id": str(row.stretch_id), "length_m": int(row.length_m),
                        "candidates": int(row.candidates), "edge_possible": int(row.edge_possible),
                        "per_100m": float(row.per_100m), "color": color,
                        "trust": trust["badge"],
                    },
                })
            folium.GeoJson(
                {"type": "FeatureCollection", "features": stretch_features},
                style_function=lambda f: {"color": f["properties"]["color"], "weight": 7,
                                          "opacity": 0.55, "lineCap": "butt"},
                tooltip=folium.GeoJsonTooltip(
                    fields=["stretch_id", "length_m", "candidates", "edge_possible", "per_100m", "trust"],
                    aliases=["Stretch", "Length (m)", "Candidates", "Edge-possible", "Per 100 m", "Trust"],
                ),
                name=f"Stretches (~{STRETCH_LENGTH_M} m)",
            ).add_to(fmap)
            if focused_stretch_id is not None:
                focused = [f for f in stretch_features if f["properties"]["stretch_id"] == focused_stretch_id]
                if focused:
                    folium.GeoJson(
                        {"type": "FeatureCollection", "features": focused},
                        style_function=lambda _f: {"color": "#eef2f3", "weight": 13, "opacity": 0.28,
                                                   "lineCap": "butt"},
                        name=f"Selected stretch ({focused_stretch_id})",
                    ).add_to(fmap)

        if rivers_display is not None:
            folium.GeoJson(
                json.loads(rivers_display.to_json()),
                style_function=lambda _f: {"color": TEAL, "weight": 2},
                name="River centerline (OSM)",
            ).add_to(fmap)

        def marker_rows(subset: pd.DataFrame) -> list:
            rows_out = []
            for r in subset.itertuples():
                area = getattr(r, "area_m2", np.nan) if HAS_AREA else np.nan
                area_txt = f" &middot; {area:.0f} m²" if pd.notna(area) else ""
                rows_out.append([
                    round(r.lat, 6), round(r.lon, 6),
                    f"{r.distance_to_river_m:.0f}m to river &middot; conf {r.confidence:.2f} &middot; "
                    f"P(built-up) {r.rf_builtup_prob:.2f}{area_txt}",
                ])
            return rows_out

        for subset, color, label in [
            (encroaching, RED, f"Candidates within {flag_distance_m}m ({len(encroaching):,})"),
            (edge_possible, AMBER, f"Edge-possible, not flagged ({len(edge_possible):,})"),
            (not_flagged, GREY, f"Passed filter, outside {flag_distance_m}m ({len(not_flagged):,})"),
        ]:
            if subset.empty:
                continue
            data = marker_rows(subset)
            marker_js = (
                "function(row){var m=L.circleMarker(new L.LatLng(row[0],row[1]),"
                f"{{radius:4,color:'{color}',fillColor:'{color}',fillOpacity:0.85,weight:1}});"
                "m.bindTooltip(row[2]);return m;}"
            )
            FastMarkerCluster(
                data=data, callback=marker_js, name=label,
                options={"maxClusterRadius": 45, "showCoverageOnHover": False, "disableClusteringAtZoom": 17},
            ).add_to(fmap)

        if region_changed:
            # Two-stage fly: first out/across to the newly selected region's
            # boundary circle, then in to the tighter extent of its actual
            # waterways — chained on Leaflet's own 'moveend' event rather than
            # a guessed delay, so the second leg only starts once the first
            # has genuinely finished.
            map_var = fmap.get_name()
            transition_script = f"""
            setTimeout(function(){{
                var m = {map_var};
                m.flyToBounds({json.dumps(aoi_bounds)}, {{duration: 1.5, padding: [20, 20]}});
                m.once('moveend', function(){{
                    m.flyToBounds({json.dumps(river_bounds)}, {{duration: 1.1, padding: [30, 30], maxZoom: 16}});
                }});
            }}, 150);
            """
            fmap.get_root().script.add_child(folium.Element(transition_script))

        folium.LayerControl(collapsed=True).add_to(fmap)
        st_folium(fmap, height=640, use_container_width=True, returned_objects=[], key="main_map")

        st.markdown(
            f"""
            <div style="padding:10px 18px 14px;">
              <div class="rd-legend-row">
                <div class="rd-legend-item"><span class="rd-swatch" style="background:{RED};"></span>Candidate (centroid within {flag_distance_m}m)</div>
                <div class="rd-legend-item"><span class="rd-swatch" style="background:{AMBER};"></span>Edge-possible (area-adjusted, not flagged)</div>
                <div class="rd-legend-item"><span class="rd-swatch" style="background:{GREY};"></span>Passed filter, not flagged</div>
                <div class="rd-legend-item"><span class="rd-swatch-line" style="background:{RED};"></span>Top-3 stretch</div>
                <div class="rd-legend-item"><span class="rd-swatch-line" style="background:{AMBER};"></span>Stretch with candidates</div>
                <div class="rd-legend-item"><span class="rd-swatch-line" style="background:{GREY};"></span>Stretch, none</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    stretch_col, note_col = st.columns([1.35, 1], gap="medium")

    with stretch_col:
        with card():
            st.markdown('<div class="rd-card-title">Stretch Priority</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="rd-sub">Which ~{STRETCH_LENGTH_M} m of river to walk first. Ranked by '
                f"centroid-flagged candidates; edge-possible carried separately. Trust: "
                f"{trust['badge']}.</div>",
                unsafe_allow_html=True,
            )
            if ranked.empty:
                st.caption("No river stretches could be built for this region.")
            else:
                display = ranked.loc[:, ["rank", "stretch_id", "length_m", "candidates",
                                          "edge_possible", "per_100m"]].head(10).copy()
                display["trust"] = trust["badge"]
                display.columns = ["#", "Stretch", "Length (m)", "Candidates", "Edge-possible",
                                   "Per 100 m", "Trust"]
                st.dataframe(display, hide_index=True, width="stretch", height=300)
                st.caption(
                    "Use the Stretches control in the toolbar to centre the map on any row. "
                    "Full ranking is in the worksheet export."
                )

    with note_col:
        limits_html = "".join(
            f"<li><b style='color:var(--text-primary);'>{k}.</b> {v}</li>" for k, v in KNOWN_LIMITS
        )
        st.markdown(
            f"""
            <div class="rd-note">
              <b>Read before acting on this map.</b> {trust['text']}
              <ul class="rd-note-list">{limits_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    with st.expander("Model: what the Random Forest actually contributes here", expanded=False):
        if not model_ready:
            st.markdown(
                f"""
                <div class="rd-note">
                  <b>No trained model found.</b> Looking for
                  <span class="mono">{MODEL_PATH.relative_to(ROOT)}</span>.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            with card():
                st.markdown(f'<div class="rd-card-title">Effect on the {region} funnel at current settings</div>',
                            unsafe_allow_html=True)
                kv = [
                    ("Rows entering the RF filter", f"{len(region_table) - removed_conf:,}"),
                    (f"Removed by P(built-up) < {material_threshold:.2f}",
                     f"{removed_mat:,} ({removed_mat / max(len(region_table) - removed_conf, 1):.1%})"),
                    ("Removed by confidence filter", f"{removed_conf:,}"),
                    ("Estimator", f"RandomForestClassifier ({model.n_estimators} trees)"),
                    ("Label", "builtup — ESA WorldCover 10 m land cover, not building presence"),
                    ("Training scope", "All three regions combined, trained once"),
                ]
                st.markdown(
                    "".join(f'<div class="rd-kv"><span>{k}</span><span>{v}</span></div>' for k, v in kv),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div style="font-size:11px;color:var(--text-tertiary);margin-top:8px;">'
                    "At the default 0.05 the RF removes under 2% of rows in every region — the "
                    "flagged count is driven by Open Buildings confidence and the distance rule, "
                    "not by this classifier.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="rd-eyebrow">Held-out evaluation &middot; reproduced 80/20 split, '
                        'all three regions</div>', unsafe_allow_html=True)

            eval_result = evaluate_model_heldout()
            if eval_result is None:
                st.markdown(
                    """
                    <div class="rd-note">
                      The held-out split can only be reproduced with all three regions' labelled
                      feature tables present (<span class="mono">*_feature_table.csv</span>). One or
                      more is missing, so no accuracy figure is shown — scoring the model on the
                      table it was trained on would not be an accuracy estimate.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                report = eval_result["report"]
                bu = report.get("built-up", {})
                c1, c2, c3, c4 = st.columns(4, gap="medium")
                with c1:
                    metric_card("Accuracy (held-out)", f"{report.get('accuracy', float('nan')):.3f}",
                                f"{eval_result['n_test']:,} unseen rows")
                with c2:
                    metric_card("Precision (built-up)", f"{bu.get('precision', float('nan')):.3f}")
                with c3:
                    metric_card("Recall (built-up)", f"{bu.get('recall', float('nan')):.3f}")
                with c4:
                    metric_card("F1 (built-up)", f"{bu.get('f1-score', float('nan')):.3f}")

                st.markdown(
                    f'<div style="font-size:11px;color:var(--text-tertiary);margin-top:8px;">'
                    f"Split reproduced as stratified 80/20, random_state=42, tables concatenated "
                    f"Kasarani → Gatharaini → Motoine. Fit on the reproduced <i>training</i> split: "
                    f"{eval_result['train_accuracy']:.3f} — a Random Forest fits its own training "
                    f"rows near-perfectly, so if this is well below 1.0 the reproduced split does "
                    f"not match the notebook's and the held-out figures above are optimistic. "
                    f"Pixel-level labels from the same 3 km circles are spatially autocorrelated; "
                    f"no spatial blocking was used.</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
                eval_left, eval_right = st.columns([1, 1], gap="medium")
                with eval_left:
                    with card():
                        st.markdown('<div class="rd-card-title">Confusion Matrix (held-out)</div>',
                                    unsafe_allow_html=True)
                        counts_cm = np.array(eval_result["confusion_matrix"])
                        row_totals = counts_cm.sum(axis=1, keepdims=True)
                        shares = np.divide(counts_cm, row_totals, out=np.zeros_like(counts_cm, dtype=float),
                                           where=row_totals > 0)
                        n_cls = counts_cm.shape[0]
                        cls_names = ["not built-up", "built-up"][:n_cls]
                        cm_fig = go.Figure(go.Heatmap(
                            z=shares, x=[f"Predicted {c}" for c in cls_names],
                            y=[f"Actually {c}" for c in cls_names],
                            colorscale=[[0, "#12191d"], [0.5, "#2c6f6a"], [1, TEAL]],
                            zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
                            text=[[f"{c:,}<br>{s:.0%}" for c, s in zip(rc, rs)]
                                  for rc, rs in zip(counts_cm, shares)],
                            texttemplate="%{text}",
                            textfont=dict(family="IBM Plex Mono, monospace", size=13, color="#eef2f3"),
                        ))
                        cm_fig.update_layout(**plotly_layout(height=280, xaxis=dict(side="top"),
                                                             yaxis=dict(autorange="reversed")))
                        st.plotly_chart(cm_fig, width="stretch", config={"displayModeBar": False})
                with eval_right:
                    with card():
                        st.markdown('<div class="rd-card-title">Held-out accuracy by region</div>',
                                    unsafe_allow_html=True)
                        st.markdown('<div class="rd-sub">One pooled model; the per-region view is '
                                    'where a spectral bias (e.g. iron-sheet roofing in Motoine) '
                                    'would show up</div>', unsafe_allow_html=True)
                        st.dataframe(pd.DataFrame(eval_result["per_region"]), hide_index=True,
                                     width="stretch")

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            with card():
                st.markdown('<div class="rd-card-title">Feature Importance</div>', unsafe_allow_html=True)
                imp = pd.Series(dict(zip(FEATURE_COLS, model.feature_importances_))).sort_values()
                imp_fig = go.Figure(go.Bar(
                    x=imp.values, y=imp.index, orientation="h", marker_color=TEAL,
                    text=[f"{v:.3f}" for v in imp.values], textposition="outside",
                ))
                imp_fig.update_layout(**plotly_layout(height=280, showlegend=False,
                                                      xaxis=dict(gridcolor="#232b30", title=None,
                                                                 range=[0, imp.max() * 1.18]),
                                                      yaxis=dict(title=None)))
                st.plotly_chart(imp_fig, width="stretch", config={"displayModeBar": False})

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="rd-note">
                  <b>What this model actually does.</b> It does not detect buildings — Google Open
                  Buildings already found those. It scores each detected point on whether its
                  Sentinel-2 signature looks built-up, as a weak secondary check. At the shipped
                  thresholds it changes almost nothing; the numbers on this dashboard are, in
                  practice, "Open Buildings points within a distance of the OSM centreline". Trained
                  once across all three regions, so any spectral bias against one region's roofing
                  is invisible in the pooled figure — check the per-region held-out table.
                </div>
                """,
                unsafe_allow_html=True,
            )

# --------------------------------------------------------- tab 2: compare areas

with compare_tab:
    with card():
        st.markdown('<div class="rd-card-title">Candidate Structures by Region — coverage, not comparison</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="rd-sub">Only Kasarani is calibrated against independent ground truth, and only '
            "at the default settings — Gatharaini and Motoine reuse that same flagging distance "
            "untested. Hatched bars are not comparable to the solid one; read them as coverage of "
            "each AOI, not as a ranking of regions. Counts are structures, not households.</div>",
            unsafe_allow_html=True,
        )

        region_compare_rows = []
        for r in REGIONS:
            t = build_region_table(r)
            if t is None:
                continue
            passed_r = t[(t["confidence"] >= confidence_threshold) & (t["rf_builtup_prob"] >= material_threshold)]
            e = passed_r[passed_r["distance_to_river_m"] <= flag_distance_m]
            trust_r = trust_state(r, at_defaults, confidence_threshold, material_threshold,
                                  int(flag_distance_m), calibrated_distance_default)
            region_compare_rows.append({
                "region": r, "screened": len(t), "passed": len(passed_r),
                "removed_material": len(t) - len(passed_r), "encroaching": len(e),
                "trust": trust_r["badge"], "solid": trust_r["key"] == "count-matched",
            })

        if region_compare_rows:
            region_cdf = pd.DataFrame(region_compare_rows)
            region_fig = go.Figure(go.Bar(
                x=region_cdf["region"], y=region_cdf["encroaching"],
                marker=dict(
                    color=[RED if s else AMBER for s in region_cdf["solid"]],
                    pattern=dict(shape=["" if s else "/" for s in region_cdf["solid"]],
                                 fillmode="overlay", fgcolor=BG, solidity=0.35),
                ),
                text=[f"{v:,}" for v in region_cdf["encroaching"]], textposition="outside",
                customdata=region_cdf["trust"],
                hovertemplate="%{x}: %{y:,} candidates<br>%{customdata}<extra></extra>",
            ))
            region_fig.update_layout(**plotly_layout(height=280, showlegend=False,
                                                     yaxis=dict(gridcolor="#232b30", title=None),
                                                     xaxis=dict(title=None)))
            st.plotly_chart(region_fig, width="stretch", config={"displayModeBar": False})
            st.markdown(
                f'<div class="rd-legend-row">'
                f'<div class="rd-legend-item"><span class="rd-swatch" style="background:{RED};"></span>'
                f'Calibrated, count-matched at default settings (Kasarani only)</div>'
                f'<div class="rd-legend-item"><span class="rd-swatch" style="background:{AMBER};"></span>'
                f'Hatched: not field-validated, or calibrated region at non-default settings</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            compare_display = region_cdf.loc[:, ["region", "screened", "removed_material", "passed",
                                                 "encroaching", "trust"]].copy()
            compare_display.columns = ["Region", "Screened", "Removed by RF filter", "Passed filters",
                                       f"Within {flag_distance_m} m", "Trust"]
            st.dataframe(compare_display, hide_index=True, width="stretch")

    if region == "Kasarani":
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        with card():
            st.markdown('<div class="rd-card-title">Candidate Structures by Locality</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="rd-sub">All {len(encroaching):,} structures flagged '
                f"(centroid within {flag_distance_m}m of a river) — grouped by whichever named locality "
                "they're closest to. Structures, not households.</div>",
                unsafe_allow_html=True,
            )

            transformer = Transformer.from_crs(WGS84, METRIC_CRS, always_xy=True)
            encroaching_metric_xy = np.array([
                transformer.transform(lon, lat) for lon, lat in zip(encroaching["lon"], encroaching["lat"])
            ]) if len(encroaching) else np.empty((0, 2))

            locality_names = list(KASARANI_AREAS.keys())
            locality_xy = np.array([
                transformer.transform(area_lon, area_lat) for area_lat, area_lon in KASARANI_AREAS.values()
            ])

            if len(encroaching_metric_xy):
                # distance from every encroaching structure to every locality centre, then
                # assign each structure to whichever centre is closest — a fixed search
                # radius (the previous approach) either missed structures sitting between
                # localities or double-counted ones near two centres at once. Nearest-centre
                # assignment guarantees every structure is counted exactly once.
                diffs = encroaching_metric_xy[:, None, :] - locality_xy[None, :, :]
                dists = np.hypot(diffs[..., 0], diffs[..., 1])
                nearest_idx = dists.argmin(axis=1)
                counts = pd.Series(nearest_idx).value_counts()
            else:
                counts = pd.Series(dtype=int)

            compare_rows = [
                {"locality": name, "encroaching_count": int(counts.get(i, 0))}
                for i, name in enumerate(locality_names)
            ]

            cdf = pd.DataFrame(compare_rows).sort_values("encroaching_count", ascending=False)
            fig = go.Figure(go.Bar(
                x=cdf["locality"], y=cdf["encroaching_count"], marker_color=RED,
                text=[f"{v:,}" for v in cdf["encroaching_count"]], textposition="outside",
                hovertemplate="%{x}: %{y:,} candidate structures (nearest)<extra></extra>",
            ))
            fig.update_layout(**plotly_layout(height=320, showlegend=False,
                                              yaxis=dict(gridcolor="#232b30", title=None),
                                              xaxis=dict(title=None)))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.markdown(
                '<div style="font-size:11px;color:var(--text-tertiary);margin-top:8px;">'
                "Locality centres are best-effort approximate coordinates, not official "
                "administrative boundaries — treat this as a directional triage view, not a "
                "precise ward-by-ward count. For planning a walk, the stretch ranking on the map "
                "tab is the better unit.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        with card():
            st.markdown('<div class="rd-card-title">Locality Totals</div>', unsafe_allow_html=True)
            st.dataframe(cdf, hide_index=True, width="stretch")
    else:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="rd-note">Locality-level breakdown is only catalogued for Kasarani — no verified '
            f"locality coordinates exist for {region}, and inventing them would manufacture precision. "
            "Switch the Region control back to Kasarani to see the by-locality view, or use the "
            f"stretch ranking on the Detection map tab for {region}.</div>",
            unsafe_allow_html=True,
        )

# ------------------------------------------------------- tab 3: method & data

with method_tab:
    left, right = st.columns([1.3, 1], gap="medium")

    with left:
        with card():
            st.markdown('<div class="rd-card-title">Pipeline</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div style="font-size:12.5px;color:var(--text-secondary);line-height:1.75;margin-top:8px;">
                <b style="color:var(--text-primary);">1 &middot; Preprocessing</b>
                (<span class="mono">01_preprocessing.ipynb</span>) — clips OSM waterways to each
                region's fixed-radius AOI, builds a 60m riparian buffer, and pulls a cloud-masked
                Sentinel-2 composite (B2/B3/B4/B8/B11/B12 + NDVI/NDBI), sampled against ESA
                WorldCover ground truth.<br><br>
                <b style="color:var(--text-primary);">2 &middot; Modelling</b>
                (<span class="mono">02_modelling.ipynb</span>) — trains one Random Forest across all
                three regions' feature tables, detects candidate structures via Google Open
                Buildings (chosen over a custom detector because the median footprint here is
                ~8m across — smaller than Sentinel-2's 10m pixels), then scores each one with the
                RF's built-up probability.<br><br>
                <b style="color:var(--text-primary);">3 &middot; Fusion &amp; report</b>
                (<span class="mono">03_fusion_and_report.ipynb</span>) — computes true distance from
                each building centroid to the nearest OSM river line, filters on confidence and RF
                probability, and sweeps a flagging distance until Kasarani's total matches its
                known field count, then applies that same distance elsewhere.<br><br>
                <b style="color:var(--text-primary);">4 &middot; This dashboard</b> — recomputes the
                filter live, cuts each river into ~400 m stretches, assigns candidates to their nearest
                stretch, and exports a verification worksheet with provenance on every row. It is a
                tool for deciding <i>where to send a field team first</i>, not a list of structures
                to act on.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        with card():
            st.markdown('<div class="rd-card-title">Known limits — who ends up on the list, and why</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:12.5px;color:var(--text-secondary);line-height:1.7;margin-top:8px;">'
                + "".join(f"<p style='margin:0 0 8px 0;'><b style='color:var(--text-primary);'>{k}.</b> {v}</p>"
                          for k, v in KNOWN_LIMITS)
                + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        if summary and "note" in summary:
            st.markdown(f'<div class="rd-note">{summary["note"]}</div>', unsafe_allow_html=True)

    with right:
        with card():
            st.markdown('<div class="rd-card-title">Provenance</div>', unsafe_allow_html=True)
            prov = [
                ("Sentinel-2 composite", S2_WINDOW),
                ("Open Buildings version", NOT_RECORDED),
                ("OSM waterways extract date", NOT_RECORDED),
                ("Labels", "ESA WorldCover built-up (10 m land cover)"),
                ("Distance basis", "building centroid → OSM river centreline"),
                ("Flagging distance origin",
                 f"{calibrated_distance_default} m — Kasarani total matched to Pamoja Trust (~700)"),
                ("Calibration source", "Pamoja Trust field count, Kasarani only"),
                ("Stretch length", f"~{STRETCH_LENGTH_M} m"),
                ("Current trust state", trust["badge"]),
                ("Counts are", "structures, not households"),
            ]
            st.markdown(
                "".join(f'<div class="rd-kv"><span>{k}</span><span>{v}</span></div>' for k, v in prov),
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="font-size:11px;color:var(--text-tertiary);margin-top:8px;">'
                "The two values marked not recorded are genuine gaps in the pipeline, not omissions "
                "in this view — a list whose input vintages are unknown cannot be fully reproduced "
                "or contested. They are written as such into every worksheet export.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        with card():
            st.markdown('<div class="rd-card-title">Artifacts</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="rd-sub">Notebook-frozen files hold the last full run at its own settings. '
                "Everything on the map and in the exports is recomputed live and may legitimately "
                "differ from them.</div>",
                unsafe_allow_html=True,
            )
            rows = []
            for r in REGIONS:
                for path, what, kind in [
                    (buildings_csv(r), f"{r} — detected structures + RF probability", "pipeline intermediate"),
                    (encroaching_csv(r), f"{r} — encroaching structures", "notebook-frozen output"),
                    (rivers_geojson(r), f"{r} — river lines (OSM)", "input — extract date not recorded"),
                    (riparian_buffer_geojson(r), f"{r} — 60 m riparian buffer", "derived"),
                    (feature_table_csv(r), f"{r} — labelled Sentinel-2 samples", "training data"),
                ]:
                    exists = path.exists()
                    rows.append({
                        "File": path.name, "Holds": what, "Kind": kind,
                        "Size": f"{path.stat().st_size / 1e6:.1f} MB" if exists else "—",
                        "Status": "present" if exists else "missing",
                    })
            rows.append({
                "File": MODEL_PATH.name,
                "Holds": "Shared Random Forest (trained on Kasarani, Gatharaini and Motoine combined)",
                "Kind": "model",
                "Size": f"{MODEL_PATH.stat().st_size / 1e6:.1f} MB" if MODEL_PATH.exists() else "—",
                "Status": "present" if MODEL_PATH.exists() else "missing",
            })
            rows.append({
                "File": SUMMARY_PATH.name,
                "Holds": "calibrated_distance_m + per-region counts (no thresholds recorded)",
                "Kind": "notebook-frozen output",
                "Size": f"{SUMMARY_PATH.stat().st_size / 1e3:.1f} KB" if SUMMARY_PATH.exists() else "—",
                "Status": "present" if SUMMARY_PATH.exists() else "missing",
            })

            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=380)
