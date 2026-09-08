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
from shapely.geometry import Point, box
from streamlit_folium import st_folium

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "processed"

MODEL_PATH = DATA_DIR / "rf_baseline.joblib"
SUMMARY_PATH = DATA_DIR / "pipeline_summary.json"

WGS84 = "EPSG:4326"

DEFAULT_LAT = -1.263295
DEFAULT_LON = 36.880376

STUDY_AREAS = {
    "kasarani": {"label": "Kasarani", "place": "Kasarani, Nairobi"},
    "gatharani": {"label": "Gatharani", "place": "Gatharani, Nairobi"},
    "motoine": {"label": "Motoine", "place": "Motoine"},
}
ALL_AREAS_KEY = "all"

NAIROBI_LOCATIONS = {
    "Kasarani": (-1.2295, 36.8908),
    "Mwiki": (-1.2154, 36.8967),
    "Sunton": (-1.2244, 36.9012),
    "Hunters": (-1.2312, 36.8874),
    "Clay City": (-1.2401, 36.8851),
    "Roysambu": (-1.2190, 36.8890),
    "Zimmerman": (-1.2080, 36.8890),
    "Githurai 44": (-1.1990, 36.9020),
    "Kahawa West": (-1.1890, 36.9150),
}
MAX_POLYGONS = 600

RISK_COLORS = {
    "High Risk (<10m)": "#e2604f",
    "Medium Risk (10m-20m)": "#f2b544",
    "Low Risk (20m-30m)": "#4dd0c4",
    "Safe Zone (>30m)": "#4a5860",
}
RISK_ORDER = ["Safe Zone (>30m)", "Low Risk (20m-30m)", "Medium Risk (10m-20m)", "High Risk (<10m)"]

TEAL = "#4dd0c4"
AMBER = "#f2b544"
RED = "#e2604f"
GREEN = "#6ad18a"
GREY = "#4a5860"

AGREEMENT_COLORS = {
    "Both flag": GREEN,
    "Model only": AMBER,
    "Rule only": RED,
    "Neither": GREY,
}

MODE_GEOMETRIC = "Geometric buffer rule"
MODE_RF = "Random Forest"
MODE_COMPARE = "Compare rule vs model"

BASEMAPS = {
    "OpenStreetMap": "OpenStreetMap",
    "Satellite (Esri)": "Esri.WorldImagery",
    "Dark (CartoDB)": "CartoDB dark_matter",
}

st.set_page_config(
    page_title="Riparian Encroachment Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def area_keys(scope: str) -> list[str]:
    return list(STUDY_AREAS) if scope == ALL_AREAS_KEY else [scope]


def scope_label(scope: str) -> str:
    if scope == ALL_AREAS_KEY:
        return "Kasarani · Gatharani · Motoine"
    return STUDY_AREAS[scope]["label"]


def area_paths(area: str) -> dict:
    return {
        "buildings": DATA_DIR / f"{area}_buildings.csv",
        "encroaching": DATA_DIR / f"{area}_encroaching_buildings.csv",
        "features": DATA_DIR / f"{area}_feature_table.csv",
        "rivers": DATA_DIR / f"{area}_rivers.geojson",
        "buffer": DATA_DIR / f"{area}_riparian_buffer.geojson",
    }


def utm_epsg_for(lon: float, lat: float) -> str:
    zone = math.floor((lon + 180) / 6) + 1
    epsg_code = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg_code}"


@st.cache_resource(show_spinner=False)
def transformer_to_metric(crs: str) -> Transformer:
    return Transformer.from_crs(WGS84, crs, always_xy=True)


@st.cache_resource(show_spinner=False)
def transformer_to_wgs84(crs: str) -> Transformer:
    return Transformer.from_crs(crs, WGS84, always_xy=True)


GEOM_COL_CANDIDATES = ("geometry", "wkt", "WKT", "geom")
LATLON_COL_CANDIDATES = (
    ("lat", "lon"), ("latitude", "longitude"),
    ("centroid_lat", "centroid_lon"), ("y", "x"),
)


def read_csv_geoms(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    df["id"] = df["id"].astype(str)

    geom_col = next((c for c in GEOM_COL_CANDIDATES if c in df.columns), None)
    if geom_col is not None:
        geoms = gpd.GeoSeries.from_wkt(df[geom_col])
        gdf = gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=geoms, crs=WGS84)
    else:
        latlon = next((pair for pair in LATLON_COL_CANDIDATES
                       if pair[0] in df.columns and pair[1] in df.columns), None)
        if latlon is None:
            raise ValueError(
                f"{path.name} has neither a WKT geometry column nor lat/lon columns — "
                "the dashboard can't place these structures on a map."
            )
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[latlon[1]], df[latlon[0]]), crs=WGS84
        )
    if "id" not in gdf.columns:
        gdf["id"] = gdf.index.astype(str)
    gdf["id"] = gdf["id"].astype(str)
    return gdf


def risk_category(dist_m: float) -> str:
    if dist_m < 10:
        return "High Risk (<10m)"
    if dist_m < 20:
        return "Medium Risk (10m-20m)"
    if dist_m < 30:
        return "Low Risk (20m-30m)"
    return "Safe Zone (>30m)"


@st.cache_data(show_spinner=False)
def load_area_buildings(area: str) -> gpd.GeoDataFrame:
    paths = area_paths(area)
    if not paths["buildings"].exists():
        return gpd.GeoDataFrame(
            {"id": pd.Series(dtype=str), "dist_to_river_m": pd.Series(dtype=float),
             "risk_category": pd.Series(dtype=str), "area": pd.Series(dtype=str),
             "rule_flag": pd.Series(dtype=bool), "total_area_m2": pd.Series(dtype=float)},
            geometry=[], crs=WGS84,
        )

    gdf = read_csv_geoms(paths["buildings"])
    if "id" not in gdf.columns:
        gdf["id"] = gdf.index.astype(str)
    gdf["id"] = gdf["id"].astype(str)
    gdf["area"] = STUDY_AREAS[area]["label"]

    cx, cy = gdf.total_bounds[[0, 1]] + (gdf.total_bounds[[2, 3]] - gdf.total_bounds[[0, 1]]) / 2
    metric_crs = utm_epsg_for(cx, cy)
    metric = gdf.to_crs(metric_crs)

    if paths["rivers"].exists():
        rivers = gpd.read_file(paths["rivers"]).to_crs(metric_crs)
        river_union = rivers.geometry.union_all()
        metric["dist_to_river_m"] = metric.geometry.distance(river_union)
    elif "dist_to_river_m" not in metric.columns:
        metric["dist_to_river_m"] = np.nan

    gdf["dist_to_river_m"] = metric["dist_to_river_m"].to_numpy()
    gdf["risk_category"] = gdf["dist_to_river_m"].map(
        lambda d: risk_category(d) if pd.notna(d) else "Safe Zone (>30m)"
    )

    if paths["encroaching"].exists():
        enc_df = pd.read_csv(paths["encroaching"])
        enc_ids = set(enc_df["id"].astype(str)) if "id" in enc_df.columns else set(enc_df.index.astype(str))
        gdf["rule_flag"] = gdf["id"].astype(str).isin(enc_ids)
    else:
        gdf["rule_flag"] = gdf["dist_to_river_m"] <= 30

    if "total_area_m2" not in gdf.columns:
        gdf["total_area_m2"] = metric.geometry.area.to_numpy()
    return gdf


@st.cache_data(show_spinner=False)
def load_scope_buildings(scope: str) -> gpd.GeoDataFrame:
    frames = [load_area_buildings(a) for a in area_keys(scope)]
    return pd.concat(frames, ignore_index=True) if frames else gpd.GeoDataFrame(geometry=[], crs=WGS84)


@st.cache_data(show_spinner=False)
def load_geojson(scope: str, kind: str) -> gpd.GeoDataFrame:
    frames = []
    for area in area_keys(scope):
        path = area_paths(area)[kind]
        if path.exists():
            gdf = gpd.read_file(path)
            if not gdf.empty:
                gdf["area"] = STUDY_AREAS[area]["label"]
                frames.append(gdf.to_crs(WGS84))
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_feature_table(area: str) -> pd.DataFrame | None:
    path = area_paths(area)["features"]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    df["id"] = df["id"].astype(str)
    if "encroachment" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "encroachment"})
    if "encroachment" not in df.columns:
        df["encroachment"] = 0
    return df


@st.cache_data(show_spinner=False)
def load_pipeline_summary() -> dict | None:
    if not SUMMARY_PATH.exists():
        return None
    try:
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def model_feature_cols(model) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    n = getattr(model, "n_features_in_", None)
    return [f"feature_{i}" for i in range(n)] if n else []


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame | None:
    model = load_model()
    if model is None:
        return None
    feat_cols = model_feature_cols(model)
    rows = []
    for area in STUDY_AREAS:
        table = load_feature_table(area)
        if table is None or not set(feat_cols).issubset(table.columns):
            continue
        scored = table.dropna(subset=feat_cols).copy()
        if scored.empty:
            continue
        if "id" not in scored.columns:
            scored["id"] = scored.index.astype(str)
        scored["id"] = scored["id"].astype(str)
        if "encroachment" not in scored.columns:
            scored["encroachment"] = 0
        scored["rf_proba"] = model.predict_proba(scored[feat_cols])[:, 1]
        scored["area"] = STUDY_AREAS[area]["label"]
        keep = ["id", "area", "rf_proba", "encroachment"]
        rows.append(scored[keep])
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def get_proba_dict(preds: pd.DataFrame | None) -> dict:
    if preds is None or preds.empty:
        return {}
    return dict(zip(zip(preds["area"], preds["id"].astype(str)), preds["rf_proba"]))


def add_rf_proba(df: pd.DataFrame, preds: pd.DataFrame | None) -> pd.DataFrame:
    if df.empty:
        df["rf_proba"] = np.nan
        return df
    if preds is None or preds.empty:
        df["rf_proba"] = np.nan
        return df
    p_dict = get_proba_dict(preds)
    df["rf_proba"] = [p_dict.get((a, str(i)), np.nan) for a, i in zip(df["area"], df["id"])]
    return df


model = load_model()
predictions = load_predictions()
model_ready = model is not None and predictions is not None
summary = load_pipeline_summary()


@st.cache_data(show_spinner=False)
def scope_default_center(scope: str) -> tuple[float, float]:
    gdf = load_scope_buildings(scope)
    if gdf.empty:
        return DEFAULT_LAT, DEFAULT_LON
    cx, cy = gdf.total_bounds[[0, 1]] + (gdf.total_bounds[[2, 3]] - gdf.total_bounds[[0, 1]]) / 2
    return float(cy), float(cx)


@st.cache_data(show_spinner=False)
def scope_metric_crs(scope: str) -> str:
    gdf = load_scope_buildings(scope)
    if gdf.empty:
        return "EPSG:32737"
    cx, cy = gdf.total_bounds[[0, 1]] + (gdf.total_bounds[[2, 3]] - gdf.total_bounds[[0, 1]]) / 2
    return utm_epsg_for(cx, cy)


@st.cache_data(show_spinner=False)
def hotspot_latlon(scope: str) -> tuple[float, float]:
    gdf = load_scope_buildings(scope)
    if gdf.empty or gdf["dist_to_river_m"].isna().all():
        return scope_default_center(scope)
    closest = gdf.loc[gdf["dist_to_river_m"].idxmin()].geometry.centroid
    crs = scope_metric_crs(scope)
    if gdf.crs == WGS84:
        return float(closest.y), float(closest.x)
    lon, lat = transformer_to_wgs84(crs).transform(closest.x, closest.y)
    return lat, lon


@st.cache_data(show_spinner=False)
def scope_risk_counts(scope: str) -> pd.Series:
    return load_scope_buildings(scope)["risk_category"].value_counts()


def fit_zoom(span_lon: float, px_width: int = 1000) -> int:
    if span_lon <= 0:
        return 12
    return max(1, int(math.floor(math.log2(360 * px_width / (256 * span_lon)))))


@st.cache_data(show_spinner="Searching OpenStreetMap...", ttl=3600)
def geocode_place_osm(query: str) -> tuple[float, float, str] | None:
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": f"{query}, Nairobi, Kenya",
                "format": "json",
                "limit": 1,
                "viewbox": "36.75,-1.35,37.00,-1.15",
                "bounded": 1,
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


def confusion_at(preds: pd.DataFrame, threshold: float) -> dict:
    predicted = (preds["rf_proba"] >= threshold).to_numpy()
    actual = preds["encroachment"].to_numpy() == 1

    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / total if total else 0.0,
        "n": total,
    }


@st.cache_data(show_spinner=False)
def threshold_sweep() -> pd.DataFrame:
    rows = []
    if predictions is not None:
        for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
            rows.append({"threshold": float(thr), **confusion_at(predictions, float(thr))})
    return pd.DataFrame(rows)


def inject_css() -> None:
    st.markdown(
        """
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap">
        <style>
        :root{
            --bg-panel:#161c20; --bg-panel-alt:#1b2226; --bg-raised:#20282d;
            --border:#2a333a; --border-soft:#232b30;
            --text-primary:#eef2f3; --text-secondary:#8a9aa3; --text-tertiary:#5c6b73;
            --accent-teal:#4dd0c4; --accent-teal-dim:#3f8f86;
            --accent-amber:#f2b544; --accent-green:#6ad18a; --accent-red:#e2604f;
        }
        html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
        .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }

        [data-testid="stAppViewContainer"] { background: #0f1417; }
        [data-testid="stSidebar"] { background: var(--bg-panel); border-right: 1px solid var(--border); }
        [data-testid="stSidebar"] .stMarkdown p { color: var(--text-secondary); }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 1.4rem; }
        .rd-eyebrow{ text-transform:uppercase; letter-spacing:.08em; font-size:11px; color:var(--text-tertiary); font-weight:600; margin-bottom:6px; }
        .rd-card{ background:var(--bg-panel); border:1px solid var(--border); border-radius:12px; padding:16px 18px; height:100%; }
        .rd-card-title{ font-size:12px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:var(--text-primary); }
        .rd-sub{ font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--text-tertiary); margin-top:2px; line-height:1.5; }
        .rd-badge{ font-family:'IBM Plex Mono',monospace; font-size:11px; padding:6px 12px; border-radius:6px; border:1px solid var(--border); color:var(--text-tertiary); display:inline-block; }
        .rd-badge-active{ background:rgba(106,209,138,0.12); border:1px solid var(--accent-green); color:var(--accent-green); }
        .rd-badge-warn{ background:rgba(242,181,68,0.10); border:1px solid var(--accent-amber); color:var(--accent-amber); }
        .rd-model-card{ padding:10px 12px; border-radius:8px; border:1px solid var(--border); background:var(--bg-panel-alt); }
        .rd-model-card-on{ border:1px solid var(--accent-green); background:rgba(106,209,138,0.08); }
        .rd-model-card-warn{ border:1px solid var(--accent-amber); background:rgba(242,181,68,0.07); }
        .rd-metric-big{ font-family:'IBM Plex Mono',monospace; font-size:44px; font-weight:700; color:var(--accent-amber); line-height:1; }
        .rd-metric-mid{ font-family:'IBM Plex Mono',monospace; font-size:25px; font-weight:600; color:var(--text-primary); }
        .rd-metric-label{ text-transform:uppercase; letter-spacing:.06em; font-size:10.5px; color:var(--text-tertiary); }
        .rd-progress-track{ margin-top:10px; height:6px; border-radius:3px; background:var(--bg-raised); overflow:hidden; }
        .rd-progress-fill{ height:100%; background:var(--accent-amber); }
        .rd-legend-row{ display:flex; gap:14px; flex-wrap:wrap; margin-top:10px; }
        .rd-legend-item{ display:flex; align-items:center; gap:6px; font-size:11px; color:var(--text-secondary); }
        .rd-swatch{ width:10px; height:10px; border-radius:2px; display:inline-block; }
        .rd-header{ display:flex; align-items:center; justify-space-between; padding:14px 4px 18px; border-bottom:1px solid var(--border); margin-bottom:18px; }
        .rd-title{ font-size:18px; font-weight:700; letter-spacing:.02em; color:var(--text-primary); }
        .rd-subtitle{ font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--text-tertiary); letter-spacing:.03em; }
        .rd-note{ border-left:2px solid var(--accent-amber); background:rgba(242,181,68,0.06); padding:12px 14px; border-radius:0 8px 8px 0; font-size:12.5px; color:var(--text-secondary); line-height:1.65; }
        .rd-kv{ display:flex; justify-content:space-between; gap:12px; padding:5px 0; border-bottom:1px solid var(--border-soft); font-size:12px; }
        .rd-kv span:first-child{ color:var(--text-tertiary); }
        .rd-kv span:last-child{ color:var(--text-primary); font-family:'IBM Plex Mono',monospace; }
        .stButton>button{ background:var(--accent-amber); color:#1a1508; border:none; font-weight:600; border-radius:8px; }
        .stButton>button:hover{ background:#f7c368; color:#1a1508; }
        [data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:10px; }
        .stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--border); }
        .stTabs [data-baseweb="tab"]{ font-size:12px; letter-spacing:.04em; text-transform:uppercase; font-weight:600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card_open(extra_style: str = "") -> None:
    st.markdown(f'<div class="rd-card" style="{extra_style}">', unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "", color: str | None = None) -> None:
    card_open()
    st.markdown(f'<div class="rd-metric-label">{label}</div>', unsafe_allow_html=True)
    style = f"margin-top:4px;{'color:' + color + ';' if color else ''}"
    st.markdown(f'<div class="rd-metric-mid" style="{style}">{value}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(
            f'<div style="font-size:11px;color:var(--text-tertiary);margin-top:2px;">{sub}</div>',
            unsafe_allow_html=True,
        )
    card_close()


def plotly_layout(**overrides) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", color="#8a9aa3", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(bgcolor="#1b2226", bordercolor="#2a333a",
                        font=dict(family="IBM Plex Mono, monospace", color="#eef2f3", size=11)),
    )
    base.update(overrides)
    return base


inject_css()

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
          <svg width="28" height="28" viewBox="0 0 30 30" fill="none">
            <circle cx="15" cy="15" r="13" stroke="#4dd0c4" stroke-width="1.4" opacity="0.35"/>
            <circle cx="15" cy="15" r="9" stroke="#4dd0c4" stroke-width="1.4" opacity="0.6"/>
            <path d="M9 17 L15 11 L21 17 Z" fill="#f2b544"/>
            <rect x="12" y="17" width="6" height="5" fill="#f2b544"/>
          </svg>
          <div>
            <div style="font-weight:600;font-size:14px;color:#eef2f3;line-height:1.2;">RIPARIAN DETECTOR</div>
            <div class="mono" style="font-size:10px;color:#5c6b73;">NAIROBI RIVERS &middot; 3 STUDY AREAS</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rd-eyebrow">Study Area</div>', unsafe_allow_html=True)
    scope_options = [STUDY_AREAS[a]["label"] for a in STUDY_AREAS] + ["All three areas"]
    scope_keys = list(STUDY_AREAS) + [ALL_AREAS_KEY]
    scope_choice = st.radio(
        "Study area", scope_options, index=0, label_visibility="collapsed",
        help="The model was trained on all three areas combined; pick one to inspect it, "
             "or all three for the aggregate view.",
    )
    scope = scope_keys[scope_options.index(scope_choice)]

    buildings = load_scope_buildings(scope)
    default_lat, default_lon = scope_default_center(scope)

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Query Location</div>', unsafe_allow_html=True)
    if "pending_jump" in st.session_state:
        st.session_state["lat"], st.session_state["lon"] = st.session_state.pop("pending_jump")
    st.session_state.setdefault("lat", default_lat)
    st.session_state.setdefault("lon", default_lon)
    if "last_scope" not in st.session_state or st.session_state["last_scope"] != scope:
        st.session_state["lat"], st.session_state["lon"] = default_lat, default_lon
        st.session_state["last_scope"] = scope

    location_mode = st.radio(
        "Location input", ["Choose a place", "Search (OpenStreetMap)", "Custom coordinates"],
        label_visibility="collapsed",
    )

    if location_mode == "Choose a place":
        place_choice = st.selectbox("Place", list(NAIROBI_LOCATIONS.keys()))
        st.session_state["lat"], st.session_state["lon"] = NAIROBI_LOCATIONS[place_choice]
        lat, lon = st.session_state["lat"], st.session_state["lon"]
        st.caption(f"{lat:.5f}, {lon:.5f}")

    elif location_mode == "Search (OpenStreetMap)":
        query = st.text_input("Place name", placeholder="e.g. Mwiki, Nairobi")
        if query:
            result = geocode_place_osm(query)
            if result is None:
                st.warning("No match found for that place — try a more specific name.")
                lat, lon = st.session_state["lat"], st.session_state["lon"]
            else:
                lat, lon, display_name = result
                st.session_state["lat"], st.session_state["lon"] = lat, lon
                st.caption(f"Found: {display_name}")
        else:
            lat, lon = st.session_state["lat"], st.session_state["lon"]
        st.caption("Free-text search via OpenStreetMap's Nominatim service — no API key required.")

    else:
        lat = st.number_input("Latitude", key="lat", format="%.6f")
        lon = st.number_input("Longitude", key="lon", format="%.6f")

    if st.button("Jump to structure closest to river", use_container_width=True):
        st.session_state["pending_jump"] = hotspot_latlon(scope)
        st.rerun()

    view_radius = st.slider("View radius (m)", min_value=100, max_value=600, value=250, step=50)

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Detection Layer</div>', unsafe_allow_html=True)
    mode_options = [MODE_GEOMETRIC, MODE_RF, MODE_COMPARE] if model_ready else [MODE_GEOMETRIC]
    mode = st.radio("Detection layer", mode_options, label_visibility="collapsed")
    if not model_ready:
        st.caption("Random Forest layers unlock once rf_baseline.joblib exists in data/processed.")

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Buffer Zone</div>', unsafe_allow_html=True)
    default_buffer = 30
    if isinstance(summary, dict):
        default_buffer = int(summary.get("buffer_m", summary.get("buffer_meters", 30)) or 30)
    buffer_m = st.slider("Riparian buffer (m)", min_value=10, max_value=60, value=default_buffer, step=5)
    st.caption(
        "Each area's riparian buffer comes from the preprocessing run; the "
        "geometric layer recomputes live at whatever distance you set."
    )

    threshold = 0.5
    if model_ready:
        st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Model Threshold</div>', unsafe_allow_html=True)
        threshold = st.slider("Flag a structure when P(encroachment) exceeds", 0.05, 0.95, 0.50, 0.05)
        st.caption("Encroaching structures are a small share of each AOI, so the 0.5 default is rarely the useful operating point.")

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Basemap</div>', unsafe_allow_html=True)
    basemap_label = st.selectbox("Basemap", list(BASEMAPS.keys()), label_visibility="collapsed")

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Detection Model</div>', unsafe_allow_html=True)
    if model_ready:
        n_scored = len(predictions)
        rf_card = f"""
          <div class="rd-model-card rd-model-card-on">
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <div style="font-size:12.5px;font-weight:600;color:var(--accent-green);">Phase 1 &middot; Random Forest</div>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="7" fill="#6ad18a"/>
              <path d="M4 7L6 9L10 5" stroke="#161c20" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <div class="rd-sub" style="margin-top:3px;">
              Trained on 3 areas &middot; {len(model_feature_cols(model))} spectral features<br>
              {n_scored:,} scored structures &middot; rf_baseline.joblib
            </div>
          </div>"""
    else:
        rf_card = """
          <div class="rd-model-card rd-model-card-warn">
            <div style="font-size:12.5px;font-weight:600;color:var(--accent-amber);">Phase 1 &middot; Random Forest</div>
            <div class="rd-sub" style="margin-top:3px;">Artifacts missing &mdash; expected at
            <span class="mono">data/processed/rf_baseline.joblib</span></div>
          </div>"""

    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;gap:8px;">
          {rf_card}
          <div class="rd-model-card">
            <div style="font-size:12.5px;font-weight:500;color:var(--text-secondary);">Phase 2 &middot; Deep Learning</div>
            <div class="rd-sub" style="margin-top:3px;">Instance counter &mdash; not yet trained</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mono" style="font-size:10px;color:#5c6b73;margin-top:16px;line-height:1.6;">
        Sources: OSM (rivers) &middot; MS Building Footprints via GEE (structures)<br>
        Sentinel-2 B4/B3/B2/B8/B11 composite (spectral features)
        </div>
        """,
        unsafe_allow_html=True,
    )

metric_crs = scope_metric_crs(scope)
to_metric = transformer_to_metric(metric_crs)
x_m, y_m = to_metric.transform(lon, lat)
query_point = Point(x_m, y_m)
view_box = box(x_m - view_radius, y_m - view_radius, x_m + view_radius, y_m + view_radius)

buildings_metric = buildings.to_crs(metric_crs) if len(buildings) else buildings
in_view = buildings_metric[buildings_metric.geometry.distance(query_point) <= view_radius].copy()
if len(in_view):
    in_view["query_dist_m"] = in_view.geometry.distance(query_point)
    in_view["rule_flag"] = in_view["rule_flag"].astype(bool)
    in_view = add_rf_proba(in_view, predictions)
else:
    in_view["query_dist_m"] = pd.Series(dtype=float)
    in_view["rule_flag"] = pd.Series(dtype=bool)
    in_view["rf_proba"] = pd.Series(dtype=float)

in_view["scored"] = in_view["rf_proba"].notna()
in_view["model_flag"] = in_view["rf_proba"].fillna(0) >= threshold

if mode == MODE_RF:
    in_view["flagged"] = in_view["model_flag"]
elif mode == MODE_COMPARE:
    in_view["flagged"] = in_view["rule_flag"] | in_view["model_flag"]
else:
    in_view["flagged"] = in_view["rule_flag"]


def agreement_label(row) -> str:
    if row["rule_flag"] and row["model_flag"]:
        return "Both flag"
    if row["model_flag"]:
        return "Model only"
    if row["rule_flag"]:
        return "Rule only"
    return "Neither"


if len(in_view):
    in_view["agreement"] = in_view.apply(agreement_label, axis=1)

total_in_view = len(in_view)
flagged_in_view = int(in_view["flagged"].sum())
rule_in_view = int(in_view["rule_flag"].sum())
model_in_view = int(in_view["model_flag"].sum())
high_risk_view = int((in_view["dist_to_river_m"] <= 10).sum())
flagged_area = float(in_view.loc[in_view["flagged"], "total_area_m2"].sum())
pct_flagged = (flagged_in_view / total_in_view * 100) if total_in_view else 0.0

citywide_total = len(buildings)
citywide_rule = int(buildings["rule_flag"].sum())
if model_ready and not buildings.empty:
    b_proba = add_rf_proba(buildings.copy(), predictions)["rf_proba"]
    citywide_model = int((b_proba.fillna(0) >= threshold).sum())
else:
    citywide_model = 0

st.markdown(
    f"""
    <div class="rd-header">
      <div>
        <div class="rd-title">RIPARIAN ENCROACHMENT DETECTOR</div>
        <div class="rd-subtitle">{scope_label(scope).upper()}</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span class="mono" style="font-size:11.5px;color:#4dd0c4;">{lat:.5f}, {lon:.5f}</span>
        <span style="width:7px;height:7px;border-radius:50%;background:#4dd0c4;display:inline-block;box-shadow:0 0 6px #4dd0c4;"></span>
        <span class="mono" style="font-size:11px;color:#8a9aa3;text-transform:uppercase;">Live</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

map_tab, aoi_tab, model_tab, method_tab = st.tabs(
    ["Detection map", "AOI overview", "Model performance", "Method & data"]
)

with map_tab:
    map_col, side_col = st.columns([2.1, 1], gap="medium")

    with map_col:
        card_open("padding:0;overflow:hidden;")
        st.markdown(
            f"""
            <div style="padding:14px 18px;border-bottom:1px solid var(--border-soft);">
              <div class="rd-card-title">{mode} &middot; {buffer_m}m buffer</div>
              <div class="rd-sub">{basemap_label} &middot; {total_in_view} structures within
              {view_radius}m of the query point &middot; {scope_label(scope)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fmap = folium.Map(location=[lat, lon], zoom_start=17,
                          tiles=BASEMAPS[basemap_label], control_scale=True)

        buffer_gdf = load_geojson(scope, "buffer")
        if len(buffer_gdf):
            buf_view = buffer_gdf.to_crs(metric_crs).geometry.union_all().intersection(view_box)
            if not buf_view.is_empty and buf_view.area > 0:
                folium.GeoJson(
                    buf_view.__geo_interface__,
                    style_function=lambda _f: {
                        "fillColor": TEAL, "color": AMBER, "weight": 1.4,
                        "dashArray": "5 4", "fillOpacity": 0.12,
                    },
                    name=f"{buffer_m}m buffer",
                ).add_to(fmap)

        drawn = in_view.sort_values("query_dist_m").head(MAX_POLYGONS)
        for _, row in drawn.to_crs(WGS84).iterrows():
            proba_txt = f"{row['rf_proba']:.2f}" if pd.notna(row["rf_proba"]) else "n/a"
            area_tag = f"{row['area']} &middot; " if scope == ALL_AREAS_KEY else ""
            if mode == MODE_GEOMETRIC:
                color = GREEN if row["rule_flag"] else RISK_COLORS.get(row["risk_category"], GREY)
                tooltip = (f"{area_tag}{row['dist_to_river_m']:.1f}m to river &middot; {row['risk_category']} &middot; "
                           f"{'INSIDE BUFFER' if row['rule_flag'] else 'outside buffer'}")
            elif mode == MODE_RF:
                color = AMBER if row["model_flag"] else (GREY if row["scored"] else "#333c42")
                tooltip = (f"{area_tag}P(encroachment) = {proba_txt} &middot; "
                           f"{'FLAGGED' if row['model_flag'] else 'not flagged'} &middot; "
                           f"{row['dist_to_river_m']:.1f}m to river")
            else:
                color = AGREEMENT_COLORS[row["agreement"]]
                tooltip = (f"{area_tag}{row['agreement']} &middot; rule {row['dist_to_river_m']:.1f}m &middot; "
                           f"model P = {proba_txt}")
            weight = 1.6 if row["flagged"] else 0.8
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _f, c=color, w=weight: {
                    "fillColor": c, "color": c, "weight": w, "fillOpacity": 0.55,
                },
                tooltip=tooltip,
            ).add_to(fmap)

        folium.CircleMarker(
            location=[lat, lon], radius=7, color=TEAL, weight=2,
            fill=True, fill_color="#0f1417", fill_opacity=1, tooltip="Query point",
        ).add_to(fmap)

        st_folium(fmap, height=420, use_container_width=True, returned_objects=[])

        if mode == MODE_GEOMETRIC:
            legend = [(GREEN, "Inside buffer"), (RED, "High risk (&lt;10m)"),
                      (AMBER, "Medium risk"), (TEAL, "Low risk"), (GREY, "Safe zone")]
        elif mode == MODE_RF:
            legend = [(AMBER, f"Model flags (P &ge; {threshold:.2f})"), (GREY, "Model clears"),
                      ("#333c42", "Not scored")]
        else:
            legend = [(c, k) for k, c in AGREEMENT_COLORS.items()]
        legend_html = "".join(
            f'<div class="rd-legend-item"><span class="rd-swatch" style="background:{c};"></span>{t}</div>'
            for c, t in legend
        )
        truncated = (
            f' &middot; drawing the {MAX_POLYGONS} nearest of {total_in_view}'
            if total_in_view > MAX_POLYGONS else ""
        )
        st.markdown(
            f"""
            <div style="padding:10px 18px 14px;">
              <div class="rd-legend-row">{legend_html}
                <div class="rd-legend-item"><span style="width:14px;height:9px;border:1px dashed #f2b544;
                background:rgba(77,208,196,0.15);display:inline-block;"></span>Buffer zone</div>
              </div>
              <div class="rd-sub">Footprints coloured by the selected detection layer{truncated}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        card_close()

    with side_col:
        card_open()
        headline_label = {
            MODE_GEOMETRIC: "Structures Inside Buffer",
            MODE_RF: "Structures Flagged by Model",
            MODE_COMPARE: "Flagged by Rule or Model",
        }[mode]
        citywide_note = (
            f"scope rule: {citywide_rule:,}" if mode == MODE_GEOMETRIC
            else f"scope model: {citywide_model:,}"
        )
        st.markdown(f'<div class="rd-metric-label">{headline_label}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display:flex;align-items:baseline;gap:8px;margin-top:4px;">
              <div class="rd-metric-big">{flagged_in_view}</div>
              <div style="font-size:12px;color:var(--text-tertiary);">/ {total_in_view} in view</div>
            </div>
            <div style="font-size:11px;color:var(--text-secondary);margin-top:4px;">
              Within {view_radius}m of the query point &middot; {buffer_m}m riparian setback</div>
            <div class="rd-progress-track"><div class="rd-progress-fill" style="width:{pct_flagged:.1f}%;"></div></div>
            <div style="display:flex;justify-content:space-between;margin-top:6px;">
              <div class="mono" style="font-size:11px;color:var(--accent-teal);">{pct_flagged:.1f}% of view</div>
              <div class="mono" style="font-size:11px;color:var(--text-tertiary);">{citywide_note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        card_close()

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        card_open()
        st.markdown('<div class="rd-card-title">Nearest Structures</div>', unsafe_allow_html=True)
        st.markdown('<div class="rd-sub">Sorted by distance to query point</div>', unsafe_allow_html=True)
        cols = {"area": "Area", "query_dist_m": "To query (m)", "dist_to_river_m": "To river (m)",
                "risk_category": "Risk tier", "rule_flag": "In buffer"}
        if model_ready:
            cols["rf_proba"] = "P(encroach)"
            cols["model_flag"] = "Model flag"
        avail_cols = [c for c in cols if c in in_view.columns]
        nearest = in_view.sort_values("query_dist_m").head(10)[avail_cols].rename(columns=cols)
        if "To query (m)" in nearest:
            nearest["To query (m)"] = nearest["To query (m)"].round(1)
            nearest["To river (m)"] = nearest["To river (m)"].round(1)
        if model_ready and "P(encroach)" in nearest:
            nearest["P(encroach)"] = nearest["P(encroach)"].round(3)
        st.dataframe(nearest, hide_index=True, use_container_width=True, height=240)
        card_close()

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    if model_ready and mode != MODE_GEOMETRIC:
        agree = int((in_view["rule_flag"] == in_view["model_flag"]).sum())
        cells = [
            (m1, "Structures In View", f"{total_in_view}", "", None),
            (m2, "Rule vs Model", f"{rule_in_view} / {model_in_view}", "flagged by each", None),
            (m3, "Rule &amp; Model Agree",
             f"{agree / total_in_view * 100:.0f}%" if total_in_view else "—",
             f"{agree} of {total_in_view} structures", None),
            (m4, "Flagged Area", f"{flagged_area:,.0f}", "m² footprint", None),
        ]
    else:
        cells = [
            (m1, "Structures In View", f"{total_in_view}", "", None),
            (m2, "Inside Buffer", f"{flagged_in_view}", f"{pct_flagged:.1f}% of view", None),
            (m3, "High Risk (&lt;10m)", f"{high_risk_view}", "", RED if high_risk_view else None),
            (m4, "Encroached Area", f"{flagged_area:,.0f}", "m² footprint, in-buffer", None),
        ]
    for col, label, value, sub, color in cells:
        with col:
            metric_card(label, value, sub, color)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    card_open()
    st.markdown(
        f'<div class="rd-card-title">Structures Near the River, by Risk Tier &middot; {scope_label(scope)}</div>',
        unsafe_allow_html=True,
    )
    risk_counts_all = scope_risk_counts(scope).reindex(RISK_ORDER).fillna(0)
    safe_count = int(risk_counts_all["Safe Zone (>30m)"])
    at_risk = risk_counts_all.reindex(["Low Risk (20m-30m)", "Medium Risk (10m-20m)", "High Risk (<10m)"])
    safe_share = f"{safe_count / citywide_total * 100:.0f}%" if citywide_total else "—"
    st.markdown(
        f'<div class="rd-sub">{int(at_risk.sum()):,} structures sit within 30m of a river ' 
        f'&middot; {safe_count:,} more ({safe_share} of {citywide_total:,} ' 
        f"total) fall beyond 30m and aren't shown on this scale</div>",
        unsafe_allow_html=True,
    )
    fig = go.Figure(
        go.Bar(
            x=at_risk.index, y=at_risk.values,
            marker_color=[RISK_COLORS[c] for c in at_risk.index],
            text=[f"{int(v):,}" for v in at_risk.values], textposition="outside",
            hovertemplate="%{x}<br>%{y:,} structures<extra></extra>",
        )
    )
    fig.update_layout(**plotly_layout(height=280, showlegend=False,
                                      yaxis=dict(gridcolor="#232b30", title=None),
                                      xaxis=dict(title=None)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    card_close()

with aoi_tab:
    scope_gdf = load_geojson(scope, "buffer") if len(load_geojson(scope, "buffer")) else buildings
    if len(scope_gdf):
        min_lon, min_lat, max_lon, max_lat = scope_gdf.total_bounds
    else:
        min_lon, min_lat, max_lon, max_lat = lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02
    pad_lon = (max_lon - min_lon) * 0.08
    pad_lat = (max_lat - min_lat) * 0.08
    zoom_to_fit = fit_zoom(max_lon - min_lon)

    centroids = buildings.geometry.centroid
    pts = pd.DataFrame({
        "lat": centroids.y.to_numpy(),
        "lon": centroids.x.to_numpy(),
        "dist_to_river_m": buildings["dist_to_river_m"].to_numpy(),
        "risk_category": buildings["risk_category"].to_numpy(),
        "rule_flag": buildings["rule_flag"].to_numpy(),
        "area": buildings["area"].to_numpy(),
        "id": buildings["id"].to_numpy(),
    })
    if model_ready and not buildings.empty:
        pts["rf_proba"] = add_rf_proba(buildings.copy(), predictions)["rf_proba"].to_numpy()
    else:
        pts["rf_proba"] = np.nan

    rule_flag_all = pts["rule_flag"]
    model_flag_all = pts["rf_proba"].fillna(0) >= threshold if model_ready else pd.Series(False, index=pts.index)

    if mode == MODE_GEOMETRIC:
        layers = [(f"Inside {buffer_m}m buffer", GREEN, pts[rule_flag_all])]
    elif mode == MODE_RF:
        layers = [(f"Model flags (P ≥ {threshold:.2f})", AMBER, pts[model_flag_all])]
    else:
        layers = [
            ("Both flag", AGREEMENT_COLORS["Both flag"], pts[rule_flag_all & model_flag_all]),
            ("Model only", AGREEMENT_COLORS["Model only"], pts[model_flag_all & ~rule_flag_all]),
            ("Rule only", AGREEMENT_COLORS["Rule only"], pts[rule_flag_all & ~model_flag_all]),
        ]
    plotted = sum(len(subset) for _, _, subset in layers)

    card_open("padding:0;overflow:hidden;")
    st.markdown(
        f"""
        <div style="padding:14px 18px;border-bottom:1px solid var(--border-soft);">
          <div class="rd-card-title">{scope_label(scope)} &middot; {mode}</div>
          <div class="rd-sub">{plotted:,} flagged structures across the scope &middot;
          {buffer_m}m buffer ribbon &middot; clustered centroids, not footprints</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview = folium.Map(
        tiles=BASEMAPS[basemap_label],
        control_scale=True,
        prefer_canvas=True,
        max_bounds=True,
        min_lat=min_lat - pad_lat, max_lat=max_lat + pad_lat,
        min_lon=min_lon - pad_lon, max_lon=max_lon + pad_lon,
        min_zoom=max(1, zoom_to_fit - 1),
    )
    overview.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    buffer_gdf = load_geojson(scope, "buffer")
    if len(buffer_gdf):
        folium.GeoJson(
            json.loads(buffer_gdf.to_json()),
            style_function=lambda _f: {
                "fillColor": TEAL, "color": AMBER, "weight": 1.1,
                "dashArray": "5 4", "fillOpacity": 0.14,
            },
            name=f"{buffer_m}m buffer",
        ).add_to(overview)

    rivers_gdf = load_geojson(scope, "rivers")
    if len(rivers_gdf):
        folium.GeoJson(
            json.loads(rivers_gdf.to_json()),
            style_function=lambda _f: {"color": TEAL, "weight": 2},
            name="River centerline",
        ).add_to(overview)

    for name, color, subset in layers:
        if subset.empty:
            continue
        rgba = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.30)"
        data = [
            [round(row.lat, 6), round(row.lon, 6),
             f"{row.area} &middot; {row.dist_to_river_m:.0f}m to river"
             + (f" &middot; P {row.rf_proba:.2f}" if pd.notna(row.rf_proba) else "")]
            for row in subset.itertuples()
        ]
        marker_js = (
            "function(row){var m=L.circleMarker(new L.LatLng(row[0],row[1]),"
            f"{{radius:4,color:'{color}',fillColor:'{color}',fillOpacity:0.85,weight:1}});"
            "m.bindTooltip(row[2]);return m;}"
        )
        cluster_js = (
            "function(cluster){return L.divIcon({html:'<div style=\"background:"
            f"{rgba};border:1.5px solid {color};color:#eef2f3;width:34px;height:34px;"
            "border-radius:50%;display:flex;align-items:center;justify-content:center;"
            "font-family:IBM Plex Mono,monospace;font-size:11px;\">'+cluster.getChildCount()+"
            "'</div>',className:'',iconSize:L.point(34,34)});}"
        )
        FastMarkerCluster(
            data=data, callback=marker_js, icon_create_function=cluster_js,
            name=f"{name} ({len(subset):,})",
            options={"maxClusterRadius": 45, "showCoverageOnHover": False,
                     "spiderfyOnMaxZoom": True, "disableClusteringAtZoom": 18},
        ).add_to(overview)

    folium.LayerControl(collapsed=False).add_to(overview)
    st_folium(overview, height=520, use_container_width=True, returned_objects=[],
              key=f"aoi_overview_{scope}")

    legend_html = "".join(
        f'<div class="rd-legend-item"><span class="rd-swatch" style="background:{c};'
        f'border-radius:50%;"></span>{n} &middot; {len(s):,}</div>'
        for n, c, s in layers
    )
    st.markdown(
        f"""
        <div style="padding:10px 18px 14px;">
          <div class="rd-legend-row">{legend_html}
            <div class="rd-legend-item"><span style="width:14px;height:0;border-top:2px solid {TEAL};
            display:inline-block;"></span>River centerline</div>
            <div class="rd-legend-item"><span style="width:14px;height:9px;border:1px dashed {AMBER};
            background:rgba(77,208,196,0.15);display:inline-block;"></span>Buffer ribbon</div>
          </div>
          <div class="rd-sub">Only flagged structures are plotted — all {citywide_total:,} structures
          would not render at this scale. Switch the sidebar's detection layer or move the threshold
          to change what is flagged; the Detection map tab still draws real footprints around your
          query point.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_close()

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4, gap="medium")
    with a1:
        metric_card("Structures In Scope", f"{citywide_total:,}",
                    f"{len(area_keys(scope))} study area{'s' if len(area_keys(scope)) > 1 else ''}")
    with a2:
        metric_card(f"Within {buffer_m}m (rule)", f"{citywide_rule:,}", "geometric distance rule")
    with a3:
        if model_ready:
            metric_card(f"Model Flags (P≥{threshold:.2f})", f"{citywide_model:,}",
                        "Random Forest", AMBER if citywide_model > 3 * citywide_rule else None)
        else:
            metric_card("High Risk (&lt;10m)", f"{int((buildings['dist_to_river_m'] <= 10).sum()):,}",
                        "scope-wide", RED)
    with a4:
        metric_card("Avg Dist to River", f"{buildings['dist_to_river_m'].mean():,.0f} m" if citywide_total else "—",
                    "mean across structures in scope")

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    card_open()
    st.markdown('<div class="rd-card-title">The Three Training Areas, Compared</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="rd-sub">Structures and rule-flagged encroachments per area &middot; '
                'the baseline model saw all three</div>', unsafe_allow_html=True)
    area_rows = []
    for a in STUDY_AREAS:
        gdf = load_area_buildings(a)
        area_rows.append({
            "Area": STUDY_AREAS[a]["label"],
            "Structures": len(gdf),
            "Inside buffer": int(gdf["rule_flag"].sum()),
            "High risk (<10m)": int((gdf["dist_to_river_m"] <= 10).sum()),
        })
    area_df = pd.DataFrame(area_rows)
    cmp_fig = go.Figure([
        go.Bar(name="Structures", x=area_df["Area"], y=area_df["Structures"],
               marker_color=GREY, hovertemplate="%{x}<br>%{y:,} structures<extra></extra>"),
        go.Bar(name="Inside buffer", x=area_df["Area"], y=area_df["Inside buffer"],
               marker_color=AMBER, hovertemplate="%{x}<br>%{y:,} inside buffer<extra></extra>"),
        go.Bar(name="High risk (<10m)", x=area_df["Area"], y=area_df["High risk (<10m)"],
               marker_color=RED, hovertemplate="%{x}<br>%{y:,} high risk<extra></extra>"),
    ])
    cmp_fig.update_layout(**plotly_layout(
        height=300, barmode="group",
        yaxis=dict(gridcolor="#232b30", title=None),
        xaxis=dict(title=None),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    ))
    st.plotly_chart(cmp_fig, use_container_width=True, config={"displayModeBar": False})
    card_close()

with model_tab:
    if not model_ready:
        st.markdown(
            f"""
            <div class="rd-note">
              <b>No trained model artifacts found.</b><br>
              The dashboard looks for <span class="mono">data/processed/rf_baseline.joblib</span>
              and the per-area feature tables
              (<span class="mono">kasarani_feature_table.csv</span>,
              <span class="mono">gatharani_feature_table.csv</span>,
              <span class="mono">motoine_feature_table.csv</span>).<br><br>
              Regenerate them with <span class="mono">02_modelling.ipynb</span>, then reload
              this page.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        live = confusion_at(predictions, threshold)
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = float(roc_auc_score(predictions["encroachment"], predictions["rf_proba"]))
        except Exception:
            roc_auc = float("nan")

        card_open()
        st.markdown('<div class="rd-card-title">Model Card &middot; trained on three areas</div>',
                    unsafe_allow_html=True)
        params = model.get_params() if hasattr(model, "get_params") else {}
        balance = predictions["encroachment"].value_counts().to_dict()
        kv = [
            ("Estimator", f"RandomForestClassifier ({params.get('n_estimators', '?')} trees, "
                          f"max_depth {params.get('max_depth', '?')}, "
                          f"class_weight {params.get('class_weight', '?')})"),
            ("Training areas", " · ".join(STUDY_AREAS[a]["label"] for a in STUDY_AREAS)),
            ("Features", ", ".join(model_feature_cols(model))),
            ("Label", "encroachment = structure intersects the riparian buffer"),
            ("Rows scored", f"{len(predictions):,} across the three feature tables"),
            ("Class balance", " · ".join(f"{k}: {v:,}" for k, v in sorted(balance.items()))),
        ]
        if isinstance(summary, dict) and summary.get("trained_at"):
            kv.append(("Trained at", str(summary["trained_at"])))
        st.markdown(
            "".join(f'<div class="rd-kv"><span>{k}</span><span>{v}</span></div>' for k, v in kv),
            unsafe_allow_html=True,
        )
        card_close()

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="rd-eyebrow">All three areas &middot; threshold {threshold:.2f}</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
        with c1:
            metric_card("Accuracy", f"{live['accuracy']:.3f}", f"{live['n']:,} rows")
        with c2:
            metric_card("ROC AUC", f"{roc_auc:.3f}" if roc_auc == roc_auc else "—",
                        "threshold-independent")
        with c3:
            metric_card("Precision", f"{live['precision']:.3f}", "of flagged, truly inside",
                        AMBER if live["precision"] < 0.5 else None)
        with c4:
            metric_card("Recall", f"{live['recall']:.3f}", "of truly inside, caught",
                        AMBER if live["recall"] < 0.5 else None)
        with c5:
            metric_card("F1", f"{live['f1']:.3f}", "encroachment class",
                        AMBER if live["f1"] < 0.5 else None)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        cm_col, curve_col = st.columns(2, gap="medium")

        with cm_col:
            card_open()
            st.markdown('<div class="rd-card-title">Confusion Matrix</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="rd-sub">Three areas combined at threshold {threshold:.2f} &middot; ' 
                "row-normalised shading, raw counts labelled</div>",
                unsafe_allow_html=True,
            )
            counts = np.array([[live["tn"], live["fp"]], [live["fn"], live["tp"]]])
            row_totals = counts.sum(axis=1, keepdims=True)
            shares = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=float),
                               where=row_totals > 0)
            cm_fig = go.Figure(
                go.Heatmap(
                    z=shares, x=["Predicted clear", "Predicted encroaching"],
                    y=["Actually clear", "Actually encroaching"],
                    colorscale=[[0, "#12191d"], [0.5, "#2c6f6a"], [1, TEAL]],
                    zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
                    text=[[f"{c:,}<br>{s:.0%}" for c, s in zip(rc, rs)]
                          for rc, rs in zip(counts, shares)],
                    texttemplate="%{text}",
                    textfont=dict(family="IBM Plex Mono, monospace", size=13, color="#eef2f3"),
                    hovertemplate="%{y} → %{x}<br>%{z:.1%} of row<extra></extra>",
                )
            )
            cm_fig.update_layout(**plotly_layout(height=280, xaxis=dict(side="top"),
                                                 yaxis=dict(autorange="reversed")))
            st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})
            card_close()

        with curve_col:
            card_open()
            st.markdown('<div class="rd-card-title">Precision &amp; Recall vs Threshold</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="rd-sub">Three areas combined &middot; the dashed line is ' 
                        "your current threshold</div>", unsafe_allow_html=True)
            sweep = threshold_sweep()
            curve = go.Figure()
            for col, color, name in (("precision", TEAL, "Precision"), ("recall", AMBER, "Recall")):
                curve.add_trace(go.Scatter(
                    x=sweep["threshold"], y=sweep[col], name=name, mode="lines",
                    line=dict(color=color, width=2),
                    hovertemplate=name + ": %{y:.3f} at %{x:.2f}<extra></extra>",
                ))
            curve.add_vline(x=threshold, line=dict(color="#5c6b73", width=1, dash="dash"))
            curve.update_layout(**plotly_layout(
                height=280,
                xaxis=dict(title="Decision threshold", gridcolor="#232b30"),
                yaxis=dict(title=None, gridcolor="#232b30", range=[0, 1]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                hovermode="x unified",
            ))
            st.plotly_chart(curve, use_container_width=True, config={"displayModeBar": False})
            card_close()

        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            card_open()
            st.markdown('<div class="rd-card-title">Feature Importance</div>', unsafe_allow_html=True)
            st.markdown('<div class="rd-sub">Mean decrease in impurity across the forest</div>',
                        unsafe_allow_html=True)
            imp = pd.Series(importances, index=model_feature_cols(model)).sort_values()
            imp_fig = go.Figure(go.Bar(
                x=imp.values, y=imp.index, orientation="h",
                marker_color=TEAL, text=[f"{v:.3f}" for v in imp.values], textposition="outside",
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))
            imp_fig.update_layout(**plotly_layout(
                height=300, showlegend=False,
                xaxis=dict(gridcolor="#232b30", title=None, range=[0, imp.max() * 1.18]),
                yaxis=dict(title=None),
            ))
            st.plotly_chart(imp_fig, use_container_width=True, config={"displayModeBar": False})
            card_close()

        pos = int((predictions["encroachment"] == 1).sum())
        n_total = len(predictions)
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="rd-note">
              <b>How to read these numbers.</b> Encroaching structures are
              {pos:,} of {n_total:,}
              ({pos / n_total * 100:.1f}%), so overall
              accuracy is dominated by the majority class and says almost nothing — a model that
              flagged nothing at all would score about
              {(1 - pos / n_total) * 100:.1f}%.
              The number that matters is the encroachment row: precision
              {live['precision']:.3f}, recall {live['recall']:.3f}, F1 {live['f1']:.3f} at the 0.5
              default.<br><br>
              Two caveats specific to this baseline. First, it is trained on the
              <b>combined</b> feature tables of Kasarani, Gatharani and Motoine — one model, three
              areas — so the numbers above are resubstitution figures on its own training rows, not
              a held-out test. Second, the features are spectral means and indices
              (NDVI/NDBI/MNDWI): nothing in a roof's colour says how far it sits from a river, so
              the geometric distance rule remains the authoritative classifier for enforcement.
              The Random Forest is the Phase 1 baseline it was scoped as; adding spatial features
              (distance-to-river, neighbourhood density) is the path to a model that earns its
              place in the decision.
            </div>
            """,
            unsafe_allow_html=True,
        )

with method_tab:
    left, right = st.columns([1.3, 1], gap="medium")

    with left:
        card_open()
        st.markdown('<div class="rd-card-title">Pipeline &middot; three study areas</div>',
                    unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:12.5px;color:var(--text-secondary);line-height:1.75;margin-top:8px;">
            <b style="color:var(--text-primary);">1 &middot; Preprocessing</b>
            (<span class="mono">notebooks/01_preprocessing.ipynb</span>) — run once per study
            area (Kasarani, Gatharani, Motoine): pulls the AOI boundary and river network from
            OSM, clips Microsoft Building Footprints via Google Earth Engine, builds a
            cloud-filtered Sentinel-2 composite (B4/B3/B2/B8/B11), and computes per-structure
            zonal statistics — band means plus NDVI, NDBI and MNDWI. Outputs land in
            <span class="mono">data/processed/{area}_*.csv|geojson</span>.<br><br>
            <b style="color:var(--text-primary);">2 &middot; Modelling</b>
            (<span class="mono">notebooks/02_modelling.ipynb</span>) — stacks the three areas'
            feature tables and trains the shared <span class="mono">rf_baseline</span>
            Random Forest against the geometric encroachment label (structure intersects the
            riparian buffer). The same run produces
            <span class="mono">pipeline_summary.json</span>.<br><br>
            <b style="color:var(--text-primary);">3 &middot; Fusion &amp; report</b>
            (<span class="mono">notebooks/03_fusion_and_report.ipynb</span>) — combines the
            rule-based and model-based detections into the reporting layer.<br><br>
            <b style="color:var(--text-primary);">This dashboard</b>
            (<span class="mono">app.py</span>) — reads the processed artifacts directly; the
            model scores each area's feature table live, so retraining is picked up on reload.
            </div>
            """,
            unsafe_allow_html=True,
        )
        card_close()

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="rd-note">
              <b>One model, three areas.</b> The sidebar's study-area selector changes what the
              maps and counters show, not what the model was trained on —
              <span class="mono">rf_baseline.joblib</span> always saw Kasarani, Gatharani and
              Motoine together. Per-area distances are computed in each area's own UTM zone,
              exactly as the preprocessing module does it.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        card_open()
        st.markdown('<div class="rd-card-title">Artifacts</div>', unsafe_allow_html=True)
        st.markdown('<div class="rd-sub">Expected under <span class="mono">data/processed/</span> '
                    '— one set per study area</div>', unsafe_allow_html=True)
        rows = []
        for area in STUDY_AREAS:
            for key, what in [
                ("buildings", "structures + geometry"),
                ("encroaching", "rule-flagged encroachers"),
                ("features", "spectral feature table"),
                ("rivers", "river network"),
                ("buffer", "riparian buffer zone"),
            ]:
                path = area_paths(area)[key]
                rows.append({
                    "Area": STUDY_AREAS[area]["label"],
                    "File": path.name,
                    "Holds": what,
                    "Status": "present" if path.exists() else "missing",
                })
        for path, what in [
            (MODEL_PATH, "trained Random Forest (all 3 areas)"),
            (SUMMARY_PATH, "pipeline summary"),
        ]:
            rows.append({"Area": "—", "File": path.name, "Holds": what,
                         "Status": "present" if path.exists() else "missing"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=330)
        card_close()

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        card_open()
        st.markdown(f'<div class="rd-card-title">Scope Totals &middot; {scope_label(scope)}</div>',
                    unsafe_allow_html=True)
        totals = [
            ("Structures", f"{citywide_total:,}"),
            (f"Within {buffer_m}m (rule)", f"{citywide_rule:,}"),
        ]
        if model_ready:
            totals.append((f"Flagged by model (P≥{threshold:.2f})", f"{citywide_model:,}"))
            p_dict = get_proba_dict(predictions)
            scored_count = sum(1 for a, i in zip(buildings["area"], buildings["id"]) if (a, str(i)) in p_dict)
            totals.append(("Scored by model", f"{scored_count:,}"))
        st.markdown(
            "".join(f'<div class="rd-kv"><span>{k}</span><span>{v}</span></div>' for k, v in totals),
            unsafe_allow_html=True,
        )
        card_close()