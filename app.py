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
from sklearn.metrics import classification_report, confusion_matrix
from streamlit_folium import st_folium

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = DATA_DIR / "rf_baseline.joblib"
SUMMARY_PATH = DATA_DIR / "pipeline_summary.json"

METRIC_CRS = "EPSG:32737"
WGS84 = "EPSG:4326"

REGIONS = ["Kasarani", "Gatharaini", "Motoine"]
# Matches 01_preprocessing.ipynb / 02_modelling.ipynb exactly — the fixed-radius
# circle each region's AOI is built from.
REGION_CENTERS = {
    "Kasarani": (36.8969, -1.2296),
    "Gatharaini": (36.95952127354356, -1.2252700532171976),
    "Motoine": (36.74237847877641, -1.3113205815273903),
}
CASE_STUDY_RADIUS_KM = 3

# Named localities within Kasarani only, for map navigation — no verified
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
AREA_COMPARE_RADIUS_M = 600
FEATURE_COLS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI"]
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_MATERIAL_OVERLAP_THRESHOLD = 0.05

TEAL = "#4dd0c4"
AMBER = "#f2b544"
RED = "#e2604f"
GREEN = "#6ad18a"
GREY = "#4a5860"

BASEMAPS = {
    "OpenStreetMap": "OpenStreetMap",
    "Satellite (Esri)": "Esri.WorldImagery",
    "Dark (CartoDB — requires API key)": "CartoDB dark_matter",
}

st.set_page_config(
    page_title="Riparian Encroachment Detector",
    page_icon="\U0001f6f0️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------- file paths


def buildings_csv(region: str) -> Path:
    return DATA_DIR / f"{region.lower()}_buildings.csv"


def encroaching_csv(region: str) -> Path:
    return DATA_DIR / f"{region.lower()}_encroaching_buildings.csv"


def riparian_buffer_geojson(region: str) -> Path:
    return DATA_DIR / f"{region.lower()}_riparian_buffer.geojson"


def rivers_geojson(region: str) -> Path:
    return DATA_DIR / f"{region.lower()}_rivers.geojson"


def feature_table_csv(region: str) -> Path:
    return DATA_DIR / f"{region.lower()}_feature_table.csv"


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
def load_river_union(region: str):
    path = rivers_geojson(region)
    if not path.exists():
        return None
    rivers = gpd.read_file(path)
    if rivers.crs is None:
        rivers = rivers.set_crs(METRIC_CRS)  # saved in EPSG:32737 by 01_preprocessing.ipynb
    else:
        rivers = rivers.to_crs(METRIC_CRS)
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


def region_aoi_wgs84(region: str) -> gpd.GeoDataFrame:
    """Reconstructs each region's fixed-radius circle exactly as
    01_preprocessing.ipynb / 02_modelling.ipynb define it — no separate AOI
    file is saved by the pipeline, so this is derived rather than read."""
    lon, lat = REGION_CENTERS[region]
    center_metric = gpd.GeoSeries([Point(lon, lat)], crs=WGS84).to_crs(METRIC_CRS)
    circle_metric = center_metric.buffer(CASE_STUDY_RADIUS_KM * 1000)
    return gpd.GeoDataFrame(geometry=circle_metric, crs=METRIC_CRS).to_crs(WGS84)


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
    sidebar sliders actually change what counts as encroaching."""
    candidates = table[
        (table["confidence"] >= confidence_threshold)
        & (table["rf_builtup_prob"] >= material_threshold)
    ]
    return candidates[candidates["distance_to_river_m"] <= flag_distance_m]


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


def fit_zoom(span_lon: float, px_width: int = 1000) -> int:
    if span_lon <= 0:
        return 12
    return max(1, int(math.floor(math.log2(360 * px_width / (256 * span_lon)))))


# ---------------------------------------------------------------- model evaluation


@st.cache_data(show_spinner=False)
def evaluate_model_on_region(region: str) -> dict | None:
    """Live evaluation against ground truth — 02_modelling.ipynb prints a
    classification report but never saves one, so this recomputes it from
    the model and the region's own labelled feature table."""
    model = load_model()
    table = load_feature_table(region)
    if model is None or table is None:
        return None
    X = table[FEATURE_COLS]
    y = table["builtup"]
    pred = model.predict(X)
    report = classification_report(y, pred, target_names=["not built-up", "built-up"],
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred).tolist()
    return {"report": report, "confusion_matrix": cm, "n": len(table)}


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
        .block-container { padding-top: 0.8rem; }
        .rd-eyebrow{ text-transform:uppercase; letter-spacing:.08em; font-size:11px; color:var(--text-tertiary); font-weight:600; margin-bottom:6px; }
        .rd-card{ background:var(--bg-panel); border:1px solid var(--border); border-radius:12px; padding:12px 16px; height:100%; }
        .rd-card-title{ font-size:12px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:var(--text-primary); }
        .rd-sub{ font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--text-tertiary); margin-top:2px; line-height:1.5; }
        .rd-metric-big{ font-family:'IBM Plex Mono',monospace; font-size:38px; font-weight:700; color:var(--accent-amber); line-height:1; }
        .rd-metric-mid{ font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:600; color:var(--text-primary); }
        .rd-metric-label{ text-transform:uppercase; letter-spacing:.06em; font-size:10.5px; color:var(--text-tertiary); }
        .rd-badge{ font-family:'IBM Plex Mono',monospace; font-size:11px; padding:6px 12px; border-radius:6px; border:1px solid var(--border); color:var(--text-tertiary); display:inline-block; }
        .rd-badge-ok{ background:rgba(106,209,138,0.12); border:1px solid var(--accent-green); color:var(--accent-green); }
        .rd-badge-warn{ background:rgba(242,181,68,0.10); border:1px solid var(--accent-amber); color:var(--accent-amber); }
        .rd-legend-row{ display:flex; gap:14px; flex-wrap:wrap; margin-top:10px; }
        .rd-legend-item{ display:flex; align-items:center; gap:6px; font-size:11px; color:var(--text-secondary); }
        .rd-swatch{ width:10px; height:10px; border-radius:50%; display:inline-block; }
        .rd-header{ display:flex; align-items:center; justify-content:space-between; padding:8px 4px 10px; border-bottom:1px solid var(--border); margin-bottom:10px; }
        .rd-title{ font-size:18px; font-weight:700; letter-spacing:.02em; color:var(--text-primary); }
        .rd-subtitle{ font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--text-tertiary); letter-spacing:.03em; }
        .rd-note{ border-left:2px solid var(--accent-amber); background:rgba(242,181,68,0.06); padding:12px 14px; border-radius:0 8px 8px 0; font-size:12.5px; color:var(--text-secondary); line-height:1.65; }
        .rd-kv{ display:flex; justify-content:space-between; gap:12px; padding:5px 0; border-bottom:1px solid var(--border-soft); font-size:12px; }
        .rd-kv span:first-child{ color:var(--text-tertiary); }
        .rd-kv span:last-child{ color:var(--text-primary); font-family:'IBM Plex Mono',monospace; }
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
        st.markdown(f'<div style="font-size:11px;color:var(--text-tertiary);margin-top:2px;">{sub}</div>',
                    unsafe_allow_html=True)
    card_close()


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


# ------------------------------------------------------------------- app state

inject_css()

summary = load_pipeline_summary()
model = load_model()
model_ready = model is not None

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
            <div class="mono" style="font-size:10px;color:#5c6b73;">NAIROBI RIVER BASIN</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rd-eyebrow">Region</div>', unsafe_allow_html=True)
    region = st.selectbox("Region", REGIONS, label_visibility="collapsed")
    if region == "Kasarani":
        st.markdown('<span class="rd-badge rd-badge-ok">Calibrated vs. Pamoja Trust field count (~700)</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="rd-badge rd-badge-warn">Untested extrapolation — no ground truth</span>',
                    unsafe_allow_html=True)

    st.markdown('<div class="rd-eyebrow" style="margin-top:12px;">Jump to Area</div>', unsafe_allow_html=True)
    if region == "Kasarani":
        area_choice = st.selectbox("Locality", list(KASARANI_AREAS.keys()), label_visibility="collapsed")
        st.session_state["map_center"] = KASARANI_AREAS[area_choice]
    else:
        st.session_state["map_center"] = (REGION_CENTERS[region][1], REGION_CENTERS[region][0])
        st.caption("No named localities catalogued for this region yet — search below, "
                   "or use the map's own pan/zoom.")

    query = st.text_input("Or search a place", placeholder="Search OpenStreetMap...")
    if query:
        result = geocode_place_osm(query, region)
        if result is None:
            st.warning("No match found — try a more specific name.")
        else:
            lat, lon, display_name = result
            st.session_state["map_center"] = (lat, lon)
            st.caption(f"Found: {display_name}")

    # Confidence and material-overlap thresholds are applied at their pipeline
    # defaults rather than exposed as sliders — the values themselves are
    # still visible on hover over each structure on the map.
    confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    material_threshold = DEFAULT_MATERIAL_OVERLAP_THRESHOLD

    st.markdown('<div class="rd-eyebrow" style="margin-top:12px;">Flagging Distance</div>', unsafe_allow_html=True)
    calibrated_distance_default = int(summary["calibrated_distance_m"]) if summary else 16
    flag_distance_m = st.slider("Flag within (m) of river", 5, 60, calibrated_distance_default, 1)
    st.caption(f"{calibrated_distance_default}m is calibrated against the Pamoja Trust field survey.")

    st.markdown('<div class="rd-eyebrow" style="margin-top:12px;">Basemap</div>', unsafe_allow_html=True)
    basemap_label = st.selectbox("Basemap", list(BASEMAPS.keys()), label_visibility="collapsed")

    st.markdown('<div class="rd-eyebrow" style="margin-top:12px;">Model</div>', unsafe_allow_html=True)
    if model_ready:
        st.markdown(
            f"""
            <div class="mono" style="font-size:10.5px;color:var(--text-tertiary);line-height:1.6;">
            RandomForestClassifier &middot; {model.n_estimators} trees<br>
            Trained once on all three regions combined<br>
            Features: {", ".join(FEATURE_COLS)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("rf_baseline.joblib not found — run 02_modelling.ipynb.")

# --------------------------------------------------------------------- compute

region_table = build_region_table(region)
data_ready = region_table is not None

if data_ready:
    encroaching = apply_filters(region_table, confidence_threshold, material_threshold, flag_distance_m)
    passed_filter = region_table[
        (region_table["confidence"] >= confidence_threshold)
        & (region_table["rf_builtup_prob"] >= material_threshold)
    ]

st.markdown(
    f"""
    <div class="rd-header">
      <div>
        <div class="rd-title">RIPARIAN ENCROACHMENT DETECTOR</div>
        <div class="rd-subtitle">{region.upper()} &middot; NAIROBI RIVER BASIN</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="width:7px;height:7px;border-radius:50%;background:#4dd0c4;display:inline-block;box-shadow:0 0 6px #4dd0c4;"></span>
        <span class="mono" style="font-size:11px;color:#8a9aa3;text-transform:uppercase;">Live</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not data_ready:
    st.markdown(
        f"""
        <div class="rd-note">
          <b>No data found for {region}.</b> Looking for
          <span class="mono">{buildings_csv(region).relative_to(ROOT)}</span> and
          <span class="mono">{rivers_geojson(region).relative_to(ROOT)}</span>.
          Run <span class="mono">01_preprocessing.ipynb</span> then
          <span class="mono">02_modelling.ipynb</span> to generate them.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

map_tab, compare_tab, model_tab, method_tab = st.tabs(
    ["Detection map", "Compare regions", "Model performance", "Method & data"]
)

# ------------------------------------------------------------- tab 1: the map

with map_tab:
    map_col, side_col = st.columns([3.2, 1], gap="medium")

    with map_col:
        card_open("padding:0;overflow:hidden;")
        st.markdown(
            f"""
            <div style="padding:14px 18px;border-bottom:1px solid var(--border-soft);">
              <div class="rd-card-title">{region} &middot; {flag_distance_m}m flagging distance</div>
              <div class="rd-sub">{len(encroaching):,} encroaching of {len(region_table):,} screened</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        center_lat, center_lon = st.session_state.get(
            "map_center", (REGION_CENTERS[region][1], REGION_CENTERS[region][0])
        )
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                          tiles=BASEMAPS[basemap_label], control_scale=True)

        aoi = region_aoi_wgs84(region)
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

        rivers_path = rivers_geojson(region)
        if rivers_path.exists():
            rivers_display = gpd.read_file(rivers_path)
            if rivers_display.crs is None:
                rivers_display = rivers_display.set_crs(METRIC_CRS)
            rivers_display = rivers_display.to_crs(WGS84)
            folium.GeoJson(
                json.loads(rivers_display.to_json()),
                style_function=lambda _f: {"color": TEAL, "weight": 2},
                name="River centerline",
            ).add_to(fmap)

        not_flagged = region_table.loc[~region_table.index.isin(encroaching.index)
                                        & region_table.index.isin(passed_filter.index)]
        for subset, color, label in [
            (encroaching, RED, f"Encroaching ({len(encroaching):,})"),
            (not_flagged, GREY, f"Passed filter, outside {flag_distance_m}m ({len(not_flagged):,})"),
        ]:
            if subset.empty:
                continue
            data = [[round(r.lat, 6), round(r.lon, 6),
                     f"{r.distance_to_river_m:.0f}m to river &middot; conf {r.confidence:.2f} &middot; "
                     f"P(built-up) {r.rf_builtup_prob:.2f}"]
                    for r in subset.itertuples()]
            marker_js = (
                "function(row){var m=L.circleMarker(new L.LatLng(row[0],row[1]),"
                f"{{radius:4,color:'{color}',fillColor:'{color}',fillOpacity:0.85,weight:1}});"
                "m.bindTooltip(row[2]);return m;}"
            )
            FastMarkerCluster(
                data=data, callback=marker_js, name=label,
                options={"maxClusterRadius": 45, "showCoverageOnHover": False, "disableClusteringAtZoom": 17},
            ).add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        st_folium(fmap, height=560, use_container_width=True, returned_objects=[], key="main_map")

        st.markdown(
            f"""
            <div style="padding:10px 18px 14px;">
              <div class="rd-legend-row">
                <div class="rd-legend-item"><span class="rd-swatch" style="background:{RED};"></span>Encroaching</div>
                <div class="rd-legend-item"><span class="rd-swatch" style="background:{GREY};"></span>Passed filter, not flagged</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        card_close()

    with side_col:
        card_open()
        st.markdown('<div class="rd-metric-label">Encroaching Structures</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rd-metric-big">{len(encroaching):,}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:11px;color:var(--text-secondary);margin-top:4px;">'
            f'of {len(region_table):,} candidate structures in {region}</div>',
            unsafe_allow_html=True,
        )
        card_close()

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        card_open()
        st.markdown('<div class="rd-card-title">Screening Funnel</div>', unsafe_allow_html=True)
        funnel = [
            ("Detected (Open Buildings)", len(region_table)),
            ("Passed confidence + material filter", len(passed_filter)),
            (f"Within {flag_distance_m}m of river", len(encroaching)),
        ]
        for label, value in funnel:
            st.markdown(
                f'<div class="rd-kv"><span>{label}</span><span>{value:,}</span></div>',
                unsafe_allow_html=True,
            )
        card_close()

# --------------------------------------------------------- tab 2: compare regions

with compare_tab:
    card_open()
    st.markdown('<div class="rd-card-title">Encroaching Structures by Region</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rd-sub">Only Kasarani is calibrated against independent ground truth — '
        "Gatharaini and Motoine reuse that same cutoff untested</div>",
        unsafe_allow_html=True,
    )

    compare_rows = []
    for r in REGIONS:
        t = build_region_table(r)
        if t is None:
            continue
        e = apply_filters(t, confidence_threshold, material_threshold, flag_distance_m)
        compare_rows.append({"region": r, "encroaching": len(e), "screened": len(t),
                             "calibrated": r == "Kasarani"})

    if compare_rows:
        cdf = pd.DataFrame(compare_rows)
        fig = go.Figure(go.Bar(
            x=cdf["region"], y=cdf["encroaching"],
            marker_color=[RED if c else AMBER for c in cdf["calibrated"]],
            text=[f"{v:,}" for v in cdf["encroaching"]], textposition="outside",
            hovertemplate="%{x}: %{y:,} encroaching<extra></extra>",
        ))
        fig.update_layout(**plotly_layout(height=320, showlegend=False,
                                          yaxis=dict(gridcolor="#232b30", title=None),
                                          xaxis=dict(title=None)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(
            f'<div class="rd-legend-row">'
            f'<div class="rd-legend-item"><span class="rd-swatch" style="background:{RED};"></span>Calibrated (Kasarani)</div>'
            f'<div class="rd-legend-item"><span class="rd-swatch" style="background:{AMBER};"></span>Untested extrapolation</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    card_close()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    card_open()
    st.markdown('<div class="rd-card-title">Region Totals</div>', unsafe_allow_html=True)
    if compare_rows:
        st.dataframe(pd.DataFrame(compare_rows), hide_index=True, width="stretch")
    card_close()

# ------------------------------------------------------- tab 3: model performance

with model_tab:
    if not model_ready:
        st.markdown(
            f"""
            <div class="rd-note">
              <b>No trained model found.</b> Looking for
              <span class="mono">{MODEL_PATH.relative_to(ROOT)}</span>.
              Regenerate with <span class="mono">02_modelling.ipynb</span>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        card_open()
        st.markdown('<div class="rd-card-title">Model Card</div>', unsafe_allow_html=True)
        kv = [
            ("Estimator", f"RandomForestClassifier ({model.n_estimators} trees)"),
            ("Features", ", ".join(FEATURE_COLS)),
            ("Label", "builtup — ESA WorldCover ground truth"),
            ("Training scope", "All three regions combined, trained once"),
        ]
        st.markdown(
            "".join(f'<div class="rd-kv"><span>{k}</span><span>{v}</span></div>' for k, v in kv),
            unsafe_allow_html=True,
        )
        card_close()

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="rd-eyebrow">Live evaluation &middot; {region} feature table</div>',
                    unsafe_allow_html=True)

        eval_result = evaluate_model_on_region(region)
        if eval_result is None:
            st.markdown(
                f"""
                <div class="rd-note">
                  No labelled feature table found for {region}
                  (<span class="mono">{feature_table_csv(region).relative_to(ROOT)}</span>) —
                  evaluation metrics can't be computed without ground truth to compare against.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            report = eval_result["report"]
            c1, c2, c3, c4 = st.columns(4, gap="medium")
            with c1:
                metric_card("Accuracy", f"{report['accuracy']:.3f}", f"{eval_result['n']:,} rows")
            with c2:
                metric_card("Precision (built-up)", f"{report['built-up']['precision']:.3f}")
            with c3:
                metric_card("Recall (built-up)", f"{report['built-up']['recall']:.3f}")
            with c4:
                metric_card("F1 (built-up)", f"{report['built-up']['f1-score']:.3f}")

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            card_open()
            st.markdown('<div class="rd-card-title">Confusion Matrix</div>', unsafe_allow_html=True)
            counts = np.array(eval_result["confusion_matrix"])
            row_totals = counts.sum(axis=1, keepdims=True)
            shares = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=float),
                               where=row_totals > 0)
            cm_fig = go.Figure(go.Heatmap(
                z=shares, x=["Predicted not built-up", "Predicted built-up"],
                y=["Actually not built-up", "Actually built-up"],
                colorscale=[[0, "#12191d"], [0.5, "#2c6f6a"], [1, TEAL]],
                zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
                text=[[f"{c:,}<br>{s:.0%}" for c, s in zip(rc, rs)] for rc, rs in zip(counts, shares)],
                texttemplate="%{text}",
                textfont=dict(family="IBM Plex Mono, monospace", size=13, color="#eef2f3"),
            ))
            cm_fig.update_layout(**plotly_layout(height=280, xaxis=dict(side="top"),
                                                 yaxis=dict(autorange="reversed")))
            st.plotly_chart(cm_fig, width="stretch", config={"displayModeBar": False})
            card_close()

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        card_open()
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
        card_close()

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="rd-note">
              <b>What this model actually does.</b> It does not detect buildings — Google Open
              Buildings already found those. It scores each detected point on whether its
              spectral signature looks built-up, as a secondary material-overlap check before a
              building counts as a real encroachment candidate. Trained once across all three
              regions, so a threshold tuned here applies everywhere rather than needing a
              region-specific retrain.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------------------- tab 4: method & data

with method_tab:
    left, right = st.columns([1.3, 1], gap="medium")

    with left:
        card_open()
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
            (<span class="mono">03_fusion_and_report.ipynb</span>) — computes true distance to
            the nearest river line per building, filters on confidence and RF probability, and
            sweeps a flagging distance to match Kasarani's known field count before applying
            that same distance elsewhere.
            </div>
            """,
            unsafe_allow_html=True,
        )
        card_close()

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        if summary and "note" in summary:
            st.markdown(f'<div class="rd-note">{summary["note"]}</div>', unsafe_allow_html=True)

    with right:
        card_open()
        st.markdown('<div class="rd-card-title">Artifacts</div>', unsafe_allow_html=True)
        rows = []
        for r in REGIONS:
            for path, what in [
                (buildings_csv(r), f"{r} — detected structures"),
                (encroaching_csv(r), f"{r} — encroaching structures"),
                (rivers_geojson(r), f"{r} — river lines"),
                (riparian_buffer_geojson(r), f"{r} — riparian buffer"),
            ]:
                exists = path.exists()
                rows.append({
                    "File": path.name, "Holds": what,
                    "Size": f"{path.stat().st_size / 1e6:.1f} MB" if exists else "—",
                    "Status": "present" if exists else "missing",
                })
        rows.append({
            "File": MODEL_PATH.name, "Holds": "Shared Random Forest (trained on Kasarani, "
                                                "Gatharaini and Motoine combined)",
            "Size": f"{MODEL_PATH.stat().st_size / 1e6:.1f} MB" if MODEL_PATH.exists() else "—",
            "Status": "present" if MODEL_PATH.exists() else "missing",
        })

        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=320)
        card_close()