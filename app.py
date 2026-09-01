from pathlib import Path
import json
import math

import geopandas as gpd
import folium
from folium.plugins import FastMarkerCluster
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pyproj import Transformer
from shapely.geometry import Point, box
from streamlit_folium import st_folium

ROOT = Path(__file__).parent
PREP_DIR = ROOT / "Preprocessing"
RESULTS_GPKG = PREP_DIR / "kasarani_encroachment_results.gpkg"
AOI_GPKG = PREP_DIR / "kasarani_aoi.gpkg"
RIVER_GPKG = PREP_DIR / "kasarani_river.gpkg"

MODEL_PATH = ROOT / "models" / "riparian_rf_model.joblib"
METADATA_PATH = ROOT / "models" / "rf_metadata.json"
PREDICTIONS_CSV = ROOT / "data" / "processed" / "kasarani_rf_predictions.csv"
LABELED_CSV = ROOT / "data" / "processed" / "kasarani_building_features_labeled.csv"

METRIC_CRS = "EPSG:32737"
WGS84 = "EPSG:4326"

DEFAULT_LAT = -1.263295
DEFAULT_LON = 36.880376
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

# Rule-vs-model agreement, using the geometric distance rule as the reference.
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
    "Dark (CartoDB)": "CartoDB dark_matter",
    "Satellite (Esri)": "Esri.WorldImagery",
    "OpenStreetMap": "OpenStreetMap",
}

st.set_page_config(
    page_title="Riparian Encroachment Detector",
    page_icon="\U0001f6f0️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------- data loading


@st.cache_data(show_spinner=False)
def load_buildings() -> gpd.GeoDataFrame:
    """Classified footprints from the preprocessing pipeline, joined to RF predictions."""
    gdf = gpd.read_file(RESULTS_GPKG, layer="buildings_classified").to_crs(METRIC_CRS)
    preds = load_predictions()
    if preds is not None:
        gdf = gdf.merge(
            preds[["id", "split", "y_true", "rf_pred", "rf_proba"]], on="id", how="left"
        )
    return gdf


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame | None:
    if not PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(PREDICTIONS_CSV, dtype={"id": str})


@st.cache_data(show_spinner=False)
def load_metadata() -> dict | None:
    if not METADATA_PATH.exists():
        return None
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_river_union():
    return gpd.read_file(RIVER_GPKG).to_crs(METRIC_CRS).geometry.union_all()


@st.cache_resource(show_spinner=False)
def get_transformer() -> Transformer:
    return Transformer.from_crs(WGS84, METRIC_CRS, always_xy=True)


@st.cache_resource(show_spinner=False)
def get_inverse_transformer() -> Transformer:
    return Transformer.from_crs(METRIC_CRS, WGS84, always_xy=True)


@st.cache_data(show_spinner=False)
def hotspot_latlon() -> tuple[float, float]:
    """Lat/lon of the structure sitting closest to the river anywhere in the AOI."""
    gdf = load_buildings()
    closest = gdf.loc[gdf["dist_to_river_m"].idxmin()].geometry.centroid
    lon, lat = get_inverse_transformer().transform(closest.x, closest.y)
    return lat, lon


@st.cache_data(show_spinner=False)
def citywide_risk_counts() -> pd.Series:
    return load_buildings()["risk_category"].value_counts()


@st.cache_data(show_spinner=False)
def buffer_polygon(buffer_m: float):
    return load_river_union().buffer(buffer_m)


@st.cache_data(show_spinner=False)
def load_aoi_wgs84() -> gpd.GeoDataFrame:
    return gpd.read_file(AOI_GPKG).to_crs(WGS84)


@st.cache_data(show_spinner=False)
def aoi_area_km2() -> float:
    return float(gpd.read_file(AOI_GPKG).to_crs(METRIC_CRS).area.sum()) / 1e6


@st.cache_data(show_spinner=False)
def river_geojson() -> dict:
    return json.loads(gpd.read_file(RIVER_GPKG).to_crs(WGS84).to_json())


@st.cache_data(show_spinner=False)
def buffer_geojson(buffer_m: float) -> dict:
    """The whole buffer ribbon, simplified — at AOI scale every vertex ships to the browser."""
    ribbon = buffer_polygon(buffer_m).simplify(3)
    return json.loads(gpd.GeoSeries([ribbon], crs=METRIC_CRS).to_crs(WGS84).to_json())


@st.cache_data(show_spinner=False)
def building_points() -> pd.DataFrame:
    """One row per footprint: centroid lat/lon plus the columns the layers filter on.

    Centroids, not polygons — the AOI view plots a few thousand structures at once,
    which per-polygon GeoJson can't carry (hence MAX_POLYGONS on the detail map).
    """
    gdf = load_buildings()
    centroids = gdf.geometry.centroid
    lon, lat = get_inverse_transformer().transform(
        centroids.x.to_numpy(), centroids.y.to_numpy()
    )
    points = pd.DataFrame({
        "lat": lat,
        "lon": lon,
        "dist_to_river_m": gdf["dist_to_river_m"].to_numpy(),
        "risk_category": gdf["risk_category"].to_numpy(),
        "rf_proba": gdf["rf_proba"].to_numpy() if "rf_proba" in gdf else np.nan,
    })
    return points


def fit_zoom(span_lon: float, px_width: int = 1000) -> int:
    """Web-Mercator zoom at which `span_lon` degrees fills `px_width` pixels."""
    if span_lon <= 0:
        return 12
    return max(1, int(math.floor(math.log2(360 * px_width / (256 * span_lon)))))


# ------------------------------------------------------------- model summaries


def confusion_at(preds: pd.DataFrame, threshold: float, split: str = "test") -> dict:
    """Confusion counts and derived rates for the model at a decision threshold.

    Defaults to the held-out split — train-split numbers are optimistic and only
    worth showing when explicitly asked for.
    """
    subset = preds if split == "all" else preds[preds["split"] == split]
    predicted = (subset["rf_proba"] >= threshold).to_numpy()
    actual = subset["y_true"].to_numpy() == 1

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
def threshold_sweep(split: str = "test") -> pd.DataFrame:
    preds = load_predictions()
    rows = []
    for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
        stats = confusion_at(preds, float(thr), split)
        rows.append({"threshold": float(thr), **stats})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ chrome/CSS


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
        .rd-header{ display:flex; align-items:center; justify-content:space-between; padding:14px 4px 18px; border-bottom:1px solid var(--border); margin-bottom:18px; }
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


# ------------------------------------------------------------------- app state

inject_css()

metadata = load_metadata()
predictions = load_predictions()
model_ready = metadata is not None and predictions is not None

buildings = load_buildings()
transformer = get_transformer()

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
            <div class="mono" style="font-size:10px;color:#5c6b73;">KASARANI &middot; NAIROBI RIVER</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rd-eyebrow">Query Coordinate</div>', unsafe_allow_html=True)
    # A queued jump is applied before the widgets render — Streamlit refuses
    # session-state writes to a widget key after that widget exists.
    if "pending_jump" in st.session_state:
        st.session_state["lat"], st.session_state["lon"] = st.session_state.pop("pending_jump")
    st.session_state.setdefault("lat", DEFAULT_LAT)
    st.session_state.setdefault("lon", DEFAULT_LON)

    lat = st.number_input("Latitude", key="lat", format="%.6f")
    lon = st.number_input("Longitude", key="lon", format="%.6f")
    if st.button("Jump to structure closest to river", width="stretch"):
        st.session_state["pending_jump"] = hotspot_latlon()
        st.rerun()
    view_radius = st.slider("View radius (m)", min_value=100, max_value=600, value=250, step=50)

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Detection Layer</div>', unsafe_allow_html=True)
    mode_options = [MODE_GEOMETRIC, MODE_RF, MODE_COMPARE] if model_ready else [MODE_GEOMETRIC]
    mode = st.radio("Detection layer", mode_options, label_visibility="collapsed")
    if not model_ready:
        st.caption("Random Forest layers unlock once the model artifacts exist — run `python train_rf.py`.")

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Buffer Zone</div>', unsafe_allow_html=True)
    default_buffer = int(metadata["buffer_meters"]) if model_ready else 30
    buffer_m = st.slider("Riparian buffer (m)", min_value=10, max_value=60, value=default_buffer, step=5)
    st.caption(
        "The proposal specifies a 60m setback and the model was trained against that "
        "label; the geometric layer recomputes live at whatever distance you set."
    )

    threshold = 0.5
    if model_ready:
        # Kept visible in every mode: the Model performance tab reads this too.
        st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Model Threshold</div>', unsafe_allow_html=True)
        threshold = st.slider("Flag a structure when P(encroachment) exceeds", 0.05, 0.95, 0.50, 0.05)
        st.caption("Encroaching structures are ~2% of the AOI, so the 0.5 default is rarely the useful operating point.")

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Basemap</div>', unsafe_allow_html=True)
    basemap_label = st.selectbox("Basemap", list(BASEMAPS.keys()), label_visibility="collapsed")

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Detection Model</div>', unsafe_allow_html=True)
    if model_ready:
        enc = metadata["metrics"]["encroachment"]
        rf_card = f"""
          <div class="rd-model-card rd-model-card-on">
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <div style="font-size:12.5px;font-weight:600;color:var(--accent-green);">Phase 1 &middot; Random Forest</div>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="7" fill="#6ad18a"/>
              <path d="M4 7L6 9L10 5" stroke="#161c20" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <div class="rd-sub" style="margin-top:3px;">
              {len(metadata['feature_cols'])} spectral features &middot; trained {metadata['trained_at'][:10]}<br>
              {metadata['n_train']:,} train / {metadata['n_test']:,} test &middot;
              encroachment F1 {enc['f1']:.3f}
            </div>
          </div>"""
    else:
        rf_card = """
          <div class="rd-model-card rd-model-card-warn">
            <div style="font-size:12.5px;font-weight:600;color:var(--accent-amber);">Phase 1 &middot; Random Forest</div>
            <div class="rd-sub" style="margin-top:3px;">Artifacts missing &mdash; run
            <span class="mono">python train_rf.py</span></div>
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
        Sources: OSM (river) &middot; MS Building Footprints via GEE (footprints)<br>
        Sentinel-2 B4/B3/B2 composite (spectral features)
        </div>
        """,
        unsafe_allow_html=True,
    )

x_m, y_m = transformer.transform(lon, lat)
query_point = Point(x_m, y_m)
view_box = box(x_m - view_radius, y_m - view_radius, x_m + view_radius, y_m + view_radius)

in_view = buildings[buildings.geometry.distance(query_point) <= view_radius].copy()
in_view["query_dist_m"] = in_view.geometry.distance(query_point)
in_view["rule_flag"] = in_view["dist_to_river_m"] <= buffer_m
if model_ready:
    in_view["model_flag"] = in_view["rf_proba"].fillna(0) >= threshold
    in_view["scored"] = in_view["rf_proba"].notna()
else:
    in_view["model_flag"] = False
    in_view["scored"] = False

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


if model_ready:
    in_view["agreement"] = (
        in_view.apply(agreement_label, axis=1) if len(in_view) else pd.Series(dtype=str)
    )

total_in_view = len(in_view)
flagged_in_view = int(in_view["flagged"].sum())
rule_in_view = int(in_view["rule_flag"].sum())
model_in_view = int(in_view["model_flag"].sum())
high_risk_view = int((in_view["dist_to_river_m"] <= 10).sum())
flagged_area = float(in_view.loc[in_view["flagged"], "total_area_m2"].sum())
pct_flagged = (flagged_in_view / total_in_view * 100) if total_in_view else 0.0

citywide_total = len(buildings)
citywide_rule = int((buildings["dist_to_river_m"] <= buffer_m).sum())
citywide_model = (
    int((buildings["rf_proba"].fillna(0) >= threshold).sum()) if model_ready else 0
)

st.markdown(
    f"""
    <div class="rd-header">
      <div>
        <div class="rd-title">RIPARIAN ENCROACHMENT DETECTOR</div>
        <div class="rd-subtitle">KASARANI &middot; NAIROBI RIVER BASIN</div>
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

# ------------------------------------------------------------- tab 1: the map

with map_tab:
    map_col, side_col = st.columns([2.1, 1], gap="medium")

    with map_col:
        card_open("padding:0;overflow:hidden;")
        st.markdown(
            f"""
            <div style="padding:14px 18px;border-bottom:1px solid var(--border-soft);">
              <div class="rd-card-title">{mode} &middot; {buffer_m}m buffer</div>
              <div class="rd-sub">{basemap_label} &middot; {total_in_view} structures within
              {view_radius}m of the query point</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fmap = folium.Map(location=[lat, lon], zoom_start=17,
                          tiles=BASEMAPS[basemap_label], control_scale=True)

        buf_geom_view = buffer_polygon(buffer_m).intersection(view_box)
        if not buf_geom_view.is_empty and buf_geom_view.area > 0:
            folium.GeoJson(
                buf_geom_view.__geo_interface__,
                style_function=lambda _f: {
                    "fillColor": TEAL, "color": AMBER, "weight": 1.4,
                    "dashArray": "5 4", "fillOpacity": 0.12,
                },
                name=f"{buffer_m}m buffer",
            ).add_to(fmap)

        drawn = in_view.sort_values("query_dist_m").head(MAX_POLYGONS)
        for _, row in drawn.to_crs(WGS84).iterrows():
            proba_txt = f"{row['rf_proba']:.2f}" if model_ready and pd.notna(row.get("rf_proba")) else "n/a"
            if mode == MODE_GEOMETRIC:
                color = GREEN if row["rule_flag"] else RISK_COLORS.get(row["risk_category"], GREY)
                tooltip = (f"{row['dist_to_river_m']:.1f}m to river &middot; {row['risk_category']} &middot; "
                           f"{'INSIDE BUFFER' if row['rule_flag'] else 'outside buffer'}")
            elif mode == MODE_RF:
                color = AMBER if row["model_flag"] else (GREY if row["scored"] else "#333c42")
                tooltip = (f"P(encroachment) = {proba_txt} &middot; "
                           f"{'FLAGGED' if row['model_flag'] else 'not flagged'} &middot; "
                           f"{row['dist_to_river_m']:.1f}m to river")
            else:
                color = AGREEMENT_COLORS[row["agreement"]]
                tooltip = (f"{row['agreement']} &middot; rule {row['dist_to_river_m']:.1f}m &middot; "
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
            f"citywide rule: {citywide_rule:,}" if mode == MODE_GEOMETRIC
            else f"citywide model: {citywide_model:,}"
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
        cols = {"query_dist_m": "To query (m)", "dist_to_river_m": "To river (m)",
                "risk_category": "Risk tier", "rule_flag": "In buffer"}
        if model_ready:
            cols["rf_proba"] = "P(encroach)"
            cols["model_flag"] = "Model flag"
        nearest = in_view.sort_values("query_dist_m").head(10)[list(cols)].rename(columns=cols)
        nearest["To query (m)"] = nearest["To query (m)"].round(1)
        nearest["To river (m)"] = nearest["To river (m)"].round(1)
        if model_ready:
            nearest["P(encroach)"] = nearest["P(encroach)"].round(3)
        st.dataframe(nearest, hide_index=True, width="stretch", height=240)
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
        '<div class="rd-card-title">Structures Near the River, by Risk Tier &middot; Kasarani AOI</div>',
        unsafe_allow_html=True,
    )
    risk_counts_all = citywide_risk_counts().reindex(RISK_ORDER).fillna(0)
    safe_count = int(risk_counts_all["Safe Zone (>30m)"])
    at_risk = risk_counts_all.reindex(["Low Risk (20m-30m)", "Medium Risk (10m-20m)", "High Risk (<10m)"])
    st.markdown(
        f'<div class="rd-sub">{int(at_risk.sum()):,} structures sit within 30m of the river '
        f'&middot; {safe_count:,} more ({safe_count / citywide_total * 100:.0f}% of {citywide_total:,} '
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
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    card_close()

# ----------------------------------------------------- tab 2: AOI overview

with aoi_tab:
    aoi = load_aoi_wgs84()
    min_lon, min_lat, max_lon, max_lat = aoi.total_bounds
    pad_lon = (max_lon - min_lon) * 0.08
    pad_lat = (max_lat - min_lat) * 0.08
    zoom_to_fit = fit_zoom(max_lon - min_lon)

    points = building_points()
    rule_flag_all = points["dist_to_river_m"] <= buffer_m
    model_flag_all = (
        points["rf_proba"].fillna(0) >= threshold
        if model_ready else pd.Series(False, index=points.index)
    )

    if mode == MODE_GEOMETRIC:
        layers = [(f"Inside {buffer_m}m buffer", GREEN, points[rule_flag_all])]
    elif mode == MODE_RF:
        layers = [(f"Model flags (P ≥ {threshold:.2f})", AMBER, points[model_flag_all])]
    else:
        layers = [
            ("Both flag", AGREEMENT_COLORS["Both flag"], points[rule_flag_all & model_flag_all]),
            ("Model only", AGREEMENT_COLORS["Model only"], points[model_flag_all & ~rule_flag_all]),
            ("Rule only", AGREEMENT_COLORS["Rule only"], points[rule_flag_all & ~model_flag_all]),
        ]
    plotted = sum(len(subset) for _, _, subset in layers)

    card_open("padding:0;overflow:hidden;")
    st.markdown(
        f"""
        <div style="padding:14px 18px;border-bottom:1px solid var(--border-soft);">
          <div class="rd-card-title">Kasarani AOI &middot; {mode}</div>
          <div class="rd-sub">{plotted:,} flagged structures across the whole study area &middot;
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

    folium.GeoJson(
        buffer_geojson(buffer_m),
        style_function=lambda _f: {
            "fillColor": TEAL, "color": AMBER, "weight": 1.1,
            "dashArray": "5 4", "fillOpacity": 0.14,
        },
        name=f"{buffer_m}m buffer",
    ).add_to(overview)

    folium.GeoJson(
        river_geojson(),
        style_function=lambda _f: {"color": TEAL, "weight": 2},
        name="River centerline",
    ).add_to(overview)

    folium.GeoJson(
        json.loads(aoi.to_json()),
        style_function=lambda _f: {
            "fillOpacity": 0, "color": TEAL, "weight": 1.2,
            "dashArray": "8 6", "opacity": 0.7,
        },
        name="AOI boundary",
    ).add_to(overview)

    for name, color, subset in layers:
        if subset.empty:
            continue
        rgba = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.30)"
        data = [
            [round(row.lat, 6), round(row.lon, 6),
             f"{row.dist_to_river_m:.0f}m to river"
             + (f" &middot; P {row.rf_proba:.2f}" if model_ready and pd.notna(row.rf_proba) else "")]
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
              key="aoi_overview")

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
            <div class="rd-legend-item"><span style="width:14px;height:9px;border:1px dashed {TEAL};
            opacity:.7;display:inline-block;"></span>AOI boundary</div>
          </div>
          <div class="rd-sub">Only flagged structures are plotted — all {citywide_total:,} footprints
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
    aoi_km2 = aoi_area_km2()
    with a1:
        metric_card("Structures Plotted", f"{plotted:,}", f"of {citywide_total:,} in the AOI")
    with a2:
        metric_card(f"Rule Flags ({buffer_m}m)", f"{citywide_rule:,}", "geometric distance rule")
    with a3:
        if model_ready:
            metric_card(f"Model Flags (P≥{threshold:.2f})", f"{citywide_model:,}",
                        "Random Forest", AMBER if citywide_model > 3 * citywide_rule else None)
        else:
            metric_card("High Risk (&lt;10m)", f"{int((buildings['dist_to_river_m'] <= 10).sum()):,}",
                        "AOI-wide", RED)
    with a4:
        metric_card("AOI Extent", f"{aoi_km2:,.0f}", "km² study area")

# ------------------------------------------------- tab 3: model performance

with model_tab:
    if not model_ready:
        st.markdown(
            f"""
            <div class="rd-note">
              <b>No trained model artifacts found.</b><br>
              The dashboard looks for <span class="mono">{MODEL_PATH.relative_to(ROOT)}</span>,
              <span class="mono">{METADATA_PATH.relative_to(ROOT)}</span> and
              <span class="mono">{PREDICTIONS_CSV.relative_to(ROOT)}</span>. These are gitignored
              build outputs, so a fresh clone won't have them.<br><br>
              Regenerate them with <span class="mono">python train_rf.py</span> (the script form of
              <span class="mono">kasarani_rf_pipeline.ipynb</span>), then reload this page.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        enc = metadata["metrics"]["encroachment"]
        live = confusion_at(predictions, threshold)

        card_open()
        st.markdown('<div class="rd-card-title">Model Card</div>', unsafe_allow_html=True)
        params = metadata["params"]
        kv = [
            ("Estimator", f"{metadata['model_type']} ({params['n_estimators']} trees, "
                          f"max_depth {params['max_depth']}, class_weight {params['class_weight']})"),
            ("Features", ", ".join(metadata["feature_cols"])),
            ("Excluded", ", ".join(metadata["excluded_cols"])),
            ("Label", f"encroachment = centroid within {metadata['buffer_meters']}m of the river"),
            ("Split", f"{metadata['n_train']:,} train / {metadata['n_test']:,} test "
                      f"(stratified, seed {params['random_state']})"),
            ("Class balance", " · ".join(f"{k}: {v:,}" for k, v in metadata["class_balance"].items())),
            ("Trained at", metadata["trained_at"]),
        ]
        st.markdown(
            "".join(f'<div class="rd-kv"><span>{k}</span><span>{v}</span></div>' for k, v in kv),
            unsafe_allow_html=True,
        )
        card_close()

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="rd-eyebrow">Held-out test set &middot; threshold {threshold:.2f}</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
        with c1:
            metric_card("Accuracy", f"{live['accuracy']:.3f}", f"{live['n']:,} test rows")
        with c2:
            metric_card("ROC AUC", f"{metadata['metrics']['roc_auc']:.3f}", "threshold-independent")
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
                f'<div class="rd-sub">Test split at threshold {threshold:.2f} &middot; '
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
            st.plotly_chart(cm_fig, width="stretch", config={"displayModeBar": False})
            card_close()

        with curve_col:
            card_open()
            st.markdown('<div class="rd-card-title">Precision &amp; Recall vs Threshold</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="rd-sub">Test split &middot; the dashed line is your current '
                        "threshold</div>", unsafe_allow_html=True)
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
            st.plotly_chart(curve, width="stretch", config={"displayModeBar": False})
            card_close()

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        card_open()
        st.markdown('<div class="rd-card-title">Feature Importance</div>', unsafe_allow_html=True)
        st.markdown('<div class="rd-sub">Mean decrease in impurity across the forest</div>',
                    unsafe_allow_html=True)
        imp = pd.Series(metadata["feature_importances"]).sort_values()
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
        st.plotly_chart(imp_fig, width="stretch", config={"displayModeBar": False})
        card_close()

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="rd-note">
              <b>How to read these numbers.</b> Encroaching structures are
              {metadata['class_balance'].get('1', 0):,} of {metadata['n_total']:,}
              ({metadata['class_balance'].get('1', 0) / metadata['n_total'] * 100:.1f}%), so overall
              accuracy is dominated by the majority class and says almost nothing — a model that
              flagged nothing at all would score about
              {(1 - metadata['class_balance'].get('1', 0) / metadata['n_total']) * 100:.1f}%.
              The number that matters is the encroachment row: precision
              {enc['precision']:.3f}, recall {enc['recall']:.3f}, F1 {enc['f1']:.3f} at the 0.5
              default.<br><br>
              That is weak, and it is a data limitation rather than a tuning problem: the tile
              carries only B4/B3/B2, so the model sees six RGB statistics plus footprint area, and
              nothing about a roof's colour tells you how far it sits from a river. The geometric
              distance rule remains the authoritative classifier for enforcement; the Random Forest
              is here as the Phase 1 baseline it was scoped as. Adding NIR/SWIR bands (NDVI/NDWI),
              texture, or neighbourhood-density features is the path to a model that earns its place
              in the decision.
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
            <b style="color:var(--text-primary);">1 &middot; Vector preprocessing</b>
            (<span class="mono">Preprocessing/Cleaning.ipynb</span>) — pulls the Nairobi River
            centerline from OSM, filters non-river waterways, clips Microsoft Building Footprints
            to the Kasarani AOI, and classifies each footprint into a risk tier by its distance to
            the river. Output: <span class="mono">kasarani_encroachment_results.gpkg</span>, which is
            what the map draws.<br><br>
            <b style="color:var(--text-primary);">2 &middot; Spectral feature extraction</b>
            (<span class="mono">build_building_features.py</span>) — zonal statistics over the
            Sentinel-2 composite: mean and standard deviation of B4/B3/B2 under each footprint,
            plus footprint area.<br><br>
            <b style="color:var(--text-primary);">3 &middot; Auto-labeling</b>
            (<span class="mono">label_buildings_by_river_distance.py</span>) — each building centroid's
            real distance to the nearest river feature; <span class="mono">encroachment = 1</span>
            within the 60m legal buffer.<br><br>
            <b style="color:var(--text-primary);">4 &middot; Random Forest</b>
            (<span class="mono">train_rf.py</span>, notebook form in
            <span class="mono">kasarani_rf_pipeline.ipynb</span>) — trains on the spectral features
            only, writes the model, its metrics, and per-building predictions that this dashboard
            reads.
            </div>
            """,
            unsafe_allow_html=True,
        )
        card_close()

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="rd-note">
              <b>Two distance numbers, deliberately.</b> The map's
              <span class="mono">dist_to_river_m</span> is polygon-edge distance from the
              preprocessing pipeline; the model's label used centroid distance clipped to the AOI.
              They disagree by a few metres on buildings that straddle the buffer line, which is why
              the citywide rule count and the model's training positives are not identical.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        card_open()
        st.markdown('<div class="rd-card-title">Artifacts</div>', unsafe_allow_html=True)
        st.markdown('<div class="rd-sub">Build outputs are gitignored — regenerate with '
                    "<span class='mono'>python train_rf.py</span></div>", unsafe_allow_html=True)
        rows = []
        for path, what in [
            (RESULTS_GPKG, "classified footprints"),
            (RIVER_GPKG, "river centerline"),
            (ROOT / "kasarani_sentinel_tile.tif", "Sentinel-2 composite"),
            (MODEL_PATH, "trained Random Forest"),
            (METADATA_PATH, "metrics + importances"),
            (PREDICTIONS_CSV, "per-building predictions"),
            (LABELED_CSV, "labeled feature table"),
        ]:
            exists = path.exists()
            rows.append({
                "File": str(path.relative_to(ROOT)),
                "Holds": what,
                "Size": f"{path.stat().st_size / 1e6:.1f} MB" if exists else "—",
                "Status": "present" if exists else "missing",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=290)
        card_close()

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        card_open()
        st.markdown('<div class="rd-card-title">AOI Totals</div>', unsafe_allow_html=True)
        totals = [
            ("Footprints", f"{citywide_total:,}"),
            (f"Within {buffer_m}m (rule)", f"{citywide_rule:,}"),
        ]
        if model_ready:
            totals.append((f"Flagged by model (P≥{threshold:.2f})", f"{citywide_model:,}"))
            totals.append(("Scored by model", f"{int(buildings['rf_proba'].notna().sum()):,}"))
        st.markdown(
            "".join(f'<div class="rd-kv"><span>{k}</span><span>{v}</span></div>' for k, v in totals),
            unsafe_allow_html=True,
        )
        card_close()
