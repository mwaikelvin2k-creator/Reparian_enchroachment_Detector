from pathlib import Path

import geopandas as gpd
import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pyproj import Transformer
from shapely.geometry import Point, box
from streamlit_folium import st_folium

DATA_DIR = Path(__file__).parent / "Preprocessing"
RESULTS_GPKG = DATA_DIR / "kasarani_encroachment_results.gpkg"
AOI_GPKG = DATA_DIR / "kasarani_aoi.gpkg"
RIVER_GPKG = DATA_DIR / "kasarani_river.gpkg"

METRIC_CRS = "EPSG:32737"
WGS84 = "EPSG:4326"

DEFAULT_LAT = -1.263295
DEFAULT_LON = 36.880376
VIEW_RADIUS_M = 250

RISK_COLORS = {
    "High Risk (<10m)": "#e2604f",
    "Medium Risk (10m-20m)": "#f2b544",
    "Low Risk (20m-30m)": "#4dd0c4",
    "Safe Zone (>30m)": "#3f8f86",
}
RISK_ORDER = ["Safe Zone (>30m)", "Low Risk (20m-30m)", "Medium Risk (10m-20m)", "High Risk (<10m)"]

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


@st.cache_data(show_spinner=False)
def load_buildings() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(RESULTS_GPKG, layer="buildings_classified")
    return gdf.to_crs(METRIC_CRS)


@st.cache_data(show_spinner=False)
def load_aoi() -> gpd.GeoDataFrame:
    return gpd.read_file(AOI_GPKG).to_crs(METRIC_CRS)


@st.cache_resource(show_spinner=False)
def load_river_union():
    river = gpd.read_file(RIVER_GPKG).to_crs(METRIC_CRS)
    return river.geometry.union_all()


@st.cache_resource(show_spinner=False)
def get_transformer() -> Transformer:
    return Transformer.from_crs(WGS84, METRIC_CRS, always_xy=True)


@st.cache_resource(show_spinner=False)
def get_inverse_transformer() -> Transformer:
    return Transformer.from_crs(METRIC_CRS, WGS84, always_xy=True)


@st.cache_data(show_spinner=False)
def citywide_risk_counts(_buildings_key: int) -> pd.Series:
    buildings = load_buildings()
    return buildings["risk_category"].value_counts()


@st.cache_data(show_spinner=False)
def buffer_polygon(buffer_m: float):
    river_union = load_river_union()
    return river_union.buffer(buffer_m)


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
        .rd-sub{ font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--text-tertiary); margin-top:2px; }
        .rd-badge{ font-family:'IBM Plex Mono',monospace; font-size:11px; padding:6px 12px; border-radius:6px; border:1px solid var(--border); color:var(--text-tertiary); display:inline-block; }
        .rd-badge-active{ background:rgba(242,181,68,0.12); border:1px solid var(--accent-amber); color:var(--accent-amber); }
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
        .stButton>button{ background:var(--accent-amber); color:#1a1508; border:none; font-weight:600; border-radius:8px; }
        .stButton>button:hover{ background:#f7c368; color:#1a1508; }
        [data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card_open(extra_style: str = "") -> None:
    st.markdown(f'<div class="rd-card" style="{extra_style}">', unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


inject_css()

buildings = load_buildings()
aoi = load_aoi()
transformer = get_transformer()
inv_transformer = get_inverse_transformer()

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
    lat = st.number_input("Latitude", value=DEFAULT_LAT, format="%.6f")
    lon = st.number_input("Longitude", value=DEFAULT_LON, format="%.6f")
    if st.button("Use closest-to-river sample point", use_container_width=True):
        lat, lon = DEFAULT_LAT, DEFAULT_LON
        st.rerun()

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Buffer Zone</div>', unsafe_allow_html=True)
    buffer_m = st.slider("Riparian buffer (m)", min_value=10, max_value=60, value=30, step=5)
    st.caption(
        "Proposal specifies a 60m setback; the current pipeline (Cleaning.ipynb) "
        "computes against a revised 30m riparian reserve. Drag to compare."
    )

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Basemap</div>', unsafe_allow_html=True)
    basemap_label = st.selectbox("Basemap", list(BASEMAPS.keys()), label_visibility="collapsed")
    st.caption("Sentinel-2 GEE export pending re-run (tile bounds mismatch) — showing a live basemap instead.")

    st.markdown('<div class="rd-eyebrow" style="margin-top:18px;">Detection Model</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:flex;flex-direction:column;gap:8px;">
          <div class="rd-badge" style="width:100%;box-sizing:border-box;">Phase 1 &middot; Random Forest &mdash; not yet trained</div>
          <div class="rd-badge" style="width:100%;box-sizing:border-box;">Phase 2 &middot; Deep Learning &mdash; not yet trained</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Counts below use the current geometric (distance-to-river) classification.")

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    st.button("Run Analysis", use_container_width=True)
    st.markdown(
        """
        <div class="mono" style="font-size:10px;color:#5c6b73;margin-top:10px;line-height:1.6;">
        Sources: OSM (river) &middot; MS Building Footprints via GEE (footprints)<br>
        57,430 structures &middot; Kasarani AOI
        </div>
        """,
        unsafe_allow_html=True,
    )

x_m, y_m = transformer.transform(lon, lat)
query_point = Point(x_m, y_m)

view_box = box(x_m - VIEW_RADIUS_M, y_m - VIEW_RADIUS_M, x_m + VIEW_RADIUS_M, y_m + VIEW_RADIUS_M)
in_view = buildings[buildings.geometry.distance(query_point) <= VIEW_RADIUS_M].copy()
in_view["inside_buffer_now"] = in_view["dist_to_river_m"] <= buffer_m
in_view["query_dist_m"] = in_view.geometry.distance(query_point)

buf_geom = buffer_polygon(buffer_m)
buf_geom_view = buf_geom.intersection(view_box)

total_in_view = len(in_view)
inside_now = int(in_view["inside_buffer_now"].sum())
high_risk_view = int((in_view["dist_to_river_m"] <= 10).sum())
encroached_area_view = float(in_view.loc[in_view["inside_buffer_now"], "total_area_m2"].sum())
pct_inside = (inside_now / total_in_view * 100) if total_in_view else 0.0

citywide_total = len(buildings)
citywide_inside = int((buildings["dist_to_river_m"] <= buffer_m).sum())

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

map_col, side_col = st.columns([2.1, 1], gap="medium")

with map_col:
    card_open("padding:0;overflow:hidden;")
    st.markdown(
        f"""
        <div style="padding:14px 18px;border-bottom:1px solid var(--border-soft);">
          <div class="rd-card-title">Satellite View &middot; {buffer_m}m Buffer Zone</div>
          <div class="rd-sub">{basemap_label} &middot; showing {total_in_view} structures within {VIEW_RADIUS_M}m of query point</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fmap = folium.Map(
        location=[lat, lon],
        zoom_start=17,
        tiles=BASEMAPS[basemap_label],
        control_scale=True,
    )

    if buf_geom_view.area > 0:
        folium.GeoJson(
            buf_geom_view.__geo_interface__,
            style_function=lambda _f: {
                "fillColor": "#4dd0c4",
                "color": "#f2b544",
                "weight": 1.4,
                "dashArray": "5 4",
                "fillOpacity": 0.12,
            },
            name=f"{buffer_m}m buffer",
        ).add_to(fmap)

    in_view_wgs = in_view.to_crs(WGS84)
    for _, row in in_view_wgs.iterrows():
        color = "#6ad18a" if row["inside_buffer_now"] else RISK_COLORS.get(row["risk_category"], "#4a5860")
        weight = 1.6 if row["inside_buffer_now"] else 0.8
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _f, c=color, w=weight: {
                "fillColor": c, "color": c, "weight": w, "fillOpacity": 0.55,
            },
            tooltip=(
                f"dist to river: {row['dist_to_river_m']:.1f}m | "
                f"{row['risk_category']} | "
                f"{'INSIDE BUFFER' if row['inside_buffer_now'] else 'outside'}"
            ),
        ).add_to(fmap)

    folium.CircleMarker(
        location=[lat, lon],
        radius=7,
        color="#4dd0c4",
        weight=2,
        fill=True,
        fill_color="#0f1417",
        fill_opacity=1,
        tooltip="Query point",
    ).add_to(fmap)

    st_folium(fmap, height=420, use_container_width=True, returned_objects=[])

    st.markdown(
        """
        <div style="padding:10px 18px 14px;">
          <div class="rd-legend-row">
            <div class="rd-legend-item"><span class="rd-swatch" style="background:#6ad18a;"></span>Inside buffer now</div>
            <div class="rd-legend-item"><span class="rd-swatch" style="background:#e2604f;"></span>High risk (&lt;10m)</div>
            <div class="rd-legend-item"><span class="rd-swatch" style="background:#f2b544;"></span>Medium risk</div>
            <div class="rd-legend-item"><span class="rd-swatch" style="background:#4a5860;"></span>Safe zone</div>
            <div class="rd-legend-item"><span style="width:14px;height:9px;border:1px dashed #f2b544;background:rgba(77,208,196,0.15);display:inline-block;"></span>Buffer zone</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_close()

with side_col:
    card_open()
    st.markdown('<div class="rd-metric-label">Structures Inside Buffer</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display:flex;align-items:baseline;gap:8px;margin-top:4px;">
          <div class="rd-metric-big">{inside_now}</div>
          <div style="font-size:12px;color:var(--text-tertiary);">/ {total_in_view} in view</div>
        </div>
        <div style="font-size:11px;color:var(--text-secondary);margin-top:4px;">Within {VIEW_RADIUS_M}m of query point &middot; {buffer_m}m riparian setback</div>
        <div class="rd-progress-track"><div class="rd-progress-fill" style="width:{pct_inside:.1f}%;"></div></div>
        <div style="display:flex;justify-content:space-between;margin-top:6px;">
          <div class="mono" style="font-size:11px;color:var(--accent-teal);">{pct_inside:.1f}% of view</div>
          <div class="mono" style="font-size:11px;color:var(--text-tertiary);">citywide: {citywide_inside}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_close()

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    card_open()
    st.markdown('<div class="rd-card-title">Nearest Structures</div>', unsafe_allow_html=True)
    st.markdown('<div class="rd-sub">Sorted by distance to query point</div>', unsafe_allow_html=True)
    nearest = (
        in_view.sort_values("query_dist_m")
        .head(8)[["query_dist_m", "dist_to_river_m", "risk_category", "inside_buffer_now"]]
        .rename(columns={
            "query_dist_m": "To query (m)",
            "dist_to_river_m": "To river (m)",
            "risk_category": "Risk tier",
            "inside_buffer_now": "In buffer",
        })
    )
    nearest["To query (m)"] = nearest["To query (m)"].round(1)
    nearest["To river (m)"] = nearest["To river (m)"].round(1)
    st.dataframe(nearest, hide_index=True, use_container_width=True, height=230)
    card_close()

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4, gap="medium")
metric_cells = [
    (m1, "Structures In View", f"{total_in_view}", ""),
    (m2, "Inside Buffer", f"{inside_now}", f"{pct_inside:.1f}% of view"),
    (m3, "High Risk (<10m)", f"{high_risk_view}", ""),
    (m4, "Encroached Area", f"{encroached_area_view:,.0f}", "m² footprint, in-buffer"),
]
for col, label, value, sub in metric_cells:
    with col:
        card_open()
        st.markdown(f'<div class="rd-metric-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rd-metric-mid" style="margin-top:4px;">{value}</div>', unsafe_allow_html=True)
        if sub:
            st.markdown(f'<div style="font-size:11px;color:var(--text-tertiary);margin-top:2px;">{sub}</div>', unsafe_allow_html=True)
        card_close()

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

card_open()
st.markdown('<div class="rd-card-title">Structures Near the River, by Risk Tier &middot; Kasarani AOI</div>', unsafe_allow_html=True)

risk_counts_all = citywide_risk_counts(citywide_total).reindex(RISK_ORDER).fillna(0)
safe_count = int(risk_counts_all["Safe Zone (>30m)"])
safe_pct = safe_count / citywide_total * 100
at_risk_tiers = ["Low Risk (20m-30m)", "Medium Risk (10m-20m)", "High Risk (<10m)"]
risk_counts = risk_counts_all.reindex(at_risk_tiers)

st.markdown(
    f'<div class="rd-sub">{int(risk_counts.sum()):,} structures sit within 30m of the river '
    f'&middot; {safe_count:,} more ({safe_pct:.0f}% of {citywide_total:,} total) fall in the safe '
    f"zone beyond 30m and aren't shown on this scale</div>",
    unsafe_allow_html=True,
)
fig = go.Figure(
    go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        marker_color=[RISK_COLORS[c] for c in risk_counts.index],
        text=[f"{int(v):,}" for v in risk_counts.values],
        textposition="outside",
    )
)
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#8a9aa3", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    height=280,
    yaxis=dict(gridcolor="#232b30", title=None),
    xaxis=dict(title=None),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
card_close()

st.markdown(
    """
    <div style="margin-top:18px;padding:14px 4px;font-size:12px;color:#5c6b73;line-height:1.6;">
    Structure counts and risk tiers above come from the project's real preprocessing pipeline
    (OSM river centerline + Microsoft Building Footprints via Google Earth Engine, cleaned and
    classified in <span class="mono">Cleaning.ipynb</span>). The Phase 1 Random Forest and Phase 2
    deep-learning models from the proposal are not yet trained — once available, their
    predicted counts and mAP metrics will populate alongside this geometric baseline.
    </div>
    """,
    unsafe_allow_html=True,
)
