"""
Train the riparian-encroachment Random Forest and write the artifacts the
Streamlit dashboard (app.py) reads.

This is the script form of kasarani_rf_pipeline.ipynb, with two additions the
dashboard needs: the evaluation numbers are written to JSON instead of only
being printed, and predictions are written out for every building (tagged
train/test) so the map can colour footprints by what the model actually said.

Inputs (real, already in the repo):
  kasarani_sentinel_tile.tif        3-band RGB Sentinel-2 composite, EPSG:32737
  kasarani_buildings_export.geojson 57k building footprints, EPSG:4326
  kasarani_river.gpkg               21 river features, EPSG:32737
  kasarani_aoi.gpkg                 AOI boundary, EPSG:32737

Outputs:
  models/riparian_rf_model.joblib               fitted RandomForestClassifier
  models/rf_metadata.json                       features, params, metrics
  data/processed/kasarani_building_features_labeled.csv
  data/processed/kasarani_rf_predictions.csv    id, split, y_true, rf_pred, rf_proba

Run:  python train_rf.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
RASTER_PATH = ROOT / "kasarani_sentinel_tile.tif"
BUILDINGS_PATH = ROOT / "kasarani_buildings_export.geojson"
RIVER_PATH = ROOT / "kasarani_river.gpkg"
AOI_PATH = ROOT / "kasarani_aoi.gpkg"

MODEL_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_PATH = MODEL_DIR / "riparian_rf_model.joblib"
METADATA_PATH = MODEL_DIR / "rf_metadata.json"
LABELED_CSV = PROCESSED_DIR / "kasarani_building_features_labeled.csv"
PREDICTIONS_CSV = PROCESSED_DIR / "kasarani_rf_predictions.csv"

BAND_NAMES = ["R", "G", "B"]  # this tile's B4, B3, B2 order
BUFFER_METERS = 60            # legal riparian buffer used for auto-labeling
TARGET_CRS = "EPSG:32737"     # UTM 37S - metric units, matches river/AOI
TEST_SIZE = 0.30
RANDOM_STATE = 42

# dist_to_river_m is deliberately excluded: `encroachment` is thresholded
# directly from it, so training on it would just re-learn the threshold.
EXCLUDE_COLUMNS = ["id", "dist_to_river_m", "encroachment"]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def build_features(buildings_raw: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-building mean/std pixel value for each raster band, plus footprint area."""
    with rasterio.open(RASTER_PATH) as src:
        raster_crs = src.crs
        affine = src.transform
        bands = [src.read(i + 1) for i in range(src.count)]

    buildings = buildings_raw[
        buildings_raw.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    dropped = len(buildings_raw) - len(buildings)
    if dropped:
        print(f"Dropped {dropped} non-polygon geometries before zonal stats.")

    buildings = buildings.to_crs(raster_crs)

    frames = []
    for band_arr, name in zip(bands, BAND_NAMES):
        stats = zonal_stats(
            buildings, band_arr, affine=affine, stats=["mean", "std"],
            nodata=np.nan, all_touched=True,
        )
        frames.append(
            pd.DataFrame(stats).rename(
                columns={"mean": f"{name}_mean", "std": f"{name}_std"}
            )
        )

    features = pd.concat(frames, axis=1)
    features["id"] = buildings["id"].values
    features["area_m2"] = buildings.geometry.area.values

    before = len(features)
    features = features.dropna()
    if before - len(features):
        print(f"Dropped {before - len(features)} buildings with no valid pixels "
              "under their footprint.")

    cols = (["id"] + [f"{n}_mean" for n in BAND_NAMES]
            + [f"{n}_std" for n in BAND_NAMES] + ["area_m2"])
    return features[cols]


def label_by_distance(buildings_raw: gpd.GeoDataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Attach real centroid-to-river distance and the 60m encroachment label."""
    buildings = buildings_raw[
        buildings_raw.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy().to_crs(TARGET_CRS)

    river = gpd.read_file(RIVER_PATH).to_crs(TARGET_CRS)
    aoi = gpd.read_file(AOI_PATH).to_crs(TARGET_CRS)

    aoi_geom = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
    before = len(buildings)
    buildings = buildings[buildings.geometry.centroid.within(aoi_geom)].copy()
    print(f"Clipped to AOI: {before} -> {len(buildings)} buildings.")

    river_union = river.union_all() if hasattr(river, "union_all") else river.unary_union
    buildings["dist_to_river_m"] = buildings.geometry.centroid.distance(river_union)
    buildings["encroachment"] = (buildings["dist_to_river_m"] <= BUFFER_METERS).astype(int)

    labels = buildings[["id", "dist_to_river_m", "encroachment"]]
    labeled = features.merge(labels, on="id", how="inner")
    dropped = len(features) - len(labeled)
    if dropped:
        print(f"Dropped {dropped} feature rows with no matching AOI-clipped building.")
    return labeled


def main() -> None:
    for path in (RASTER_PATH, BUILDINGS_PATH, RIVER_PATH, AOI_PATH):
        require_file(path)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    buildings_raw = gpd.read_file(BUILDINGS_PATH)
    print(f"Read {len(buildings_raw)} building footprints.")

    features = build_features(buildings_raw)
    labeled = label_by_distance(buildings_raw, features)
    labeled.to_csv(LABELED_CSV, index=False)
    print(f"Wrote {len(labeled)} labeled rows to {LABELED_CSV}")
    print(labeled["encroachment"].value_counts().to_string())

    feature_cols = [c for c in labeled.columns if c not in EXCLUDE_COLUMNS]
    X = labeled[feature_cols]
    y = labeled["encroachment"]

    idx_train, idx_test = train_test_split(
        labeled.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    params = dict(
        n_estimators=100, max_depth=15, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight="balanced",
    )
    model = RandomForestClassifier(**params)
    model.fit(X.loc[idx_train], y.loc[idx_train])
    print(f"Trained on {len(idx_train)} samples, testing on {len(idx_test)}.")

    y_test = y.loc[idx_test]
    y_pred = model.predict(X.loc[idx_test])
    proba_test = model.predict_proba(X.loc[idx_test])[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(classification_report(y_test, y_pred, zero_division=0))

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": "RandomForestClassifier",
        "params": params,
        "feature_cols": feature_cols,
        "excluded_cols": EXCLUDE_COLUMNS,
        "buffer_meters": BUFFER_METERS,
        "n_total": int(len(labeled)),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().items()},
        "metrics": {
            "accuracy": float(report["accuracy"]),
            "roc_auc": float(roc_auc_score(y_test, proba_test)),
            "encroachment": {
                "precision": float(report["1"]["precision"]),
                "recall": float(report["1"]["recall"]),
                "f1": float(report["1"]["f1-score"]),
                "support": int(report["1"]["support"]),
            },
            "non_encroachment": {
                "precision": float(report["0"]["precision"]),
                "recall": float(report["0"]["recall"]),
                "f1": float(report["0"]["f1-score"]),
                "support": int(report["0"]["support"]),
            },
            "macro_f1": float(report["macro avg"]["f1-score"]),
        },
        "confusion_matrix": cm.tolist(),
        "feature_importances": {
            col: float(imp) for col, imp in zip(feature_cols, model.feature_importances_)
        },
    }

    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Score every building so the map can show predictions anywhere, with the
    # split tagged - test-set rows are the only honest read on performance.
    test_ids = set(idx_test)
    predictions = pd.DataFrame({
        "id": labeled["id"],
        "split": np.where(labeled.index.isin(test_ids), "test", "train"),
        "dist_to_river_m": labeled["dist_to_river_m"],
        "y_true": y,
        "rf_pred": model.predict(X),
        "rf_proba": model.predict_proba(X)[:, 1],
    })
    predictions.to_csv(PREDICTIONS_CSV, index=False)

    print(f"\nModel saved:       {MODEL_PATH}")
    print(f"Metadata saved:    {METADATA_PATH}")
    print(f"Predictions saved: {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
