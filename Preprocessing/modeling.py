"""
Modeling module for structure/land-cover feature classification.
Trains Random Forest classifiers on spectral profiles and evaluates
performance, for either a binary target (encroachment, is_building) or a
multi-class target (land_cover).

FIXES applied (vs original):
- model_name now stored in metadata (required by visualize.py)
- Baseline accuracy / class distribution added to binary metrics
- Average precision added to binary metrics
- Optional spatial block train/test split (reduces optimistic bias from
  spatial autocorrelation)
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, classification_report, confusion_matrix,
    average_precision_score,
)


def _spatial_block_split(df: pd.DataFrame, feature_cols: list, label_col: str,
                         test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Split data by spatial blocks (UTM easting bands) to reduce
    optimistic bias from spatial autocorrelation.

    Requires df to have 'x_centroid' and 'y_centroid' columns (polygon
    centroids in metric CRS). Falls back to random stratified split if
    centroids are missing."""
    if "x_centroid" not in df.columns or "y_centroid" not in df.columns:
        warnings.warn(
            "Spatial split requested but x_centroid/y_centroid not found. "
            "Falling back to random stratified split."
        )
        return None

    rng = np.random.default_rng(random_state)
    df_sorted = df.sort_values("x_centroid").reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * test_size)
    offset = rng.integers(0, max(1, n_test))
    test_idx = np.arange(offset, n, max(1, n_test))[:n_test]
    test_mask = df_sorted.index.isin(test_idx)

    train_df = df_sorted[~test_mask]
    test_df = df_sorted[test_mask]

    if label_col in train_df.columns and label_col in test_df.columns:
        if train_df[label_col].nunique() < 2 or test_df[label_col].nunique() < 2:
            warnings.warn(
                "Spatial block split created a single-class subset. "
                "Falling back to random stratified split."
            )
            return None

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[label_col]
    y_test = test_df[label_col]
    id_train = train_df["id"]
    id_test = test_df["id"]
    return X_train, X_test, y_train, y_test, id_train, id_test


def train_random_forest(df: pd.DataFrame, feature_cols: list, label_col: str,
                         test_size: float = 0.2, random_state: int = 42,
                         rf_params: dict | None = None,
                         spatial_split: bool = False):
    if df is None or df.empty:
        raise ValueError("Provided training table is empty.")

    rf_params = rf_params or dict(n_estimators=300, max_depth=10, class_weight="balanced", n_jobs=-1)
    X, y = df[feature_cols], df[label_col]

    split_result = None
    if spatial_split:
        split_result = _spatial_block_split(df, feature_cols, label_col, test_size, random_state)

    if split_result is not None:
        X_train, X_test, y_train, y_test, id_train, id_test = split_result
    else:
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X, y, df["id"], test_size=test_size, stratify=stratify, random_state=random_state
        )

    df = df.copy()
    df["split"] = "train"
    df.loc[df["id"].isin(id_test), "split"] = "test"

    model = RandomForestClassifier(random_state=random_state, **rf_params)
    model.fit(X_train, y_train)

    return model, df, X_train, X_test, y_train, y_test, rf_params


def evaluate_binary_model(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    majority_class = int(pd.Series(y_test).mode()[0])
    baseline_acc = (y_test == majority_class).mean()
    class_dist = pd.Series(y_test).value_counts(normalize=True).to_dict()
    class_dist = {str(k): float(v) for k, v in class_dist.items()}

    return {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "baseline_accuracy": float(baseline_acc),
        "majority_class": majority_class,
        "class_distribution": class_dist,
    }


def evaluate_multiclass_model(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)

    majority_class = pd.Series(y_test).mode()[0]
    baseline_acc = (y_test == majority_class).mean()
    class_dist = pd.Series(y_test).value_counts(normalize=True).to_dict()
    class_dist = {str(k): float(v) for k, v in class_dist.items()}

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {k: v for k, v in report.items() if k in model.classes_},
        "confusion_matrix": confusion_matrix(y_test, pred, labels=model.classes_).tolist(),
        "class_labels": list(model.classes_),
        "baseline_accuracy": float(baseline_acc),
        "majority_class": str(majority_class),
        "class_distribution": class_dist,
    }


def run_modeling(training_table: pd.DataFrame, feature_cols: list, output_dir: str | Path,
                  label_col: str = "encroachment", model_name: str = "model",
                  task_type: str = "binary", rf_params: dict | None = None,
                  spatial_split: bool = False) -> dict:
    """Trains, evaluates, and exports one model."""
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    data_dir = output_dir / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    model, df, X_train, X_test, y_train, y_test, rf_params = train_random_forest(
        training_table, feature_cols, label_col=label_col, rf_params=rf_params,
        spatial_split=spatial_split,
    )

    if task_type == "multiclass":
        metrics = evaluate_multiclass_model(model, X_test, y_test)
    else:
        metrics = evaluate_binary_model(model, X_test, y_test)

    joblib.dump(model, models_dir / f"{model_name}.joblib")

    metadata = {
        "model_type": "RandomForestClassifier",
        "model_name": model_name,
        "task_type": task_type,
        "label_col": label_col,
        "features_used": feature_cols,
        "params": rf_params,
        "metrics": metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "spatial_split": spatial_split,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(models_dir / f"{model_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    df.to_csv(data_dir / f"{model_name}_predictions.csv", index=False)

    return {"model": model, "metrics": metrics, "training_table": df}