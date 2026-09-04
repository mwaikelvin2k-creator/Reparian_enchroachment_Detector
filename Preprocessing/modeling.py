"""
Modeling module for structure/land-cover feature classification.
Trains Random Forest classifiers on spectral profiles and evaluates
performance, for either a binary target (encroachment, is_building) or a
multi-class target (land_cover).
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, classification_report, confusion_matrix,
)


def train_random_forest(df: pd.DataFrame, feature_cols: list, label_col: str,
                         test_size: float = 0.2, random_state: int = 42,
                         rf_params: dict | None = None):
    if df is None or df.empty:
        raise ValueError("Provided training table is empty.")

    rf_params = rf_params or dict(n_estimators=300, max_depth=10, class_weight="balanced", n_jobs=-1)
    X, y = df[feature_cols], df[label_col]

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
    return {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def evaluate_multiclass_model(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {k: v for k, v in report.items() if k in model.classes_},
        "confusion_matrix": confusion_matrix(y_test, pred, labels=model.classes_).tolist(),
        "class_labels": list(model.classes_),
    }


def run_modeling(training_table: pd.DataFrame, feature_cols: list, output_dir: str | Path,
                  label_col: str = "encroachment", model_name: str = "model",
                  task_type: str = "binary", rf_params: dict | None = None) -> dict:
    """Trains, evaluates, and exports one model. task_type is 'binary' or
    'multiclass' and selects which evaluation function runs. model_name
    keeps each task's artifacts from overwriting another's in the same
    output_dir."""
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    data_dir = output_dir / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    model, df, X_train, X_test, y_train, y_test, rf_params = train_random_forest(
        training_table, feature_cols, label_col=label_col, rf_params=rf_params
    )

    if task_type == "multiclass":
        metrics = evaluate_multiclass_model(model, X_test, y_test)
    else:
        metrics = evaluate_binary_model(model, X_test, y_test)

    joblib.dump(model, models_dir / f"{model_name}.joblib")

    metadata = {
        "model_type": "RandomForestClassifier",
        "task_type": task_type,
        "label_col": label_col,
        "features_used": feature_cols,
        "params": rf_params,
        "metrics": metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(models_dir / f"{model_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    df.to_csv(data_dir / f"{model_name}_predictions.csv", index=False)

    return {"model": model, "metrics": metrics, "training_table": df}