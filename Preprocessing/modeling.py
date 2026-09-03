"""
Modeling module for structure/land-cover feature classification.
Trains Random Forest classifiers on spectral profiles and evaluates performance.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def train_random_forest(df: pd.DataFrame, feature_cols: list, label_col: str = "encroachment",
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

    df["split"] = "train"
    df.loc[df["id"].isin(id_test), "split"] = "test"

    model = RandomForestClassifier(random_state=random_state, **rf_params)
    model.fit(X_train, y_train)

    return model, df, X_train, X_test, y_train, y_test, rf_params


def evaluate_model(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def run_modeling(training_table: pd.DataFrame, feature_cols: list, buffer_m: float,
                  output_dir: str | Path, rf_params: dict | None = None) -> dict:
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    data_dir = output_dir / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    model, df, X_train, X_test, y_train, y_test, rf_params = train_random_forest(
        training_table, feature_cols, rf_params=rf_params
    )
    metrics = evaluate_model(model, X_test, y_test)

    # Export Model & Artifacts
    joblib.dump(model, models_dir / "spectral_rf_model.joblib")
    
    metadata = {
        "model_type": "RandomForestClassifier",
        "features_used": feature_cols,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {"model": model, "metrics": metrics, "training_table": df}