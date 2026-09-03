"""
Model training, evaluation, and export functions for the
riparian-encroachment pipeline. Operates on a training table produced by
preprocessing.run_preprocessing() — kept separate so a bad/empty table
shows up before any model-fitting is attempted.

Requires: pandas, scikit-learn, joblib
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
    """Stratified split + fit. Falls back to a plain split if the minority
    class is too small to stratify safely."""
    rf_params = rf_params or dict(n_estimators=300, max_depth=10,
                                   class_weight="balanced", n_jobs=-1)
    X, y = df[feature_cols], df[label_col]

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df["id"], test_size=test_size, stratify=stratify, random_state=random_state,
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


def export_model_artifacts(model, df: pd.DataFrame, feature_cols: list, metrics: dict,
                            rf_params: dict, buffer_m: float,
                            models_dir: Path, data_dir: Path) -> None:
    """Writes the model + metadata + predictions in the schema a dashboard reads."""
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "riparian_rf_model.joblib")

    X = df[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    predictions = pd.DataFrame({
        "id": df["id"], "split": df["split"], "y_true": df["encroachment"],
        "rf_pred": (proba >= 0.5).astype(int), "rf_proba": proba,
    })
    predictions.to_csv(data_dir / "rf_predictions.csv", index=False)

    labeled_cols = ["id", "split"] + feature_cols + ["centroid_dist_to_river_m", "encroachment"]
    df[labeled_cols].to_csv(data_dir / "building_features_labeled.csv", index=False)

    class_balance = {str(k): int(v) for k, v in df["encroachment"].value_counts().items()}
    metadata = {
        "model_type": "RandomForestClassifier",
        "params": rf_params,
        "feature_cols": feature_cols,
        "buffer_meters": buffer_m,
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "n_total": int(len(df)),
        "class_balance": class_balance,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {"encroachment": {k: metrics[k] for k in ("precision", "recall", "f1")},
                    "roc_auc": metrics["roc_auc"]},
        "feature_importances": dict(zip(feature_cols, map(float, model.feature_importances_))),
    }
    with open(models_dir / "rf_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def run_modeling(training_table: pd.DataFrame, feature_cols: list, buffer_m: float,
                  output_dir: str | Path, rf_params: dict | None = None) -> dict:
    """Train, evaluate, and export the model from an already-built training
    table."""
    output_dir = Path(output_dir)

    model, training_table, X_train, X_test, y_train, y_test, rf_params = train_random_forest(
        training_table, feature_cols, rf_params=rf_params
    )
    metrics = evaluate_model(model, X_test, y_test)

    export_model_artifacts(
        model, training_table, feature_cols, metrics, rf_params, buffer_m,
        models_dir=output_dir / "models", data_dir=output_dir / "data",
    )

    return {"model": model, "metrics": metrics, "training_table": training_table}
