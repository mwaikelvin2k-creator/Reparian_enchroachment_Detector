# File: Modelling/modelling_utils.py
"""
Random Forest training pipeline for the Nairobi riparian encroachment
project. Handles the spatial split, training, evaluation, and persistence
for both the binary (built-up) phase and the later multiclass phase.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = ["B4", "B3", "B2", "B8", "ndvi", "ndwi", "distance_to_river_m"]

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BINARY_MODEL_PATH = MODELS_DIR / "rf_built_up_binary.pkl"
MULTICLASS_MODEL_PATH = MODELS_DIR / "rf_landcover_multiclass.pkl"


# ---------------------------------------------------------------------------
# Step 1 — Spatial train/test split
# ---------------------------------------------------------------------------

class SpatialSplitter:
    """Splits by tile_id (whole tiles go to train or test, never mixed),
    since adjacent pixels within a tile are highly correlated and a
    random row split would let the model 'cheat' by seeing near-duplicate
    neighbors of test pixels during training."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame, group_col: str = "tile_id"):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


# ---------------------------------------------------------------------------
# Step 2 — Class imbalance handling
# ---------------------------------------------------------------------------

class ClassBalancer:
    """Optional stratified subsampling of the majority class. Kept separate
    from class_weight so you can choose either approach, or both, per run."""

    def __init__(self, majority_ratio: float = 4.0, random_state: int = 42):
        self.majority_ratio = majority_ratio  # majority:minority ratio to keep
        self.random_state = random_state

    def subsample(self, df: pd.DataFrame, label_col: str = "built_up") -> pd.DataFrame:
        minority = df[df[label_col] == 1]
        majority = df[df[label_col] == 0]

        n_majority_keep = int(len(minority) * self.majority_ratio)
        n_majority_keep = min(n_majority_keep, len(majority))

        majority_sampled = majority.sample(n=n_majority_keep, random_state=self.random_state)
        combined = pd.concat([minority, majority_sampled])
        return combined.sample(frac=1, random_state=self.random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 — RF training and evaluation
# ---------------------------------------------------------------------------

class RFTrainer:
    """Wraps RandomForestClassifier training, evaluation, and feature
    importance reporting. Works for both the binary and multiclass label,
    since nothing here depends on how many classes there are."""

    def __init__(
        self,
        feature_columns: list[str] = FEATURE_COLUMNS,
        n_estimators: int = 200,
        max_depth: int | None = None,
        class_weight: str | None = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.feature_columns = feature_columns
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, train_df: pd.DataFrame, label_col: str):
        X_train = train_df[self.feature_columns]
        y_train = train_df[label_col]
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, test_df: pd.DataFrame, label_col: str) -> dict:
        X_test = test_df[self.feature_columns]
        y_test = test_df[label_col]
        y_pred = self.model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        print(classification_report(y_test, y_pred))
        print("Confusion matrix:\n", cm)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")

        return {"report": report, "confusion_matrix": cm}

    def feature_importance(self) -> pd.DataFrame:
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4 — Persistence (pickling)
# ---------------------------------------------------------------------------

class ModelPersister:
    """Saves/loads a trained model to disk. This is the 'pickling' step —
    it lets the trained model be reused later without retraining, which is
    a prerequisite for deployment but not deployment itself."""

    @staticmethod
    def save(model, path: str):
        joblib.dump(model, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path: str):
        return joblib.load(path)