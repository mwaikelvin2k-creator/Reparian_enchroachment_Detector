"""
Visualization module for model performance reporting.
Reads artifacts produced by modeling.py (metadata JSON, predictions CSV,
model joblib) and generates publication-ready figures.

Supports both binary (encroachment, building detection) and multiclass
(land cover) tasks.
"""

from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.facecolor"] = "white"

CMAP_BINARY = "Blues"
CMAP_MULTICLASS = "YlOrRd"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_artifacts(output_dir: str | Path, model_name: str) -> dict:
    """Load metadata JSON, predictions CSV, and joblib model."""
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    data_dir = output_dir / "data"

    meta_path = models_dir / f"{model_name}_metadata.json"
    pred_path = data_dir / f"{model_name}_predictions.csv"
    model_path = models_dir / f"{model_name}.joblib"

    artifacts = {}
    if meta_path.exists():
        with open(meta_path) as f:
            artifacts["metadata"] = json.load(f)
    else:
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    if pred_path.exists():
        artifacts["predictions"] = pd.read_csv(pred_path)
    else:
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    if model_path.exists():
        artifacts["model"] = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    return artifacts


# ---------------------------------------------------------------------------
# Binary-task plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray,
                                  ax: plt.Axes | None = None,
                                  title: str = "Confusion Matrix") -> plt.Axes:
    """Normalised + raw confusion matrix for a binary task."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap=CMAP_BINARY,
                xticklabels=["Non-encroaching", "Encroaching"],
                yticklabels=["Non-encroaching", "Encroaching"],
                vmin=0, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   ax: plt.Axes | None = None,
                   title: str = "ROC Curve") -> plt.Axes:
    """ROC curve with AUC annotation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = np.trapz(tpr, fpr)

    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_precision_recall(y_true: np.ndarray, y_proba: np.ndarray,
                          ax: plt.Axes | None = None,
                          title: str = "Precision–Recall Curve") -> plt.Axes:
    """Precision–Recall curve with AP annotation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()

    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(baseline, color="k", linestyle="--", lw=1, alpha=0.5,
               label=f"Baseline = {baseline:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    return ax


def plot_feature_importance(model, feature_cols: list,
                            ax: plt.Axes | None = None,
                            title: str = "Feature Importance",
                            top_n: int = 15) -> plt.Axes:
    """Horizontal bar chart of Gini importance (MDI)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    y_pos = np.arange(len(idx))

    ax.barh(y_pos, importances[idx], align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_cols[i] for i in idx])
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Impurity (Gini)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_probability_distribution(y_true: np.ndarray, y_proba: np.ndarray,
                                  ax: plt.Axes | None = None,
                                  title: str = "Prediction Probability Distribution") -> plt.Axes:
    """Histogram of predicted probabilities by true class."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.hist(y_proba[y_true == 0], bins=30, alpha=0.6, label="Non-encroaching (0)", color="steelblue")
    ax.hist(y_proba[y_true == 1], bins=30, alpha=0.6, label="Encroaching (1)", color="coral")
    ax.axvline(0.5, color="black", linestyle="--", lw=1, label="Threshold = 0.5")
    ax.set_xlabel("Predicted Probability (Class 1)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Multiclass-task plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix_multiclass(y_true: np.ndarray, y_pred: np.ndarray,
                                     class_labels: list,
                                     ax: plt.Axes | None = None,
                                     title: str = "Confusion Matrix") -> plt.Axes:
    """Normalised confusion matrix for multiclass."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap=CMAP_MULTICLASS,
                xticklabels=class_labels, yticklabels=class_labels,
                vmin=0, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_per_class_metrics(per_class: dict, ax: plt.Axes | None = None,
                           title: str = "Per-Class Metrics") -> plt.Axes:
    """Grouped bar chart of precision / recall / f1 per class."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    classes = list(per_class.keys())
    precision = [per_class[c]["precision"] for c in classes]
    recall = [per_class[c]["recall"] for c in classes]
    f1 = [per_class[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    ax.bar(x - width, precision, width, label="Precision", color="steelblue")
    ax.bar(x, recall, width, label="Recall", color="seagreen")
    ax.bar(x + width, f1, width, label="F1", color="coral")

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# High-level report builders
# ---------------------------------------------------------------------------

def build_binary_report(artifacts: dict, save_path: str | Path | None = None) -> plt.Figure:
    """2x2 grid: confusion matrix, ROC, PR curve, feature importance."""
    meta = artifacts["metadata"]
    df = artifacts["predictions"]
    model = artifacts["model"]
    feature_cols = meta["features_used"]

    test_df = df[df["split"] == "test"]
    y_true = test_df[meta["label_col"]].values
    X_test = test_df[feature_cols].values
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_confusion_matrix_binary(y_true, y_pred, ax=axes[0, 0],
                                  title=f"Confusion Matrix — {meta['model_name']}")
    plot_roc_curve(y_true, y_proba, ax=axes[0, 1],
                   title=f"ROC Curve (AUC = {meta['metrics']['roc_auc']:.3f})")
    plot_precision_recall(y_true, y_proba, ax=axes[1, 0],
                          title="Precision–Recall Curve")
    plot_feature_importance(model, feature_cols, ax=axes[1, 1],
                            title="Feature Importance (MDI)")

    fig.suptitle(f"Model Performance Report — {meta['model_name']}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved binary report to {save_path}")
    return fig


def build_multiclass_report(artifacts: dict, save_path: str | Path | None = None) -> plt.Figure:
    """2x2 grid: confusion matrix, per-class metrics, feature importance, summary stats."""
    meta = artifacts["metadata"]
    df = artifacts["predictions"]
    model = artifacts["model"]
    feature_cols = meta["features_used"]
    class_labels = meta["metrics"]["class_labels"]

    test_df = df[df["split"] == "test"]
    y_true = test_df[meta["label_col"]].values
    X_test = test_df[feature_cols].values
    y_pred = model.predict(X_test)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_confusion_matrix_multiclass(y_true, y_pred, class_labels, ax=axes[0, 0],
                                      title=f"Confusion Matrix — {meta['model_name']}")
    plot_per_class_metrics(meta["metrics"]["per_class"], ax=axes[0, 1],
                           title="Per-Class Precision / Recall / F1")
    plot_feature_importance(model, feature_cols, ax=axes[1, 0],
                            title="Feature Importance (MDI)")

    # Summary text panel
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = (
        f"Overall Accuracy: {meta['metrics']['accuracy']:.3f}\n"
        f"Macro Precision:  {meta['metrics']['macro_precision']:.3f}\n"
        f"Macro Recall:     {meta['metrics']['macro_recall']:.3f}\n"
        f"Macro F1:         {meta['metrics']['macro_f1']:.3f}\n\n"
        f"Training samples: {meta['n_train']}\n"
        f"Test samples:     {meta['n_test']}\n"
        f"Features used:    {len(feature_cols)}\n"
        f"Model:            {meta['model_type']}"
    )
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    ax.set_title("Summary Statistics")

    fig.suptitle(f"Model Performance Report — {meta['model_name']}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved multiclass report to {save_path}")
    return fig


def build_all_reports(output_dir: str | Path,
                      model_names: list[str] | None = None) -> dict[str, plt.Figure]:
    """Convenience wrapper that discovers the three standard models and
    builds a report for each.  Returns a dict of figure handles."""
    output_dir = Path(output_dir)
    if model_names is None:
        model_names = ["encroachment_rf", "building_detector_rf", "land_cover_rf"]

    figures = {}
    for name in model_names:
        try:
            artifacts = load_artifacts(output_dir, name)
            task_type = artifacts["metadata"]["task_type"]
            save_path = output_dir / "figures" / f"{name}_report.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if task_type == "multiclass":
                fig = build_multiclass_report(artifacts, save_path=save_path)
            else:
                fig = build_binary_report(artifacts, save_path=save_path)
            figures[name] = fig
        except FileNotFoundError as exc:
            print(f"Skipping {name}: {exc}")

    return figures


# ---------------------------------------------------------------------------
# CLI / direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <output_dir> [model_name]")
        print("Example: python visualize.py ./output/kasarani encroachment_rf")
        sys.exit(1)

    out_dir = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else None

    if name:
        artifacts = load_artifacts(out_dir, name)
        task = artifacts["metadata"]["task_type"]
        if task == "multiclass":
            build_multiclass_report(artifacts)
        else:
            build_binary_report(artifacts)
        plt.show()
    else:
        figs = build_all_reports(out_dir)
        for name, fig in figs.items():
            print(f"Rendered report for {name}")
        plt.show()