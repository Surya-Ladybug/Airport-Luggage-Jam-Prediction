"""
evaluate.py
-----------
Model evaluation and all plotting functions for the jam-prediction
project. Every plotting function saves its figure into the images/
folder and also returns the matplotlib Figure object (useful if a
notebook / interactive session wants to display it too).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

sns.set_theme(style="whitegrid", context="talk")

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")


def _ensure_images_dir():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def plot_class_distribution(y, save_name="class_distribution.png"):
    """Bar chart of the (imbalanced) target class distribution."""
    _ensure_images_dir()

    counts = y.value_counts().sort_index()
    labels = ["No jam (0)", "Jam (1)"]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(
        labels,
        counts.values,
        color=["#4C72B0", "#C44E52"],
        width=0.55
    )

    ymax = counts.max()

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + ymax * 0.01,
            f"{val:,}\n({val/len(y)*100:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

    ax.set_title(
        "Class Distribution of Jam Events",
        fontsize=18,
        fontweight="bold",
        pad=18
    )

    ax.set_ylabel("Number of Records", fontsize=14)
    ax.set_ylim(0, ymax * 1.12)

    fig.tight_layout()

    fig.savefig(
        os.path.join(IMAGES_DIR, save_name),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return fig


def plot_correlation_heatmap(df_numeric,
                             save_name="correlation_heatmap.png",
                             top_n=10):
    """
    Correlation heatmap of the top variables most correlated with jam_event.
    """

    _ensure_images_dir()

    corr = df_numeric.corr(numeric_only=True)

    top_features = (
        corr["jam_event"]
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    corr_subset = df_numeric[top_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_subset,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size":10},
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap (Top 10 Features vs. jam_event)",
        fontsize=17,
        fontweight="bold",
        pad=18
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()

    fig.savefig(
        os.path.join(IMAGES_DIR, save_name),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return fig


def plot_feature_importance(model, feature_names, save_name="feature_importance.png", top_n=15):
    """Horizontal bar chart of the top_n Random Forest feature importances."""
    _ensure_images_dir()
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    importances.sort_values().plot(kind="barh", ax=ax, color="#55A868")
    ax.set_title(f"Random Forest Feature Importance (Top {top_n})")
    ax.set_xlabel("Importance (Gini)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, save_name), dpi=150)
    plt.close(fig)
    return fig, importances


def plot_confusion_matrix(y_true, y_pred, save_name="confusion_matrix.png"):
    """Confusion matrix heatmap (counts)."""
    _ensure_images_dir()
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No jam", "Jam"], yticklabels=["No jam", "Jam"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, save_name), dpi=150)
    plt.close(fig)
    return fig, cm


def plot_roc_curve(y_true, y_proba, save_name="roc_curve.png"):
    """ROC curve with AUC annotation."""
    _ensure_images_dir()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#C44E52", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Jam Event Prediction")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, save_name), dpi=150)
    plt.close(fig)
    return fig, auc


def plot_precision_recall_curve(y_true, y_proba, save_name="precision_recall_curve.png"):
    """
    Precision-Recall curve. This is included in addition to the ROC curve
    because jam_event is a rare-event (~0.6% positive) problem, where PR
    curves are considerably more informative than ROC curves about
    real-world performance.
    """
    _ensure_images_dir()
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#4C72B0", lw=2, label=f"PR curve (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Jam Event Prediction")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, save_name), dpi=150)
    plt.close(fig)
    return fig, ap


def plot_numeric_distributions_by_class(df, columns, save_name="feature_distributions.png"):
    """
    Small grid of boxplots comparing key engineering variables (e.g. gap,
    rigidity, weight) between jam and non-jam records, to support the
    discussion of which variables are most influential.
    """
    _ensure_images_dir()
    n = len(columns)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(columns):
        sns.boxplot(x="jam_event", y=col, data=df, ax=axes[i], palette=["#4C72B0", "#C44E52"])
        axes[i].set_xlabel("jam_event")
        axes[i].set_title(col)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Key Feature Distributions by Jam Outcome", y=1.02, fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, save_name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute the full set of evaluation metrics required by the brief."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "classification_report": classification_report(y_true, y_pred, target_names=["No jam", "Jam"], zero_division=0),
    }
    return metrics