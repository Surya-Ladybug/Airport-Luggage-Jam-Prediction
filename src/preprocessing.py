"""
preprocessing.py
-----------------
Loading, cleaning, and feature preparation for the luggage jam-prediction
project (Deliverable 3b - Design for Industry 4.0, TU Clausthal).

The functions in this module are intentionally kept simple and explicit,
since the goal of this project is an interpretable, easy-to-present
Random Forest classifier rather than a heavily engineered pipeline.
"""

import pandas as pd
import numpy as np


# ==========================================================
# CONSTANTS
# ==========================================================

ID_COLUMNS = ["domain", "timestamp", "cycle"]

LEAKAGE_COLUMNS = [
    "delay_s",
    "cycle_time_s",
    "energy_kwh",
    "throughput_bph",
    "quality_score",
    "efficiency_pct",
]

TARGET_COLUMN = "jam_event"

CATEGORICAL_COLUMNS = [
    "unit_id",
    "counter",
    "bag_type",
    "fault_type",
]


# ==========================================================
# DATA LOADING
# ==========================================================

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)


# ==========================================================
# EDA SUMMARY
# ==========================================================

def eda_summary(df: pd.DataFrame) -> dict:
    """Return summary statistics for EDA."""

    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str),
        "missing_values": df.isna().sum(),
        "missing_pct": (df.isna().sum() / len(df) * 100).round(2),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": df[TARGET_COLUMN].value_counts(),
        "class_distribution_pct": (
            df[TARGET_COLUMN].value_counts(normalize=True) * 100
        ).round(3),
        "describe_numeric": df.describe().T,
    }


# ==========================================================
# CLEANING
# ==========================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values and duplicates."""

    df = df.drop_duplicates().copy()

    sensor_cols = [c for c in df.columns if df[c].isna().any()]

    for col in sensor_cols:

        df[col] = df.groupby("unit_id")[col].transform(
            lambda s: s.fillna(s.median())
        )

        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features."""

    df = df.copy()

    df["bag_volume_m3"] = (
        df["bag_length_mm"]
        * df["bag_width_mm"]
        * df["bag_height_mm"]
    ) / 1e9

    df["weight_per_volume"] = (
        df["bag_weight_kg"]
        / df["bag_volume_m3"].replace(0, np.nan)
    )

    df["weight_per_volume"] = df["weight_per_volume"].fillna(
        df["weight_per_volume"].median()
    )

    safe_speed = df["belt_speed_mps"].replace(0, np.nan)

    df["gap_per_speed_s"] = (
        df["gap_to_prev_bag_mm"] / 1000
    ) / safe_speed

    df["gap_per_speed_s"] = df["gap_per_speed_s"].fillna(
        df["gap_per_speed_s"].median()
    )

    return df


# ==========================================================
# ENCODING
# ==========================================================

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""

    return pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=False,
    )


# ==========================================================
# TRAINING PREPROCESSING
# ==========================================================

def build_feature_table(df: pd.DataFrame):
    """
    Complete preprocessing pipeline used during training.

    Returns
    -------
    X : Feature dataframe
    y : Target series
    """

    df = clean_data(df)
    df = engineer_features(df)

    drop_cols = ID_COLUMNS + LEAKAGE_COLUMNS

    df = df.drop(
        columns=[c for c in drop_cols if c in df.columns]
    )

    df = encode_categoricals(df)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    return X, y


# ==========================================================
# PREDICTION PREPROCESSING
# ==========================================================

def prepare_prediction_input(user_input: dict, feature_names: list):
    """
    Prepare a single user input for prediction using the
    same preprocessing pipeline as the training data.
    """

    df = pd.DataFrame([user_input])

    df = engineer_features(df)

    drop_cols = ID_COLUMNS + LEAKAGE_COLUMNS

    df = df.drop(
        columns=[c for c in drop_cols if c in df.columns],
        errors="ignore",
    )

    df = encode_categoricals(df)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df.reindex(columns=feature_names, fill_value=0)

    return df