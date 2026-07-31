"""
main.py
-------
Entry point for Deliverable 3b: "Predict jams and collisions at the
transfer points" - Airport Luggage Handling System
(Design for Industry 4.0, TU Clausthal, SoSe 2026).

Running this script end-to-end will:

1. Load the luggage dataset.
2. Run and print an Exploratory Data Analysis (EDA) summary.
3. Clean the data and engineer a small set of features.
4. Encode categorical variables and build the model-ready feature table.
5. Split the data into train/test sets (stratified).
6. Train a Random Forest classifier.
7. Evaluate the model (accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
   confusion matrix, classification report, feature importance).
8. Save all plots into the images/ folder.
9. Print a final results summary to the console.

Usage:
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from src.preprocessing import (
    load_data,
    eda_summary,
    clean_data,
    engineer_features,
    encode_categoricals,
    build_feature_table,
    prepare_prediction_input,
    TARGET_COLUMN,
)
from src.train_model import split_data, train_random_forest
from src.evaluate import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_numeric_distributions_by_class,
    compute_metrics,
)

DATA_PATH = "data/luggage_dataset.csv"


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_eda(df: pd.DataFrame):
    print_header("1. EXPLORATORY DATA ANALYSIS (EDA)")
    summary = eda_summary(df)

    print(f"\nDataset shape: {summary['shape'][0]} rows x {summary['shape'][1]} columns")

    print("\nVariable types:")
    print(summary["dtypes"])

    print("\nMissing values per column (only columns with missing values shown):")
    missing = summary["missing_values"]
    missing_pct = summary["missing_pct"]
    missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    print(missing_report[missing_report["missing_count"] > 0])

    print(f"\nDuplicate rows: {summary['duplicate_rows']}")

    print("\nClass distribution (jam_event):")
    print(summary["class_distribution"])
    print(summary["class_distribution_pct"])

    print("\nStatistical summary (numeric columns):")
    print(summary["describe_numeric"])

    return summary


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df_raw = load_data(DATA_PATH)

    # ------------------------------------------------------------------
    # 2. EDA
    # ------------------------------------------------------------------
    run_eda(df_raw)

    # EDA plots (before cleaning removes anything relevant to visualize)
    plot_class_distribution(df_raw[TARGET_COLUMN])

    numeric_df = df_raw.select_dtypes(include="number")
    plot_correlation_heatmap(numeric_df)

    plot_numeric_distributions_by_class(
        df_raw,
        columns=["bag_rigidity_factor", "gap_to_prev_bag_mm", "bag_weight_kg",
                 "inter_arrival_s", "belt_speed_mps", "vibration_rms_g"],
    )

    # ------------------------------------------------------------------
    # 3-6. Cleaning, feature engineering, encoding, feature table
    # ------------------------------------------------------------------
    print_header("2. DATA CLEANING & FEATURE ENGINEERING")
    df_clean = clean_data(df_raw)
    print(f"Rows after removing duplicates: {len(df_clean)}")
    print("Missing values remaining after imputation:", int(df_clean.isna().sum().sum()))

    df_features = engineer_features(df_clean)
    print("Engineered features added: bag_volume_m3, weight_per_volume, gap_per_speed_s")

    X, y = build_feature_table(df_raw)
    print(f"\nFinal feature table shape: {X.shape}")
    print(f"Number of features after one-hot encoding: {X.shape[1]}")

    # ------------------------------------------------------------------
    # 7. Train/test split
    # ------------------------------------------------------------------
    print_header("3. TRAIN / TEST SPLIT")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)
    print(f"Training set: {X_train.shape[0]} rows ({y_train.sum()} jam events)")
    print(f"Test set: {X_test.shape[0]} rows ({y_test.sum()} jam events)")

    # ------------------------------------------------------------------
    # 8. Train model
    # ------------------------------------------------------------------
    print_header("4. MODEL TRAINING (Random Forest)")
    model = train_random_forest(X_train, y_train, random_state=42)
    print("Random Forest trained with 300 trees, max_depth=12, class_weight='balanced'.")

    # ------------------------------------------------------------------
    # 9. Evaluation
    # ------------------------------------------------------------------
    print_header("5. MODEL EVALUATION")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Precision (jam=1)  : {metrics['precision']:.4f}")
    print(f"Recall (jam=1)     : {metrics['recall']:.4f}")
    print(f"F1 Score (jam=1)   : {metrics['f1_score']:.4f}")
    print(f"ROC AUC            : {metrics['roc_auc']:.4f}")
    print(f"Average Precision  : {metrics['average_precision']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    # ------------------------------------------------------------------
    # 10. Plots (confusion matrix, ROC, PR curve, feature importance)
    # ------------------------------------------------------------------
    print_header("6. GENERATING GRAPHS (saved to images/)")
    _, cm = plot_confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    _, auc = plot_roc_curve(y_test, y_proba)
    _, ap = plot_precision_recall_curve(y_test, y_proba)
    _, importances = plot_feature_importance(model, X.columns, top_n=15)

    print("\nTop 10 most important features:")
    print(importances.sort_values(ascending=False).head(10))

    print("\nAll graphs saved in the 'images/' folder:")
    print(" - class_distribution.png")
    print(" - correlation_heatmap.png")
    print(" - feature_distributions.png")
    print(" - confusion_matrix.png")
    print(" - roc_curve.png")
    print(" - precision_recall_curve.png")
    print(" - feature_importance.png")

    results = {
        "model": model,
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc_auc": auc,
        "average_precision": ap,
        "feature_importance": importances,
        "feature_names": list(X.columns),

        # Used by the Streamlit app
        "prepare_prediction_input": prepare_prediction_input,

        # Additional outputs
        "dataset_shape": X.shape,
        "raw_dataset": df_raw,
        "processed_dataset": df_features,
        "X_test": X_test,
        "y_test": y_test,
    }

    print_header("DONE")
    print("Deliverable 3b pipeline completed successfully.")

    return results


if __name__ == "__main__":
    main()