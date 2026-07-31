"""
train_model.py
---------------
Model selection, training, and the train/test split used for the
jam-prediction Random Forest classifier.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def split_data(X, y, test_size: float = 0.25, random_state: int = 42):
    """
    Stratified train/test split.

    Stratification on the target is important here because jam_event is
    heavily imbalanced (roughly 0.6% positive class). A plain random split
    could easily leave the test set with too few positive examples to
    evaluate reliably.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Why Random Forest (as preferred by the assignment brief):

    - Interpretability: feature importances and individual trees can be
      explained directly in a presentation, unlike e.g. gradient boosting
      internals or neural networks.
    - Robustness to the mixed feature types in this dataset (continuous
      sensor readings, geometric measurements, and one-hot encoded
      categories) without requiring feature scaling.
    - Handles non-linear interactions (e.g. "heavy AND rigid AND small
      gap") naturally, which matters here since jams are described in the
      brief as arising from a combination of factors rather than any
      single variable.
    - class_weight='balanced' lets the forest compensate for the strong
      class imbalance (~0.6% jam events) without needing synthetic
      oversampling (e.g. SMOTE), keeping the pipeline simple.

    n_estimators=300 and a modest max_depth are used to keep the model
    stable without overfitting to the rare positive class.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model