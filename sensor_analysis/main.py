"""
anomaly_classifier_rf
=====================

This module provides a simple example of how to train a machine learning model to
detect anomalous events in a rocket telemetry dataset using a Random Forest
classifier.  The dataset is stored in ``dtl1.csv`` and contains sensor
measurements taken during a rocket flight along with several event marker
columns (e.g., ``LDA``, ``Apogee``, ``N-O``, ``Drogue``, ``Main``, ``AUX``).

An event marker column is ``0`` for all normal rows and takes a non‑zero value
when a specific event has occurred.  Because there are only four rows in the
entire dataset where an event happens, the data is extremely imbalanced.  To
teach the classifier to recognise these rare anomalies we oversample the
positive class by duplicating the anomalous rows multiple times before
training.  Without oversampling the model will almost always predict the
majority (normal) class and never flag an anomaly.

The steps performed by this script are:

* Load the telemetry CSV file into a pandas DataFrame.
* Create a binary ``Anomaly`` label that is ``1`` if any of the event
  indicator columns is non‑zero and ``0`` otherwise.
* Oversample the anomaly rows by repeating them ``N_OVERSAMPLES`` times.
* Split the data into training and testing sets.
* Fit a RandomForestClassifier to the training data.
* Evaluate the model on the test set and on the original (non‑oversampled)
  dataset.
* Print the classification metrics and feature importances.

```
uv run main.py
```

"""

import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the telemetry data and create an ``Anomaly`` column.

    Parameters
    ----------
    csv_path: str
        Path to the ``dtl1.csv`` file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the original columns plus a binary
        ``Anomaly`` indicator.
    """
    df = pd.read_csv("./dtl1.csv")

    # Identify whether any of the event marker columns is non‑zero on each row.
    event_columns = ["LDA", "Apogee", "N-O", "Drogue", "Main", "AUX"]
    df["Anomaly"] = (df[event_columns].sum(axis=1) != 0).astype(int)
    return df


def oversample_anomalies(df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
    """Duplicate the anomaly rows to balance the dataset.

    Because the dataset only contains four anomalous rows, the training
    distribution will be highly skewed towards normal observations.  To
    mitigate this we duplicate each anomaly row ``n_samples`` times so
    that the positive class appears in the training data frequently enough
    for the Random Forest to learn meaningful patterns.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame containing the telemetry data with an ``Anomaly`` column.
    n_samples: int, default ``100``
        The number of times to repeat each anomaly row.  Setting this to a
        larger value increases the representation of anomalies in the training
        set.  For example, if there are four anomalous rows and
        ``n_samples=100``, the oversampled dataset will contain 400 anomalous
        rows in addition to all of the normal rows.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the original normal rows and the
        oversampled anomaly rows.
    """
    anomalies = df[df["Anomaly"] == 1]
    normals = df[df["Anomaly"] == 0]

    # Duplicate the anomaly rows n_samples times.  Resetting the index
    # prevents duplicate indices when concatenating.
    anomalies_oversampled = pd.concat([anomalies] * n_samples, ignore_index=True)

    # Combine the oversampled anomalies with the normal observations.
    balanced_df = pd.concat([normals, anomalies_oversampled], ignore_index=True)
    return balanced_df


def train_random_forest(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> RandomForestClassifier:
    """Train a RandomForestClassifier on the provided data.

    Parameters
    ----------
    X: pd.DataFrame
        The feature matrix used to train the model.
    y: pd.Series
        The binary target labels corresponding to each row in ``X``.
    random_state: int, default ``42``
        A seed to control the randomness of the forest, ensuring
        reproducibility.

    Returns
    -------
    RandomForestClassifier
        The fitted classifier.
    """
    clf = RandomForestClassifier(
        n_estimators=200, random_state=random_state
    )
    clf.fit(X, y)
    return clf


def evaluate_model(
    clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """Print evaluation metrics for a trained classifier on test data.

    Parameters
    ----------
    clf: RandomForestClassifier
        The trained model to evaluate.
    X_test: pd.DataFrame
        The feature matrix for testing.
    y_test: pd.Series
        The ground truth labels for ``X_test``.
    """
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Test set accuracy: {:.4f}".format(acc))
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)


def evaluate_on_original(
    clf: RandomForestClassifier, df: pd.DataFrame
) -> None:
    """Evaluate the model on the original (non‑oversampled) dataset.

    This function provides insight into how the model performs when exposed
    to the extremely imbalanced real data.  Because the oversampled model
    tends to overpredict anomalies to compensate for the prior imbalance,
    you may observe a few false positives.

    Parameters
    ----------
    clf: RandomForestClassifier
        The trained classifier.
    df: pd.DataFrame
        The original DataFrame with features and the ``Anomaly`` label.
    """
    feature_cols = ["T", "Alt", "FAlt", "FVeloc"]
    X_orig = df[feature_cols]
    y_orig = df["Anomaly"]
    y_pred_orig = clf.predict(X_orig)

    cm = confusion_matrix(y_orig, y_pred_orig)
    report = classification_report(y_orig, y_pred_orig)

    print("\nEvaluation on original dataset:")
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)


def main(csv_path: str, oversample_factor: int = 100) -> None:
    """Run the full training and evaluation pipeline.

    Parameters
    ----------
    csv_path: str
        Path to the telemetry CSV file.
    oversample_factor: int, default ``100``
        The number of times to duplicate anomaly rows for oversampling.
    """
    # Load and prepare the data
    df = load_and_prepare_data(csv_path)

    # Oversample anomalies to create a more balanced training set
    balanced_df = oversample_anomalies(df, oversample_factor)

    feature_cols = ["T", "Alt", "FAlt", "FVeloc"]
    X = balanced_df[feature_cols]
    y = balanced_df["Anomaly"]

    # Split into training and test subsets.  Using stratify on y ensures that
    # the test set contains both normal and oversampled anomaly samples.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train the Random Forest classifier
    clf = train_random_forest(X_train, y_train)

    # Evaluate on the oversampled test set
    evaluate_model(clf, X_test, y_test)

    # Evaluate on the original dataset to see how the model behaves in
    # practice
    evaluate_on_original(clf, df)

    # Print feature importances to interpret which sensors influence the
    # classifier most.  Larger values indicate more influential features.
    print("\nFeature importances:")
    for feature, importance in zip(feature_cols, clf.feature_importances_):
        print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a random forest classifier to detect anomalies in rocket telemetry data."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="dtl1.csv",
        help="Path to the dtl1.csv dataset.  Defaults to 'dtl1.csv' in the current directory."
    )
    parser.add_argument(
        "--oversample", type=int, default=100, help="Factor by which to duplicate anomaly rows."
    )
    args = parser.parse_args()
    main(args.csv_path, args.oversample)
