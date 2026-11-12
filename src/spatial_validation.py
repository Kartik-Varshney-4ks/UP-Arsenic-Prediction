from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "Arsenic_Classification_Dataset.csv"
)

OUTPUT_DIR = ROOT / "docs" / "spatial_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "arsenic_detected"
LAT = "Latitude"
LON = "Longitude"


def main():
    df = pd.read_csv(DATA_PATH)

    required = {TARGET, LAT, LON}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns are missing: {sorted(missing)}"
        )

    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    y = df[TARGET].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            "Expected both target classes, 0 and 1."
        )

    X = df.drop(columns=[TARGET]).apply(
        pd.to_numeric, errors="coerce"
    )

    X = X.dropna(axis=1, how="all")

    coordinates = df[[LAT, LON]].apply(
        pd.to_numeric, errors="coerce"
    )

    valid_coordinates = coordinates.notna().all(axis=1)

    df = df.loc[valid_coordinates].reset_index(drop=True)
    X = X.loc[valid_coordinates].reset_index(drop=True)
    y = y.loc[valid_coordinates].reset_index(drop=True)
    coordinates = coordinates.loc[
        valid_coordinates
    ].reset_index(drop=True)

    # Create approximate geographic groups.
    # The 0.25-degree grid is a preliminary choice, not a
    # universal or scientifically optimized spatial scale.
    df["spatial_group"] = (
        coordinates[LAT].round(2).astype(str)
        + "_"
        + coordinates[LON].round(2).astype(str)
    )

    groups = df["spatial_group"]

    if groups.nunique() < 5:
        raise ValueError(
            "Too few geographic groups for spatial validation."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("\nSPATIAL SPLIT SUMMARY")
    print("Total samples:", len(X))
    print("Training samples:", len(train_idx))
    print("Testing samples:", len(test_idx))
    print("Unique spatial groups:", groups.nunique())
    print("Training groups:", groups.iloc[train_idx].nunique())
    print("Testing groups:", groups.iloc[test_idx].nunique())
    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())
    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "One split contains only one class. "
            "Try a different grouping scale or split seed."
        )

    # This is a fresh model trained on the spatial training set.
    # Its parameters are a reasonable starting point, but should
    # ideally be tuned using spatially separated validation data.
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": "XGBoost",
        "Validation": "Spatial group holdout",
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC_Average_Precision": average_precision_score(
            y_test, probabilities
        ),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        OUTPUT_DIR / "spatial_validation_metrics.csv",
        index=False,
    )

    predictions_df = pd.DataFrame(
        {
            "Latitude": coordinates.iloc[test_idx][LAT].to_numpy(),
            "Longitude": coordinates.iloc[test_idx][LON].to_numpy(),
            "Actual": y_test.to_numpy(),
            "Predicted": predictions,
            "Predicted_Probability": probabilities,
            "Spatial_Group": groups.iloc[test_idx].to_numpy(),
        }
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "spatial_validation_predictions.csv",
        index=False,
    )

    confusion = confusion_matrix(y_test, predictions)
    pd.DataFrame(
        confusion,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(
        OUTPUT_DIR / "spatial_confusion_matrix.csv"
    )

    print("\nSPATIAL VALIDATION METRICS")
    print(metrics_df.round(4).to_string(index=False))

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("\nCONFUSION MATRIX")
    print(confusion)

    print("\nSaved files to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()