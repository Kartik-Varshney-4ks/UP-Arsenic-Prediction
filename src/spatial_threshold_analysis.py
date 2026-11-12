from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from pyproj import CRS, Transformer
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Arsenic_Classification_Dataset.csv"
)

OUTPUT_DIR = ROOT / "docs" / "spatial_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "arsenic_detected"
GRID_KM = 25
N_SPLITS = 5


def build_model():
    return Pipeline(
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


def main():
    df = pd.read_csv(DATA_PATH)

    required = [TARGET, "Latitude", "Longitude"]

    missing = [
        col for col in required if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for col in required:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna(subset=required).reset_index(drop=True)

    y = df[TARGET].astype(int)

    X = df.drop(columns=[TARGET]).apply(
        pd.to_numeric, errors="coerce"
    )

    X = X.dropna(axis=1, how="all")

    centre_lat = df["Latitude"].median()
    centre_lon = df["Longitude"].median()

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={centre_lat} "
        f"+lon_0={centre_lon} +datum=WGS84 "
        f"+units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True,
    )

    x_coord, y_coord = transformer.transform(
        df["Longitude"].to_numpy(),
        df["Latitude"].to_numpy(),
    )

    cell_size = GRID_KM * 1000

    cell_x = np.floor(
        np.asarray(x_coord) / cell_size
    ).astype(int)

    cell_y = np.floor(
        np.asarray(y_coord) / cell_size
    ).astype(int)

    groups = pd.Series(
        [
            f"{gx}_{gy}"
            for gx, gy in zip(cell_x, cell_y)
        ]
    )

    splitter = GroupKFold(n_splits=N_SPLITS)

    probabilities = np.full(len(df), np.nan)
    fold_ids = np.full(len(df), -1)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups),
        start=1,
    ):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() < 2:
            raise ValueError(
                f"Fold {fold} training set has only one class."
            )

        if y_test.nunique() < 2:
            raise ValueError(
                f"Fold {fold} test set has only one class. "
                "ROC-independent threshold analysis is not "
                "possible for this fold."
            )

        model = build_model()

        model.fit(
            X.iloc[train_idx],
            y_train,
        )

        probabilities[test_idx] = model.predict_proba(
            X.iloc[test_idx]
        )[:, 1]

        fold_ids[test_idx] = fold

        print(
            f"Completed fold {fold}: "
            f"{len(train_idx)} train, "
            f"{len(test_idx)} test observations"
        )

    if np.isnan(probabilities).any():
        raise RuntimeError(
            "Some observations do not have out-of-fold predictions."
        )

    threshold_rows = []

    thresholds = np.round(
        np.arange(0.10, 0.91, 0.05), 2
    )

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        ).ravel()

        threshold_rows.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(
                    y, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y, predictions, zero_division=0
                ),
                "F1_Score": f1_score(
                    y, predictions, zero_division=0
                ),
                "Balanced_Accuracy": balanced_accuracy_score(
                    y, predictions
                ),
                "True_Negatives": tn,
                "False_Positives": fp,
                "False_Negatives": fn,
                "True_Positives": tp,
            }
        )

    results = pd.DataFrame(threshold_rows)

    results.to_csv(
        OUTPUT_DIR / "spatial_threshold_results.csv",
        index=False,
    )

    prediction_df = pd.DataFrame(
        {
            "Latitude": df["Latitude"],
            "Longitude": df["Longitude"],
            "Actual": y,
            "Out_of_Fold_Probability": probabilities,
            "Fold": fold_ids,
        }
    )

    prediction_df.to_csv(
        OUTPUT_DIR / "spatial_oof_predictions.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        results["Threshold"],
        results["Precision"],
        marker="o",
        label="Precision",
    )

    plt.plot(
        results["Threshold"],
        results["Recall"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        results["Threshold"],
        results["F1_Score"],
        marker="o",
        label="F1-score",
    )

    plt.plot(
        results["Threshold"],
        results["Balanced_Accuracy"],
        marker="o",
        label="Balanced accuracy",
    )

    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.title(
        "Spatial Out-of-Fold Threshold Analysis (25 km)"
    )

    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "spatial_threshold_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    best_f1_row = results.loc[
        results["F1_Score"].idxmax()
    ]

    print("\nTHRESHOLD ANALYSIS")
    print(results.round(4).to_string(index=False))

    print("\nThreshold with highest pooled F1-score:")
    print(best_f1_row.to_string())

    print(
        "\nNote: These are pooled out-of-fold results. "
        "Threshold selection and final performance reporting "
        "should use separate validation and evaluation data "
        "to avoid optimistic estimates."
    )

    print("\nSaved files to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()