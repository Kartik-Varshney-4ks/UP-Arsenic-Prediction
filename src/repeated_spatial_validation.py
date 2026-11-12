from pathlib import Path

import numpy as np
import pandas as pd

from pyproj import CRS, Transformer

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

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
GRID_SIZES_KM = [25, 50]
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


def calculate_metrics(y_true, predictions, probabilities):
    result = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "PR_AUC_Average_Precision": average_precision_score(
            y_true, probabilities
        ),
    }

    if y_true.nunique() == 2:
        result["ROC_AUC"] = roc_auc_score(
            y_true, probabilities
        )
    else:
        result["ROC_AUC"] = np.nan

    return result


def main():
    df = pd.read_csv(DATA_PATH)

    required = [
        TARGET,
        "Latitude",
        "Longitude",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="raise"
    )

    df["Latitude"] = pd.to_numeric(
        df["Latitude"], errors="coerce"
    )

    df["Longitude"] = pd.to_numeric(
        df["Longitude"], errors="coerce"
    )

    df = df.dropna(
        subset=[TARGET, "Latitude", "Longitude"]
    ).reset_index(drop=True)

    y = df[TARGET].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            "The target must contain both binary classes."
        )

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

    all_results = []

    for grid_km in GRID_SIZES_KM:
        cell_size = grid_km * 1000

        cell_x = np.floor(
            x_coord / cell_size
        ).astype(int)

        cell_y = np.floor(
            y_coord / cell_size
        ).astype(int)

        groups = pd.Series(
            [
                f"{gx}_{gy}"
                for gx, gy in zip(cell_x, cell_y)
            ]
        )

        number_of_groups = groups.nunique()

        print(f"\n{'=' * 55}")
        print(f"GRID SIZE: {grid_km} km")
        print("Observations:", len(df))
        print("Unique geographic cells:", number_of_groups)

        if number_of_groups < N_SPLITS:
            print("Skipping: not enough geographic cells.")
            continue

        splitter = GroupKFold(
            n_splits=N_SPLITS
        )

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups=groups),
            start=1,
        ):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                print(
                    f"Fold {fold}: skipped because a split "
                    "contains only one target class."
                )
                continue

            model = build_model()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            metrics = calculate_metrics(
                y_test,
                predictions,
                probabilities,
            )

            row = {
                "Grid_km": grid_km,
                "Fold": fold,
                "Train_samples": len(train_idx),
                "Test_samples": len(test_idx),
                "Train_cells": groups.iloc[
                    train_idx
                ].nunique(),
                "Test_cells": groups.iloc[
                    test_idx
                ].nunique(),
                "Positive_rate_train": y_train.mean(),
                "Positive_rate_test": y_test.mean(),
                **metrics,
            }

            all_results.append(row)

            print(
                f"Fold {fold}: "
                f"F1={metrics['F1_Score']:.3f}, "
                f"Recall={metrics['Recall']:.3f}, "
                f"ROC-AUC={metrics['ROC_AUC']:.3f}"
            )

    if not all_results:
        raise RuntimeError(
            "No valid folds were evaluated. "
            "Inspect geographic grouping and class distributions."
        )

    results_df = pd.DataFrame(all_results)

    detailed_path = (
        OUTPUT_DIR / "repeated_spatial_fold_results.csv"
    )

    results_df.to_csv(
        detailed_path,
        index=False,
    )

    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
        "PR_AUC_Average_Precision",
    ]

    summary_df = (
        results_df.groupby("Grid_km")[metric_columns]
        .agg(["mean", "std"])
        .round(4)
    )

    summary_df.to_csv(
        OUTPUT_DIR / "repeated_spatial_summary.csv"
    )

    print("\n\nREPEATED SPATIAL VALIDATION SUMMARY")
    print(summary_df.to_string())

    print("\nDetailed fold results saved to:")
    print(detailed_path)

    print(
        "\nSummary saved to:",
        OUTPUT_DIR / "repeated_spatial_summary.csv",
    )

    print(
        "\nInterpretation warning: GroupKFold evaluates "
        "held-out geographic cells, but does not guarantee "
        "that every test cell is far from every training cell. "
        "The grid sizes and study extent must be justified."
    )


if __name__ == "__main__":
    main()