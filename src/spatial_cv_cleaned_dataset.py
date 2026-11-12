from pathlib import Path

import numpy as np
import pandas as pd

from pyproj import CRS, Transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "Arsenic_Spatial_Model_Dataset.csv"
)

OUTPUT_DIR = ROOT / "docs" / "spatial_validation"

TARGET = "arsenic_detected"
GRID_SIZE_KM = 25
N_SPLITS = 5
RANDOM_STATE = 42


def make_spatial_groups(df):
    center_lat = float(df["Latitude"].median())
    center_lon = float(df["Longitude"].median())

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} "
        f"+lon_0={center_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True,
    )

    x_m, y_m = transformer.transform(
        df["Longitude"].to_numpy(),
        df["Latitude"].to_numpy(),
    )

    grid_size_m = GRID_SIZE_KM * 1000

    grid_x = np.floor(
        (x_m - np.min(x_m)) / grid_size_m
    ).astype(int)

    grid_y = np.floor(
        (y_m - np.min(y_m)) / grid_size_m
    ).astype(int)

    grid_pairs = pd.DataFrame({
        "grid_x": grid_x,
        "grid_y": grid_y,
    })

    groups = pd.factorize(
        pd.MultiIndex.from_frame(grid_pairs)
    )[0]

    return groups


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required = [
        TARGET,
        "Latitude",
        "Longitude",
    ]

    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Target or coordinate columns contain missing values."
        )

    y = df[TARGET].astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    # Coordinates are used to construct groups.
    # They are excluded from model predictors.
    excluded = {
        TARGET,
        "Latitude",
        "Longitude",
    }

    feature_columns = [
        col for col in df.columns if col not in excluded
    ]

    X = df[feature_columns].copy()
    groups = make_spatial_groups(df)

    unique_groups = np.unique(groups)

    if len(unique_groups) < N_SPLITS:
        raise ValueError(
            f"Only {len(unique_groups)} spatial groups available."
        )

    print("\n===== SPATIAL CROSS-VALIDATION =====")
    print("Dataset rows:", len(df))
    print("Predictor features:", len(feature_columns))
    print("Spatial grid size:", GRID_SIZE_KM, "km")
    print("Unique spatial groups:", len(unique_groups))
    print("Cross-validation folds:", N_SPLITS)

    cv = GroupKFold(n_splits=N_SPLITS)

    fold_results = []
    oof_records = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"Skipping fold {fold}: only one target class.")
            continue

        model = build_model()
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "Fold": fold,
            "Train_Rows": len(train_idx),
            "Test_Rows": len(test_idx),
            "Test_Spatial_Groups": len(
                np.unique(groups[test_idx])
            ),
            "Accuracy": accuracy_score(
                y_test, predictions
            ),
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
            "ROC_AUC": roc_auc_score(
                y_test, probabilities
            ),
            "PR_AUC_AP": average_precision_score(
                y_test, probabilities
            ),
        }

        fold_results.append(metrics)

        oof_records.append(pd.DataFrame({
            "Row_Index": test_idx,
            "Fold": fold,
            "Actual": y_test.to_numpy(),
            "Predicted": predictions,
            "Probability": probabilities,
        }))

        print(
            f"Fold {fold}: "
            f"Accuracy={metrics['Accuracy']:.4f}, "
            f"F1={metrics['F1_Score']:.4f}, "
            f"Recall={metrics['Recall']:.4f}, "
            f"ROC-AUC={metrics['ROC_AUC']:.4f}, "
            f"PR-AUC={metrics['PR_AUC_AP']:.4f}"
        )

    if not fold_results:
        raise RuntimeError("No valid folds were evaluated.")

    results_df = pd.DataFrame(fold_results)

    results_df.to_csv(
        OUTPUT_DIR / "cleaned_spatial_cv_fold_results.csv",
        index=False,
    )

    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
        "PR_AUC_AP",
    ]

    summary = pd.DataFrame({
        "Metric": metric_columns,
        "Mean": [
            results_df[col].mean()
            for col in metric_columns
        ],
        "Std": [
            results_df[col].std(ddof=1)
            if len(results_df) > 1 else 0.0
            for col in metric_columns
        ],
    })

    summary.to_csv(
        OUTPUT_DIR / "cleaned_spatial_cv_summary.csv",
        index=False,
    )

    pd.concat(
        oof_records,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "cleaned_spatial_cv_oof_predictions.csv",
        index=False,
    )

    print("\n===== CROSS-VALIDATION SUMMARY =====")

    for _, row in summary.iterrows():
        print(
            f"{row['Metric']}: "
            f"{row['Mean']:.4f} +/- {row['Std']:.4f}"
        )

    print("\nResults saved in:")
    print(OUTPUT_DIR)

    print(
        "\nNote: GroupKFold holds out spatial grid cells, "
        "but does not guarantee a minimum distance between "
        "training and test observations."
    )


if __name__ == "__main__":
    main()