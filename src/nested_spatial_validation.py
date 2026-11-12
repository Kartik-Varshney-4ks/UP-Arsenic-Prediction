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
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Arsenic_Spatial_Model_Dataset.csv"
)

OUTPUT_DIR = ROOT / "docs" / "nested_spatial_validation"

TARGET = "arsenic_detected"
GRID_SIZE_KM = 25

OUTER_FOLDS = 5
INNER_FOLDS = 4

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

RANDOM_STATE = 42


# ============================================================
# SPATIAL GROUPING
# ============================================================

def create_spatial_groups(df):
    """
    Create projected square grid groups.

    Coordinates determine group membership but are excluded
    from the model predictors.
    """

    required = ["Latitude", "Longitude"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing coordinate column: {col}")

    if df[required].isna().any().any():
        raise ValueError("Coordinates contain missing values.")

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

    grid_cells = pd.DataFrame({
        "grid_x": grid_x,
        "grid_y": grid_y,
    })

    groups = pd.factorize(
        pd.MultiIndex.from_frame(grid_cells)
    )[0]

    return groups


# ============================================================
# MODEL
# ============================================================

def build_model():
    """
    Imputation is fitted within each training fold.
    """

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
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


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(
            y_true, predictions
        ),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1_Score": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Specificity": (
            tn / (tn + fp) if (tn + fp) else 0.0
        ),
        "ROC_AUC": roc_auc_score(
            y_true, probabilities
        ),
        "PR_AUC_AP": average_precision_score(
            y_true, probabilities
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# INNER SPATIAL VALIDATION: SELECT THRESHOLD
# ============================================================

def select_threshold(X, y, groups):
    """
    Generate inner out-of-fold predictions, then choose the
    threshold with the highest mean inner-fold F1-score.

    Threshold selection occurs only inside the outer training
    data. The outer test fold is not used for selection.
    """

    unique_groups = np.unique(groups)

    if len(unique_groups) < INNER_FOLDS:
        raise ValueError(
            "Insufficient inner spatial groups."
        )

    inner_cv = GroupKFold(
        n_splits=INNER_FOLDS
    )

    fold_probabilities = []
    fold_labels = []
    fold_ids = []

    for inner_fold, (train_idx, valid_idx) in enumerate(
        inner_cv.split(X, y, groups),
        start=1,
    ):
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        if y_train.nunique() < 2:
            raise ValueError(
                f"Inner fold {inner_fold} training set "
                "contains only one class."
            )

        model = build_model()

        model.fit(
            X.iloc[train_idx],
            y_train,
        )

        probabilities = model.predict_proba(
            X.iloc[valid_idx]
        )[:, 1]

        fold_probabilities.append(probabilities)
        fold_labels.append(y_valid.to_numpy())

        fold_ids.append(
            np.full(len(valid_idx), inner_fold)
        )

    all_probabilities = np.concatenate(
        fold_probabilities
    )

    all_labels = np.concatenate(
        fold_labels
    )

    all_fold_ids = np.concatenate(
        fold_ids
    )

    threshold_scores = []

    for threshold in THRESHOLDS:
        fold_f1_scores = []

        for fold_id in range(1, INNER_FOLDS + 1):
            mask = all_fold_ids == fold_id

            fold_f1_scores.append(
                f1_score(
                    all_labels[mask],
                    (
                        all_probabilities[mask]
                        >= threshold
                    ).astype(int),
                    zero_division=0,
                )
            )

        threshold_scores.append({
            "Threshold": threshold,
            "Mean_Inner_F1": float(
                np.mean(fold_f1_scores)
            ),
        })

    threshold_df = pd.DataFrame(
        threshold_scores
    )

    # Deterministic tie-breaking: choose the larger threshold.
    threshold_df = threshold_df.sort_values(
        ["Mean_Inner_F1", "Threshold"],
        ascending=[False, False],
    )

    selected_threshold = float(
        threshold_df.iloc[0]["Threshold"]
    )

    return selected_threshold, threshold_df


# ============================================================
# NESTED SPATIAL CROSS-VALIDATION
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = [
        TARGET,
        "Latitude",
        "Longitude",
    ]

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    y = df[TARGET].astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    # Exclude coordinates from predictors.
    excluded_columns = {
        TARGET,
        "Latitude",
        "Longitude",
    }

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    X = df[feature_columns].copy()

    groups = create_spatial_groups(df)

    unique_groups = np.unique(groups)

    if len(unique_groups) < OUTER_FOLDS:
        raise ValueError(
            "Not enough spatial groups for outer validation."
        )

    print("\n===== NESTED SPATIAL VALIDATION =====")
    print("Observations:", len(df))
    print("Predictor features:", len(feature_columns))
    print("Spatial grid size (km):", GRID_SIZE_KM)
    print("Spatial groups:", len(unique_groups))
    print("Outer folds:", OUTER_FOLDS)
    print("Inner folds:", INNER_FOLDS)

    outer_cv = GroupKFold(
        n_splits=OUTER_FOLDS
    )

    outer_results = []
    outer_predictions = []
    threshold_selection_records = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups),
        start=1,
    ):
        print(
            f"\n----- OUTER FOLD {outer_fold} -----"
        )

        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        train_groups = groups[train_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            raise ValueError(
                f"Outer fold {outer_fold} has only one class "
                "in its training or test partition."
            )

        # Select threshold using outer training data only.
        selected_threshold, inner_scores = select_threshold(
            X_train,
            y_train,
            train_groups,
        )

        print(
            "Selected inner threshold:",
            selected_threshold,
        )

        inner_scores.insert(
            0,
            "Outer_Fold",
            outer_fold,
        )

        threshold_selection_records.append(
            inner_scores
        )

        # Refit the model on the complete outer training set.
        outer_model = build_model()

        outer_model.fit(
            X_train,
            y_train,
        )

        # Evaluate only once on the untouched outer test fold.
        probabilities = outer_model.predict_proba(
            X_test
        )[:, 1]

        fold_metrics = calculate_metrics(
            y_test.to_numpy(),
            probabilities,
            selected_threshold,
        )

        fold_metrics["Outer_Fold"] = outer_fold
        fold_metrics["Train_Rows"] = len(train_idx)
        fold_metrics["Test_Rows"] = len(test_idx)
        fold_metrics["Selected_Threshold"] = selected_threshold

        outer_results.append(fold_metrics)

        predictions = (
            probabilities >= selected_threshold
        ).astype(int)

        outer_predictions.append(
            pd.DataFrame({
                "Row_Index": test_idx,
                "Outer_Fold": outer_fold,
                "Actual": y_test.to_numpy(),
                "Probability": probabilities,
                "Selected_Threshold": selected_threshold,
                "Predicted": predictions,
            })
        )

        print(
            f"Accuracy={fold_metrics['Accuracy']:.4f}, "
            f"Precision={fold_metrics['Precision']:.4f}, "
            f"Recall={fold_metrics['Recall']:.4f}, "
            f"F1={fold_metrics['F1_Score']:.4f}, "
            f"ROC-AUC={fold_metrics['ROC_AUC']:.4f}, "
            f"PR-AUC={fold_metrics['PR_AUC_AP']:.4f}"
        )

    # Save outer-fold results.
    results_df = pd.DataFrame(
        outer_results
    )

    results_df.to_csv(
        OUTPUT_DIR / "nested_spatial_outer_fold_results.csv",
        index=False,
    )

    # Save all untouched outer-fold predictions.
    predictions_df = pd.concat(
        outer_predictions,
        ignore_index=True,
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "nested_spatial_oof_predictions.csv",
        index=False,
    )

    # Save inner threshold selection records.
    threshold_df = pd.concat(
        threshold_selection_records,
        ignore_index=True,
    )

    threshold_df.to_csv(
        OUTPUT_DIR / "nested_spatial_inner_threshold_scores.csv",
        index=False,
    )

    # Summarize outer-fold metrics.
    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "Specificity",
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
        OUTPUT_DIR / "nested_spatial_summary.csv",
        index=False,
    )

    print("\n===== NESTED SPATIAL VALIDATION SUMMARY =====")

    for _, row in summary.iterrows():
        print(
            f"{row['Metric']}: "
            f"{row['Mean']:.4f} +/- {row['Std']:.4f}"
        )

    print("\nSelected thresholds by outer fold:")
    print(
        results_df[
            ["Outer_Fold", "Selected_Threshold"]
        ].to_string(index=False)
    )

    print("\nResults saved to:")
    print(OUTPUT_DIR)

    print(
        "\nInterpretation: Each outer test fold is excluded from "
        "threshold selection. However, these results estimate "
        "generalization to held-out grid groups, not necessarily "
        "to distant or independent regions. The spatial grid "
        "does not enforce a geographic buffer."
    )


if __name__ == "__main__":
    main()