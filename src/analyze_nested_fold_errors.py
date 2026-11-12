from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "Arsenic_Spatial_Model_Dataset.csv"
)

PREDICTIONS_PATH = (
    ROOT / "docs" / "nested_spatial_validation"
    / "nested_spatial_oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "docs" / "nested_spatial_validation"

TARGET = "arsenic_detected"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    required = [
        "Row_Index",
        "Outer_Fold",
        "Actual",
        "Predicted",
        "Probability",
    ]

    missing = [
        col for col in required
        if col not in predictions.columns
    ]

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    if not predictions["Row_Index"].is_unique:
        raise ValueError("Duplicate row indices in predictions.")

    if len(predictions) != len(data):
        raise ValueError(
            "Prediction count does not match dataset row count."
        )

    # Match predictions to their original observations.
    predictions = predictions.copy()

    row_indices = predictions["Row_Index"].astype(int).to_numpy()

    if (
        (row_indices < 0).any()
        or (row_indices >= len(data)).any()
    ):
        raise ValueError("Prediction row index is out of range.")

    predictions["Dataset_Target"] = (
        data.iloc[row_indices][TARGET].to_numpy()
    )

    if not (
        predictions["Actual"].to_numpy()
        == predictions["Dataset_Target"].to_numpy()
    ).all():
        raise ValueError(
            "Prediction labels do not match the dataset."
        )

    # Error type for every prediction.
    predictions["Error_Type"] = np.select(
        [
            (predictions["Actual"] == 1)
            & (predictions["Predicted"] == 1),

            (predictions["Actual"] == 0)
            & (predictions["Predicted"] == 0),

            (predictions["Actual"] == 0)
            & (predictions["Predicted"] == 1),

            (predictions["Actual"] == 1)
            & (predictions["Predicted"] == 0),
        ],
        [
            "True Positive",
            "True Negative",
            "False Positive",
            "False Negative",
        ],
        default="Unknown",
    )

    # Summarize errors by outer fold.
    fold_summary = []

    for fold, group in predictions.groupby(
        "Outer_Fold",
        sort=True,
    ):
        tn = int(
            (
                (group["Actual"] == 0)
                & (group["Predicted"] == 0)
            ).sum()
        )

        fp = int(
            (
                (group["Actual"] == 0)
                & (group["Predicted"] == 1)
            ).sum()
        )

        fn = int(
            (
                (group["Actual"] == 1)
                & (group["Predicted"] == 0)
            ).sum()
        )

        tp = int(
            (
                (group["Actual"] == 1)
                & (group["Predicted"] == 1)
            ).sum()
        )

        fold_summary.append({
            "Outer_Fold": int(fold),
            "Observations": len(group),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "False_Positive_Rate": (
                fp / (fp + tn) if (fp + tn) else np.nan
            ),
            "False_Negative_Rate": (
                fn / (fn + tp) if (fn + tp) else np.nan
            ),
        })

    fold_summary_df = pd.DataFrame(fold_summary)

    fold_summary_df.to_csv(
        OUTPUT_DIR / "nested_fold_error_summary.csv",
        index=False,
    )

    # Save each observation and its prediction error.
    predictions.to_csv(
        OUTPUT_DIR / "nested_fold_prediction_errors.csv",
        index=False,
    )

    # Compare feature values for correct and incorrect predictions.
    excluded = {
        TARGET,
        "Latitude",
        "Longitude",
    }

    feature_columns = [
        col for col in data.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(data[col])
    ]

    error_records = []

    for fold, group in predictions.groupby(
        "Outer_Fold",
        sort=True,
    ):
        original_rows = group["Row_Index"].astype(int).to_numpy()

        fold_data = data.iloc[original_rows].copy()

        actual = group["Actual"].to_numpy()
        predicted = group["Predicted"].to_numpy()

        fold_data["Prediction_Status"] = np.where(
            actual == predicted,
            "Correct",
            "Incorrect",
        )

        fold_data["Outer_Fold"] = int(fold)

        for feature in feature_columns:
            correct_values = fold_data.loc[
                fold_data["Prediction_Status"] == "Correct",
                feature,
            ].dropna()

            incorrect_values = fold_data.loc[
                fold_data["Prediction_Status"] == "Incorrect",
                feature,
            ].dropna()

            if (
                len(correct_values) == 0
                or len(incorrect_values) == 0
            ):
                continue

            correct_median = float(correct_values.median())
            incorrect_median = float(incorrect_values.median())

            error_records.append({
                "Outer_Fold": int(fold),
                "Feature": feature,
                "Correct_Count": len(correct_values),
                "Incorrect_Count": len(incorrect_values),
                "Correct_Median": correct_median,
                "Incorrect_Median": incorrect_median,
                "Median_Difference_Incorrect_Minus_Correct": (
                    incorrect_median - correct_median
                ),
            })

    error_features_df = pd.DataFrame(error_records)

    error_features_df.to_csv(
        OUTPUT_DIR / "nested_fold_error_feature_comparison.csv",
        index=False,
    )

    # Rank absolute median differences for Fold 4.
    fold4 = error_features_df[
        error_features_df["Outer_Fold"] == 4
    ].copy()

    if not fold4.empty:
        fold4["Absolute_Median_Difference"] = (
            fold4[
                "Median_Difference_Incorrect_Minus_Correct"
            ].abs()
        )

        fold4 = fold4.sort_values(
            "Absolute_Median_Difference",
            ascending=False,
        )

    print("\n===== ERROR COUNTS BY OUTER FOLD =====")
    print(fold_summary_df.round(4).to_string(index=False))

    print("\n===== FOLD 4: LARGEST MEDIAN FEATURE DIFFERENCES =====")

    if fold4.empty:
        print("Insufficient data to compare feature medians.")
    else:
        print(
            fold4[
                [
                    "Feature",
                    "Correct_Count",
                    "Incorrect_Count",
                    "Correct_Median",
                    "Incorrect_Median",
                    "Median_Difference_Incorrect_Minus_Correct",
                ]
            ]
            .head(10)
            .round(4)
            .to_string(index=False)
        )

    print("\nSaved error summary:")
    print(OUTPUT_DIR / "nested_fold_error_summary.csv")

    print("\nSaved observation-level errors:")
    print(OUTPUT_DIR / "nested_fold_prediction_errors.csv")

    print("\nSaved feature comparisons:")
    print(OUTPUT_DIR / "nested_fold_error_feature_comparison.csv")

    print(
        "\nInterpretation note: Differences in feature medians "
        "are descriptive, not evidence that a feature caused "
        "prediction errors."
    )


if __name__ == "__main__":
    main()