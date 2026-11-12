from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "docs"
    / "spatial_validation"
    / "cleaned_spatial_cv_oof_predictions.csv"
)

OUTPUT_PATH = (
    ROOT
    / "docs"
    / "spatial_validation"
    / "threshold_per_fold.csv"
)

THRESHOLDS = [0.20, 0.35, 0.50]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        "Fold",
        "Actual",
        "Probability",
    ]

    missing = [
        col for col in required if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Fold, Actual, or Probability contains missing values."
        )

    results = []

    for fold, fold_df in df.groupby("Fold", sort=True):
        y_true = fold_df["Actual"].astype(int).to_numpy()
        probabilities = fold_df["Probability"].astype(float).to_numpy()

        for threshold in THRESHOLDS:
            y_pred = (
                probabilities >= threshold
            ).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            ).ravel()

            results.append({
                "Fold": int(fold),
                "Threshold": threshold,
                "Observations": len(fold_df),
                "Positive_Observations": int(y_true.sum()),
                "Precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "F1_Score": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "Balanced_Accuracy": balanced_accuracy_score(
                    y_true,
                    y_pred,
                ),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            })

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n===== THRESHOLD PERFORMANCE BY SPATIAL FOLD =====")

    print(
        results_df.round(4).to_string(index=False)
    )

    print("\n===== MEAN METRICS BY THRESHOLD =====")

    summary = (
        results_df
        .groupby("Threshold")[
            [
                "Precision",
                "Recall",
                "F1_Score",
                "Balanced_Accuracy",
            ]
        ]
        .agg(["mean", "std"])
        .round(4)
    )

    print(summary.to_string())

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print(
        "\nNote: This is a diagnostic comparison using the "
        "existing out-of-fold predictions. It does not provide "
        "an independent evaluation of a selected threshold."
    )


if __name__ == "__main__":
    main()