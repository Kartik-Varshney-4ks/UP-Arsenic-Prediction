from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "docs"
    / "spatial_validation"
    / "cleaned_spatial_cv_oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "docs" / "spatial_validation"

TARGET = "Actual"
PROBABILITY = "Probability"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Out-of-fold predictions not found: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required = [TARGET, PROBABILITY]

    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Actual labels or probabilities contain missing values."
        )

    y_true = df[TARGET].astype(int).to_numpy()
    probabilities = df[PROBABILITY].astype(float).to_numpy()

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Actual labels must contain only 0 and 1.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    thresholds = np.arange(0.10, 0.91, 0.05)

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append({
            "Threshold": round(float(threshold), 2),
            "Accuracy": accuracy_score(
                y_true, predictions
            ),
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
            "Specificity": (
                tn / (tn + fp) if (tn + fp) else 0.0
            ),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

    results_df = pd.DataFrame(results)

    output_path = (
        OUTPUT_DIR / "cleaned_spatial_threshold_analysis.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\n===== SPATIAL THRESHOLD ANALYSIS =====")
    print("Out-of-fold observations:", len(df))

    print("\nThreshold comparison:")
    print(
        results_df.round(4).to_string(index=False)
    )

    best_f1 = results_df.loc[
        results_df["F1_Score"].idxmax()
    ]

    recall_80 = results_df[
        results_df["Recall"] >= 0.80
    ]

    print("\nHighest observed F1-score threshold:")
    print(best_f1.to_string())

    if not recall_80.empty:
        best_precision_at_recall = recall_80.loc[
            recall_80["Precision"].idxmax()
        ]

        print(
            "\nHighest-precision threshold among those "
            "with recall >= 0.80:"
        )

        print(best_precision_at_recall.to_string())
    else:
        print(
            "\nNo tested threshold achieved recall >= 0.80."
        )

    print("\nSaved results to:")
    print(output_path)

    print(
        "\nIMPORTANT: These out-of-fold predictions can be used "
        "for exploratory threshold analysis. Choosing and reporting "
        "the best threshold on the same predictions gives an "
        "optimistic estimate. Final threshold selection should be "
        "separated from final evaluation."
    )


if __name__ == "__main__":
    main()