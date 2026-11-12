from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "docs"
    / "spatial_validation"
    / "spatial_threshold_results.csv"
)

OUTPUT_PATH = (
    ROOT
    / "docs"
    / "spatial_validation"
    / "threshold_recommendations.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    required = [
        "Threshold",
        "Precision",
        "Recall",
        "F1_Score",
        "Balanced_Accuracy",
        "False_Positives",
        "False_Negatives",
        "True_Positives",
        "True_Negatives",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # Select the threshold with the highest observed F1-score.
    best_f1 = df.loc[
        df["F1_Score"].idxmax()
    ].copy()

    # Select the highest threshold that achieves at least 80% recall.
    recall_candidates = df[
        df["Recall"] >= 0.80
    ]

    if not recall_candidates.empty:
        high_recall = recall_candidates.sort_values(
            "Threshold"
        ).iloc[-1].copy()
    else:
        high_recall = None

    selected_rows = []

    best_f1["Selection_Rule"] = (
        "Highest observed F1-score"
    )
    selected_rows.append(best_f1)

    if high_recall is not None:
        high_recall["Selection_Rule"] = (
            "Highest threshold with recall >= 0.80"
        )

        if (
            high_recall["Threshold"]
            != best_f1["Threshold"]
        ):
            selected_rows.append(high_recall)

    result = pd.DataFrame(selected_rows)

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSELECTED THRESHOLD COMPARISON")
    print(
        result[
            [
                "Selection_Rule",
                "Threshold",
                "Precision",
                "Recall",
                "F1_Score",
                "Balanced_Accuracy",
                "False_Positives",
                "False_Negatives",
                "True_Positives",
                "True_Negatives",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print(
        "\nWarning: These thresholds are exploratory. "
        "Final threshold selection should use training/validation "
        "data, with a separate spatial test set reserved for "
        "final performance evaluation."
    )


if __name__ == "__main__":
    main()