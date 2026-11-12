from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

VALIDATION_DIR = ROOT / "docs" / "spatial_validation"
REPORT_DIR = ROOT / "docs" / "research_results"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = (
    VALIDATION_DIR / "cleaned_spatial_cv_summary.csv"
)

FOLD_FILE = (
    VALIDATION_DIR / "threshold_per_fold.csv"
)

THRESHOLD_FILE = (
    VALIDATION_DIR / "cleaned_spatial_threshold_analysis.csv"
)


def main():
    for file_path in [
        SUMMARY_FILE,
        FOLD_FILE,
        THRESHOLD_FILE,
    ]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required results file not found: {file_path}"
            )

    spatial_summary = pd.read_csv(SUMMARY_FILE)
    fold_results = pd.read_csv(FOLD_FILE)
    threshold_results = pd.read_csv(THRESHOLD_FILE)

    # Save the main spatial cross-validation summary.
    spatial_summary.to_csv(
        REPORT_DIR / "spatial_cv_summary_for_research.csv",
        index=False,
    )

    # Save the threshold comparison for the three selected values.
    selected_thresholds = threshold_results[
        threshold_results["Threshold"].round(2).isin(
            [0.20, 0.35, 0.50]
        )
    ].copy()

    selected_thresholds.to_csv(
        REPORT_DIR / "selected_threshold_comparison.csv",
        index=False,
    )

    # Save fold-level results for the same thresholds.
    selected_fold_results = fold_results[
        fold_results["Threshold"].round(2).isin(
            [0.20, 0.35, 0.50]
        )
    ].copy()

    selected_fold_results.to_csv(
        REPORT_DIR / "threshold_fold_comparison.csv",
        index=False,
    )

    # Summarize fold-level threshold performance.
    metrics = [
        "Precision",
        "Recall",
        "F1_Score",
        "Balanced_Accuracy",
    ]

    threshold_summary = (
        selected_fold_results
        .groupby("Threshold")[metrics]
        .agg(["mean", "std"])
        .round(4)
    )

    threshold_summary.to_csv(
        REPORT_DIR / "threshold_summary_mean_std.csv"
    )

    print("\n===== RESEARCH RESULTS EXPORT =====")

    print("\nSpatial cross-validation summary:")
    print(spatial_summary.to_string(index=False))

    print("\nThreshold comparison:")
    print(
        threshold_summary.to_string()
    )

    print("\nFiles saved in:")
    print(REPORT_DIR)

    print(
        "\nNote: These reports summarize existing experiments. "
        "They do not represent independent final test results."
    )


if __name__ == "__main__":
    main()