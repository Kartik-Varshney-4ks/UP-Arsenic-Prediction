from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Arsenic_Spatial_Model_Dataset.csv"
)

RESULTS_PATH = (
    ROOT
    / "docs"
    / "nested_spatial_validation"
    / "nested_spatial_outer_fold_results.csv"
)

PREDICTIONS_PATH = (
    ROOT
    / "docs"
    / "nested_spatial_validation"
    / "nested_spatial_oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "docs" / "nested_spatial_validation"


def main():
    for path in [DATA_PATH, RESULTS_PATH, PREDICTIONS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    data = pd.read_csv(DATA_PATH)
    results = pd.read_csv(RESULTS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    if len(data) != len(predictions):
        raise ValueError(
            "Dataset and prediction row counts differ. "
            "Check that predictions correspond to this dataset."
        )

    # Verify that saved row indices match the original dataset.
    predictions = predictions.copy()

    if not predictions["Row_Index"].is_unique:
        raise ValueError("Repeated Row_Index values found.")

    predictions["Latitude"] = data.loc[
        predictions["Row_Index"], "Latitude"
    ].to_numpy()

    predictions["Longitude"] = data.loc[
        predictions["Row_Index"], "Longitude"
    ].to_numpy()

    predictions["Dataset_Target"] = data.loc[
        predictions["Row_Index"], "arsenic_detected"
    ].to_numpy()

    if not (
        predictions["Actual"].to_numpy()
        == predictions["Dataset_Target"].to_numpy()
    ).all():
        raise ValueError(
            "Prediction labels do not match the dataset target."
        )

    rows = []

    for fold, group in predictions.groupby("Outer_Fold", sort=True):
        rows.append({
            "Outer_Fold": int(fold),
            "Observations": len(group),
            "Positive_Count": int(group["Actual"].sum()),
            "Positive_Percent": round(
                group["Actual"].mean() * 100, 2
            ),
            "Latitude_Mean": group["Latitude"].mean(),
            "Latitude_Min": group["Latitude"].min(),
            "Latitude_Max": group["Latitude"].max(),
            "Longitude_Mean": group["Longitude"].mean(),
            "Longitude_Min": group["Longitude"].min(),
            "Longitude_Max": group["Longitude"].max(),
        })

    geography = pd.DataFrame(rows)

    report = geography.merge(
        results,
        on="Outer_Fold",
        how="left",
        validate="one_to_one",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "nested_fold_geography_diagnostics.csv"

    report.to_csv(output_path, index=False)

    display_columns = [
        "Outer_Fold",
        "Observations",
        "Positive_Count",
        "Positive_Percent",
        "Latitude_Mean",
        "Longitude_Mean",
        "Selected_Threshold",
        "Precision",
        "Recall",
        "F1_Score",
    ]

    print("\n===== OUTER FOLD GEOGRAPHIC DIAGNOSTICS =====")

    print(
        report[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nFull report saved to:")
    print(output_path)


if __name__ == "__main__":
    main()