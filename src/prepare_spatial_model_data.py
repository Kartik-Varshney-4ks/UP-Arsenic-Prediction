from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "processed" / "Arsenic_Classification_Dataset_Clean.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "Arsenic_Spatial_Model_Dataset.csv"
AUDIT_DIR = ROOT / "docs" / "data_audit"

TARGET = "arsenic_detected"
CONSTANT_FEATURE = "soil_observations_2020"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print("\n===== SPATIAL MODEL DATA PREPARATION =====")
    print("Original rows:", len(df))
    print("Original columns:", len(df.columns))

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    # Remove the known constant feature.
    if CONSTANT_FEATURE in df.columns:
        df = df.drop(columns=[CONSTANT_FEATURE])
        print("Removed constant feature:", CONSTANT_FEATURE)

    # Identify remaining constant or all-missing predictors.
    features = [c for c in df.columns if c != TARGET]

    audit = []

    for col in features:
        unique_count = int(df[col].nunique(dropna=True))
        audit.append({
            "Feature": col,
            "Unique_Nonmissing_Values": unique_count,
            "Missing_Count": int(df[col].isna().sum()),
            "Is_Constant_or_All_Missing": unique_count <= 1
        })

    audit_df = pd.DataFrame(audit)

    audit_df.to_csv(
        AUDIT_DIR / "spatial_model_feature_audit.csv",
        index=False
    )

    constant_features = audit_df.loc[
        audit_df["Is_Constant_or_All_Missing"],
        "Feature"
    ].tolist()

    if constant_features:
        df = df.drop(columns=constant_features)
        print("Removed additional constant features:", constant_features)

    # Report missing values. Do not impute them here.
    missing_report = pd.DataFrame({
        "Feature": df.columns,
        "Missing_Count": [
            int(df[col].isna().sum()) for col in df.columns
        ]
    })

    missing_report["Missing_Percent"] = (
        missing_report["Missing_Count"] / len(df) * 100
    ).round(2)

    missing_report.to_csv(
        AUDIT_DIR / "spatial_model_missing_values.csv",
        index=False
    )

    # Identify repeated feature vectors without deleting them.
    features = [c for c in df.columns if c != TARGET]

    duplicate_mask = df.duplicated(
        subset=features,
        keep=False
    )

    duplicate_rows = df.loc[duplicate_mask].copy()

    duplicate_rows.to_csv(
        AUDIT_DIR / "duplicate_feature_vectors_for_review.csv",
        index=False
    )

    print("\nRows with repeated feature vectors:", int(duplicate_mask.sum()))

    # Save the prepared dataset.
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n===== PREPARATION SUMMARY =====")
    print("Prepared rows:", len(df))
    print("Prepared columns:", len(df.columns))
    print("Predictor count:", len(df.columns) - 1)

    print("\nTarget distribution:")
    print(df[TARGET].value_counts().sort_index().to_string())

    print("\nFeatures with missing values:")
    nonzero_missing = missing_report[
        missing_report["Missing_Count"] > 0
    ]

    if nonzero_missing.empty:
        print("No missing values found.")
    else:
        print(nonzero_missing.to_string(index=False))

    print("\nPrepared dataset saved to:")
    print(OUTPUT_PATH)

    print("\nAudit reports saved to:")
    print(AUDIT_DIR)

    print("\nMissing values have not been imputed.")
    print("Perform imputation within each training fold during validation.")


if __name__ == "__main__":
    main()