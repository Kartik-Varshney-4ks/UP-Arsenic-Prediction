from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Arsenic_Classification_Dataset.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
AUDIT_DIR = ROOT / "docs" / "data_audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "arsenic_detected"


def main():
    df = pd.read_csv(INPUT_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found."
        )

    original_rows = len(df)

    # Preserve all copies of duplicated rows in an audit file.
    duplicate_mask = df.duplicated(keep=False)

    df.loc[duplicate_mask].to_csv(
        AUDIT_DIR / "duplicate_rows_before_cleaning.csv",
        index=False,
    )

    # Remove only exact duplicate rows.
    cleaned = df.drop_duplicates().reset_index(drop=True)

    removed_rows = original_rows - len(cleaned)

    # Check feature columns for constant or all-missing values.
    feature_columns = [
        col for col in cleaned.columns
        if col != TARGET
    ]

    feature_audit = []

    for col in feature_columns:
        non_missing = cleaned[col].dropna()

        feature_audit.append(
            {
                "Feature": col,
                "Unique_Nonmissing_Values": non_missing.nunique(),
                "Missing_Count": int(cleaned[col].isna().sum()),
                "Is_Constant": non_missing.nunique() <= 1,
                "Is_All_Missing": non_missing.empty,
            }
        )

    audit_df = pd.DataFrame(feature_audit)

    audit_df.to_csv(
        AUDIT_DIR / "cleaned_feature_audit.csv",
        index=False,
    )

    output_path = (
        OUTPUT_DIR / "Arsenic_Classification_Dataset_Clean.csv"
    )

    cleaned.to_csv(output_path, index=False)

    print("\nCLEANING SUMMARY")
    print("Original rows:", original_rows)
    print("Rows after exact deduplication:", len(cleaned))
    print("Exact duplicate rows removed:", removed_rows)
    print("Original columns:", len(df.columns))
    print("Cleaned columns:", len(cleaned.columns))

    print("\nTARGET DISTRIBUTION AFTER CLEANING")
    print(
        cleaned[TARGET]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\nCONSTANT FEATURES")
    print(
        audit_df.loc[
            audit_df["Is_Constant"] | audit_df["Is_All_Missing"]
        ].to_string(index=False)
    )

    print("\nCleaned dataset saved to:")
    print(output_path)

    print("\nAudit files saved to:")
    print(AUDIT_DIR)

    print(
        "\nNote: The original dataset has not been modified. "
        "The cleaned dataset is for controlled experiments; "
        "duplicate handling should be documented in the research."
    )


if __name__ == "__main__":
    main()