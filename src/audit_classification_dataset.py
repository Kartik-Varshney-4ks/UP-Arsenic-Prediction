from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Arsenic_Classification_Dataset.csv"
)

OUTPUT_DIR = ROOT / "docs" / "data_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "arsenic_detected"


def main():
    df = pd.read_csv(DATA_PATH)

    print("\n1. DATASET SHAPE")
    print(df.shape)

    print("\n2. COLUMN NAMES")
    for column in df.columns:
        print(column)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found."
        )

    print("\n3. TARGET DISTRIBUTION")
    print(
        df[TARGET]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\n4. TARGET PERCENTAGES")
    print(
        (
            df[TARGET]
            .value_counts(normalize=True, dropna=False)
            .sort_index()
            .mul(100)
        ).round(2).to_string()
    )

    print("\n5. MISSING VALUES")
    missing = pd.DataFrame(
        {
            "Missing_Count": df.isna().sum(),
            "Missing_Percent": (
                df.isna().mean() * 100
            ),
        }
    ).sort_values(
        "Missing_Count",
        ascending=False,
    )

    print(missing.to_string())

    print("\n6. DUPLICATE ROWS")
    print("Exact duplicate rows:", df.duplicated().sum())

    print("\n7. DUPLICATE FEATURE VECTORS")
    feature_columns = [
        column for column in df.columns
        if column != TARGET
    ]

    print(
        "Duplicate feature vectors:",
        df.duplicated(
            subset=feature_columns,
            keep=False,
        ).sum(),
    )

    print("\n8. NUMERIC FEATURE SUMMARY")
    print(
        df.drop(columns=[TARGET])
        .describe()
        .T
        .round(3)
        .to_string()
    )

    # Save audit outputs.
    missing.to_csv(
        OUTPUT_DIR / "missing_values.csv"
    )

    df.describe(include="all").T.to_csv(
        OUTPUT_DIR / "dataset_summary.csv"
    )

    duplicates = df.loc[
        df.duplicated(keep=False)
    ]

    duplicates.to_csv(
        OUTPUT_DIR / "duplicate_rows.csv",
        index=False,
    )

    print("\nAudit files saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()