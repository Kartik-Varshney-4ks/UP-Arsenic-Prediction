from pathlib import Path

import numpy as np
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

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found."
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    features = df.drop(columns=[TARGET]).apply(
        pd.to_numeric,
        errors="coerce",
    )

    # 1. Count distinct values and detect constant features.
    feature_rows = []

    for column in features.columns:
        values = features[column].dropna()

        feature_rows.append(
            {
                "Feature": column,
                "Non_Missing": int(values.size),
                "Missing": int(features[column].isna().sum()),
                "Unique_Values": int(values.nunique()),
                "Constant_Among_Nonmissing": (
                    values.nunique() <= 1
                ),
                "Minimum": (
                    values.min() if not values.empty else np.nan
                ),
                "Maximum": (
                    values.max() if not values.empty else np.nan
                ),
            }
        )

    feature_audit = pd.DataFrame(feature_rows)

    feature_audit.to_csv(
        OUTPUT_DIR / "feature_quality_audit.csv",
        index=False,
    )

    print("\n1. FEATURE QUALITY")
    print(
        feature_audit.sort_values(
            ["Constant_Among_Nonmissing", "Unique_Values"],
            ascending=[False, True],
        ).to_string(index=False)
    )

    # 2. Examine exact duplicate rows.
    exact_duplicates = df.loc[
        df.duplicated(keep=False)
    ].copy()

    exact_duplicates.to_csv(
        OUTPUT_DIR / "exact_duplicate_records.csv",
        index=False,
    )

    print("\n2. EXACT DUPLICATE RECORDS")
    print(exact_duplicates.to_string(index=False))

    # 3. Examine repeated feature vectors and their labels.
    feature_columns = [
        column for column in df.columns
        if column != TARGET
    ]

    duplicated_features = df.loc[
        df.duplicated(
            subset=feature_columns,
            keep=False,
        )
    ].copy()

    duplicated_features.to_csv(
        OUTPUT_DIR / "duplicate_feature_vectors.csv",
        index=False,
    )

    print("\n3. DUPLICATE FEATURE VECTORS")
    print(duplicated_features.to_string(index=False))

    if not duplicated_features.empty:
        label_consistency = (
            duplicated_features
            .groupby(feature_columns, dropna=False)[TARGET]
            .agg(
                Rows="size",
                Unique_Target_Labels="nunique",
            )
            .reset_index()
        )

        print("\n4. DUPLICATE LABEL CONSISTENCY")
        print(
            label_consistency.to_string(index=False)
        )

        label_consistency.to_csv(
            OUTPUT_DIR / "duplicate_label_consistency.csv",
            index=False,
        )

    # 4. Examine univariate feature-target associations.
    correlations = []

    for column in features.columns:
        temp = pd.DataFrame(
            {
                "feature": features[column],
                "target": df[TARGET],
            }
        ).dropna()

        if (
            temp["feature"].nunique() < 2
            or temp["target"].nunique() < 2
        ):
            correlation = np.nan
        else:
            correlation = temp["feature"].corr(
                temp["target"],
                method="spearman",
            )

        correlations.append(
            {
                "Feature": column,
                "Spearman_Correlation_With_Target": correlation,
                "Non_Missing_Pairs": len(temp),
            }
        )

    association_df = pd.DataFrame(correlations)

    association_df["Absolute_Correlation"] = (
        association_df[
            "Spearman_Correlation_With_Target"
        ].abs()
    )

    association_df = association_df.sort_values(
        "Absolute_Correlation",
        ascending=False,
        na_position="last",
    )

    association_df.to_csv(
        OUTPUT_DIR / "feature_target_associations.csv",
        index=False,
    )

    print("\n5. STRONGEST UNIVARIATE ASSOCIATIONS")
    print(
        association_df.head(15).round(4).to_string(index=False)
    )

    print("\nSaved diagnostic files to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()