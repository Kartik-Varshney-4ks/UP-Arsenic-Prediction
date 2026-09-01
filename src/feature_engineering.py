from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "UP_Arsenic_Cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CLASSIFICATION_PATH = (
    OUTPUT_DIR
    / "Arsenic_Classification_Dataset.csv"
)

REGRESSION_PATH = (
    OUTPUT_DIR
    / "Arsenic_Regression_Dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{INPUT_PATH}"
        )

    return pd.read_csv(INPUT_PATH)


# ============================================================
# CREATE TARGET VARIABLES
# ============================================================

def create_targets(df):

    # Arsenic detection target
    df["arsenic_detected"] = (
        df["As"] > 0
    ).astype(int)

    # Log transformed arsenic concentration
    # Useful for the strongly right-skewed concentration data
    df["As_log1p"] = np.log1p(df["As"])

    return df


# ============================================================
# SELECT FEATURES
# ============================================================

def select_features(df):

    feature_columns = [
        # Geography
        "Latitude",
        "Longitude",

        # Groundwater chemistry
        "pH",
        "EC",
        "CO3",
        "HCO3",
        "Cl",
        "F",
        "SO4",
        "NO3",
        "PO4",
        "Total_Hardness",
        "Ca",
        "Mg",
        "Na",
        "K",
        "Fe",
        "U",

        # Rainfall
        "rainfall_mean_annual_mm",
        "rainfall_mean_monsoon_mm",
        "rainfall_mean_premonsoon_mm",
        "rainfall_mean_postmonsoon_mm",
        "rainfall_mean_winter_mm",

        # Soil moisture
        "soil_moisture_mean_2020",
        "soil_moisture_median_2020",
        "soil_moisture_std_2020",
        "soil_moisture_volume_mean_2020",
        "soil_moisture_pct_mean_2020",
        "soil_observations_2020",
    ]

    available_features = [
        column
        for column in feature_columns
        if column in df.columns
    ]

    return available_features


# ============================================================
# CREATE CLASSIFICATION DATASET
# ============================================================

def create_classification_dataset(
    df,
    features
):

    columns = (
        features
        + ["arsenic_detected"]
    )

    classification_df = df[columns].copy()

    return classification_df


# ============================================================
# CREATE REGRESSION DATASET
# ============================================================

def create_regression_dataset(
    df,
    features
):

    positive_arsenic = df[
        df["As"] > 0
    ].copy()

    columns = (
        features
        + ["As", "As_log1p"]
    )

    regression_df = positive_arsenic[
        columns
    ].copy()

    return regression_df


# ============================================================
# SAVE DATASETS
# ============================================================

def save_datasets(
    classification_df,
    regression_df
):

    classification_df.to_csv(
        CLASSIFICATION_PATH,
        index=False
    )

    regression_df.to_csv(
        REGRESSION_PATH,
        index=False
    )

    print(
        f"\nClassification dataset saved to:"
        f"\n{CLASSIFICATION_PATH}"
    )

    print(
        f"\nRegression dataset saved to:"
        f"\n{REGRESSION_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("UP ARSENIC PREDICTION - FEATURE ENGINEERING")
    print("=" * 70)

    df = load_data()

    print(
        f"\nOriginal dataset shape: "
        f"{df.shape}"
    )

    df = create_targets(df)

    features = select_features(df)

    print(
        f"\nNumber of selected features: "
        f"{len(features)}"
    )

    print("\nSelected features:")

    for feature in features:
        print(f"  - {feature}")

    classification_df = (
        create_classification_dataset(
            df,
            features
        )
    )

    regression_df = (
        create_regression_dataset(
            df,
            features
        )
    )

    print(
        f"\nClassification dataset shape: "
        f"{classification_df.shape}"
    )

    print(
        f"Regression dataset shape: "
        f"{regression_df.shape}"
    )

    print("\nClassification target distribution:")

    print(
        classification_df[
            "arsenic_detected"
        ].value_counts()
    )

    save_datasets(
        classification_df,
        regression_df
    )

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)