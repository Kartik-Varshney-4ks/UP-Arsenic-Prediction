from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "UP_Arsenic_Climate_Environmental_Master_Working.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "UP_Arsenic_Cleaned.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("=" * 70)
    print("UP ARSENIC PREDICTION - DATA CLEANING")
    print("=" * 70)

    print(f"\nOriginal rows    : {df.shape[0]}")
    print(f"Original columns : {df.shape[1]}")

    return df


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    return df


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(
        f"\nExact duplicate rows removed: "
        f"{before - after}"
    )

    return df


# ============================================================
# DATA TYPE CONVERSION
# ============================================================

def convert_numeric_columns(df):

    numeric_columns = [
        "S_No",
        "Longitude",
        "Latitude",
        "Year",
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
        "As",
        "U",
        "rainfall_mean_annual_mm",
        "rainfall_mean_monsoon_mm",
        "rainfall_mean_premonsoon_mm",
        "rainfall_mean_postmonsoon_mm",
        "rainfall_mean_winter_mm",
        "soil_moisture_mean_2020",
        "soil_moisture_median_2020",
        "soil_moisture_std_2020",
        "soil_moisture_volume_mean_2020",
        "soil_moisture_pct_mean_2020",
        "soil_observations_2020",
    ]

    for column in numeric_columns:

        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text_columns(df):

    text_columns = [
        "State",
        "District",
        "Location",
        "District_std",
        "District_canonical",
        "IMD_subdivision",
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    return df


# ============================================================
# VALIDATE ARSENIC VALUES
# ============================================================

def validate_arsenic(df):

    if "As" not in df.columns:
        raise ValueError(
            "Arsenic column 'As' is missing."
        )

    invalid = (df["As"] < 0).sum()

    print(
        f"\nNegative arsenic values: {invalid}"
    )

    if invalid > 0:
        print(
            "WARNING: Negative arsenic values "
            "were found."
        )

    print("\nArsenic summary:")
    print(df["As"].describe())

    return df


# ============================================================
# VALIDATE GEOGRAPHICAL VALUES
# ============================================================

def validate_coordinates(df):

    if "Latitude" in df.columns:

        invalid_lat = (
            ~df["Latitude"].between(-90, 90)
        ).sum()

        print(
            f"\nInvalid latitude values: "
            f"{invalid_lat}"
        )

    if "Longitude" in df.columns:

        invalid_lon = (
            ~df["Longitude"].between(-180, 180)
        ).sum()

        print(
            f"Invalid longitude values: "
            f"{invalid_lon}"
        )

    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):

    print("\nMissing values before treatment:")

    print(
        df.isnull()
        .sum()
        .loc[lambda x: x > 0]
    )

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------
    # We do NOT invent location information.
    # Keep missing Location values as missing.
    #
    # --------------------------------------------------------

    # --------------------------------------------------------
    # PO4
    # --------------------------------------------------------
    # Median imputation will be performed later during the
    # modelling pipeline if PO4 is used as a feature.
    #
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Soil moisture
    # --------------------------------------------------------
    # Missing soil moisture values are retained here.
    # Modelling pipeline will handle them appropriately.
    #
    # --------------------------------------------------------

    print(
        "\nMissing values have NOT been blindly filled."
    )

    print(
        "They will be handled according to the modelling "
        "pipeline and documented."
    )

    return df


# ============================================================
# SORT DATA
# ============================================================

def sort_data(df):

    sort_columns = []

    if "District" in df.columns:
        sort_columns.append("District")

    if "S_No" in df.columns:
        sort_columns.append("S_No")

    if sort_columns:
        df = df.sort_values(
            sort_columns
        )

    return df


# ============================================================
# SAVE CLEAN DATA
# ============================================================

def save_data(df):

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nCleaned dataset saved to:\n"
        f"{OUTPUT_PATH}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    df = load_data()

    df = clean_column_names(df)

    df = remove_duplicates(df)

    df = convert_numeric_columns(df)

    df = clean_text_columns(df)

    df = validate_arsenic(df)

    df = validate_coordinates(df)

    df = handle_missing_values(df)

    df = sort_data(df)

    save_data(df)

    print("\n" + "=" * 70)
    print("DATA CLEANING COMPLETED")
    print("=" * 70)

    print(
        f"\nFinal shape: {df.shape}"
    )