from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "UP_Arsenic_Climate_Environmental_Master_Working.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "EDA"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the project master dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("UP ARSENIC PREDICTION - DATA VALIDATION")
    print("=" * 70)

    print("\nDataset loaded successfully.")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# BASIC INFORMATION
# ============================================================

def basic_information(df):

    print("\n" + "=" * 70)
    print("1. BASIC INFORMATION")
    print("=" * 70)

    print("\nData types:")
    print(df.dtypes)

    print("\nDataset shape:")
    print(df.shape)

    print("\nNumber of unique values:")
    print(df.nunique())


# ============================================================
# MISSING VALUES
# ============================================================

def missing_value_analysis(df):

    print("\n" + "=" * 70)
    print("2. MISSING VALUE ANALYSIS")
    print("=" * 70)

    missing_count = df.isnull().sum()
    missing_percentage = (
        df.isnull().mean() * 100
    )

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

    missing_report = (
        missing_report[
            missing_report["missing_count"] > 0
        ]
        .sort_values(
            "missing_count",
            ascending=False
        )
    )

    if missing_report.empty:
        print("\nNo missing values found.")

    else:
        print("\nColumns containing missing values:")
        print(missing_report)

    missing_report.to_csv(
        REPORT_DIR / "Missing_Value_Report.csv"
    )

    print(
        f"\nMissing-value report saved to: "
        f"{REPORT_DIR / 'Missing_Value_Report.csv'}"
    )


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def duplicate_analysis(df):

    print("\n" + "=" * 70)
    print("3. DUPLICATE ANALYSIS")
    print("=" * 70)

    duplicate_rows = df.duplicated().sum()

    print(f"\nDuplicate rows: {duplicate_rows}")

    if "Location" in df.columns:
        duplicate_locations = df["Location"].duplicated().sum()

        print(
            f"Duplicate Location values: "
            f"{duplicate_locations}"
        )


# ============================================================
# DISTRICT ANALYSIS
# ============================================================

def district_analysis(df):

    print("\n" + "=" * 70)
    print("4. DISTRICT ANALYSIS")
    print("=" * 70)

    if "District" in df.columns:

        districts = (
            df["District"]
            .dropna()
            .unique()
        )

        print(
            f"\nNumber of unique districts: "
            f"{len(districts)}"
        )

        print("\nDistricts:")
        print(sorted(districts))


# ============================================================
# ARSENIC ANALYSIS
# ============================================================

def arsenic_analysis(df):

    print("\n" + "=" * 70)
    print("5. ARSENIC ANALYSIS")
    print("=" * 70)

    if "As" not in df.columns:
        raise ValueError(
            "Arsenic column 'As' was not found."
        )

    arsenic = df["As"]

    print("\nArsenic statistics:")

    print(f"Count  : {arsenic.count()}")
    print(f"Mean   : {arsenic.mean():.2f}")
    print(f"Median : {arsenic.median():.2f}")
    print(f"Std    : {arsenic.std():.2f}")
    print(f"Minimum: {arsenic.min():.2f}")
    print(f"Maximum: {arsenic.max():.2f}")

    print(
        f"\nZero arsenic observations: "
        f"{(arsenic == 0).sum()}"
    )

    print(
        f"Non-zero arsenic observations: "
        f"{(arsenic > 0).sum()}"
    )


# ============================================================
# GEOGRAPHICAL VALIDATION
# ============================================================

def geographical_validation(df):

    print("\n" + "=" * 70)
    print("6. GEOGRAPHICAL VALIDATION")
    print("=" * 70)

    if "Latitude" in df.columns:

        invalid_latitude = (
            (~df["Latitude"].between(-90, 90))
            .sum()
        )

        print(
            f"\nInvalid latitude values: "
            f"{invalid_latitude}"
        )

    if "Longitude" in df.columns:

        invalid_longitude = (
            (~df["Longitude"].between(-180, 180))
            .sum()
        )

        print(
            f"Invalid longitude values: "
            f"{invalid_longitude}"
        )


# ============================================================
# YEAR VALIDATION
# ============================================================

def year_validation(df):

    print("\n" + "=" * 70)
    print("7. YEAR VALIDATION")
    print("=" * 70)

    if "Year" in df.columns:

        print("\nYears present:")
        print(
            sorted(
                df["Year"]
                .dropna()
                .unique()
            )
        )


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

def numerical_summary(df):

    print("\n" + "=" * 70)
    print("8. NUMERICAL SUMMARY")
    print("=" * 70)

    numerical = df.select_dtypes(
        include="number"
    )

    summary = numerical.describe().T

    print(summary)

    summary.to_csv(
        REPORT_DIR / "Numerical_Summary.csv"
    )

    print(
        f"\nNumerical summary saved to: "
        f"{REPORT_DIR / 'Numerical_Summary.csv'}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    basic_information(df)

    missing_value_analysis(df)

    duplicate_analysis(df)

    district_analysis(df)

    arsenic_analysis(df)

    geographical_validation(df)

    year_validation(df)

    numerical_summary(df)

    print("\n" + "=" * 70)
    print("DATA VALIDATION COMPLETED")
    print("=" * 70)