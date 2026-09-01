from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

EDA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "EDA"
)

EDA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("=" * 70)
    print("UP ARSENIC PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# ARSENIC SUMMARY
# ============================================================

def arsenic_summary(df):

    arsenic = df["As"]

    summary = pd.DataFrame({
        "Metric": [
            "Count",
            "Mean",
            "Median",
            "Standard Deviation",
            "Minimum",
            "Maximum",
            "Zero observations",
            "Non-zero observations"
        ],
        "Value": [
            arsenic.count(),
            arsenic.mean(),
            arsenic.median(),
            arsenic.std(),
            arsenic.min(),
            arsenic.max(),
            (arsenic == 0).sum(),
            (arsenic > 0).sum()
        ]
    })

    print("\nARSENIC SUMMARY")
    print(summary)

    summary.to_csv(
        EDA_DIR / "Arsenic_Summary.csv",
        index=False
    )


# ============================================================
# ARSENIC DISTRIBUTION
# ============================================================

def arsenic_distribution(df):

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["As"],
        bins=50,
        kde=True
    )

    plt.title("Distribution of Arsenic Concentration")
    plt.xlabel("Arsenic (As)")
    plt.ylabel("Number of Observations")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "Arsenic_Distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# LOG ARSENIC DISTRIBUTION
# ============================================================

def arsenic_log_distribution(df):

    non_zero = df.loc[df["As"] > 0, "As"]

    plt.figure(figsize=(10, 6))

    sns.histplot(
        non_zero,
        bins=50,
        kde=True
    )

    plt.xscale("log")

    plt.title(
        "Distribution of Non-Zero Arsenic "
        "(Log Scale)"
    )

    plt.xlabel("Arsenic (log scale)")
    plt.ylabel("Number of Observations")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "Arsenic_Log_Distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# DISTRICT-WISE ARSENIC
# ============================================================

def district_analysis(df):

    district_stats = (
        df.groupby("District")["As"]
        .agg(
            observations="count",
            mean_arsenic="mean",
            median_arsenic="median",
            maximum_arsenic="max",
            non_zero_arsenic=lambda x: (x > 0).sum()
        )
        .sort_values(
            "mean_arsenic",
            ascending=False
        )
    )

    print("\nTOP 15 DISTRICTS BY MEAN ARSENIC")
    print(district_stats.head(15))

    district_stats.to_csv(
        EDA_DIR / "District_Arsenic_Statistics.csv"
    )

    top15 = district_stats.head(15)

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=top15.reset_index(),
        x="mean_arsenic",
        y="District"
    )

    plt.title(
        "Top 15 Districts by Mean Arsenic Concentration"
    )

    plt.xlabel("Mean Arsenic")
    plt.ylabel("District")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "Top_15_Districts_Arsenic.png",
        dpi=300
    )

    plt.close()


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):

    numerical_df = df.select_dtypes(
        include="number"
    )

    correlation = numerical_df.corr()

    correlation.to_csv(
        EDA_DIR / "Correlation_Matrix.csv"
    )

    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Correlation Matrix - Numerical Variables"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "Correlation_Heatmap.png",
        dpi=300
    )

    plt.close()


# ============================================================
# ARSENIC CORRELATIONS
# ============================================================

def arsenic_correlations(df):

    numerical_df = df.select_dtypes(
        include="number"
    )

    correlation = (
        numerical_df
        .corr()["As"]
        .sort_values(
            ascending=False
        )
    )

    print("\nVARIABLES CORRELATED WITH ARSENIC")
    print(correlation)

    correlation.to_csv(
        EDA_DIR / "Arsenic_Correlations.csv"
    )


# ============================================================
# ENVIRONMENTAL VARIABLES
# ============================================================

def environmental_analysis(df):

    environmental_columns = [
        "rainfall_mean_annual_mm",
        "rainfall_mean_monsoon_mm",
        "rainfall_mean_premonsoon_mm",
        "rainfall_mean_postmonsoon_mm",
        "rainfall_mean_winter_mm",
        "soil_moisture_mean_2020",
        "soil_moisture_median_2020",
        "soil_moisture_std_2020",
        "soil_moisture_volume_mean_2020",
        "soil_moisture_pct_mean_2020"
    ]

    available = [
        column
        for column in environmental_columns
        if column in df.columns
    ]

    correlation = (
        df[available + ["As"]]
        .corr()["As"]
        .sort_values(
            ascending=False
        )
    )

    print("\nENVIRONMENTAL VARIABLES vs ARSENIC")
    print(correlation)

    correlation.to_csv(
        EDA_DIR / "Environmental_Arsenic_Correlation.csv"
    )


# ============================================================
# GEOGRAPHICAL SCATTER
# ============================================================

def geographical_plot(df):

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["As"],
        s=25,
        alpha=0.7
    )

    plt.colorbar(
        scatter,
        label="Arsenic Concentration"
    )

    plt.title(
        "Geographical Distribution of Arsenic Observations"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "Arsenic_Geographical_Distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MISSING VALUES
# ============================================================

def missing_value_plot(df):

    missing = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[missing > 0]

    if missing.empty:
        print("\nNo missing values found.")
        return

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=missing.values,
        y=missing.index
    )

    plt.title("Missing Values by Column")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Column")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "Missing_Values.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    arsenic_summary(df)

    arsenic_distribution(df)

    arsenic_log_distribution(df)

    district_analysis(df)

    correlation_analysis(df)

    arsenic_correlations(df)

    environmental_analysis(df)

    geographical_plot(df)

    missing_value_plot(df)

    print("\n" + "=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nEDA outputs saved in:\n{EDA_DIR}"
    )