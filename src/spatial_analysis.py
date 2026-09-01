"""
UP ARSENIC PREDICTION PROJECT
SPATIAL ANALYSIS

Analyzes the geographic distribution of arsenic
concentration across Uttar Pradesh.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "UP_Arsenic_Climate_Environmental_Master_Working.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "EDA"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("UP ARSENIC PREDICTION - SPATIAL ANALYSIS")
print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found at:\n{DATA_PATH}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading master dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "As",
    "District",
    "Latitude",
    "Longitude"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"\nRequired columns missing:\n"
        f"{missing_columns}"
    )


# ============================================================
# BASIC SPATIAL DATA CHECK
# ============================================================

spatial_df = df.dropna(
    subset=[
        "Latitude",
        "Longitude",
        "As"
    ]
).copy()


print(
    f"\nValid spatial observations: "
    f"{len(spatial_df)}"
)


# ============================================================
# DISTRICT SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("DISTRICT-LEVEL ARSENIC SUMMARY")
print("-" * 70)


district_summary = (
    df.groupby("District")
    .agg(
        observations=("As", "count"),

        mean_arsenic=("As", "mean"),

        median_arsenic=("As", "median"),

        maximum_arsenic=("As", "max"),

        non_zero_arsenic=(
            "As",
            lambda x: (x > 0).sum()
        )
    )
)


# ============================================================
# DETECTION PERCENTAGE
# ============================================================

district_summary[
    "detection_percentage"
] = (
    district_summary[
        "non_zero_arsenic"
    ]
    /
    district_summary[
        "observations"
    ]
    * 100
)


# ============================================================
# SORT BY MEAN ARSENIC
# ============================================================

district_summary = (
    district_summary
    .sort_values(
        "mean_arsenic",
        ascending=False
    )
)


print(
    district_summary.head(15)
)


# ============================================================
# SAVE DISTRICT SUMMARY
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "District_Arsenic_Summary.csv"
)

district_summary.to_csv(
    summary_path
)


# ============================================================
# SPATIAL SCATTER PLOT
# ============================================================

print("\nGenerating spatial arsenic distribution...")

plt.figure(
    figsize=(10, 8)
)

scatter = plt.scatter(
    spatial_df["Longitude"],
    spatial_df["Latitude"],
    c=spatial_df["As"],
    s=35,
    alpha=0.75
)

plt.colorbar(
    scatter,
    label="Arsenic Concentration"
)

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "Spatial Distribution of Arsenic "
    "Concentration in Uttar Pradesh"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


spatial_plot_path = (
    OUTPUT_DIR
    / "Arsenic_Spatial_Distribution.png"
)


plt.savefig(
    spatial_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TOP 15 DISTRICTS
# ============================================================

print(
    "\nGenerating district arsenic comparison..."
)


top_districts = (
    district_summary
    .head(15)
    .sort_values(
        "mean_arsenic"
    )
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(
    top_districts.index,
    top_districts[
        "mean_arsenic"
    ]
)


plt.xlabel(
    "Mean Arsenic Concentration"
)

plt.ylabel(
    "District"
)

plt.title(
    "Top 15 Uttar Pradesh Districts "
    "by Mean Arsenic Concentration"
)

plt.tight_layout()


district_plot_path = (
    OUTPUT_DIR
    / "Top_15_Districts_Arsenic.png"
)


plt.savefig(
    district_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DETECTION RATE BY DISTRICT
# ============================================================

print(
    "\nGenerating arsenic detection rate plot..."
)


top_detection = (
    district_summary
    .sort_values(
        "detection_percentage",
        ascending=False
    )
    .head(15)
    .sort_values(
        "detection_percentage"
    )
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(
    top_detection.index,
    top_detection[
        "detection_percentage"
    ]
)


plt.xlabel(
    "Arsenic Detection (%)"
)

plt.ylabel(
    "District"
)

plt.title(
    "Top 15 Districts by Arsenic "
    "Detection Percentage"
)

plt.tight_layout()


detection_plot_path = (
    OUTPUT_DIR
    / "Top_15_Districts_Detection_Rate.png"
)


plt.savefig(
    detection_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COORDINATE RANGE
# ============================================================

print("\n" + "-" * 70)
print("GEOGRAPHIC COVERAGE")
print("-" * 70)

print(
    f"Latitude range : "
    f"{spatial_df['Latitude'].min():.4f} "
    f"to "
    f"{spatial_df['Latitude'].max():.4f}"
)

print(
    f"Longitude range: "
    f"{spatial_df['Longitude'].min():.4f} "
    f"to "
    f"{spatial_df['Longitude'].max():.4f}"
)

print(
    f"Districts covered: "
    f"{df['District'].nunique()}"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SPATIAL ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print(
    f"\n1. District summary:"
    f"\n   {summary_path}"
)

print(
    f"\n2. Spatial distribution:"
    f"\n   {spatial_plot_path}"
)

print(
    f"\n3. Top district plot:"
    f"\n   {district_plot_path}"
)

print(
    f"\n4. Detection rate plot:"
    f"\n   {detection_plot_path}"
)

print("\n" + "=" * 70)