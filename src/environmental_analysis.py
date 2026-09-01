"""
UP ARSENIC PREDICTION PROJECT
ENVIRONMENTAL ANALYSIS

Analyzes relationships between environmental variables
(rainfall and soil moisture) and arsenic concentration.
"""

from pathlib import Path

import numpy as np
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
print("UP ARSENIC PREDICTION - ENVIRONMENTAL ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found at:\n{DATA_PATH}"
    )


print("\nLoading master dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "As",

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


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nRequired environmental columns missing:\n"
        f"{missing_columns}"
    )


# ============================================================
# ENVIRONMENTAL VARIABLES
# ============================================================

rainfall_variables = [
    "rainfall_mean_annual_mm",
    "rainfall_mean_monsoon_mm",
    "rainfall_mean_premonsoon_mm",
    "rainfall_mean_postmonsoon_mm",
    "rainfall_mean_winter_mm"
]


soil_variables = [
    "soil_moisture_mean_2020",
    "soil_moisture_median_2020",
    "soil_moisture_std_2020",
    "soil_moisture_volume_mean_2020",
    "soil_moisture_pct_mean_2020"
]


environmental_variables = (
    rainfall_variables
    + soil_variables
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "-" * 70)
print("ENVIRONMENTAL VARIABLES VS ARSENIC")
print("-" * 70)


correlation_data = df[
    ["As"] + environmental_variables
].copy()


correlations = (
    correlation_data
    .corr(numeric_only=True)["As"]
    .drop("As")
    .sort_values(
        ascending=False
    )
)


print("\nCorrelation with arsenic:")

print(correlations)


# ============================================================
# SAVE CORRELATIONS
# ============================================================

correlation_output = pd.DataFrame({
    "Environmental_Variable":
        correlations.index,

    "Correlation_with_As":
        correlations.values,

    "Absolute_Correlation":
        correlations.abs().values
})


correlation_path = (
    OUTPUT_DIR
    / "Environmental_Arsenic_Correlations.csv"
)


correlation_output.to_csv(
    correlation_path,
    index=False
)


# ============================================================
# RAINFALL CORRELATION PLOT
# ============================================================

print(
    "\nGenerating rainfall correlation plot..."
)


rainfall_corr = (
    correlations[
        correlations.index.isin(
            rainfall_variables
        )
    ]
    .sort_values()
)


plt.figure(
    figsize=(10, 6)
)


plt.barh(
    rainfall_corr.index,
    rainfall_corr.values
)


plt.xlabel(
    "Correlation with Arsenic"
)

plt.ylabel(
    "Rainfall Variable"
)

plt.title(
    "Rainfall Variables vs Arsenic Concentration"
)

plt.axvline(
    x=0,
    linestyle="--"
)

plt.tight_layout()


rainfall_corr_path = (
    OUTPUT_DIR
    / "Rainfall_Arsenic_Correlation.png"
)


plt.savefig(
    rainfall_corr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SOIL MOISTURE CORRELATION PLOT
# ============================================================

print(
    "Generating soil moisture correlation plot..."
)


soil_corr = (
    correlations[
        correlations.index.isin(
            soil_variables
        )
    ]
    .sort_values()
)


plt.figure(
    figsize=(10, 6)
)


plt.barh(
    soil_corr.index,
    soil_corr.values
)


plt.xlabel(
    "Correlation with Arsenic"
)

plt.ylabel(
    "Soil Moisture Variable"
)

plt.title(
    "Soil Moisture Variables vs Arsenic Concentration"
)

plt.axvline(
    x=0,
    linestyle="--"
)

plt.tight_layout()


soil_corr_path = (
    OUTPUT_DIR
    / "Soil_Moisture_Arsenic_Correlation.png"
)


plt.savefig(
    soil_corr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ENVIRONMENTAL CORRELATION HEATMAP
# ============================================================

print(
    "Generating environmental correlation matrix..."
)


matrix = correlation_data.corr(
    numeric_only=True
)


plt.figure(
    figsize=(12, 10)
)


image = plt.imshow(
    matrix,
    aspect="auto"
)


plt.colorbar(
    image,
    label="Correlation"
)


plt.xticks(
    range(len(matrix.columns)),
    matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(matrix.index)),
    matrix.index
)


plt.title(
    "Environmental Variables and Arsenic "
    "Correlation Matrix"
)


plt.tight_layout()


heatmap_path = (
    OUTPUT_DIR
    / "Environmental_Correlation_Matrix.png"
)


plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RAINFALL VS ARSENIC
# ============================================================

print(
    "\nGenerating rainfall vs arsenic plots..."
)


for variable in rainfall_variables:

    plot_df = df[
        [variable, "As"]
    ].dropna()

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        plot_df[variable],
        plot_df["As"],
        alpha=0.6
    )

    plt.xlabel(
        variable
    )

    plt.ylabel(
        "Arsenic Concentration"
    )

    plt.title(
        f"{variable} vs Arsenic"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    safe_name = variable.replace(
        "rainfall_",
        ""
    )

    output_path = (
        OUTPUT_DIR
        / f"{safe_name}_vs_Arsenic.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SOIL MOISTURE VS ARSENIC
# ============================================================

print(
    "Generating soil moisture vs arsenic plots..."
)


for variable in soil_variables:

    plot_df = df[
        [variable, "As"]
    ].dropna()

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        plot_df[variable],
        plot_df["As"],
        alpha=0.6
    )

    plt.xlabel(
        variable
    )

    plt.ylabel(
        "Arsenic Concentration"
    )

    plt.title(
        f"{variable} vs Arsenic"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    safe_name = variable.replace(
        "soil_moisture_",
        ""
    )

    output_path = (
        OUTPUT_DIR
        / f"{safe_name}_vs_Arsenic.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# ENVIRONMENTAL SUMMARY STATISTICS
# ============================================================

print(
    "\nGenerating environmental summary..."
)


summary = df[
    ["As"] + environmental_variables
].describe().T


summary_path = (
    OUTPUT_DIR
    / "Environmental_Summary_Statistics.csv"
)


summary.to_csv(
    summary_path
)


# ============================================================
# STRONGEST ENVIRONMENTAL RELATIONSHIPS
# ============================================================

print("\n" + "-" * 70)
print("STRONGEST ENVIRONMENTAL RELATIONSHIPS")
print("-" * 70)


strongest = (
    correlation_output
    .sort_values(
        "Absolute_Correlation",
        ascending=False
    )
)


print(
    strongest.head(10)
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "Environmental_Analysis_Report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "UP ARSENIC PREDICTION PROJECT\n"
    )

    file.write(
        "ENVIRONMENTAL ANALYSIS REPORT\n"
    )

    file.write(
        "=" * 60
    )

    file.write("\n\n")

    file.write(
        "Environmental variables analyzed:\n\n"
    )

    for variable in environmental_variables:

        file.write(
            f"- {variable}\n"
        )

    file.write(
        "\n\nCorrelations with arsenic:\n\n"
    )

    for _, row in correlation_output.iterrows():

        file.write(
            f"{row['Environmental_Variable']}: "
            f"{row['Correlation_with_As']:.6f}\n"
        )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("ENVIRONMENTAL ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated outputs:")

print(
    f"\n1. {correlation_path}"
)

print(
    f"\n2. {rainfall_corr_path}"
)

print(
    f"\n3. {soil_corr_path}"
)

print(
    f"\n4. {heatmap_path}"
)

print(
    f"\n5. {summary_path}"
)

print(
    f"\n6. {report_path}"
)

print("\n" + "=" * 70)