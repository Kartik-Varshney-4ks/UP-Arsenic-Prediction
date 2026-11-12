"""
HOTSPOT ANALYSIS
================

Compares groundwater arsenic hotspots between 2023 and 2024.

Hotspot threshold:
    As >= 10 ppb

Spatial comparison:
    Latitude/Longitude grid cells

Outputs:
    - hotspot map for 2023
    - hotspot map for 2024
    - hotspot change map
    - hotspot change CSV
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
    / "src"
    / "UP_Arsenic_Temporal_2023_2024.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "docs"
    / "hotspot_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

ARSENIC_THRESHOLD = 10.0

# Approximately 5 km grid at UP latitude.
# Smaller value = finer spatial resolution.
GRID_SIZE = 0.05


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset loaded: {df.shape}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    required_columns = [
        "Year",
        "Latitude",
        "Longitude",
        "As_ppb"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    data = df.copy()

    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="coerce"
    )

    data["Latitude"] = pd.to_numeric(
        data["Latitude"],
        errors="coerce"
    )

    data["Longitude"] = pd.to_numeric(
        data["Longitude"],
        errors="coerce"
    )

    data["As_ppb"] = pd.to_numeric(
        data["As_ppb"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "Year",
            "Latitude",
            "Longitude",
            "As_ppb"
        ]
    )

    # Hotspot classification
    data["Hotspot"] = (
        data["As_ppb"] >= ARSENIC_THRESHOLD
    ).astype(int)

    return data


# ============================================================
# CREATE SPATIAL GRID
# ============================================================

def create_grid(df):

    data = df.copy()

    data["Grid_Lat"] = (
        (data["Latitude"] / GRID_SIZE)
        .round()
        * GRID_SIZE
    )

    data["Grid_Lon"] = (
        (data["Longitude"] / GRID_SIZE)
        .round()
        * GRID_SIZE
    )

    return data


# ============================================================
# AGGREGATE HOTSPOTS BY GRID
# ============================================================

def create_year_grid(df, year):

    year_data = df[
        df["Year"] == year
    ].copy()

    if year_data.empty:

        raise ValueError(
            f"No data available for {year}"
        )

    # A grid cell is considered a hotspot if
    # at least one observation in that cell
    # exceeds the threshold.

    grid = (
        year_data
        .groupby(
            [
                "Grid_Lat",
                "Grid_Lon"
            ]
        )
        .agg(
            Observations=(
                "As_ppb",
                "count"
            ),

            Maximum_As=(
                "As_ppb",
                "max"
            ),

            Mean_As=(
                "As_ppb",
                "mean"
            ),

            Hotspot_Count=(
                "Hotspot",
                "sum"
            )
        )
        .reset_index()
    )

    grid["Hotspot"] = (
        grid["Hotspot_Count"] > 0
    ).astype(int)

    return grid


# ============================================================
# CREATE HOTSPOT MAP
# ============================================================

def create_hotspot_map(
    grid,
    year
):

    output_file = (
        OUTPUT_DIR
        / f"hotspot_map_{year}.png"
    )

    plt.figure(
        figsize=(10, 8)
    )

    normal = grid[
        grid["Hotspot"] == 0
    ]

    hotspots = grid[
        grid["Hotspot"] == 1
    ]

    # Non-hotspot cells
    plt.scatter(
        normal["Grid_Lon"],
        normal["Grid_Lat"],
        s=18,
        alpha=0.35,
        label="Non-hotspot"
    )

    # Hotspot cells
    plt.scatter(
        hotspots["Grid_Lon"],
        hotspots["Grid_Lat"],
        s=55,
        alpha=0.85,
        label=f"Hotspot (As ≥ {ARSENIC_THRESHOLD} ppb)"
    )

    plt.xlabel(
        "Longitude"
    )

    plt.ylabel(
        "Latitude"
    )

    plt.title(
        f"Groundwater Arsenic Hotspots — {year}"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved hotspot map: {output_file}"
    )


# ============================================================
# COMPARE TWO YEARS
# ============================================================

def compare_years(
    grid_2023,
    grid_2024
):

    # Create common spatial grid index

    g23 = grid_2023[
        [
            "Grid_Lat",
            "Grid_Lon",
            "Hotspot",
            "Observations",
            "Maximum_As",
            "Mean_As"
        ]
    ].copy()

    g24 = grid_2024[
        [
            "Grid_Lat",
            "Grid_Lon",
            "Hotspot",
            "Observations",
            "Maximum_As",
            "Mean_As"
        ]
    ].copy()

    # Rename year-specific columns

    g23 = g23.rename(
        columns={
            "Hotspot": "Hotspot_2023",
            "Observations": "Observations_2023",
            "Maximum_As": "Maximum_As_2023",
            "Mean_As": "Mean_As_2023"
        }
    )

    g24 = g24.rename(
        columns={
            "Hotspot": "Hotspot_2024",
            "Observations": "Observations_2024",
            "Maximum_As": "Maximum_As_2024",
            "Mean_As": "Mean_As_2024"
        }
    )

    # Combine spatial cells

    comparison = pd.merge(
        g23,
        g24,
        on=[
            "Grid_Lat",
            "Grid_Lon"
        ],
        how="outer"
    )

    # Missing cells mean no observation
    # in that year.

    comparison["Hotspot_2023"] = (
        comparison["Hotspot_2023"]
        .fillna(0)
        .astype(int)
    )

    comparison["Hotspot_2024"] = (
        comparison["Hotspot_2024"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # HOTSPOT CHANGE CLASSIFICATION
    # --------------------------------------------------------

    def classify_change(row):

        h23 = row["Hotspot_2023"]
        h24 = row["Hotspot_2024"]

        if h23 == 0 and h24 == 0:

            return "Non-hotspot"

        if h23 == 0 and h24 == 1:

            return "New hotspot"

        if h23 == 1 and h24 == 0:

            return "Disappeared hotspot"

        if h23 == 1 and h24 == 1:

            return "Persistent hotspot"

        return "Unknown"

    comparison["Change_Type"] = (
        comparison.apply(
            classify_change,
            axis=1
        )
    )

    return comparison


# ============================================================
# CREATE CHANGE MAP
# ============================================================

def create_change_map(
    comparison
):

    output_file = (
        OUTPUT_DIR
        / "hotspot_change_2023_2024.png"
    )

    plt.figure(
        figsize=(11, 9)
    )

    categories = [
        "Non-hotspot",
        "New hotspot",
        "Disappeared hotspot",
        "Persistent hotspot"
    ]

    markers = {
        "Non-hotspot": "o",
        "New hotspot": "^",
        "Disappeared hotspot": "x",
        "Persistent hotspot": "s"
    }

    sizes = {
        "Non-hotspot": 12,
        "New hotspot": 65,
        "Disappeared hotspot": 65,
        "Persistent hotspot": 60
    }

    for category in categories:

        subset = comparison[
            comparison["Change_Type"] == category
        ]

        if subset.empty:

            continue

        plt.scatter(
            subset["Grid_Lon"],
            subset["Grid_Lat"],
            s=sizes[category],
            marker=markers[category],
            alpha=0.75,
            label=category
        )

    plt.xlabel(
        "Longitude"
    )

    plt.ylabel(
        "Latitude"
    )

    plt.title(
        "Groundwater Arsenic Hotspot Change: 2023 → 2024"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved change map: {output_file}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    comparison
):

    output_file = (
        OUTPUT_DIR
        / "hotspot_change_2023_2024.csv"
    )

    comparison.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved hotspot comparison:\n{output_file}"
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    comparison
):

    print(
        "\n========================================"
    )

    print(
        "HOTSPOT CHANGE SUMMARY"
    )

    print(
        "========================================\n"
    )

    summary = (
        comparison["Change_Type"]
        .value_counts()
    )

    for category, count in summary.items():

        print(
            f"{category}: {count}"
        )

    print(
        "\nThreshold:"
    )

    print(
        f"As >= {ARSENIC_THRESHOLD} ppb"
    )

    print(
        f"\nGrid size:"
    )

    print(
        f"{GRID_SIZE} degrees"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "GROUNDWATER ARSENIC HOTSPOT ANALYSIS"
    )

    print(
        "========================================\n"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    df = prepare_data(
        df
    )

    print(
        f"Clean observations: {len(df)}"
    )

    print(
        f"Years: "
        f"{sorted(df['Year'].unique())}"
    )

    # --------------------------------------------------------
    # CREATE GRID
    # --------------------------------------------------------

    df = create_grid(
        df
    )

    # --------------------------------------------------------
    # CREATE YEAR-SPECIFIC GRIDS
    # --------------------------------------------------------

    grid_2023 = create_year_grid(
        df,
        2023
    )

    grid_2024 = create_year_grid(
        df,
        2024
    )

    print(
        f"\n2023 spatial cells: "
        f"{len(grid_2023)}"
    )

    print(
        f"2024 spatial cells: "
        f"{len(grid_2024)}"
    )

    # --------------------------------------------------------
    # MAPS
    # --------------------------------------------------------

    create_hotspot_map(
        grid_2023,
        2023
    )

    create_hotspot_map(
        grid_2024,
        2024
    )

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    comparison = compare_years(
        grid_2023,
        grid_2024
    )

    # --------------------------------------------------------
    # CHANGE MAP
    # --------------------------------------------------------

    create_change_map(
        comparison
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        comparison
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_summary(
        comparison
    )

    print(
        "\n========================================"
    )

    print(
        "HOTSPOT ANALYSIS COMPLETE"
    )

    print(
        "========================================"
    )