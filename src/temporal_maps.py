"""
Temporal Arsenic Risk Mapping

Creates geographic maps of observed arsenic concentration
for different years.

Current supported years:
    2023
    2024

This module does NOT modify the trained ML model.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = BASE_DIR / "docs" / "temporal_maps"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(file_path):
    """
    Load arsenic dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_columns(df):
    """
    Check whether required temporal/spatial columns exist.
    """

    required_columns = [
        "Latitude",
        "Longitude",
        "Year",
        "As"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_temporal_data(df):
    """
    Clean the columns required for temporal mapping.
    """

    validate_columns(df)

    data = df.copy()

    data["Latitude"] = pd.to_numeric(
        data["Latitude"],
        errors="coerce"
    )

    data["Longitude"] = pd.to_numeric(
        data["Longitude"],
        errors="coerce"
    )

    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="coerce"
    )

    data["As"] = pd.to_numeric(
        data["As"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "Latitude",
            "Longitude",
            "Year",
            "As"
        ]
    )

    return data


# ============================================================
# CREATE MAP
# ============================================================

def create_year_map(
    df,
    year,
    output_file=None
):
    """
    Create arsenic concentration map for one year.
    """

    year_data = df[
        df["Year"] == year
    ].copy()

    if year_data.empty:

        raise ValueError(
            f"No data available for year {year}"
        )

    plt.figure(
        figsize=(10, 8)
    )

    scatter = plt.scatter(
        year_data["Longitude"],
        year_data["Latitude"],
        c=year_data["As"],
        s=25,
        alpha=0.75
    )

    plt.colorbar(
        scatter,
        label="Arsenic concentration (As)"
    )

    plt.xlabel(
        "Longitude"
    )

    plt.ylabel(
        "Latitude"
    )

    plt.title(
        f"Groundwater Arsenic Distribution — {int(year)}"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    if output_file is None:

        output_file = (
            OUTPUT_DIR
            / f"arsenic_map_{int(year)}.png"
        )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Map saved: {output_file}"
    )


# ============================================================
# CREATE ALL YEAR MAPS
# ============================================================

def create_all_maps(
    df,
    years=None
):
    """
    Create maps for all requested years.
    """

    if years is None:

        years = sorted(
            df["Year"]
            .dropna()
            .unique()
        )

    for year in years:

        create_year_map(
            df,
            year
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "TEMPORAL ARSENIC MAPPING"
    )

    print(
        "========================================\n"
    )

    # --------------------------------------------------------
    # CHANGE THIS PATH IF YOUR DATASET HAS A DIFFERENT NAME
    # --------------------------------------------------------

    dataset_path = (
        DATA_DIR
        / "processed"
        / "UP_Arsenic_Climate_Environmental_Master_Working.csv"
    )

    print(
        f"Loading dataset:\n{dataset_path}\n"
    )

    df = load_dataset(
        dataset_path
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Available years: "
        f"{sorted(df['Year'].dropna().unique())}\n"
    )

    df = prepare_temporal_data(
        df
    )

    print(
        f"Cleaned data shape: {df.shape}\n"
    )

    create_all_maps(
        df
    )

    print(
        "\n========================================"
    )

    print(
        "TEMPORAL MAP GENERATION COMPLETE"
    )

    print(
        "========================================"
    )