"""
HOTSPOT PERSISTENCE ANALYSIS
============================

Analyzes arsenic hotspots that remain present from 2023 to 2024.

Input:
    docs/hotspot_analysis/hotspot_change_2023_2024.csv

Output:
    - persistent_hotspots_2023_2024.csv
    - persistent_hotspots_2023_2024.png
    - persistence_summary.csv
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "docs"
    / "hotspot_analysis"
    / "hotspot_change_2023_2024.csv"
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
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded hotspot change data: {df.shape}"
    )

    return df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(df):

    required_columns = [
        "Grid_Lat",
        "Grid_Lon",
        "Hotspot_2023",
        "Hotspot_2024",
        "Change_Type"
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


# ============================================================
# IDENTIFY PERSISTENT HOTSPOTS
# ============================================================

def identify_persistent_hotspots(df):

    persistent = df[
        (
            df["Hotspot_2023"] == 1
        )
        &
        (
            df["Hotspot_2024"] == 1
        )
    ].copy()

    # Explicitly label them
    persistent["Persistence"] = (
        "Persistent hotspot"
    )

    return persistent


# ============================================================
# CALCULATE PERSISTENCE METRICS
# ============================================================

def calculate_persistence_metrics(
    df,
    persistent
):

    hotspots_2023 = (
        df["Hotspot_2023"] == 1
    ).sum()

    hotspots_2024 = (
        df["Hotspot_2024"] == 1
    ).sum()

    persistent_count = len(
        persistent
    )

    if hotspots_2023 > 0:

        persistence_rate = (
            persistent_count
            / hotspots_2023
            * 100
        )

    else:

        persistence_rate = 0.0

    metrics = pd.DataFrame(
        {
            "Metric": [
                "2023 hotspot cells",
                "2024 hotspot cells",
                "Persistent hotspot cells",
                "2023-to-2024 persistence rate (%)"
            ],

            "Value": [
                hotspots_2023,
                hotspots_2024,
                persistent_count,
                round(
                    persistence_rate,
                    2
                )
            ]
        }
    )

    return metrics


# ============================================================
# SAVE PERSISTENT HOTSPOTS
# ============================================================

def save_persistent_hotspots(
    persistent
):

    output_file = (
        OUTPUT_DIR
        / "persistent_hotspots_2023_2024.csv"
    )

    persistent.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved persistent hotspot data:\n"
        f"{output_file}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    metrics
):

    output_file = (
        OUTPUT_DIR
        / "persistence_summary.csv"
    )

    metrics.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved persistence summary:\n"
        f"{output_file}"
    )


# ============================================================
# CREATE PERSISTENCE MAP
# ============================================================

def create_persistence_map(
    persistent
):

    output_file = (
        OUTPUT_DIR
        / "persistent_hotspots_2023_2024.png"
    )

    plt.figure(
        figsize=(10, 8)
    )

    if persistent.empty:

        plt.text(
            0.5,
            0.5,
            "No persistent hotspots detected",
            ha="center",
            va="center",
            fontsize=14
        )

        plt.axis(
            "off"
        )

    else:

        plt.scatter(
            persistent["Grid_Lon"],
            persistent["Grid_Lat"],
            s=70,
            alpha=0.85
        )

        plt.xlabel(
            "Longitude"
        )

        plt.ylabel(
            "Latitude"
        )

        plt.title(
            "Persistent Groundwater Arsenic Hotspots\n"
            "2023 → 2024"
        )

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
        f"Saved persistence map:\n"
        f"{output_file}"
    )


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    metrics,
    persistent
):

    print(
        "\n========================================"
    )

    print(
        "HOTSPOT PERSISTENCE REPORT"
    )

    print(
        "========================================\n"
    )

    for _, row in metrics.iterrows():

        print(
            f"{row['Metric']}: "
            f"{row['Value']}"
        )

    print(
        "\nPersistent hotspot cells:"
    )

    print(
        len(persistent)
    )

    if not persistent.empty:

        print(
            "\nPersistent hotspot locations:"
        )

        print(
            persistent[
                [
                    "Grid_Lat",
                    "Grid_Lon"
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\n========================================"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "GROUNDWATER ARSENIC HOTSPOT PERSISTENCE"
    )

    print(
        "========================================\n"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_data(
        df
    )

    # --------------------------------------------------------
    # IDENTIFY PERSISTENCE
    # --------------------------------------------------------

    persistent = (
        identify_persistent_hotspots(
            df
        )
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = (
        calculate_persistence_metrics(
            df,
            persistent
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_persistent_hotspots(
        persistent
    )

    save_summary(
        metrics
    )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    create_persistence_map(
        persistent
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print_report(
        metrics,
        persistent
    )

    print(
        "\nPersistence analysis complete."
    )