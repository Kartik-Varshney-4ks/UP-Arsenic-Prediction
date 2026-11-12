"""
HOTSPOT EXPANSION AND CONTRACTION ANALYSIS
==========================================

Analyzes spatial changes in groundwater arsenic hotspots
between 2023 and 2024.

Definitions:

Expansion:
    Non-hotspot in 2023 -> Hotspot in 2024

Contraction:
    Hotspot in 2023 -> Non-hotspot in 2024

Persistence:
    Hotspot in 2023 -> Hotspot in 2024

Non-hotspot:
    Non-hotspot in both years

Input:
    docs/hotspot_analysis/hotspot_change_2023_2024.csv

Outputs:
    expansion_contraction_2023_2024.csv
    expansion_contraction_summary.csv
    expansion_contraction_map_2023_2024.png
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
        f"Loaded data: {df.shape}"
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
        "Hotspot_2024"
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
# CLASSIFY SPATIAL CHANGE
# ============================================================

def classify_changes(df):

    result = df.copy()

    conditions = [

        (
            (result["Hotspot_2023"] == 0)
            &
            (result["Hotspot_2024"] == 1)
        ),

        (
            (result["Hotspot_2023"] == 1)
            &
            (result["Hotspot_2024"] == 0)
        ),

        (
            (result["Hotspot_2023"] == 1)
            &
            (result["Hotspot_2024"] == 1)
        ),

        (
            (result["Hotspot_2023"] == 0)
            &
            (result["Hotspot_2024"] == 0)
        )
    ]

    categories = [
        "Expansion",
        "Contraction",
        "Persistent",
        "Non-hotspot"
    ]

    result["Spatial_Change"] = (
        pd.Series(
            "Unknown",
            index=result.index
        )
    )

    for condition, category in zip(
        conditions,
        categories
    ):

        result.loc[
            condition,
            "Spatial_Change"
        ] = category

    return result


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(df):

    total_cells = len(df)

    expansion = (
        df["Spatial_Change"] == "Expansion"
    ).sum()

    contraction = (
        df["Spatial_Change"] == "Contraction"
    ).sum()

    persistent = (
        df["Spatial_Change"] == "Persistent"
    ).sum()

    non_hotspot = (
        df["Spatial_Change"] == "Non-hotspot"
    ).sum()

    hotspot_2023 = (
        df["Hotspot_2023"] == 1
    ).sum()

    hotspot_2024 = (
        df["Hotspot_2024"] == 1
    ).sum()

    net_change = (
        hotspot_2024
        - hotspot_2023
    )

    # Expansion/contraction ratio
    if contraction > 0:

        expansion_contraction_ratio = (
            expansion / contraction
        )

    else:

        expansion_contraction_ratio = None

    # Percentages relative to all spatial cells
    expansion_pct = (
        expansion / total_cells * 100
        if total_cells > 0
        else 0
    )

    contraction_pct = (
        contraction / total_cells * 100
        if total_cells > 0
        else 0
    )

    metrics = pd.DataFrame(
        {
            "Metric": [
                "Total spatial cells",
                "2023 hotspot cells",
                "2024 hotspot cells",
                "Expansion cells",
                "Contraction cells",
                "Persistent hotspot cells",
                "Non-hotspot cells",
                "Net hotspot change",
                "Expansion cells (%)",
                "Contraction cells (%)",
                "Expansion / Contraction ratio"
            ],

            "Value": [
                total_cells,
                hotspot_2023,
                hotspot_2024,
                expansion,
                contraction,
                persistent,
                non_hotspot,
                net_change,
                round(
                    expansion_pct,
                    2
                ),
                round(
                    contraction_pct,
                    2
                ),
                (
                    round(
                        expansion_contraction_ratio,
                        3
                    )
                    if expansion_contraction_ratio
                    is not None
                    else "N/A"
                )
            ]
        }
    )

    return metrics


# ============================================================
# SAVE CHANGE DATA
# ============================================================

def save_change_data(df):

    output_file = (
        OUTPUT_DIR
        / "expansion_contraction_2023_2024.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved:\n{output_file}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(metrics):

    output_file = (
        OUTPUT_DIR
        / "expansion_contraction_summary.csv"
    )

    metrics.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved:\n{output_file}"
    )


# ============================================================
# CREATE MAP
# ============================================================

def create_change_map(df):

    output_file = (
        OUTPUT_DIR
        / "expansion_contraction_map_2023_2024.png"
    )

    plt.figure(
        figsize=(11, 9)
    )

    categories = [
        "Expansion",
        "Contraction",
        "Persistent",
        "Non-hotspot"
    ]

    markers = {
        "Expansion": "^",
        "Contraction": "v",
        "Persistent": "s",
        "Non-hotspot": "o"
    }

    sizes = {
        "Expansion": 75,
        "Contraction": 75,
        "Persistent": 55,
        "Non-hotspot": 12
    }

    for category in categories:

        subset = df[
            df["Spatial_Change"] == category
        ]

        if subset.empty:

            continue

        plt.scatter(
            subset["Grid_Lon"],
            subset["Grid_Lat"],
            marker=markers[category],
            s=sizes[category],
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
        "Groundwater Arsenic Hotspot Expansion "
        "and Contraction\n2023 → 2024"
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
        f"Saved map:\n{output_file}"
    )


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(metrics):

    print(
        "\n========================================"
    )

    print(
        "HOTSPOT EXPANSION / CONTRACTION REPORT"
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
        "ARSENIC HOTSPOT EXPANSION & CONTRACTION"
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
    # CLASSIFY
    # --------------------------------------------------------

    df = classify_changes(
        df
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        df
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_change_data(
        df
    )

    save_summary(
        metrics
    )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    create_change_map(
        df
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print_report(
        metrics
    )

    print(
        "\nExpansion/contraction analysis complete."
    )