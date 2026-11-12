from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "sampling_aware_hotspot_change_2023_2024.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "sampling_aware_summary.csv"
)

THRESHOLD = 10.0
GRID_SIZE = 0.05


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Loaded data:", df.shape)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "Year",
    "Latitude",
    "Longitude",
    "As_ppb"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

for col in [
    "Year",
    "Latitude",
    "Longitude",
    "As_ppb"
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "Year",
        "Latitude",
        "Longitude",
        "As_ppb"
    ]
)


df["Year"] = df["Year"].astype(int)


# ============================================================
# HOTSPOT CLASSIFICATION
# ============================================================

df["Hotspot"] = (
    df["As_ppb"] >= THRESHOLD
).astype(int)


# ============================================================
# CREATE SPATIAL GRID
# ============================================================

df["Grid_Lat"] = (
    (df["Latitude"] / GRID_SIZE)
    .round()
    * GRID_SIZE
)

df["Grid_Lon"] = (
    (df["Longitude"] / GRID_SIZE)
    .round()
    * GRID_SIZE
)


# ============================================================
# CREATE YEAR-WISE GRID STATUS
# ============================================================

year_grid = (
    df.groupby(
        [
            "Year",
            "Grid_Lat",
            "Grid_Lon"
        ]
    )
    .agg(
        Observations=("As_ppb", "count"),
        Max_As_ppb=("As_ppb", "max"),
        Mean_As_ppb=("As_ppb", "mean"),
        Hotspot=("Hotspot", "max")
    )
    .reset_index()
)


# ============================================================
# SPLIT YEARS
# ============================================================

data_2023 = year_grid[
    year_grid["Year"] == 2023
].copy()

data_2024 = year_grid[
    year_grid["Year"] == 2024
].copy()


# ============================================================
# RENAME COLUMNS
# ============================================================

data_2023 = data_2023.rename(
    columns={
        "Observations": "Observations_2023",
        "Max_As_ppb": "Max_As_2023",
        "Mean_As_ppb": "Mean_As_2023",
        "Hotspot": "Hotspot_2023"
    }
)

data_2024 = data_2024.rename(
    columns={
        "Observations": "Observations_2024",
        "Max_As_ppb": "Max_As_2024",
        "Mean_As_ppb": "Mean_As_2024",
        "Hotspot": "Hotspot_2024"
    }
)


# ============================================================
# MERGE SPATIAL GRIDS
# ============================================================

comparison = pd.merge(
    data_2023,
    data_2024,
    on=[
        "Grid_Lat",
        "Grid_Lon"
    ],
    how="outer"
)


# ============================================================
# SAMPLING STATUS
# ============================================================

comparison["Sampled_2023"] = (
    comparison["Observations_2023"]
    .notna()
    .astype(int)
)

comparison["Sampled_2024"] = (
    comparison["Observations_2024"]
    .notna()
    .astype(int)
)


# ============================================================
# FILL HOTSPOT VALUES ONLY AFTER SAMPLING STATUS
# ============================================================

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


# ============================================================
# SAMPLING-AWARE CLASSIFICATION
# ============================================================

def classify(row):

    sampled_23 = row["Sampled_2023"]
    sampled_24 = row["Sampled_2024"]

    hotspot_23 = row["Hotspot_2023"]
    hotspot_24 = row["Hotspot_2024"]

    # Neither year sampled
    if sampled_23 == 0 and sampled_24 == 0:
        return "Not sampled"

    # Only 2024 sampled
    if sampled_23 == 0 and sampled_24 == 1:

        if hotspot_24 == 1:
            return "Newly sampled hotspot"

        return "Newly sampled non-hotspot"

    # Only 2023 sampled
    if sampled_23 == 1 and sampled_24 == 0:

        if hotspot_23 == 1:
            return "No 2024 observation"

        return "No 2024 observation"

    # Both years sampled
    if sampled_23 == 1 and sampled_24 == 1:

        if hotspot_23 == 0 and hotspot_24 == 0:
            return "Observed non-hotspot"

        if hotspot_23 == 0 and hotspot_24 == 1:
            return "Expansion"

        if hotspot_23 == 1 and hotspot_24 == 0:
            return "Contraction"

        if hotspot_23 == 1 and hotspot_24 == 1:
            return "Persistent hotspot"

    return "Unknown"


comparison["Change_Type"] = comparison.apply(
    classify,
    axis=1
)


# ============================================================
# SAVE RESULT
# ============================================================

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"Saved sampling-aware analysis: {OUTPUT_FILE}"
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    comparison["Change_Type"]
    .value_counts()
    .rename_axis("Change_Type")
    .reset_index(
        name="Spatial_Cells"
    )
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nSampling-aware hotspot summary:")
print(summary.to_string(index=False))


# ============================================================
# ADDITIONAL METRICS
# ============================================================

both_sampled = comparison[
    (comparison["Sampled_2023"] == 1)
    &
    (comparison["Sampled_2024"] == 1)
]


persistent = both_sampled[
    (both_sampled["Hotspot_2023"] == 1)
    &
    (both_sampled["Hotspot_2024"] == 1)
]


expansion = both_sampled[
    (both_sampled["Hotspot_2023"] == 0)
    &
    (both_sampled["Hotspot_2024"] == 1)
]


contraction = both_sampled[
    (both_sampled["Hotspot_2023"] == 1)
    &
    (both_sampled["Hotspot_2024"] == 0)
]


non_hotspot = both_sampled[
    (both_sampled["Hotspot_2023"] == 0)
    &
    (both_sampled["Hotspot_2024"] == 0)
]


print("\nBoth-year sampled cells:")
print(
    len(both_sampled)
)


print("\nPersistent hotspots:")
print(
    len(persistent)
)


print("\nExpansion:")
print(
    len(expansion)
)


print("\nContraction:")
print(
    len(contraction)
)


print("\nObserved non-hotspots:")
print(
    len(non_hotspot)
)


print("\nAnalysis complete.")