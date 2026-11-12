from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Arsenic_Classification_Dataset.csv"
)

OUTPUT_DIR = ROOT / "docs" / "spatial_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SIZES_KM = [10, 25, 50, 100]


def main():
    df = pd.read_csv(DATA_PATH)

    required = [
        "Latitude",
        "Longitude",
        "arsenic_detected",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(subset=required).copy()

    valid_coordinates = (
        df["Latitude"].between(-90, 90)
        & df["Longitude"].between(-180, 180)
    )

    df = df.loc[valid_coordinates].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid coordinate observations found.")

    # Estimate a suitable local projection from the dataset centre.
    centre_lat = df["Latitude"].median()
    centre_lon = df["Longitude"].median()

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={centre_lat} "
        f"+lon_0={centre_lon} +datum=WGS84 "
        f"+units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True,
    )

    x, y = transformer.transform(
        df["Longitude"].to_numpy(),
        df["Latitude"].to_numpy(),
    )

    df["projected_x_m"] = x
    df["projected_y_m"] = y

    print("\nDATASET SUMMARY")
    print("Valid observations:", len(df))
    print("Median latitude:", round(centre_lat, 5))
    print("Median longitude:", round(centre_lon, 5))

    summary = []

    for grid_km in GRID_SIZES_KM:
        cell_size_m = grid_km * 1000

        grid_x = np.floor(
            df["projected_x_m"] / cell_size_m
        ).astype(int)

        grid_y = np.floor(
            df["projected_y_m"] / cell_size_m
        ).astype(int)

        group = (
            grid_x.astype(str)
            + "_"
            + grid_y.astype(str)
        )

        group_sizes = group.value_counts()

        row = {
            "Grid_km": grid_km,
            "Observations": len(df),
            "Unique_cells": int(group.nunique()),
            "Singleton_cells": int(
                (group_sizes == 1).sum()
            ),
            "Cells_with_multiple_samples": int(
                (group_sizes > 1).sum()
            ),
            "Largest_cell_size": int(
                group_sizes.max()
            ),
            "Median_cell_size": float(
                group_sizes.median()
            ),
        }

        summary.append(row)

        cell_data = df.copy()
        cell_data["Spatial_Cell"] = group

        cell_data.to_csv(
            OUTPUT_DIR
            / f"spatial_grid_{grid_km}km.csv",
            index=False,
        )

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUTPUT_DIR / "spatial_grid_size_comparison.csv",
        index=False,
    )

    print("\nSPATIAL GRID COMPARISON")
    print(summary_df.to_string(index=False))

    print("\nSaved diagnostic files to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()