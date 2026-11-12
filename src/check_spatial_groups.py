from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Arsenic_Classification_Dataset.csv"
)

OUTPUT_DIR = ROOT / "docs" / "spatial_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)

    required = ["Latitude", "Longitude", "arsenic_detected"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Latitude"] = pd.to_numeric(
        df["Latitude"], errors="coerce"
    )

    df["Longitude"] = pd.to_numeric(
        df["Longitude"], errors="coerce"
    )

    df["arsenic_detected"] = pd.to_numeric(
        df["arsenic_detected"], errors="coerce"
    )

    df = df.dropna(subset=required).copy()

    print("\nDATASET OVERVIEW")
    print("Rows with valid coordinates:", len(df))
    print("Unique exact coordinate pairs:",
          df[["Latitude", "Longitude"]].drop_duplicates().shape[0])

    for decimals in [1, 2, 3]:
        group_col = f"grid_{decimals}dp"

        df[group_col] = (
            df["Latitude"].round(decimals).astype(str)
            + "_"
            + df["Longitude"].round(decimals).astype(str)
        )

        group_sizes = df.groupby(group_col).size()

        print(f"\nGRID PRECISION: {decimals} DECIMAL PLACES")
        print("Unique groups:", group_sizes.size)
        print("Groups with one sample:",
              int((group_sizes == 1).sum()))
        print("Groups with multiple samples:",
              int((group_sizes > 1).sum()))
        print("Largest group size:", int(group_sizes.max()))
        print("Median group size:", float(group_sizes.median()))

        group_summary = (
            df.groupby(group_col)
            .agg(
                sample_count=("arsenic_detected", "size"),
                positive_count=("arsenic_detected", "sum"),
                latitude_mean=("Latitude", "mean"),
                longitude_mean=("Longitude", "mean"),
            )
            .reset_index()
        )

        group_summary.to_csv(
            OUTPUT_DIR / f"spatial_groups_{decimals}dp.csv",
            index=False,
        )

    print("\nFiles saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()