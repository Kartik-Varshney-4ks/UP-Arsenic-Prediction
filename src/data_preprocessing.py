from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input dataset
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "UP_Arsenic_Climate_Environmental_Master_Working.csv"
)


def load_dataset():
    """Load the integrated arsenic-environmental dataset."""
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 60)
    print("UP ARSENIC PREDICTION PROJECT")
    print("=" * 60)

    print(f"\nDataset loaded successfully!")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nFirst 5 records:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    df = load_dataset()