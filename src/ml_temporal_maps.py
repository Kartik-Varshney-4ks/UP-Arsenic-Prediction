"""
ML-Based Temporal Arsenic Risk Maps

Uses the existing trained XGBoost pipeline to generate
arsenic detection probabilities for observations and
creates separate maps for each available year.

The trained model is NOT modified.
"""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_arsenic_classification_model.joblib"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "UP_Arsenic_Climate_Environmental_Master_Working.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "docs"
    / "temporal_maps"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "Latitude",
    "Longitude",
    "pH",
    "EC",
    "CO3",
    "HCO3",
    "Cl",
    "F",
    "SO4",
    "NO3",
    "PO4",
    "Total_Hardness",
    "Ca",
    "Mg",
    "Na",
    "K",
    "Fe",
    "U",
    "rainfall_mean_annual_mm",
    "rainfall_mean_monsoon_mm",
    "rainfall_mean_premonsoon_mm",
    "rainfall_mean_postmonsoon_mm",
    "rainfall_mean_winter_mm",
    "soil_moisture_mean_2020",
    "soil_moisture_median_2020",
    "soil_moisture_std_2020",
    "soil_moisture_volume_mean_2020",
    "soil_moisture_pct_mean_2020",
    "soil_observations_2020",
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    return model


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
# VALIDATE DATA
# ============================================================

def validate_data(df):

    required = FEATURES + [
        "Year",
        "Latitude",
        "Longitude"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns:\n"
            + "\n".join(missing)
        )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    df
):

    X = df[FEATURES].copy()

    # Existing pipeline already handles preprocessing
    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    result = df.copy()

    result["ML_Probability"] = probabilities

    result["ML_Prediction"] = predictions

    return result


# ============================================================
# CREATE TEMPORAL MAP
# ============================================================

def create_prediction_map(
    df,
    year
):

    year_data = df[
        df["Year"] == year
    ].copy()

    if year_data.empty:

        print(
            f"No observations for {year}"
        )

        return

    output_file = (
        OUTPUT_DIR
        / f"ml_risk_map_{int(year)}.png"
    )

    plt.figure(
        figsize=(10, 8)
    )

    scatter = plt.scatter(
        year_data["Longitude"],
        year_data["Latitude"],
        c=year_data["ML_Probability"],
        s=30,
        alpha=0.8
    )

    plt.colorbar(
        scatter,
        label="Predicted arsenic detection probability"
    )

    plt.xlabel(
        "Longitude"
    )

    plt.ylabel(
        "Latitude"
    )

    plt.title(
        f"ML-Predicted Arsenic Risk — {int(year)}"
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
        f"Saved: {output_file}"
    )


# ============================================================
# SAVE PREDICTION DATA
# ============================================================

def save_predictions(df):

    output_file = (
        OUTPUT_DIR
        / "temporal_ml_predictions.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Prediction dataset saved:\n{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "ML TEMPORAL ARSENIC MAPPING"
    )

    print(
        "========================================\n"
    )

    model = load_model()

    df = load_data()

    validate_data(
        df
    )

    print(
        "\nGenerating ML predictions..."
    )

    result = generate_predictions(
        model,
        df
    )

    print(
        "\nPrediction summary:"
    )

    print(
        result["ML_Prediction"]
        .value_counts()
    )

    print(
        "\nAvailable years:"
    )

    print(
        sorted(
            result["Year"]
            .dropna()
            .unique()
        )
    )

    # --------------------------------------------------------
    # CREATE MAP FOR EVERY AVAILABLE YEAR
    # --------------------------------------------------------

    for year in sorted(
        result["Year"]
        .dropna()
        .unique()
    ):

        create_prediction_map(
            result,
            year
        )

    # --------------------------------------------------------
    # SAVE COMPLETE RESULTS
    # --------------------------------------------------------

    save_predictions(
        result
    )

    print(
        "\n========================================"
    )

    print(
        "ML TEMPORAL MAPPING COMPLETE"
    )

    print(
        "========================================"
    )