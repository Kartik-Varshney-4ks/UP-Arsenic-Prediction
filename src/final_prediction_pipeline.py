"""
UP ARSENIC PREDICTION PROJECT
FINAL PREDICTION PIPELINE

Purpose:
    Load the trained classification model and provide
    a reusable prediction function for the Streamlit app.

Prediction:
    0 = Arsenic not detected
    1 = Arsenic detected
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_arsenic_classification_model.joblib"
)


# ============================================================
# FEATURE LIST
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
    "soil_observations_2020"
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found at:\n{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    return model


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(data):
    """
    Convert user input into the exact feature
    structure expected by the trained model.
    """

    if isinstance(data, dict):

        input_data = data.copy()

    elif isinstance(data, pd.DataFrame):

        input_data = data.to_dict(
            orient="records"
        )[0]

    else:

        raise TypeError(
            "Input must be a dictionary "
            "or pandas DataFrame."
        )


    missing_features = [
        feature
        for feature in FEATURES
        if feature not in input_data
    ]


    if missing_features:

        raise ValueError(
            "Missing required features:\n"
            f"{missing_features}"
        )


    row = {
        feature: input_data[feature]
        for feature in FEATURES
    }


    prepared = pd.DataFrame(
        [row],
        columns=FEATURES
    )


    # Convert infinite values to NaN.
    # The trained pipeline handles missing values.
    prepared = prepared.replace(
        [np.inf, -np.inf],
        np.nan
    )


    return prepared


# ============================================================
# PREDICT
# ============================================================

def predict_arsenic_risk(data):

    model = load_model()

    X = prepare_input(
        data
    )


    prediction = model.predict(
        X
    )[0]


    probability = model.predict_proba(
        X
    )[0, 1]


    if prediction == 1:

        risk_label = (
            "Arsenic Detected"
        )

    else:

        risk_label = (
            "Arsenic Not Detected"
        )


    return {
        "prediction": int(prediction),

        "probability": float(
            probability
        ),

        "risk_label": risk_label
    }


# ============================================================
# TEST PIPELINE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "UP ARSENIC PREDICTION - "
        "FINAL PREDICTION PIPELINE"
    )
    print("=" * 70)


    print(
        "\nLoading trained model..."
    )

    model = load_model()

    print(
        "Model loaded successfully."
    )


    print(
        f"\nNumber of expected features: "
        f"{len(FEATURES)}"
    )


    # --------------------------------------------------------
    # Example input
    # --------------------------------------------------------

    example_input = {

        "Latitude": 27.1767,

        "Longitude": 78.0081,

        "pH": 7.2,

        "EC": 650,

        "CO3": 10,

        "HCO3": 250,

        "Cl": 100,

        "F": 0.8,

        "SO4": 50,

        "NO3": 20,

        "PO4": 1,

        "Total_Hardness": 300,

        "Ca": 80,

        "Mg": 30,

        "Na": 60,

        "K": 5,

        "Fe": 1.2,

        "U": 0.02,

        "rainfall_mean_annual_mm": 800,

        "rainfall_mean_monsoon_mm": 650,

        "rainfall_mean_premonsoon_mm": 100,

        "rainfall_mean_postmonsoon_mm": 50,

        "rainfall_mean_winter_mm": 20,

        "soil_moisture_mean_2020": 0.25,

        "soil_moisture_median_2020": 0.24,

        "soil_moisture_std_2020": 0.05,

        "soil_moisture_volume_mean_2020": 0.25,

        "soil_moisture_pct_mean_2020": 25,

        "soil_observations_2020": 100
    }


    print(
        "\nRunning example prediction..."
    )


    result = predict_arsenic_risk(
        example_input
    )


    print("\n" + "-" * 70)
    print("PREDICTION RESULT")
    print("-" * 70)


    print(
        f"\nPrediction: "
        f"{result['prediction']}"
    )

    print(
        f"Risk label: "
        f"{result['risk_label']}"
    )

    print(
        f"Detection probability: "
        f"{result['probability']:.4f}"
    )


    print("\n" + "=" * 70)
    print(
        "FINAL PREDICTION PIPELINE "
        "TEST COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)