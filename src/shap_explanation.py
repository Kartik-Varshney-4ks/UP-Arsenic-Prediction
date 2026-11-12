"""
UP ARSENIC PREDICTION PROJECT
SHAP EXPLANATION MODULE

Purpose:
    Explain individual arsenic predictions using SHAP.

Model structure:
    sklearn Pipeline
        |
        ├── preprocessor
        |
        └── XGBoost model

This module:
    1. Loads the trained model
    2. Applies the same preprocessing used during training
    3. Extracts the XGBoost model
    4. Calculates SHAP values
    5. Returns feature-level contributions
"""

from pathlib import Path

import joblib
import pandas as pd
import shap

from src.final_prediction_pipeline import (
    prepare_input
)


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
# LOAD TRAINED MODEL
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
# EXTRACT XGBOOST MODEL
# ============================================================

def get_xgboost_model(model):

    """
    Extract the XGBoost classifier from
    the trained sklearn Pipeline.
    """

    xgb_model = model.named_steps[
        "model"
    ]

    return xgb_model


# ============================================================
# PREPROCESS INPUT
# ============================================================

def transform_input(
    model,
    data
):

    """
    Apply exactly the same preprocessing
    that was used during model training.
    """

    # Prepare input according to
    # the project's feature list.
    prepared = prepare_input(
        data
    )

    # Get preprocessing component
    preprocessor = model.named_steps[
        "preprocessor"
    ]

    # Transform input
    transformed = preprocessor.transform(
        prepared
    )

    return transformed


# ============================================================
# GET FEATURE NAMES
# ============================================================

def get_feature_names(model):

    """
    Get feature names after preprocessing.
    """

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    return list(
        feature_names
    )


# ============================================================
# GENERATE SHAP EXPLANATION
# ============================================================

def explain_prediction(
    data
):

    """
    Generate SHAP feature contributions
    for one arsenic prediction.

    Parameters
    ----------
    data : dict or pandas DataFrame
        Input groundwater observation.

    Returns
    -------
    pandas.DataFrame
        SHAP feature contribution table.
    """

    # --------------------------------------------------------
    # Load complete trained pipeline
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # Extract XGBoost model
    # --------------------------------------------------------

    xgb_model = get_xgboost_model(
        model
    )


    # --------------------------------------------------------
    # Apply preprocessing
    # --------------------------------------------------------

    transformed_input = transform_input(
        model,
        data
    )


    # --------------------------------------------------------
    # Get feature names
    # --------------------------------------------------------

    feature_names = get_feature_names(
        model
    )


    # --------------------------------------------------------
    # Create SHAP TreeExplainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        xgb_model
    )


    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        transformed_input
    )


    # --------------------------------------------------------
    # Handle different SHAP output formats
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        values = shap_values[-1][0]

    else:

        values = shap_values[0]


    # --------------------------------------------------------
    # Create explanation DataFrame
    # --------------------------------------------------------

    explanation = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": values
        }
    )


    # --------------------------------------------------------
    # Calculate absolute contribution
    # --------------------------------------------------------

    explanation[
        "Absolute SHAP"
    ] = (
        explanation[
            "SHAP Value"
        ].abs()
    )


    # --------------------------------------------------------
    # Sort by importance
    # --------------------------------------------------------

    explanation = explanation.sort_values(
        "Absolute SHAP",
        ascending=False
    )


    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    explanation = explanation.reset_index(
        drop=True
    )


    return explanation


# ============================================================
# TEST THE SHAP MODULE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "UP ARSENIC PREDICTION"
    )

    print(
        "SHAP EXPLANATION TEST"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Example groundwater observation
    # --------------------------------------------------------

    example_input = {

        "Latitude": 27.1767,

        "Longitude": 78.0081,


        # ----------------------------------------------------
        # Water Chemistry
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Rainfall
        # ----------------------------------------------------

        "rainfall_mean_annual_mm": 800,

        "rainfall_mean_monsoon_mm": 650,

        "rainfall_mean_premonsoon_mm": 100,

        "rainfall_mean_postmonsoon_mm": 50,

        "rainfall_mean_winter_mm": 20,


        # ----------------------------------------------------
        # Soil Moisture
        # ----------------------------------------------------

        "soil_moisture_mean_2020": 0.25,

        "soil_moisture_median_2020": 0.24,

        "soil_moisture_std_2020": 0.05,

        "soil_moisture_volume_mean_2020": 0.25,

        "soil_moisture_pct_mean_2020": 25,

        "soil_observations_2020": 100
    }


    # --------------------------------------------------------
    # Run SHAP
    # --------------------------------------------------------

    print(
        "\nRunning SHAP explanation..."
    )


    try:

        result = explain_prediction(
            example_input
        )


        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print("\n")

        print(
            "=" * 70
        )

        print(
            "TOP SHAP FEATURE CONTRIBUTIONS"
        )

        print(
            "=" * 70
        )


        print(
            result[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )


        # ----------------------------------------------------
        # Completion message
        # ----------------------------------------------------

        print("\n")

        print(
            "=" * 70
        )

        print(
            "SHAP TEST COMPLETED SUCCESSFULLY"
        )

        print(
            "=" * 70
        )


    except Exception as error:

        print("\n")

        print(
            "=" * 70
        )

        print(
            "SHAP TEST FAILED"
        )

        print(
            "=" * 70
        )


        print(
            f"\nError:\n{error}"
        )