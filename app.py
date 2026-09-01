"""
UP ARSENIC PREDICTION
STREAMLIT APPLICATION

Research-oriented arsenic detection prediction
for groundwater observations in Uttar Pradesh.
"""

from pathlib import Path

import streamlit as st
import pandas as pd

from src.final_prediction_pipeline import (
    predict_arsenic_risk,
    FEATURES
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UP Arsenic Prediction",
    page_icon="💧",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_arsenic_classification_model.joblib"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "💧 Uttar Pradesh Arsenic Prediction"
)

st.markdown(
    """
    ### Groundwater Arsenic Detection Prediction

    This application uses groundwater chemistry,
    geographic coordinates, rainfall and soil-moisture
    variables to estimate the probability of arsenic
    detection.
    """
)


st.info(
    """
    **Research / Educational Tool**

    This prediction is based on the machine-learning model
    developed for this project. It should not be treated as
    a laboratory measurement, medical diagnosis, or official
    regulatory determination.
    """
)


# ============================================================
# MODEL CHECK
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        "Trained model was not found.\n\n"
        "Expected location:\n"
        f"{MODEL_PATH}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🧪 Input Parameters"
)

st.sidebar.caption(
    "Enter groundwater and environmental measurements."
)


# ============================================================
# GEOGRAPHIC INPUTS
# ============================================================

st.sidebar.subheader(
    "📍 Geographic Information"
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=27.1767,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=78.0081,
    format="%.6f"
)


# ============================================================
# WATER CHEMISTRY
# ============================================================

st.sidebar.subheader(
    "🧪 Water Chemistry"
)

pH = st.sidebar.number_input(
    "pH",
    value=7.2,
    min_value=0.0,
    max_value=14.0
)

EC = st.sidebar.number_input(
    "EC",
    value=650.0,
    min_value=0.0
)

CO3 = st.sidebar.number_input(
    "CO3",
    value=10.0,
    min_value=0.0
)

HCO3 = st.sidebar.number_input(
    "HCO3",
    value=250.0,
    min_value=0.0
)

Cl = st.sidebar.number_input(
    "Cl",
    value=100.0,
    min_value=0.0
)

F = st.sidebar.number_input(
    "F",
    value=0.8,
    min_value=0.0
)

SO4 = st.sidebar.number_input(
    "SO4",
    value=50.0,
    min_value=0.0
)

NO3 = st.sidebar.number_input(
    "NO3",
    value=20.0,
    min_value=0.0
)

PO4 = st.sidebar.number_input(
    "PO4",
    value=1.0,
    min_value=0.0
)

Total_Hardness = st.sidebar.number_input(
    "Total Hardness",
    value=300.0,
    min_value=0.0
)

Ca = st.sidebar.number_input(
    "Ca",
    value=80.0,
    min_value=0.0
)

Mg = st.sidebar.number_input(
    "Mg",
    value=30.0,
    min_value=0.0
)

Na = st.sidebar.number_input(
    "Na",
    value=60.0,
    min_value=0.0
)

K = st.sidebar.number_input(
    "K",
    value=5.0,
    min_value=0.0
)

Fe = st.sidebar.number_input(
    "Fe",
    value=1.2,
    min_value=0.0
)

U = st.sidebar.number_input(
    "U",
    value=0.02,
    min_value=0.0
)


# ============================================================
# RAINFALL INPUTS
# ============================================================

st.sidebar.subheader(
    "🌧️ Rainfall"
)

rainfall_mean_annual_mm = st.sidebar.number_input(
    "Annual Rainfall (mm)",
    value=800.0,
    min_value=0.0
)

rainfall_mean_monsoon_mm = st.sidebar.number_input(
    "Monsoon Rainfall (mm)",
    value=650.0,
    min_value=0.0
)

rainfall_mean_premonsoon_mm = st.sidebar.number_input(
    "Pre-monsoon Rainfall (mm)",
    value=100.0,
    min_value=0.0
)

rainfall_mean_postmonsoon_mm = st.sidebar.number_input(
    "Post-monsoon Rainfall (mm)",
    value=50.0,
    min_value=0.0
)

rainfall_mean_winter_mm = st.sidebar.number_input(
    "Winter Rainfall (mm)",
    value=20.0,
    min_value=0.0
)


# ============================================================
# SOIL MOISTURE INPUTS
# ============================================================

st.sidebar.subheader(
    "🌱 Soil Moisture"
)

soil_moisture_mean_2020 = st.sidebar.number_input(
    "Mean Soil Moisture",
    value=0.25,
    min_value=0.0
)

soil_moisture_median_2020 = st.sidebar.number_input(
    "Median Soil Moisture",
    value=0.24,
    min_value=0.0
)

soil_moisture_std_2020 = st.sidebar.number_input(
    "Soil Moisture Std",
    value=0.05,
    min_value=0.0
)

soil_moisture_volume_mean_2020 = st.sidebar.number_input(
    "Soil Moisture Volume Mean",
    value=0.25,
    min_value=0.0
)

soil_moisture_pct_mean_2020 = st.sidebar.number_input(
    "Soil Moisture Percentage Mean",
    value=25.0,
    min_value=0.0
)

soil_observations_2020 = st.sidebar.number_input(
    "Soil Observations",
    value=100.0,
    min_value=0.0
)


# ============================================================
# CREATE INPUT DICTIONARY
# ============================================================

input_data = {

    "Latitude": latitude,
    "Longitude": longitude,

    "pH": pH,
    "EC": EC,
    "CO3": CO3,
    "HCO3": HCO3,
    "Cl": Cl,
    "F": F,
    "SO4": SO4,
    "NO3": NO3,
    "PO4": PO4,
    "Total_Hardness": Total_Hardness,
    "Ca": Ca,
    "Mg": Mg,
    "Na": Na,
    "K": K,
    "Fe": Fe,
    "U": U,

    "rainfall_mean_annual_mm":
        rainfall_mean_annual_mm,

    "rainfall_mean_monsoon_mm":
        rainfall_mean_monsoon_mm,

    "rainfall_mean_premonsoon_mm":
        rainfall_mean_premonsoon_mm,

    "rainfall_mean_postmonsoon_mm":
        rainfall_mean_postmonsoon_mm,

    "rainfall_mean_winter_mm":
        rainfall_mean_winter_mm,

    "soil_moisture_mean_2020":
        soil_moisture_mean_2020,

    "soil_moisture_median_2020":
        soil_moisture_median_2020,

    "soil_moisture_std_2020":
        soil_moisture_std_2020,

    "soil_moisture_volume_mean_2020":
        soil_moisture_volume_mean_2020,

    "soil_moisture_pct_mean_2020":
        soil_moisture_pct_mean_2020,

    "soil_observations_2020":
        soil_observations_2020
}


# ============================================================
# MAIN CONTENT
# ============================================================

st.header(
    "🔬 Prediction"
)

st.write(
    "Review the input values in the sidebar and "
    "click the button below."
)


if st.button(
    "🔮 Predict Arsenic Risk",
    type="primary",
    use_container_width=True
):

    try:

        result = predict_arsenic_risk(
            input_data
        )


        prediction = result[
            "prediction"
        ]

        probability = result[
            "probability"
        ]

        risk_label = result[
            "risk_label"
        ]


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.subheader(
            "Prediction Result"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Prediction",
                risk_label
            )


        with col2:

            st.metric(
                "Detection Probability",
                f"{probability * 100:.2f}%"
            )


        with col3:

            st.metric(
                "Model",
                "XGBoost"
            )


        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        st.subheader(
            "📊 Arsenic Detection Probability"
        )

        st.progress(
            probability
        )


        if prediction == 1:

            st.warning(
                "⚠️ The model predicts that arsenic "
                "may be detected in the supplied "
                "conditions."
            )

        else:

            st.success(
                "✅ The model predicts that arsenic "
                "is not detected under the supplied "
                "conditions."
            )


        # ----------------------------------------------------
        # INPUT SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "📋 Input Summary"
        )


        input_table = pd.DataFrame(
            {
                "Feature": list(
                    input_data.keys()
                ),

                "Value": list(
                    input_data.values()
                )
            }
        )


        st.dataframe(
            input_table,
            use_container_width=True,
            hide_index=True
        )


    except Exception as error:

        st.error(
            f"Prediction failed:\n\n{error}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header(
    "ℹ️ About This Model"
)

st.markdown(
    """
    **Model:** XGBoost Classification

    **Prediction task:** Arsenic detection

    **Training dataset:** Uttar Pradesh groundwater
    observations combined with environmental variables.

    **Model evaluation:** The project compares Logistic
    Regression, Random Forest and XGBoost before selecting
    the best-performing classification model.

    **Important:** A machine-learning prediction is not a
    substitute for laboratory testing of a water sample.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UP Arsenic Prediction Project | "
    "Research / Educational Use"
)