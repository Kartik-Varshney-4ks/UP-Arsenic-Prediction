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

# ============================================================
# RESEARCH EVALUATION DASHBOARD
# ============================================================

st.divider()
st.header("📊 Research Evaluation Dashboard")
st.caption(
    "Nested spatial validation results for the groundwater arsenic "
    "classification research."
)

RESEARCH_DIR = BASE_DIR / "docs" / "research_results"
SPATIAL_DIR = BASE_DIR / "docs" / "nested_spatial_validation"

summary_path = RESEARCH_DIR / "final_nested_spatial_summary.csv"
fold_path = RESEARCH_DIR / "final_nested_spatial_fold_results.csv"
analysis_path = RESEARCH_DIR / "final_nested_spatial_fold_analysis.csv"
map_path = SPATIAL_DIR / "fold4_geographic_error_map.png"


def _find_column(dataframe, possible_names):
    """Return the first matching column, ignoring case and spaces."""
    normalized = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in dataframe.columns
    }
    for name in possible_names:
        key = name.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


if summary_path.exists():
    summary_df = pd.read_csv(summary_path)
    st.subheader("Overall Spatial Validation Performance")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    metric_col = _find_column(summary_df, ["Metric", "Metrics"])
    mean_col = _find_column(summary_df, ["Mean", "Mean_Score"])
    std_col = _find_column(summary_df, ["Std", "Standard_Deviation"])

    if metric_col and mean_col:
        metrics = summary_df.set_index(metric_col)[mean_col].to_dict()
        metric_display = [
            ("Accuracy", "Accuracy"),
            ("Balanced_Accuracy", "Balanced accuracy"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1_Score", "F1 score"),
            ("ROC_AUC", "ROC-AUC"),
            ("PR_AUC_AP", "PR-AUC"),
        ]
        cards = []
        for key, label in metric_display:
            for actual_key, value in metrics.items():
                if str(actual_key).strip().lower().replace(" ", "_") == key.lower():
                    try:
                        cards.append((label, float(value)))
                    except (TypeError, ValueError):
                        pass
                    break

        for start in range(0, len(cards), 4):
            columns = st.columns(min(4, len(cards) - start))
            for column, (label, value) in zip(columns, cards[start:start + 4]):
                column.metric(label, f"{value:.3f}")

    st.info(
        "These are mean outer-fold results from nested spatially grouped "
        "cross-validation, not results from an independent external test set."
    )
else:
    st.warning(f"Research summary file not found: {summary_path}")


if fold_path.exists():
    folds_df = pd.read_csv(fold_path)
    st.subheader("Performance Across Spatial Folds")
    st.dataframe(folds_df, use_container_width=True, hide_index=True)

    fold_col = _find_column(folds_df, ["Outer_Fold", "Fold"])
    chart_candidates = [
        column for column in ["Accuracy", "Precision", "Recall", "F1_Score"]
        if column in folds_df.columns
    ]
    if fold_col and chart_candidates:
        chart_df = folds_df.set_index(fold_col)[chart_candidates]
        st.markdown("**Comparison of selected metrics across folds**")
        st.bar_chart(chart_df)

    st.download_button(
        "⬇️ Download fold results CSV",
        data=folds_df.to_csv(index=False).encode("utf-8"),
        file_name="nested_spatial_fold_results.csv",
        mime="text/csv",
        key="download_nested_fold_results",
    )
else:
    st.warning(f"Fold results file not found: {fold_path}")


if analysis_path.exists():
    analysis_df = pd.read_csv(analysis_path)
    with st.expander("🔎 Fold-level geography and error analysis"):
        st.dataframe(analysis_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download fold analysis CSV",
            data=analysis_df.to_csv(index=False).encode("utf-8"),
            file_name="nested_spatial_fold_analysis.csv",
            mime="text/csv",
            key="download_nested_fold_analysis",
        )
else:
    st.caption("Optional fold-level analysis file was not found.")


st.subheader("🗺️ Geographic Prediction-Error Map")
if map_path.exists():
    st.image(
        str(map_path),
        caption=(
            "Prediction outcomes for spatial validation Fold 4. "
            "This is not a confirmed statewide arsenic-contamination map."
        ),
        use_container_width=True,
    )
else:
    st.warning(f"Geographic error map not found: {map_path}")


st.subheader("⚠️ Research Interpretation and Limitations")
st.markdown(
    """
    - Spatially grouped validation evaluates performance across groups of nearby observations.
    - The validation did not enforce a geographic buffer between training and test observations.
    - Performance varies between folds and may differ in other locations.
    - Predictions estimate detection risk; they do not measure arsenic concentration.
    - Laboratory testing is required to determine arsenic concentration in a specific water sample.
    - These research metrics describe the evaluated validation pipeline. They should not be assumed to
      validate the live prediction pipeline until the model and feature configurations are confirmed to match.
    """
)

