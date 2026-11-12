from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "models" / "model_comparison" / "xgboost_model.joblib"
)

DATA_PATH = (
    ROOT / "data" / "processed" / "Arsenic_Classification_Dataset.csv"
)

OUTPUT_DIR = ROOT / "docs" / "shap_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "arsenic_detected"
SAMPLE_SIZE = 300
RANDOM_STATE = 42


# ============================================================
# 2. LOAD MODEL AND DATA
# ============================================================

print("=" * 65)
print("UP ARSENIC PREDICTION - SHAP EXPLAINABILITY")
print("=" * 65)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("\nLoading model...")
pipeline = joblib.load(MODEL_PATH)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )

# Separate features and target.
X = df.drop(columns=[TARGET_COLUMN]).copy()
y = df[TARGET_COLUMN].copy()

# Match the model's expected feature order.
expected_features = list(pipeline.feature_names_in_)

missing_features = [
    col for col in expected_features
    if col not in X.columns
]

if missing_features:
    raise ValueError(
        f"Dataset is missing model features: {missing_features}"
    )

X = X[expected_features]

# Convert features to numeric values.
for col in X.columns:
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce",
    )

print("\nDataset shape:", X.shape)
print("Number of features:", len(expected_features))
print("Model pipeline steps:", list(pipeline.named_steps.keys()))


# ============================================================
# 3. SAMPLE OBSERVATIONS
# ============================================================

sample_size = min(SAMPLE_SIZE, len(X))

X_sample = X.sample(
    n=sample_size,
    random_state=RANDOM_STATE,
)

y_sample = y.loc[X_sample.index]

print("SHAP sample size:", len(X_sample))


# ============================================================
# 4. EXTRACT THE IMPUTER AND XGBOOST MODEL
# ============================================================

if "imputer" not in pipeline.named_steps:
    raise ValueError(
        "The expected 'imputer' step was not found."
    )

if "model" not in pipeline.named_steps:
    raise ValueError(
        "The expected 'model' step was not found."
    )

imputer = pipeline.named_steps["imputer"]
xgb_model = pipeline.named_steps["model"]

# Apply the saved imputer.
X_transformed = imputer.transform(X_sample)

X_transformed = np.asarray(X_transformed)

# Preserve original feature names for interpretation.
feature_names = list(X_sample.columns)

X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names,
    index=X_sample.index,
)

print("\nTransformed feature shape:", X_transformed.shape)


# ============================================================
# 5. CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(xgb_model)

shap_output = explainer.shap_values(
    X_transformed_df
)

# Handle SHAP outputs from different versions.
if isinstance(shap_output, list):
    shap_values = (
        shap_output[1]
        if len(shap_output) > 1
        else shap_output[0]
    )
elif isinstance(shap_output, shap.Explanation):
    shap_values = shap_output.values
else:
    shap_values = np.asarray(shap_output)

shap_values = np.asarray(shap_values)

# For binary classification, some SHAP versions return:
# samples x features x classes.
if shap_values.ndim == 3:
    class_index = (
        1 if shap_values.shape[-1] > 1 else 0
    )

    shap_values = shap_values[:, :, class_index]

if shap_values.ndim != 2:
    raise ValueError(
        f"Unexpected SHAP shape: {shap_values.shape}"
    )

if shap_values.shape[1] != len(feature_names):
    raise ValueError(
        "SHAP feature count does not match input feature count."
    )

print("SHAP values shape:", shap_values.shape)


# ============================================================
# 6. SAVE FEATURE IMPORTANCE
# ============================================================

importance_values = np.mean(
    np.abs(shap_values),
    axis=0,
)

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Mean_Absolute_SHAP": importance_values,
    }
).sort_values(
    "Mean_Absolute_SHAP",
    ascending=False,
)

importance_path = (
    OUTPUT_DIR / "shap_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False,
)

print("\nTop 10 features by mean absolute SHAP:")
print(
    importance_df.head(10).to_string(index=False)
)


# ============================================================
# 7. CREATE SHAP SUMMARY PLOT
# ============================================================

print("\nCreating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_transformed_df,
    feature_names=feature_names,
    max_display=20,
    show=False,
)

plt.title(
    "SHAP Summary: Arsenic Detection Model"
)

plt.tight_layout()

summary_path = (
    OUTPUT_DIR / "shap_summary.png"
)

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 8. CREATE FEATURE IMPORTANCE BAR CHART
# ============================================================

print("Creating feature importance plot...")

top_features = (
    importance_df.head(15)
    .sort_values("Mean_Absolute_SHAP")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Mean_Absolute_SHAP"],
)

plt.xlabel("Mean absolute SHAP value")
plt.ylabel("Feature")

plt.title(
    "Top 15 Features Influencing Arsenic Predictions"
)

plt.tight_layout()

importance_plot_path = (
    OUTPUT_DIR / "shap_feature_importance.png"
)

plt.savefig(
    importance_plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 9. SAVE SAMPLE PREDICTIONS
# ============================================================

print("Saving sample predictions...")

predicted_classes = pipeline.predict(X_sample)

prediction_output = X_sample.copy()

prediction_output["Actual_Arsenic_Detected"] = (
    y_sample.to_numpy()
)

prediction_output["Predicted_Arsenic_Detected"] = (
    predicted_classes
)

if hasattr(pipeline, "predict_proba"):
    probabilities = pipeline.predict_proba(X_sample)

    classes = list(pipeline.classes_)

    if 1 in classes:
        positive_class_index = classes.index(1)

        prediction_output[
            "Predicted_Risk_Probability"
        ] = probabilities[:, positive_class_index]

prediction_path = (
    OUTPUT_DIR / "shap_sample_predictions.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False,
)


# ============================================================
# 10. PRINT EVALUATION SUMMARY
# ============================================================

print("\nSample prediction report:")
print(
    classification_report(
        y_sample,
        predicted_classes,
        zero_division=0,
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_sample,
        predicted_classes,
    )
)


# ============================================================
# 11. COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nGenerated files:")

print(f"Feature importance CSV: {importance_path}")
print(f"SHAP summary plot: {summary_path}")
print(f"Feature importance plot: {importance_plot_path}")
print(f"Sample predictions: {prediction_path}")

print(
    "\nNote: SHAP describes model contributions, "
    "not causal relationships."
)