"""
UP ARSENIC PREDICTION PROJECT
Regression Model Evaluation

Evaluates the selected regression model for predicting
arsenic concentration (As).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "Arsenic_Regression_Dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_arsenic_regression_model.joblib"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "EDA"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("UP ARSENIC PREDICTION - REGRESSION MODEL EVALUATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nRegression dataset not found:\n{DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nRegression model not found:\n{MODEL_PATH}\n"
        "\nRun regression_model.py first."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading regression dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# TARGET
# ============================================================

TARGET = "As"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

DROP_COLUMNS = [
    "As",
    "S_No",
    "State",
    "District",
    "Location",
    "District_std",
    "District_canonical",
    "IMD_subdivision",
    "Year",
    "arsenic_detected",
    "soil_moisture_available",
    "rainfall_available",
]

DROP_COLUMNS = [
    column
    for column in DROP_COLUMNS
    if column in df.columns
]

X = df.drop(
    columns=DROP_COLUMNS
)

y = df[TARGET]


print(f"\nFeature count: {X.shape[1]}")
print(f"Target: {TARGET}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "-" * 70)
print("TRAIN / TEST SPLIT")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading best regression model...")

model = joblib.load(MODEL_PATH)

print("Regression model loaded successfully!")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION MODEL PERFORMANCE")
print("=" * 70)

print(f"\nModel loaded from:")
print(MODEL_PATH)

print("\nMetrics:")

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],
    "Value": [
        mae,
        rmse,
        r2
    ]
})

metrics_path = (
    OUTPUT_DIR
    / "Regression_Metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "Actual_As": y_test.values,
    "Predicted_As": y_pred,
    "Residual": y_test.values - y_pred
})

predictions_path = (
    OUTPUT_DIR
    / "Regression_Test_Predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

print("\nGenerating Actual vs Predicted plot...")

plt.figure(figsize=(8, 7))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

# Perfect prediction line
min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Arsenic Concentration")
plt.ylabel("Predicted Arsenic Concentration")

plt.title(
    "Actual vs Predicted Arsenic Concentration"
)

plt.tight_layout()

actual_predicted_path = (
    OUTPUT_DIR
    / "Actual_vs_Predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RESIDUAL PLOT
# ============================================================

print("Generating residual plot...")

residuals = y_test.values - y_pred

plt.figure(figsize=(8, 7))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Arsenic Concentration")
plt.ylabel("Residual")

plt.title(
    "Residual Plot - Arsenic Regression"
)

plt.tight_layout()

residual_path = (
    OUTPUT_DIR
    / "Regression_Residuals.png"
)

plt.savefig(
    residual_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TEXT REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "Regression_Evaluation_Report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "UP ARSENIC PREDICTION PROJECT\n"
    )

    file.write(
        "REGRESSION MODEL EVALUATION REPORT\n"
    )

    file.write("=" * 60)
    file.write("\n\n")

    file.write(
        f"Dataset rows: {len(df)}\n"
    )

    file.write(
        f"Training rows: {len(X_train)}\n"
    )

    file.write(
        f"Testing rows: {len(X_test)}\n\n"
    )

    file.write(
        f"MAE : {mae:.6f}\n"
    )

    file.write(
        f"RMSE: {rmse:.6f}\n"
    )

    file.write(
        f"R2  : {r2:.6f}\n"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print(
    f"1. {metrics_path}"
)

print(
    f"2. {predictions_path}"
)

print(
    f"3. {actual_predicted_path}"
)

print(
    f"4. {residual_path}"
)

print(
    f"5. {report_path}"
)

print("\n" + "=" * 70)