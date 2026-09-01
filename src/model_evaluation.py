"""
UP ARSENIC PREDICTION PROJECT
Final Model Evaluation

Evaluates the selected XGBoost classification model
using a stratified train/test split.
"""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "Arsenic_Classification_Dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_arsenic_classification_model.joblib"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "EDA"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("UP ARSENIC PREDICTION - FINAL MODEL EVALUATION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# TARGET
# ============================================================

TARGET = "arsenic_detected"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found in dataset."
    )


# ============================================================
# REMOVE NON-MODELING COLUMNS
# ============================================================

DROP_COLUMNS = [
    "arsenic_detected",
    "As",
    "S_No",
    "State",
    "District",
    "Location",
    "District_std",
    "District_canonical",
    "IMD_subdivision",
    "soil_moisture_available",
    "rainfall_available",
]


DROP_COLUMNS = [
    col for col in DROP_COLUMNS
    if col in df.columns
]


X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]


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
    random_state=42,
    stratify=y
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
else:
    y_prob = None


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

if y_prob is not None:
    roc_auc = roc_auc_score(y_test, y_prob)
else:
    roc_auc = None


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(f"\nModel: XGBoost")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

if roc_auc is not None:
    print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "-" * 70)
print("CLASSIFICATION REPORT")
print("-" * 70)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "No Arsenic Detected",
        "Arsenic Detected"
    ],
    zero_division=0
)

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = OUTPUT_DIR / "Final_Model_Classification_Report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("UP ARSENIC PREDICTION\n")
    file.write("FINAL MODEL CLASSIFICATION REPORT\n")
    file.write("=" * 60 + "\n\n")

    file.write("Model: XGBoost\n\n")

    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall   : {recall:.4f}\n")
    file.write(f"F1 Score : {f1:.4f}\n")

    if roc_auc is not None:
        file.write(f"ROC-AUC  : {roc_auc:.4f}\n")

    file.write("\n")
    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Arsenic",
        "Arsenic Detected"
    ]
)

disp.plot(ax=ax)

ax.set_title("XGBoost Confusion Matrix")

plt.tight_layout()

cm_path = OUTPUT_DIR / "XGBoost_Confusion_Matrix.png"

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

if y_prob is not None:

    print("Generating ROC curve...")

    fig, ax = plt.subplots(figsize=(7, 6))

    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        ax=ax
    )

    ax.set_title(
        f"XGBoost ROC Curve (AUC = {roc_auc:.3f})"
    )

    plt.tight_layout()

    roc_path = OUTPUT_DIR / "XGBoost_ROC_Curve.png"

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics_path = OUTPUT_DIR / "Final_Model_Metrics.csv"

metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION COMPLETED")
print("=" * 70)

print("\nOutputs saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print(f"1. {report_path.name}")
print(f"2. {metrics_path.name}")
print(f"3. {cm_path.name}")

if y_prob is not None:
    print(f"4. {roc_path.name}")

print("\n" + "=" * 70)