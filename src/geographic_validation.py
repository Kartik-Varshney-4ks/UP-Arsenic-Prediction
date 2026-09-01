"""
UP ARSENIC PREDICTION PROJECT
GEOGRAPHIC VALIDATION

Purpose:
    Evaluate arsenic classification using a district-based
    train/test split instead of a random row-based split.

The original master dataset is used so that geographic
information such as District, Latitude and Longitude
remains available.

Target:
    arsenic_detected
        1 = As > 0
        0 = As = 0
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "UP_Arsenic_Climate_Environmental_Master_Working.csv"
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
print("UP ARSENIC PREDICTION - GEOGRAPHIC VALIDATION")
print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nMaster dataset not found at:\n{DATA_PATH}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading master dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "As",
    "District"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"\nRequired columns missing: "
        f"{missing_columns}"
    )


# ============================================================
# CREATE CLASSIFICATION TARGET
# ============================================================

print("\nCreating arsenic detection target...")

df["arsenic_detected"] = (
    df["As"] > 0
).astype(int)


print("\nTarget distribution:")

print(
    df["arsenic_detected"].value_counts()
    .sort_index()
)


# ============================================================
# REMOVE ROWS WITHOUT DISTRICT
# ============================================================

before_rows = len(df)

df = df.dropna(
    subset=["District"]
).copy()

after_rows = len(df)

if before_rows != after_rows:

    print(
        f"\nRemoved {before_rows - after_rows} "
        "rows without District information."
    )


# ============================================================
# DEFINE TARGET
# ============================================================

TARGET = "arsenic_detected"


# ============================================================
# DROP NON-MODELING COLUMNS
# ============================================================

DROP_COLUMNS = [
    "As",
    "arsenic_detected",

    # Identification / geographic grouping
    "S_No",
    "State",
    "District",
    "Location",

    # Derived geographic text fields
    "District_std",
    "District_canonical",
    "IMD_subdivision",

    # Time identifier
    "Year",

    # Boolean availability indicators
    "soil_moisture_available",
    "rainfall_available"
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

groups = df["District"]


# ============================================================
# KEEP NUMERIC FEATURES ONLY
# ============================================================

print("\nChecking feature types...")

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

X = X[numeric_columns]


print(
    f"Numerical features: {len(numeric_columns)}"
)

print(
    numeric_columns
)


# ============================================================
# REMOVE COMPLETELY EMPTY FEATURES
# ============================================================

empty_columns = [
    column
    for column in X.columns
    if X[column].isna().all()
]

if empty_columns:

    print(
        "\nRemoving completely empty columns:"
    )

    print(empty_columns)

    X = X.drop(
        columns=empty_columns
    )


# ============================================================
# REMOVE INFINITE VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# DISTRICT-BASED TRAIN / TEST SPLIT
# ============================================================

print("\n" + "-" * 70)
print("DISTRICT-BASED TRAIN / TEST SPLIT")
print("-" * 70)


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

train_districts = (
    groups.iloc[train_idx]
)

test_districts = (
    groups.iloc[test_idx]
)


print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows : {len(X_test)}"
)

print(
    f"Training districts: "
    f"{train_districts.nunique()}"
)

print(
    f"Testing districts : "
    f"{test_districts.nunique()}"
)


# ============================================================
# CHECK DISTRICT OVERLAP
# ============================================================

train_district_set = set(
    train_districts.unique()
)

test_district_set = set(
    test_districts.unique()
)

overlap = (
    train_district_set
    .intersection(test_district_set)
)


print(
    f"\nDistrict overlap: {len(overlap)}"
)


if len(overlap) > 0:

    print(
        "WARNING: Some districts appear "
        "in both sets."
    )

else:

    print(
        "No district overlap detected."
    )


# ============================================================
# CREATE MODEL PIPELINE
# ============================================================

print("\n" + "-" * 70)
print("CREATING XGBOOST MODEL")
print("-" * 70)


model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining geographic validation model...")

model.fit(
    X_train,
    y_train
)

print(
    "Training completed successfully."
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


try:

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

except ValueError:

    roc_auc = np.nan


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("GEOGRAPHIC VALIDATION RESULTS")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame({

    "Validation_Type": [
        "District Group Split"
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1_Score": [
        f1
    ],

    "ROC_AUC": [
        roc_auc
    ],

    "Training_Rows": [
        len(X_train)
    ],

    "Testing_Rows": [
        len(X_test)
    ],

    "Training_Districts": [
        train_districts.nunique()
    ],

    "Testing_Districts": [
        test_districts.nunique()
    ],

    "District_Overlap": [
        len(overlap)
    ]
})


RESULT_PATH = (
    OUTPUT_DIR
    / "Geographic_Validation_Results.csv"
)


results_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "District": test_districts.values,

    "Actual": y_test.values,

    "Predicted": y_pred,

    "Probability": y_probability

})


PREDICTION_PATH = (
    OUTPUT_DIR
    / "Geographic_Test_Predictions.csv"
)


prediction_df.to_csv(
    PREDICTION_PATH,
    index=False
)


# ============================================================
# SAVE TEST DISTRICTS
# ============================================================

district_df = pd.DataFrame({

    "District": sorted(
        test_district_set
    )

})


DISTRICT_PATH = (
    OUTPUT_DIR
    / "Geographic_Test_Districts.csv"
)


district_df.to_csv(
    DISTRICT_PATH,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

GEOGRAPHIC_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "geographic_validation_model.joblib"
)


GEOGRAPHIC_MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    model,
    GEOGRAPHIC_MODEL_PATH
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"\n1. Results:"
    f"\n   {RESULT_PATH}"
)

print(
    f"\n2. Predictions:"
    f"\n   {PREDICTION_PATH}"
)

print(
    f"\n3. Test districts:"
    f"\n   {DISTRICT_PATH}"
)

print(
    f"\n4. Geographic model:"
    f"\n   {GEOGRAPHIC_MODEL_PATH}"
)


print("\n" + "=" * 70)
print("GEOGRAPHIC VALIDATION COMPLETED SUCCESSFULLY")
print("=" * 70)