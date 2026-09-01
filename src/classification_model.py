from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Arsenic_Classification_Dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "EDA"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Classification dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("ARSENIC DETECTION CLASSIFICATION")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")

    return df


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(df):

    target = "arsenic_detected"

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTRAIN / TEST SPLIT")
    print("-" * 70)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline(X):

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    pipeline,
    X_train,
    y_train
):

    print("\nTRAINING RANDOM FOREST...")

    pipeline.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    return pipeline


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    pipeline,
    X_test,
    y_test
):

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
        ]
    })

    metrics.to_csv(
        REPORT_DIR
        / "Classification_Model_Metrics.csv",
        index=False
    )

    return predictions


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(pipeline):

    model_path = (
        MODEL_DIR
        / "arsenic_detection_random_forest.joblib"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    print(
        f"\nModel saved to:\n{model_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = (
        split_data(X, y)
    )

    pipeline = build_pipeline(
        X_train
    )

    pipeline = train_model(
        pipeline,
        X_train,
        y_train
    )

    evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    save_model(
        pipeline
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION MODEL COMPLETED")
    print("=" * 70)