from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "Arsenic_Classification_Dataset.csv"
MODEL_DIR = ROOT / "models" / "model_comparison"
OUTPUT_DIR = ROOT / "docs" / "model_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "arsenic_detected"

MODEL_FILES = {
    "XGBoost": "xgboost_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "CatBoost": "catboost_model.joblib",
    "LightGBM": "lightgbm_model.joblib",
}


def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")

    if df[TARGET].isna().any():
        raise ValueError("The target contains missing values.")

    X = df.drop(columns=[TARGET]).apply(
        pd.to_numeric, errors="coerce"
    )

    X = X.dropna(axis=1, how="all")
    y = df[TARGET].astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Expected a binary target containing 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    results = []

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            print(f"Skipping {model_name}: model file not found.")
            continue

        model = joblib.load(model_path)

        # Align the evaluation features with the saved model.
        model_features = getattr(model, "feature_names_in_", None)

        if model_features is not None:
            X_train_model = X_train.reindex(columns=model_features)
            X_test_model = X_test.reindex(columns=model_features)
        else:
            X_train_model = X_train
            X_test_model = X_test

        # Saved pipelines may already include their own imputer.
        y_pred = model.predict(X_test_model)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_model)[:, 1]
        else:
            y_score = None

        row = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced_Accuracy": balanced_accuracy_score(
                y_test, y_pred
            ),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1_Score": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "ROC_AUC": (
                roc_auc_score(y_test, y_score)
                if y_score is not None
                else np.nan
            ),
            "PR_AUC_Average_Precision": (
                average_precision_score(y_test, y_score)
                if y_score is not None
                else np.nan
            ),
        }

        results.append(row)

    if not results:
        raise RuntimeError("No saved models were available for evaluation.")

    results_df = pd.DataFrame(results).sort_values(
        "F1_Score", ascending=False
    )

    output_path = OUTPUT_DIR / "advanced_evaluation.csv"
    results_df.to_csv(output_path, index=False)

    print("\nADVANCED MODEL EVALUATION")
    print(results_df.round(4).to_string(index=False))

    print("\nSaved results to:")
    print(output_path)

    print(
        "\nNote: This evaluation uses the same random split "
        "configuration as the training script. It is not an "
        "independent validation experiment if the models were "
        "trained on this same split."
    )


if __name__ == "__main__":
    main()