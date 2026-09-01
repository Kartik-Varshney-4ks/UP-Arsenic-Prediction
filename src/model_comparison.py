from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "EDA"
)

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("UP ARSENIC PREDICTION - MODEL COMPARISON")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(df):

    target = "arsenic_detected"

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found."
        )

    # Target
    y = df[target]

    # Remove target
    X = df.drop(columns=[target])

    # --------------------------------------------------------
    # IMPORTANT:
    # Remove arsenic concentration itself if present.
    # We must NOT use As to predict arsenic_detected.
    # --------------------------------------------------------

    if "As" in X.columns:
        X = X.drop(columns=["As"])

        print("\nRemoved 'As' from features to prevent target leakage.")

    # Remove row identifier
    if "S_No" in X.columns:
        X = X.drop(columns=["S_No"])

    # Year is not useful for this classification baseline
    if "Year" in X.columns:
        X = X.drop(columns=["Year"])

    print("\nFinal feature shape:")
    print(X.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

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

    print("\n" + "-" * 70)
    print("TRAIN / TEST SPLIT")
    print("-" * 70)

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(X):

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
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
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
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

    return preprocessor


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    models = {}

    # --------------------------------------------------------
    # 1. Logistic Regression
    # --------------------------------------------------------

    models["Logistic Regression"] = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    # --------------------------------------------------------
    # 2. Random Forest
    # --------------------------------------------------------

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # --------------------------------------------------------
    # 3. XGBoost
    # --------------------------------------------------------

    try:

        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
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

        print("\nXGBoost detected.")

    except ImportError:

        print(
            "\nXGBoost is not installed."
        )

        print(
            "Using Gradient Boosting instead."
        )

        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

    return models


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

def evaluate_models(
    models,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):

    results = {}

    trained_models = {}

    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 70)

    for name, model in models.items():

        print(f"\nTraining: {name}")
        print("-" * 50)

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

        # Train
        pipeline.fit(
            X_train,
            y_train
        )

        # Predictions
        predictions = pipeline.predict(
            X_test
        )

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        # Metrics
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

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }

        trained_models[name] = pipeline

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

    return results, trained_models


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    results_df = pd.DataFrame(
        results
    ).T

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    output_path = (
        REPORT_DIR
        / "Model_Comparison.csv"
    )

    results_df.to_csv(
        output_path
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)

    print(
        results_df.round(4)
    )

    print(
        f"\nResults saved to:\n{output_path}"
    )

    return results_df


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(
    results_df,
    trained_models
):

    best_model_name = results_df.index[0]

    best_model = trained_models[
        best_model_name
    ]

    model_path = (
        MODEL_DIR
        / "best_arsenic_classification_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"\nBest model: {best_model_name}"
    )

    print(
        f"F1 Score: "
        f"{results_df.loc[best_model_name, 'F1 Score']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{results_df.loc[best_model_name, 'ROC-AUC']:.4f}"
    )

    print(
        f"\nBest model saved to:\n{model_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    X, y = prepare_data(df)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(X, y)

    preprocessor = create_preprocessor(
        X_train
    )

    models = create_models()

    results, trained_models = evaluate_models(
        models,
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    results_df = save_results(
        results
    )

    save_best_model(
        results_df,
        trained_models
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 70)