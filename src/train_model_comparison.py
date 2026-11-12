"""
UP Arsenic Prediction
Compare Random Forest, XGBoost, LightGBM, and CatBoost.

Dataset:
data/processed/Arsenic_Classification_Dataset.csv

Target:
arsenic_detected
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed"
    / "Arsenic_Classification_Dataset.csv"
)

MODEL_DIR = ROOT / "models" / "model_comparison"
RESULT_DIR = ROOT / "docs" / "model_comparison"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "arsenic_detected"
RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load and validate dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=[TARGET]).copy()

# Target must be binary.
target_values = set(df[TARGET].unique())

if not target_values.issubset({0, 1}):
    raise ValueError(
        f"Expected binary target values 0 and 1, "
        f"but found: {target_values}"
    )

# Remove rows with invalid numeric target values, if any.
df[TARGET] = df[TARGET].astype(int)

# Prevent non-numeric target leakage and remove identifiers
# only if they are known to be non-predictive identifiers.
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

# Convert all features to numeric where possible.
# This dataset is expected to contain numerical features.
for column in X.columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# Remove completely empty features.
empty_columns = X.columns[X.isna().all()].tolist()

if empty_columns:
    print("Removing completely empty columns:", empty_columns)
    X = X.drop(columns=empty_columns)

if y.nunique() != 2:
    raise ValueError(
        "Both target classes must be present."
    )

print("\nDataset loaded successfully.")
print("Rows:", len(X))
print("Features:", X.shape[1])
print("\nTarget distribution:")
print(y.value_counts().sort_index())
print("\nTarget proportions:")
print(y.value_counts(normalize=True).sort_index().round(4))

# --------------------------------------------------
# 3. Split data
# --------------------------------------------------

# This is a preliminary stratified random holdout.
# Nearby groundwater observations may be correlated.
# A separate spatial holdout is recommended for research
# claims about performance in previously unseen locations.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# --------------------------------------------------
# 4. Define models
# --------------------------------------------------

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        class_weight="balanced",
        verbosity=-1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "CatBoost": CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        verbose=False,
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        thread_count=-1,
    ),
}

# --------------------------------------------------
# 5. Train, evaluate, and save models
# --------------------------------------------------

results = []
fitted_models = {}

for name, estimator in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    # Imputation is fitted only on the training data.
    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        ("model", clone(estimator)),
    ])

    try:
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        probabilities = None

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1_Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC_AUC": (
                roc_auc_score(y_test, probabilities)
                if probabilities is not None
                else np.nan
            ),
        }

        results.append(metrics)
        fitted_models[name] = pipeline

        safe_name = name.lower().replace(" ", "_")

        # Save model separately from your existing models.
        model_path = MODEL_DIR / f"{safe_name}_model.joblib"
        joblib.dump(pipeline, model_path)

        # Save predictions for further analysis.
        prediction_output = pd.DataFrame({
            "Actual": y_test.to_numpy(),
            "Predicted": predictions,
        })

        if probabilities is not None:
            prediction_output["Probability_Class_1"] = probabilities

        prediction_output.to_csv(
            RESULT_DIR / f"{safe_name}_predictions.csv",
            index=False,
        )

        # Save confusion matrix.
        cm = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(cm, interpolation="nearest")
        ax.figure.colorbar(image, ax=ax)

        ax.set(
            title=f"{name} - Confusion Matrix",
            xlabel="Predicted label",
            ylabel="Actual label",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
        )

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center"
                )

        fig.tight_layout()
        fig.savefig(
            RESULT_DIR / f"{safe_name}_confusion_matrix.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(pd.Series(metrics).to_string())
        print("\nClassification report:")
        print(classification_report(
            y_test, predictions, zero_division=0
        ))
        print("\nSaved model:", model_path)

    except Exception as exc:
        print(f"ERROR training {name}: {exc}")
        raise

# --------------------------------------------------
# 6. Save comparison table
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="F1_Score",
    ascending=False,
)

comparison_path = RESULT_DIR / "model_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(comparison.round(4).to_string(index=False))

print("\nComparison saved to:", comparison_path)

# --------------------------------------------------
# 7. Create comparison chart
# --------------------------------------------------

chart_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC",
]

chart_data = comparison.set_index("Model")[chart_metrics]

ax = chart_data.plot(
    kind="bar",
    figsize=(12, 6),
)

ax.set_title("Arsenic Classification Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

chart_path = RESULT_DIR / "model_comparison.png"
plt.savefig(chart_path, dpi=200, bbox_inches="tight")
plt.close()

print("Comparison chart saved to:", chart_path)
print("\nPhase 1 training completed.")
