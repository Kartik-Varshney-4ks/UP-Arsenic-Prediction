from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "Arsenic_Regression_Dataset.csv"
)

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "EDA"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("ARSENIC CONCENTRATION REGRESSION")
print("=" * 70)


df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


target = "As"

if target not in df.columns:
    raise ValueError(
        "Target column 'As' not found."
    )


drop_columns = [
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

drop_columns = [
    col for col in drop_columns
    if col in df.columns
]

X = df.drop(columns=drop_columns)
y = df[target]


print(f"\nFeatures: {X.shape[1]}")
print(f"Target: {target}")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "category", "bool"]
).columns.tolist()


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


models = {
    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


results = {}

trained_models = {}


for name, estimator in models.items():

    print("\n" + "-" * 70)
    print(f"Training: {name}")
    print("-" * 70)

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            estimator
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    trained_models[name] = pipeline

    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")


results_df = pd.DataFrame(results).T

results_df = results_df.sort_values(
    "RMSE"
)


print("\n" + "=" * 70)
print("REGRESSION MODEL COMPARISON")
print("=" * 70)

print(
    results_df.round(4)
)


results_path = (
    OUTPUT_DIR
    / "Regression_Model_Comparison.csv"
)

results_df.to_csv(
    results_path
)


best_model_name = results_df.index[0]

best_model = trained_models[
    best_model_name
]


best_model_path = (
    MODEL_DIR
    / "best_arsenic_regression_model.joblib"
)

joblib.dump(
    best_model,
    best_model_path
)


print(
    f"\nBest regression model: "
    f"{best_model_name}"
)

print(
    f"Saved to:\n{best_model_path}"
)

print("\nREGRESSION COMPLETED")