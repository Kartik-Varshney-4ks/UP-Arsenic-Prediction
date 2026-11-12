from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "UP_Arsenic_Temporal_2023_2024.csv"
OUTPUT_DIR = ROOT / "docs" / "temporal_analysis"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    "Year",
    "District",
    "Longitude",
    "Latitude",
    "pH",
    "EC",
    "As_ppb",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

numeric_columns = [
    "Year",
    "Longitude",
    "Latitude",
    "pH",
    "EC",
    "As_ppb",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df = df.dropna(
    subset=[
        "Year",
        "District",
        "Longitude",
        "Latitude",
        "As_ppb",
    ]
).copy()

df["Year"] = df["Year"].astype(int)
df["District"] = df["District"].astype(str).str.strip()

df = df[df["District"] != ""]

print("=" * 60)
print("UP ARSENIC PREDICTION: LSTM EXPERIMENT")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("Available years:", sorted(df["Year"].unique()))

print("\nArsenic concentration summary:")
print(df["As_ppb"].describe())


# ============================================================
# 3. AGGREGATE MEASUREMENTS BY DISTRICT AND YEAR
# ============================================================

annual = (
    df.groupby(
        ["District", "Year"],
        as_index=False,
    )
    .agg(
        As_ppb=("As_ppb", "mean"),
        pH=("pH", "mean"),
        EC=("EC", "mean"),
        Latitude=("Latitude", "mean"),
        Longitude=("Longitude", "mean"),
        Sample_Count=("As_ppb", "size"),
    )
    .sort_values(
        ["District", "Year"]
    )
    .reset_index(drop=True)
)

print("\nDistrict-year records:", len(annual))

print("\nNumber of yearly observations per district:")
print(
    annual.groupby("District")["Year"]
    .nunique()
    .value_counts()
    .sort_index()
)

years = sorted(annual["Year"].unique())

if len(years) < 2:
    raise SystemExit(
        "\nNot enough distinct years for a temporal experiment."
    )

if 2023 not in years or 2024 not in years:
    raise SystemExit(
        "\nThis script expects observations from both 2023 and 2024."
    )


# ============================================================
# 4. BUILD DISTRICT-LEVEL 2023 -> 2024 SEQUENCES
# ============================================================

feature_columns = [
    "As_ppb",
    "pH",
    "EC",
    "Latitude",
    "Longitude",
]

sequences = []
targets = []
metadata = []

for district, group in annual.groupby("District"):

    group = group.sort_values("Year")

    row_2023 = group[group["Year"] == 2023]
    row_2024 = group[group["Year"] == 2024]

    # Require exactly one aggregated record per year.
    if len(row_2023) != 1 or len(row_2024) != 1:
        continue

    input_row = row_2023.iloc[0]
    target_row = row_2024.iloc[0]

    sequences.append(
        input_row[feature_columns].to_numpy(dtype=float)
    )

    targets.append(
        float(target_row["As_ppb"])
    )

    metadata.append(
        {
            "District": district,
            "Input_Year": 2023,
            "Target_Year": 2024,
            "Actual_As_ppb": float(target_row["As_ppb"]),
            "Input_As_ppb": float(input_row["As_ppb"]),
        }
    )

if len(sequences) < 10:
    raise SystemExit(
        "\nSTOP: Fewer than 10 districts have valid observations "
        "in both 2023 and 2024. Collect more data before training "
        "an LSTM."
    )

X = np.asarray(sequences, dtype=float)
y = np.asarray(targets, dtype=float)
meta = pd.DataFrame(metadata)

print("\nValid district sequences:", len(X))
print("Input shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 5. SPLIT DISTRICTS INTO TRAINING AND TEST SETS
# ============================================================

# Deterministic split by district name.
# This prevents rows from the same district appearing in both sets.
order = np.argsort(meta["District"].to_numpy())

X = X[order]
y = y[order]
meta = meta.iloc[order].reset_index(drop=True)

split_index = int(len(X) * 0.8)

if split_index < 2 or len(X) - split_index < 2:
    raise SystemExit(
        "\nSTOP: Insufficient districts for training and testing."
    )

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

meta_test = meta.iloc[split_index:].copy()

print("\nTraining districts:", len(X_train))
print("Testing districts:", len(X_test))


# ============================================================
# 6. PREPROCESS INPUT FEATURES
# ============================================================

n_features = X_train.shape[-1]

imputer = SimpleImputer(
    strategy="median"
)

scaler = StandardScaler()

X_train_flat = imputer.fit_transform(
    X_train
)

X_test_flat = imputer.transform(
    X_test
)

X_train_scaled = scaler.fit_transform(
    X_train_flat
).reshape(
    X_train.shape[0],
    1,
    n_features,
)

X_test_scaled = scaler.transform(
    X_test_flat
).reshape(
    X_test.shape[0],
    1,
    n_features,
)

# Scale target values for neural-network training.
y_mean = float(np.mean(y_train))
y_std = float(np.std(y_train))

if y_std == 0:
    y_std = 1.0

y_train_scaled = (
    y_train - y_mean
) / y_std


# ============================================================
# 7. BUILD AND TRAIN LSTM
# ============================================================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(1, n_features)
        ),
        tf.keras.layers.LSTM(
            16,
            activation="tanh",
        ),
        tf.keras.layers.Dense(
            8,
            activation="relu",
        ),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

print("\nStarting LSTM training...")

model.fit(
    X_train_scaled,
    y_train_scaled,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1,
)


# ============================================================
# 8. GENERATE TEST PREDICTIONS
# ============================================================

predictions_scaled = model.predict(
    X_test_scaled,
    verbose=0,
).reshape(-1)

predictions = (
    predictions_scaled * y_std
) + y_mean

# A simple baseline predicts the training-set mean.
baseline_predictions = np.full(
    shape=len(y_test),
    fill_value=y_mean,
    dtype=float,
)


# ============================================================
# 9. EVALUATE MODEL AND BASELINE
# ============================================================

def calculate_metrics(actual, predicted):
    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "R2": float(
            r2_score(actual, predicted)
        ) if len(actual) >= 2 else np.nan,
    }


lstm_metrics = calculate_metrics(
    y_test,
    predictions,
)

baseline_metrics = calculate_metrics(
    y_test,
    baseline_predictions,
)

metrics_df = pd.DataFrame(
    [
        {
            "Model": "LSTM",
            **lstm_metrics,
            "Train_Samples": len(y_train),
            "Test_Samples": len(y_test),
        },
        {
            "Model": "Training_Mean_Baseline",
            **baseline_metrics,
            "Train_Samples": len(y_train),
            "Test_Samples": len(y_test),
        },
    ]
)

metrics_df["Evaluation_Note"] = (
    "Exploratory district-level 2023-to-2024 prediction; "
    "not validated long-term forecasting."
)

metrics_path = OUTPUT_DIR / "lstm_metrics.csv"

metrics_df.to_csv(
    metrics_path,
    index=False,
)


# ============================================================
# 10. SAVE PREDICTIONS
# ============================================================

meta_test["LSTM_Predicted_As_ppb"] = predictions

meta_test["Baseline_Predicted_As_ppb"] = (
    baseline_predictions
)

meta_test["LSTM_Absolute_Error"] = np.abs(
    meta_test["Actual_As_ppb"]
    - meta_test["LSTM_Predicted_As_ppb"]
)

meta_test["Baseline_Absolute_Error"] = np.abs(
    meta_test["Actual_As_ppb"]
    - meta_test["Baseline_Predicted_As_ppb"]
)

predictions_path = (
    OUTPUT_DIR / "lstm_predictions.csv"
)

meta_test.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# 11. SAVE MODEL AND PREPROCESSING PARAMETERS
# ============================================================

model_path = MODEL_DIR / "lstm_model.keras"

model.save(model_path)

preprocessing_path = (
    MODEL_DIR / "lstm_preprocessing.npz"
)

np.savez(
    preprocessing_path,
    imputer_statistics=imputer.statistics_,
    scaler_mean=scaler.mean_,
    scaler_scale=scaler.scale_,
    target_mean=np.array(y_mean),
    target_std=np.array(y_std),
    feature_columns=np.array(feature_columns),
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(metrics_df.to_string(index=False))

print("\nSaved files:")

print(f"LSTM model: {model_path}")
print(f"Preprocessing: {preprocessing_path}")
print(f"Metrics: {metrics_path}")
print(f"Predictions: {predictions_path}")

print("\nExperiment completed successfully.")

print(
    "\nIMPORTANT: Only two annual snapshots are available. "
    "This experiment does not establish reliable long-term "
    "forecasting performance."
)