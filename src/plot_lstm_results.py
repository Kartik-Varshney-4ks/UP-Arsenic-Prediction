from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    ROOT / "docs" / "temporal_analysis" / "lstm_predictions.csv"
)

OUTPUT_DIR = ROOT / "docs" / "temporal_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Predictions file not found: {RESULTS_PATH}"
    )

df = pd.read_csv(RESULTS_PATH)

required_columns = [
    "Actual_As_ppb",
    "LSTM_Predicted_As_ppb",
    "Baseline_Predicted_As_ppb",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df = df.dropna(
    subset=required_columns
).copy()

if df.empty:
    raise ValueError(
        "No valid prediction records are available."
    )

# --------------------------------------------------
# Plot 1: Actual vs LSTM predictions
# --------------------------------------------------

plt.figure(figsize=(9, 7))

plt.scatter(
    df["Actual_As_ppb"],
    df["LSTM_Predicted_As_ppb"],
    alpha=0.8,
    label="LSTM predictions",
)

plt.scatter(
    df["Actual_As_ppb"],
    df["Baseline_Predicted_As_ppb"],
    alpha=0.6,
    marker="x",
    label="Mean baseline",
)

max_value = max(
    df["Actual_As_ppb"].max(),
    df["LSTM_Predicted_As_ppb"].max(),
    df["Baseline_Predicted_As_ppb"].max(),
)

min_value = min(
    df["Actual_As_ppb"].min(),
    df["LSTM_Predicted_As_ppb"].min(),
    df["Baseline_Predicted_As_ppb"].min(),
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect prediction",
)

plt.xlabel("Actual arsenic concentration (ppb)")
plt.ylabel("Predicted arsenic concentration (ppb)")
plt.title("LSTM: Actual vs Predicted Arsenic Concentrations")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot1_path = OUTPUT_DIR / "lstm_actual_vs_predicted.png"
plt.savefig(plot1_path, dpi=300)
plt.close()

# --------------------------------------------------
# Plot 2: Absolute error comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    range(len(df)),
    df["LSTM_Absolute_Error"],
    marker="o",
    label="LSTM absolute error",
)

plt.plot(
    range(len(df)),
    df["Baseline_Absolute_Error"],
    marker="x",
    label="Baseline absolute error",
)

plt.xlabel("Test district index")
plt.ylabel("Absolute error (ppb)")
plt.title("LSTM vs Mean Baseline: Absolute Prediction Error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot2_path = OUTPUT_DIR / "lstm_error_comparison.png"
plt.savefig(plot2_path, dpi=300)
plt.close()

print("Visualizations created successfully!")
print(f"Saved: {plot1_path}")
print(f"Saved: {plot2_path}")
print(f"Number of test districts: {len(df)}")