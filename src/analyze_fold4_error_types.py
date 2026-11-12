from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "Arsenic_Spatial_Model_Dataset.csv"
PRED_PATH = ROOT / "docs" / "nested_spatial_validation" / "nested_spatial_oof_predictions.csv"
OUT_DIR = ROOT / "docs" / "nested_spatial_validation"

data = pd.read_csv(DATA_PATH)
pred = pd.read_csv(PRED_PATH)

required = {"Row_Index", "Outer_Fold", "Actual", "Predicted", "Probability"}
missing = required - set(pred.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

idx = pd.to_numeric(pred["Row_Index"], errors="raise")
if idx.isna().any() or not np.equal(idx, np.floor(idx)).all():
    raise ValueError("Row_Index contains invalid values.")
idx = idx.astype(int)

if idx.duplicated().any():
    raise ValueError("Row_Index contains duplicates.")
if not idx.between(0, len(data) - 1).all():
    raise ValueError("Row_Index is outside the dataset range.")

aligned = data.iloc[idx.to_numpy()].reset_index(drop=True)
actual = pred["Actual"].astype(int).to_numpy()
expected = aligned["arsenic_detected"].astype(int).to_numpy()

if not np.array_equal(actual, expected):
    raise ValueError("Row_Index does not match the dataset labels. Stop and inspect the validation script.")

print("SUCCESS: Row_Index mapping verified.")

result = aligned.copy()
result["Row_Index"] = idx.to_numpy()
result["Outer_Fold"] = pred["Outer_Fold"].to_numpy()
result["Actual"] = actual
result["Predicted"] = pred["Predicted"].astype(int).to_numpy()
result["Probability"] = pd.to_numeric(pred["Probability"], errors="raise").to_numpy()

result["Error_Type"] = np.select(
    [
        (result["Actual"] == 1) & (result["Predicted"] == 1),
        (result["Actual"] == 0) & (result["Predicted"] == 0),
        (result["Actual"] == 0) & (result["Predicted"] == 1),
        (result["Actual"] == 1) & (result["Predicted"] == 0),
    ],
    ["True Positive", "True Negative", "False Positive", "False Negative"],
    default="Unknown",
)

counts = pd.crosstab(result["Outer_Fold"], result["Error_Type"])
counts.to_csv(OUT_DIR / "fold_error_type_counts.csv")

print("\nERROR COUNTS BY FOLD")
print(counts.to_string())

fold4 = result[result["Outer_Fold"] == 4].copy()
excluded = {"arsenic_detected", "Latitude", "Longitude", "Row_Index", "Outer_Fold", "Actual", "Predicted", "Probability"}
features = [
    c for c in aligned.select_dtypes(include=np.number).columns
    if c not in excluded
]

comparisons = []

def compare_groups(frame, group_a, group_b, name):
    a_frame = frame[frame["Error_Type"] == group_a]
    b_frame = frame[frame["Error_Type"] == group_b]
    for feature in features:
        a = pd.to_numeric(a_frame[feature], errors="coerce").dropna()
        b = pd.to_numeric(b_frame[feature], errors="coerce").dropna()
        if len(a) and len(b):
            comparisons.append({
                "Comparison": name,
                "Feature": feature,
                "Group_A": group_a,
                "Group_B": group_b,
                "Group_A_Count": len(a),
                "Group_B_Count": len(b),
                "Group_A_Median": a.median(),
                "Group_B_Median": b.median(),
                "Median_Difference_B_minus_A": b.median() - a.median(),
            })

compare_groups(fold4[fold4["Actual"] == 0], "True Negative", "False Positive", "False Positives vs True Negatives")
compare_groups(fold4[fold4["Actual"] == 1], "True Positive", "False Negative", "False Negatives vs True Positives")

comparison_df = pd.DataFrame(comparisons)
comparison_df.to_csv(OUT_DIR / "fold4_error_type_feature_comparison.csv", index=False)

for name in comparison_df["Comparison"].unique():
    subset = comparison_df[comparison_df["Comparison"] == name].copy()
    subset["Absolute_Difference"] = subset["Median_Difference_B_minus_A"].abs()
    print(f"\nFOLD 4: {name}")
    print(subset.sort_values("Absolute_Difference", ascending=False).head(12).drop(columns="Absolute_Difference").round(4).to_string(index=False))

print("\nFOLD 4 PROBABILITY SUMMARY")
print(fold4.groupby("Error_Type")["Probability"].agg(["count", "mean", "median", "min", "max"]).round(4).to_string())

result[result["Actual"] != result["Predicted"]].to_csv(OUT_DIR / "fold_prediction_errors_verified.csv", index=False)

print("\nAnalysis complete. Output files saved in:")
print(OUT_DIR)
