from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(**file**).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "Arsenic_Spatial_Model_Dataset.csv"

PREDICTIONS_PATH = ROOT / "docs" / "nested_spatial_validation" / "nested_spatial_oof_predictions.csv"

OUTPUT_DIR = ROOT / "docs" / "nested_spatial_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("NESTED SPATIAL FOLD ERROR ANALYSIS")
print("=" * 65)

data = pd.read_csv(DATA_PATH)
pred = pd.read_csv(PREDICTIONS_PATH)

print(f"\nDataset rows: {len(data)}")
print(f"Prediction rows: {len(pred)}")

required = {
"Row_Index",
"Outer_Fold",
"Actual",
"Predicted",
"Probability",
}

missing = required - set(pred.columns)

if missing:
raise ValueError(f"Missing prediction columns: {sorted(missing)}")

if "arsenic_detected" not in data.columns:
raise ValueError("Dataset target column is missing.")

# Validate Row_Index before using it.

row_index = pd.to_numeric(pred["Row_Index"], errors="raise")

if row_index.isna().any():
raise ValueError("Row_Index contains missing values.")

if not np.all(row_index == np.floor(row_index)):
raise ValueError("Row_Index contains non-integer values.")

row_index = row_index.astype(int)

if row_index.duplicated().any():
raise ValueError("Row_Index contains duplicate values.")

# This check assumes Row_Index represents zero-based row positions.

# It verifies the range and actual labels before analysis.

if not row_index.between(0, len(data) - 1).all():
raise ValueError(
"Row_Index is outside the dataset's zero-based row range. "
"Inspect nested_spatial_validation.py before proceeding."
)

data_y = data["arsenic_detected"].astype(int).to_numpy()
actual_y = pred["Actual"].astype(int).to_numpy()

expected_y = data_y[row_index.to_numpy()]

if not np.array_equal(expected_y, actual_y):
mismatches = int(np.sum(expected_y != actual_y))

```
raise ValueError(
    f"Row_Index label verification failed for {mismatches} records. "
    "Row_Index may not represent current dataset row positions. "
    "Do not interpret the error analysis until its meaning is verified."
)
```

print("\nRow_Index verification passed.")
print("Every prediction's Actual label matches its indexed dataset row.")

# Attach the original dataset features using verified row positions.

features = data.iloc[row_index.to_numpy()].reset_index(drop=True).copy()

combined = features.copy()

combined["Row_Index"] = row_index.to_numpy()
combined["Outer_Fold"] = pred["Outer_Fold"].to_numpy()
combined["Actual"] = actual_y
combined["Predicted"] = pred["Predicted"].astype(int).to_numpy()
combined["Probability"] = pd.to_numeric(
pred["Probability"], errors="raise"
).to_numpy()
combined["Selected_Threshold"] = pd.to_numeric(
pred["Selected_Threshold"], errors="raise"
).to_numpy()

# Confirm predictions are binary.

if not combined["Predicted"].isin([0, 1]).all():
raise ValueError("Predicted contains values other than 0 and 1.")

# Classify outcomes.

combined["Error_Type"] = np.select(
[
(combined["Actual"] == 1) & (combined["Predicted"] == 1),
(combined["Actual"] == 0) & (combined["Predicted"] == 0),
(combined["Actual"] == 0) & (combined["Predicted"] == 1),
(combined["Actual"] == 1) & (combined["Predicted"] == 0),
],
[
"True Positive",
"True Negative",
"False Positive",
"False Negative",
],
default="Unknown",
)

# Summarize errors by fold.

counts = pd.crosstab(
combined["Outer_Fold"],
combined["Error_Type"],
)

for label in [
"True Negative",
"False Positive",
"False Negative",
"True Positive",
]:
if label not in counts.columns:
counts[label] = 0

counts["Observations"] = counts[
[
"True Negative",
"False Positive",
"False Negative",
"True Positive",
]
].sum(axis=1)

negative_total = counts["True Negative"] + counts["False Positive"]
positive_total = counts["True Positive"] + counts["False Negative"]

counts["False_Positive_Rate"] = (
counts["False Positive"] / negative_total.replace(0, np.nan)
)

counts["False_Negative_Rate"] = (
counts["False Negative"] / positive_total.replace(0, np.nan)
)

counts.to_csv(OUTPUT_DIR / "fold_error_type_counts.csv")

print("\n===== ERROR-TYPE COUNTS BY FOLD =====")
print(counts.round(4).to_string())

# Select numerical predictors, excluding target, IDs, and coordinates.

excluded = {
"arsenic_detected",
"Row_Index",
"Outer_Fold",
"Actual",
"Predicted",
"Probability",
"Selected_Threshold",
"Latitude",
"Longitude",
}

numeric_features = [
column
for column in features.select_dtypes(include=np.number).columns
if column not in excluded
]

comparison_rows = []

def compare_groups(frame, group_a, group_b, comparison_name):
a_data = frame[frame["Error_Type"] == group_a]
b_data = frame[frame["Error_Type"] == group_b]

```
for feature in numeric_features:
    a = pd.to_numeric(a_data[feature], errors="coerce").dropna()
    b = pd.to_numeric(b_data[feature], errors="coerce").dropna()

    if a.empty or b.empty:
        continue

    comparison_rows.append({
        "Comparison": comparison_name,
        "Feature": feature,
        "Group_A": group_a,
        "Group_B": group_b,
        "Group_A_Count": len(a),
        "Group_B_Count": len(b),
        "Group_A_Median": a.median(),
        "Group_B_Median": b.median(),
        "Median_Difference_B_minus_A": b.median() - a.median(),
    })
```

# Analyze Fold 4.

fold4 = combined[combined["Outer_Fold"] == 4].copy()

compare_groups(
fold4[fold4["Actual"] == 0],
"True Negative",
"False Positive",
"Fold 4: False Positives vs True Negatives",
)

compare_groups(
fold4[fold4["Actual"] == 1],
"True Positive",
"False Negative",
"Fold 4: False Negatives vs True Positives",
)

comparison_df = pd.DataFrame(comparison_rows)

comparison_path = OUTPUT_DIR / "fold4_error_type_feature_comparison.csv"

if not comparison_df.empty:
comparison_df.to_csv(comparison_path, index=False)

```
for name in comparison_df["Comparison"].unique():
    subset = comparison_df[comparison_df["Comparison"] == name].copy()

    subset["Absolute_Difference"] = (
        subset["Median_Difference_B_minus_A"].abs()
    )

    print(f"\n===== {name.upper()} =====")

    print(
        subset.sort_values("Absolute_Difference", ascending=False)
        .head(12)
        .drop(columns="Absolute_Difference")
        .round(4)
        .to_string(index=False)
    )
```

else:
print("\nNo feature comparisons could be generated.")

# Summarize prediction probabilities by error type.

print("\n===== FOLD 4 PROBABILITY SUMMARY =====")

print(
fold4.groupby("Error_Type")["Probability"]
.agg(["count", "mean", "median", "min", "max"])
.round(4)
.to_string()
)

# Save verified observation-level errors.

errors = combined[combined["Actual"] != combined["Predicted"]].copy()

errors.to_csv(
OUTPUT_DIR / "fold_prediction_errors_verified.csv",
index=False,
)

print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)

print("\nSaved files:")
print(OUTPUT_DIR / "fold_error_type_counts.csv")
print(comparison_path)
print(OUTPUT_DIR / "fold_prediction_errors_verified.csv")

print(
"\nFeature median differences are descriptive; "
"they do not establish causation."
)
