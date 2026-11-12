import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

data = pd.read_csv("data/processed/Arsenic_Spatial_Model_Dataset.csv")
pred = pd.read_csv("docs/nested_spatial_validation/nested_spatial_oof_predictions.csv")

idx = pred["Row_Index"].astype(int)
assert (data.iloc[idx]["arsenic_detected"].to_numpy() == pred["Actual"].to_numpy()).all(), "Row mapping mismatch"

fold = pred[pred["Outer_Fold"] == 4].copy()
idx4 = fold["Row_Index"].astype(int)

fold["Latitude"] = data.iloc[idx4]["Latitude"].to_numpy()
fold["Longitude"] = data.iloc[idx4]["Longitude"].to_numpy()

fold["Error_Type"] = (
fold["Actual"].astype(str) + "_" + fold["Predicted"].astype(str)
).map({"0_0": "TN", "0_1": "FP", "1_0": "FN", "1_1": "TP"})

print(fold["Error_Type"].value_counts())

fig, ax = plt.subplots(figsize=(10, 8))
categories = ["TN", "FP", "FN", "TP"]
[ax.scatter(fold.loc[fold["Error_Type"] == c, "Longitude"], fold.loc[fold["Error_Type"] == c, "Latitude"], label=f"{c} (n={(fold['Error_Type'] == c).sum()})", s=35) for c in categories]

ax.set_title("Fold 4: Geographic Distribution of Prediction Errors")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()

out = Path("docs/nested_spatial_validation")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fold4_geographic_error_map.png", dpi=300, bbox_inches="tight")
plt.close(fig)
fold.to_csv(out / "fold4_geographic_error_points.csv", index=False)

print("Map saved: docs/nested_spatial_validation/fold4_geographic_error_map.png")
