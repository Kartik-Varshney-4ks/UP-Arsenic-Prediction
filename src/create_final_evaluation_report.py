from pathlib import Path
import pandas as pd

base = Path("docs/nested_spatial_validation")
out = Path("docs/research_results")
out.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(base / "nested_spatial_summary.csv")
folds = pd.read_csv(base / "nested_spatial_outer_fold_results.csv")
geo = pd.read_csv(base / "nested_fold_geography_diagnostics.csv")

print("\nOVERALL NESTED SPATIAL VALIDATION")
print(summary.to_string(index=False))

print("\nOUTER FOLD RESULTS")
print(folds.to_string(index=False))

print("\nFOLD GEOGRAPHY DIAGNOSTICS")
print(geo.to_string(index=False))

summary.to_csv(out / "final_nested_spatial_summary.csv", index=False)
folds.to_csv(out / "final_nested_spatial_fold_results.csv", index=False)

merged = folds.merge(geo, on="Outer_Fold", how="left", suffixes=("", "_geo"))
merged.to_csv(out / "final_nested_spatial_fold_analysis.csv", index=False)

print("\nSUCCESS: Final evaluation files created.")
print("docs/research_results/final_nested_spatial_summary.csv")
print("docs/research_results/final_nested_spatial_fold_results.csv")
print("docs/research_results/final_nested_spatial_fold_analysis.csv")
