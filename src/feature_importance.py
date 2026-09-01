from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "Arsenic_Classification_Dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_arsenic_classification_model.joblib"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "EDA"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 70)


df = pd.read_csv(DATA_PATH)

target = "arsenic_detected"

drop_columns = [
    "arsenic_detected",
    "As",
    "S_No",
    "State",
    "District",
    "Location",
    "District_std",
    "District_canonical",
    "IMD_subdivision",
    "soil_moisture_available",
    "rainfall_available",
    "Year",
]

drop_columns = [
    col for col in drop_columns
    if col in df.columns
]

X = df.drop(columns=drop_columns)

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

importance = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nTOP 20 FEATURES")
print(
    importance_df.head(20).to_string(index=False)
)


csv_path = (
    OUTPUT_DIR
    / "XGBoost_Feature_Importance.csv"
)

importance_df.to_csv(
    csv_path,
    index=False
)


top_features = importance_df.head(15).sort_values(
    "Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top XGBoost Features for Arsenic Detection")

plt.tight_layout()

figure_path = (
    OUTPUT_DIR
    / "XGBoost_Feature_Importance.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSaved:")
print(csv_path)
print(figure_path)

print("\nFEATURE IMPORTANCE COMPLETED")