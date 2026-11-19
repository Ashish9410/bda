
from pathlib import Path

import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "monthly_foreign_arrivals_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ml"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_FILE = OUTPUT_DIR / "tourism_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "model_comparison.csv"
BEST_MODEL_FILE = MODEL_DIR / "best_tourism_model.pkl"


# ==========================================
# 2. PROJECT CONFIGURATION
# ==========================================

TARGET = "foreign_tourist_arrivals"

FEATURES = [
    "year",
    "month",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_3",
    "rolling_mean_3"
]

TEST_YEAR = 2024

print("\n==========================================")
print(" TOURISM INTELLIGENCE - ML MODEL TRAINING")
print("==========================================")


# ==========================================
# 3. LOAD FEATURE ENGINEERED DATASET
# ==========================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Feature dataset not found: {DATA_FILE}\n"
        "Run prepare_ml_dataset.py first."
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")

print("Total Records:", len(df))
print("Available Years:", sorted(df["year"].unique()))


# ==========================================
# 4. DATA VALIDATION
# ==========================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=required_columns
).copy()

df = df[
    df[TARGET] >= 0
].copy()

df = df.sort_values(
    ["year", "month"]
).reset_index(drop=True)

print("\nData validation completed.")

print("Usable Records:", len(df))


# ==========================================
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

print("\nPreparing training and testing datasets...")

train_df = df[
    df["year"] < TEST_YEAR
].copy()

test_df = df[
    df["year"] == TEST_YEAR
].copy()

if train_df.empty or test_df.empty:
    raise ValueError(
        "Training or testing dataset is empty. "
        "Check the available years."
    )

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

print("Training Years:", sorted(train_df["year"].unique()))
print("Testing Year:", TEST_YEAR)


# ==========================================
# 6. DEFINE MACHINE LEARNING MODELS
# ==========================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

}


# ==========================================
# 7. TRAIN AND EVALUATE MODELS
# ==========================================

results = []
predictions = {}

print("\n==========================================")
print(" MODEL TRAINING STARTED")
print("==========================================")

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

    y_pred = np.maximum(
        y_pred,
        0
    )

    # Evaluation metrics

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    ) if len(y_test) >= 2 else np.nan

    results.append({

        "model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2_Score": round(r2, 4)

    })

    predictions[model_name] = y_pred

    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")


# ==========================================
# 8. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE"
).reset_index(drop=True)

print("\n==========================================")
print(" MODEL PERFORMANCE COMPARISON")
print("==========================================")

print(results_df.to_string(index=False))

results_df.to_csv(
    METRICS_FILE,
    index=False
)


# ==========================================
# 9. SELECT BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["model"]

best_model = models[best_model_name]

best_predictions = predictions[best_model_name]

print("\nBest Model:", best_model_name)


# ==========================================
# 10. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    BEST_MODEL_FILE
)

print("\nBest model saved successfully!")

print(BEST_MODEL_FILE)


# ==========================================
# 11. EXPORT PREDICTIONS
# ==========================================

prediction_df = test_df[
    ["year", "month", TARGET]
].copy()

prediction_df = prediction_df.rename(
    columns={
        TARGET: "actual_arrivals"
    }
)

prediction_df["predicted_arrivals"] = (
    np.round(best_predictions).astype("int64")
)

prediction_df["prediction_error"] = (
    prediction_df["actual_arrivals"] -
    prediction_df["predicted_arrivals"]
)

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)

print("\n==========================================")
print(" ACTUAL VS PREDICTED ARRIVALS")
print("==========================================")

print(
    prediction_df.to_string(index=False)
)


# ==========================================
# 12. FINAL SUMMARY
# ==========================================

print("\n==========================================")
print(" MACHINE LEARNING COMPLETED")
print("==========================================")

print("Best Model:", best_model_name)

print("Model Comparison:", METRICS_FILE)

print("Predictions:", PREDICTION_FILE)

print("Saved Model:", BEST_MODEL_FILE)

print("\nTourism Demand ML Pipeline Completed!")
