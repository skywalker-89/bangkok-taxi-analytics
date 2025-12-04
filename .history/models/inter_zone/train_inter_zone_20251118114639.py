import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:8882"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "bangkok_taxi_inter_zone"
mlflow.set_experiment(EXPERIMENT_NAME)
REGISTERED_MODEL_NAME = "inter_zone_model"

# --- Data File Paths ---
X_PATH = "X_features_inter_zone.csv"
Y_PATH = "y_target_inter_zone.csv"

# --- Model & Artifact Files ---
MODEL_FILE = "xgboost_inter_zone_model.pkl"
ENCODER_FILE = "inter_zone_encoders.pkl"
METADATA_FILE = "inter_zone_model_metadata.json"


def load_data():
    """Loads feature and target CSVs from disk."""
    print(f"📚 [1/7] Loading data from {X_PATH} and {Y_PATH}...")

    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print(f"❌ Error: Data files not found. Run prep script first.")
        return None, None

    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)

    print(f"✓ Loaded {len(X)} rows.")
    return X, y


def encode_features(X):
    """Label encodes H3 features and saves encoders."""
    print("🔑 [2/7] Label encoding H3 zones...")
    X_encoded = X.copy()

    le_start = LabelEncoder()
    le_end = LabelEncoder()

    X_encoded["start_h3"] = le_start.fit_transform(X["start_h3"])
    X_encoded["end_h3"] = le_end.fit_transform(X["end_h3"])

    # Save encoders
    encoders = {"start_h3_encoder": le_start, "end_h3_encoder": le_end}
    joblib.dump(encoders, ENCODER_FILE)
    print(f"✓ Encoders saved to {ENCODER_FILE}")

    return X_encoded


def train():
    """
    Loads data, trains XGBRegressor, and logs to MLflow.
    """
    X, y = load_data()
    if X is None:
        return

    X_encoded = encode_features(X)

    # --- 1. Split Data ---
    print(" splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # --- Start MLflow Run ---
    with mlflow.start_run() as run:
        print(f"🚀 [3/7] Starting MLflow Run ID: {run.info.run_id}")

        # --- 2. Set Parameters (from notebook) ---
        # These are the 'best_params' from your notebook
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "tree_method": "hist",
            "early_stopping_rounds": 50,
            "enable_categorical": True,  # For H3 zones
        }

        print(f"Logging parameters: {params}")
        mlflow.log_params(params)

        # --- 3. Train Model ---
        print("🧠 [4/7] Training XGBoost model...")

        # Tell XGBoost which columns are categorical
        X_train["start_h3"] = X_train["start_h3"].astype("category")
        X_train["end_h3"] = X_train["end_h3"].astype("category")
        X_test["start_h3"] = X_test["start_h3"].astype("category")
        X_test["end_h3"] = X_test["end_h3"].astype("category")

        model = xgb.XGBRegressor(**params)

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

        # --- 4. Evaluation ---
        print("📈 [5/7] Evaluating model on test set...")
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\n--- Test Metrics ---")
        print(f"  Test MAE:  {mae:.3f} minutes")
        print(f"  Test RMSE: {rmse:.3f} minutes")
        print(f"  Test R2:   {r2:.3f}")

        # --- 5. Log Metrics to MLflow ---
        print(" [6/7] Logging metrics to MLflow...")
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        # --- 6. Save Artifacts & Log to MLflow ---
        print(f"💾 [7/7] Saving artifacts and registering model...")

        # Save model with joblib (as in notebook)
        joblib.dump(model, MODEL_FILE)

        # Save metadata (as in notebook)
        metadata = {
            "model_type": "XGBRegressor",
            "target": "travel_time_minutes",
            "features": list(X_encoded.columns),
            "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        }
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

        # Log all artifacts
        mlflow.log_artifact(MODEL_FILE)
        mlflow.log_artifact(ENCODER_FILE)
        mlflow.log_artifact(METADATA_FILE)

        # --- 7. Model Registration ---
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",  # Separate from joblib, this is for MLflow's native format
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(5),
        )

        print(f"\n✅ Successfully completed run {run.info.run_id}")


if __name__ == "__main__":
    train()
