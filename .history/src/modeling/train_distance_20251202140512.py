import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature

# ✅ PRO IMPORT: Use central DB connector
from src.utils.db import get_engine

# --- Configuration ---
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "bangkok_taxi_trip_distance"
mlflow.set_experiment(EXPERIMENT_NAME)

REGISTERED_MODEL_NAME = "trip_distance_model"
FEATURE_TABLE_NAME = "features_trip_distance"
TARGET_COL = "total_trip_distance_km"


def load_data_from_db():
    """Fetches features directly from PostgreSQL."""
    print(f"📡 Fetching training data from table: {FEATURE_TABLE_NAME}...")
    engine = get_engine()

    # Read the whole table (Consider limits or chunking for massive data)
    df = pd.read_sql(f"SELECT * FROM {FEATURE_TABLE_NAME}", engine)

    # Drop non-feature columns if they exist (like identifiers)
    # Adjust this list based on what you actually want to train on
    ignore_cols = ["VehicleID", "trip_id"]
    actual_ignore = [c for c in ignore_cols if c in df.columns]

    if actual_ignore:
        df = df.drop(columns=actual_ignore)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"✅ Loaded {len(X)} rows from DB.")
    return X, y


def train():
    """Main training pipeline."""
    X, y = load_data_from_db()

    # 1. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Define Model & Grid Search
    xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
    }

    print(f"🚀 Starting Grid Search for {EXPERIMENT_NAME}...")

    # Start MLflow Run
    with mlflow.start_run():
        # Log Data Lineage (Where did this data come from?)
        mlflow.set_tag("training_data_source", FEATURE_TABLE_NAME)
        mlflow.set_tag("model_type", "xgboost_regressor")
        mlflow.set_tag("developer", "papangkorn")

        grid_search = GridSearchCV(
            estimator=xgb_reg,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_absolute_error",
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        print(f"✅ Best parameters: {grid_search.best_params_}")
        mlflow.log_params(grid_search.best_params_)

        best_xgb = grid_search.best_estimator_

        # 3. Evaluation
        print("Evaluating model...")
        y_pred = best_xgb.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"  Test MAE: {mae:.4f}")
        print(f"  Test RMSE: {rmse:.4f}")
        print(f"  Test R2: {r2:.4f}")

        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        # 4. Infer Signature (The "Pro" Step)
        # This tells MLflow: "This model expects these columns as inputs"
        signature = infer_signature(X_test, y_pred)

        # 5. Log Model with Signature
        print(f"💾 Logging model to MLflow as '{REGISTERED_MODEL_NAME}'...")
        mlflow.xgboost.log_model(
            xgb_model=best_xgb,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,  # <--- Secure Input Schema
            input_example=X_test.iloc[
                :5
            ],  # Optional: Helps other devs see example inputs
        )

        print("🎉 Training Complete!")


if __name__ == "__main__":
    train()
