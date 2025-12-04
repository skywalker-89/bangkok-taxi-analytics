import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
from src.utils.db import get_engine


# --- MLflow Configuration ---
load_dotenv()  # Load .env file for credentials
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_experiment("bangkok_taxi_trip_distance")


# --- Data File Paths ---
FEATURE_TABLE_NAME = "features_trip_distance"
TARGET_COL = "total_trip_distance_km"


def load_data_from_db():
    print(f"📡 Fetching training data from {FEATURE_TABLE_NAME}...")
    engine = get_engine()

    # Read the whole table
    df = pd.read_sql(f"SELECT * FROM {FEATURE_TABLE_NAME}", engine)

    # Split X and y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"✅ Loaded {len(X)} rows.")
    return X, y


def train():
    """
    Loads data, trains a model using GridSearchCV,
    and logs the results with MLflow.
    """
    df_X, df_y = load_data_from_db()
    if df_X is None:
        return

    # --- 1. Prepare Data ---
    print("Preparing features and target...")

    # --- FIX: Drop h3_end as well ---
    # It's an 'object' (string) column and not a feature
    X = df_X.drop(columns=["VehicleID", "trip_id", "h3_end"])
    y = df_y["total_trip_distance_km"]

    # --- FIX: Encode h3_start and convert to category BEFORE split ---
    print("Encoding categorical features...")
    if "h3_start" in X.columns:
        encoder = LabelEncoder()
        X["h3_start_encoded"] = encoder.fit_transform(X["h3_start"])
        X = X.drop(columns=["h3_start"])  # Drop the original text column

        # Convert to category dtype *before* train_test_split
        X["h3_start_encoded"] = X["h3_start_encoded"].astype("category")

    # --- END FIX ---

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Start MLflow Run ---
    with mlflow.start_run() as run:
        print(f"🚀 Starting MLflow Run ID: {run.info.run_id}")

        # --- 2. GridSearchCV ---
        xgb_reg = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            enable_categorical=True,  # This is correct
        )

        # Using a small, reasonable grid
        param_grid = {
            "max_depth": [5, 7],
            "learning_rate": [0.1],
            "n_estimators": [200],
            "subsample": [0.8],
        }

        print(f"Logging parameter grid: {param_grid}")
        mlflow.log_param("param_grid", param_grid)

        grid_search = GridSearchCV(
            estimator=xgb_reg,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_absolute_error",
            verbose=2,
        )

        print("Starting GridSearchCV.fit()...")
        grid_search.fit(X_train, y_train)

        print("GridSearch complete.")
        print(f"Best parameters found: {grid_search.best_params_}")

        mlflow.log_params(grid_search.best_params_)
        best_xgb = grid_search.best_estimator_

        # --- 3. Evaluation ---
        print("Evaluating model on test set...")
        y_pred = best_xgb.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Test MAE: {mae:.2f}")
        print(f"Test RMSE: {rmse:.2f}")
        print(f"Test R2: {r2:.3f}")

        # --- 4. Log Metrics to MLflow ---
        print("Logging metrics to MLflow...")
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        # --- 5. Model Logging & Registration ---
        print(f"Logging and registering model as '{REGISTERED_MODEL_NAME}'...")
        mlflow.xgboost.log_model(
            xgb_model=best_xgb,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # --- 6. Save model file ---
        model_filename = "best_xgb_trip_distance.model"
        best_xgb.save_model(model_filename)
        print(f"Saved model file to {model_filename}")
        mlflow.log_artifact(model_filename)

        print(f"\n✅ Successfully completed run {run.info.run_id}")


if __name__ == "__main__":
    train()
