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

# ✅ FIX: Define a clean artifact directory
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_data_from_db():
    print(f"📡 Fetching training data from {FEATURE_TABLE_NAME}...")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {FEATURE_TABLE_NAME}", engine)

    # Drop IDs if present
    drop_cols = ["VehicleID", "trip_id"]
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f"✅ Loaded {len(X)} rows.")
    return X, y


def train():
    X, y = load_data_from_db()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    # Reduced grid for speed
    param_grid = {"n_estimators": [100], "max_depth": [5], "learning_rate": [0.1]}

    print(f"🚀 Starting Training for {EXPERIMENT_NAME}...")

    with mlflow.start_run():
        mlflow.set_tag("training_data_source", FEATURE_TABLE_NAME)
        mlflow.set_tag("model_type", "xgboost_regressor")

        grid_search = GridSearchCV(
            estimator=xgb_reg,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_absolute_error",
            verbose=1,
        )

        grid_search.fit(X_train, y_train)
        best_xgb = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)

        # Evaluation
        y_pred = best_xgb.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"  Test MAE: {mae:.4f}")
        mlflow.log_metrics({"test_mae": mae, "test_rmse": rmse, "test_r2": r2})

        # Signature
        signature = infer_signature(X_test, y_pred)

        print(f"💾 Logging model to MLflow...")
        mlflow.xgboost.log_model(
            xgb_model=best_xgb,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
        )

        print("🎉 Training Complete!")


if __name__ == "__main__":
    train()
