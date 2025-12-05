import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from src.utils.db import get_engine

# --- Configuration ---
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "bangkok_taxi_inter_zone"
mlflow.set_experiment(EXPERIMENT_NAME)
REGISTERED_MODEL_NAME = "inter_zone_model"

FEATURE_TABLE_NAME = "features_inter_zone"
TARGET_COL = "travel_time_minutes"

# Artifacts
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
START_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "inter_zone_start_encoder.pkl")
END_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "inter_zone_end_encoder.pkl")


def load_data_from_db():
    print(f"📡 Fetching data from {FEATURE_TABLE_NAME}...")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {FEATURE_TABLE_NAME}", engine)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"✅ Loaded {len(X)} rows.")
    return X, y


def train():
    X, y = load_data_from_db()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 7],
        "learning_rate": [0.05, 0.1],
    }

    print(f"🚀 Starting Grid Search for {EXPERIMENT_NAME}...")

    with mlflow.start_run():
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

        best_xgb = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)

        # Evaluate
        print("Evaluating model...")
        y_pred = best_xgb.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"  Test MAE: {mae:.4f}")
        print(f"  Test RMSE: {rmse:.4f}")
        print(f"  Test R2: {r2:.4f}")

        mlflow.log_metrics({"test_mae": mae, "test_rmse": rmse, "test_r2": r2})

        # Log Encoders
        print("💾 Logging encoders...")
        if os.path.exists(START_ENCODER_PATH):
            mlflow.log_artifact(START_ENCODER_PATH, artifact_path="encoders")
        if os.path.exists(END_ENCODER_PATH):
            mlflow.log_artifact(END_ENCODER_PATH, artifact_path="encoders")

        # Log Model
        signature = infer_signature(X_test, y_pred)
        mlflow.xgboost.log_model(
            xgb_model=best_xgb,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=X_test.iloc[:5],
        )
        print("🎉 Training Complete!")


if __name__ == "__main__":
    train()
