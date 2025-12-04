import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:8882"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "bangkok_taxi_trip_duration"
mlflow.set_experiment(EXPERIMENT_NAME)
REGISTERED_MODEL_NAME = "trip_duration_model"

# --- Data File Paths ---
X_PATH = "X_features_duration.csv"
Y_PATH = "y_target_duration.csv"


def load_data():
    """Loads feature and target CSVs from disk."""
    print(f"📚 [1/6] Loading data from {X_PATH} and {Y_PATH}...")

    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print(f"❌ Error: Data files not found. Run prep script first.")
        return None, None

    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)["duration_minutes"]

    print(f"✓ Loaded {len(X)} rows.")
    return X, y


def train():
    """
    Loads data, trains a model using GridSearchCV,
    and logs the results with MLflow.
    """
    X, y = load_data()
    if X is None:
        return

    # --- 1. Split Data ---
    print(" splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Start MLflow Run ---
    with mlflow.start_run() as run:
        print(f"🚀 [2/6] Starting MLflow Run ID: {run.info.run_id}")

        # --- 2. GridSearchCV ---
        xgb_reg = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            enable_categorical=True,
        )

        # Using a fast, simple grid. Update this as needed.
        param_grid = {
            "max_depth": [7],
            "learning_rate": [0.1],
            "n_estimators": [200],
            "subsample": [0.8],
        }

        print(f"Logging parameter grid: {param_grid}")
        mlflow.log_param("param_grid", param_grid)

        # Tell XGBoost which columns are categorical
        X_train["start_h3_zone"] = X_train["start_h3_zone"].astype("category")
        X_train["end_h3_zone"] = X_train["end_h3_zone"].astype("category")
        X_test["start_h3_zone"] = X_test["start_h3_zone"].astype("category")
        X_test["end_h3_zone"] = X_test["end_h3_zone"].astype("category")

        grid_search = GridSearchCV(
            estimator=xgb_reg,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_absolute_error",
            verbose=2,
        )

        print("🧠 [3/6] Starting GridSearchCV.fit()...")
        grid_search.fit(X_train, y_train)

        print("✓ GridSearch complete.")
        print(f"Best parameters: {grid_search.best_params_}")

        # Log the best params
        mlflow.log_params(grid_search.best_params_)

        best_xgb = grid_search.best_estimator_

        # --- 3. Evaluation ---
        print("📈 [4/6] Evaluating model on test set...")
        y_pred = best_xgb.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\n--- Test Metrics ---")
        print(f"  Test MAE:  {mae:.3f} minutes")
        print(f"  Test RMSE: {rmse:.3f} minutes")
        print(f"  Test R2:   {r2:.3f}")

        # --- 4. Log Metrics to MLflow ---
        print(" [5/6] Logging metrics to MLflow...")
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        # --- 5. Model Logging & Registration ---
        print(f"💾 [6/6] Logging and registering model as '{REGISTERED_MODEL_NAME}'...")
        mlflow.xgboost.log_model(
            xgb_model=best_xgb,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(5),
        )

        # Also log the encoders
        mlflow.log_artifact("duration_start_zone_encoder.pkl")
        mlflow.log_artifact("duration_end_zone_encoder.pkl")

        print(f"\n✅ Successfully completed run {run.info.run_id}")


if __name__ == "__main__":
    train()
