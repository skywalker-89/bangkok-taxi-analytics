import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- MLflow Configuration ---
# Set this to the server you're running (likely from Docker)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# This will create the experiment if it doesn't exist
EXPERIMENT_NAME = "bangkok_taxi_trip_distance"
mlflow.set_experiment(EXPERIMENT_NAME)

# Model registry name
REGISTERED_MODEL_NAME = "trip_distance_model"

# --- Data File Paths ---
# We're reading from CSVs, just like your notebook
X_PATH = "X_features_distance.csv"
Y_PATH = "y_target_distance.csv"


def load_data():
    """Loads feature and target CSVs from disk."""
    print(f"Loading data from {X_PATH} and {Y_PATH}...")

    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print(f"❌ Error: Data files not found.")
        print(f"Please make sure '{X_PATH}' and '{Y_PATH}' are in the same directory,")
        print(f"or run the model_data_prep script first.")
        return None, None

    df_X = pd.read_csv(X_PATH)
    df_y = pd.read_csv(Y_PATH)

    print(f"✅ Loaded {len(df_X)} rows.")
    return df_X, df_y


def train():
    """
    Loads data, trains a model using GridSearchCV,
    and logs the results with MLflow.
    """
    df_X, df_y = load_data()
    if df_X is None:
        return

    # --- 1. Prepare Data (Logic from Notebook) ---
    print("Preparing features and target...")
    X = df_X.drop(columns=["VehicleID", "trip_id"])
    y = df_y["total_trip_distance_km"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Start MLflow Run ---
    with mlflow.start_run() as run:
        print(f"🚀 Starting MLflow Run ID: {run.info.run_id}")

        # --- 2. GridSearchCV (Logic from Notebook) ---
        xgb_reg = xgb.XGBRegressor(
            objective="reg:squarederror", random_state=42, n_jobs=-1
        )

        # NOTE: Your notebook snippet didn't show the param_grid.
        # Using a small, reasonable grid here.
        # --- !! UPDATE THIS GRID TO MATCH YOUR NOTEBOOK !! ---
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

        # Log the best parameters found by the grid search
        mlflow.log_params(grid_search.best_params_)

        best_xgb = grid_search.best_estimator_

        # --- 3. Evaluation (Logic from Notebook) ---
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

        # --- 6. Save model file (Logic from Notebook) ---
        # We do this *in addition* to MLflow to match your notebook
        model_filename = "best_xgb_trip_distance.model"
        best_xgb.save_model(model_filename)
        print(f"Saved model file to {model_filename}")
        mlflow.log_artifact(model_filename)  # Also save this file as an MLflow artifact

        print(f"\n✅ Successfully completed run {run.info.run_id}")


if __name__ == "__main__":
    train()
