import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv  # Make sure this is here

# --- MLflow Configuration ---
load_dotenv()  # Load .env file for credentials
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "bangkok_taxi_pickup_rate"
mlflow.set_experiment(EXPERIMENT_NAME)
REGISTERED_MODEL_NAME = "pickup_rate_model"

# --- Data File Paths ---
X_PATH = "X_features_5min.csv"
Y_PATH = "y_target_5min.csv"
META_PATH = "meta_5min.csv"

# --- Model Artifacts ---
ENCODER_FILE = "pickup_rate_h3_encoder.pkl"


def load_data():
    """Loads feature, target, and metadata CSVs."""
    print("📚 [1/8] Loading data...")

    if not all(os.path.exists(p) for p in [X_PATH, Y_PATH, META_PATH]):
        print("❌ Error: Data files not found. Run prep script first.")
        return None, None, None

    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH).iloc[:, 0]  # Get y as a Series
    meta = pd.read_csv(META_PATH)

    # The prep script saves the column as 'time_bin'
    meta["time_bin"] = pd.to_datetime(meta["time_bin"])

    print(f"✓ Loaded {len(X)} rows.")
    return X, y, meta


def split_data(X, y, meta):
    """Splits data based on time (85% train, 15% valid)."""
    print("⏰ [2/8] Creating time-based split...")

    # Use 'time_bin' to split
    split_time = meta["time_bin"].quantile(0.85)

    train_indices = meta[meta["time_bin"] < split_time].index
    valid_indices = meta[meta["time_bin"] >= split_time].index

    X_train, X_valid = X.loc[train_indices], X.loc[valid_indices]
    y_train, y_valid = y.loc[train_indices], y.loc[valid_indices]
    meta_train, meta_valid = meta.loc[train_indices], meta.loc[valid_indices]

    print(f"✓ Train: {len(X_train)}, Valid: {len(X_valid)}")
    return X_train, X_valid, y_train, y_valid, meta_train, meta_valid


def train():
    """
    Loads data, trains an XGBoost classifier, and logs to MLflow.
    """
    # --- 1. Load Data ---
    X, y, meta = load_data()
    if X is None:
        return

    # --- 2. Copy features from Meta to X ---
    print("🔑 [3/8] Label encoding and copying features...")

    # Fit the encoder on ALL H3 cells *before* splitting
    h3_encoder = LabelEncoder()
    X["h3_cell"] = h3_encoder.fit_transform(meta["h3_cell"])

    # --- FIX ---
    # Copy the time features from meta to X
    X["hour_of_day"] = meta["hour_of_day"]
    X["day_of_week"] = meta["day_of_week"]
    X["is_weekend"] = meta["is_weekend"]
    # --- END FIX ---

    # --- 3. Split Data ---
    (X_train, X_valid, y_train, y_valid, meta_train, meta_valid) = split_data(
        X, y, meta
    )
    # The X_train and X_valid dataframes now contain all the categorical columns

    # --- 4. Save Encoder ---
    print("💾 [4/8] Saving H3 encoder...")
    joblib.dump(h3_encoder, ENCODER_FILE)
    print(f"✓ Saved encoder to {ENCODER_FILE}")

    # --- 5. Start MLflow Run ---
    with mlflow.start_run() as run:
        print(f"\n🚀 [5/8] Starting MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("data_split_time", meta_train["time_bin"].max())
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_valid))

        # --- 6. Train Model ---
        print("🧠 [6/8] Training XGBoost model...")

        # These features will be set as 'category' type
        CATEGORICAL_FEATURES = ["h3_cell", "hour_of_day", "day_of_week", "is_weekend"]

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "random_state": 42,
            "enable_categorical": True,  # Tell XGBoost to use category features
            "n_jobs": -1,
        }
        mlflow.log_params(params)

        # Convert feature types for XGBoost
        X_train[CATEGORICAL_FEATURES] = X_train[CATEGORICAL_FEATURES].astype("category")
        X_valid[CATEGORICAL_FEATURES] = X_valid[CATEGORICAL_FEATURES].astype("category")

        model = xgb.XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=50,
            verbose=100,
        )

        # --- 7. Evaluation ---
        print("📈 [7/8] Evaluating model and logging metrics...")
        y_pred_proba = model.predict_proba(X_valid)[:, 1]

        auc = roc_auc_score(y_valid, y_pred_proba)
        pr_auc = average_precision_score(y_valid, y_pred_proba)
        brier = brier_score_loss(y_valid, y_pred_proba)

        print(f"\n--- Validation Metrics ---")
        print(f"  ROC AUC:          {auc:.4f}")
        print(f"  Precision-Recall (PR) AUC: {pr_auc:.4f}")
        print(f"  Brier Score Loss: {brier:.4f}")

        mlflow.log_metric("valid_roc_auc", auc)
        mlflow.log_metric("valid_pr_auc", pr_auc)
        mlflow.log_metric("valid_brier_loss", brier)

        # --- 8. Log Artifacts & Register Model ---
        print("💾 [8/8] Logging artifacts and registering model...")

        # Log the encoder
        mlflow.log_artifact(ENCODER_FILE)

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.iloc[:5],  # Log an example input
        )

        print(f"\n✅ Successfully completed run {run.info.run_id}")


if __name__ == "__main__":
    train()
