import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:8882"
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
    print(f"📚 [1/8] Loading data...")

    if not all(os.path.exists(p) for p in [X_PATH, Y_PATH, META_PATH]):
        print(f"❌ Error: Data files not found. Run prep script first.")
        return None, None, None

    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH).iloc[:, 0]  # Get y as a Series
    meta = pd.read_csv(META_PATH)
    meta["time_bin"] = pd.to_datetime(meta["time_bin"])

    print(f"✓ Loaded {len(X)} rows.")
    return X, y, meta


def train():
    """
    Loads data, trains XGBoost classifier with time-series split,
    and logs results to MLflow.
    """
    X, y, meta = load_data()
    if X is None:
        return

    # --- 1. Define Features ---
    # (h3_cell is in meta, not X)
    CATEGORICAL_FEATURES = ["dow", "hour", "minute_of_day"]
    TARGET = "y"

    # --- 2. Time-Based Split (CRITICAL) ---
    print("⏰ [2/8] Creating time-based split...")
    # Use dates from your notebook. Update these as needed.
    TRAIN_END_DATE = meta["time_bin"].max() - pd.to_timedelta("3D")
    VALID_END_DATE = meta["time_bin"].max() - pd.to_timedelta("1D")

    train_idx = meta[meta["time_bin"] < TRAIN_END_DATE].index
    valid_idx = meta[
        (meta["time_bin"] >= TRAIN_END_DATE) & (meta["time_bin"] < VALID_END_DATE)
    ].index

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]
    meta_train, meta_valid = meta.loc[train_idx], meta.loc[valid_idx]

    print(f"✓ Train: {len(X_train)}, Valid: {len(X_valid)}")

    # --- 3. Label Encode H3 Cell (Categorical) ---
    print("🔑 [3/8] Label encoding H3 cells...")
    h3_encoder = LabelEncoder()

    # Add h3_cell from meta to X for training
    X_train["h3_cell"] = h3_encoder.fit_transform(meta_train["h3_cell"])
    X_valid["h3_cell"] = h3_encoder.transform(meta_valid["h3_cell"])

    CATEGORICAL_FEATURES.append("h3_cell")
    joblib.dump(h3_encoder, ENCODER_FILE)
    print(f"✓ H3 encoder saved to {ENCODER_FILE}")

    # --- 4. Handle Class Imbalance (CRITICAL) ---
    print("⚖️ [4/8] Calculating class imbalance...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"✓ scale_pos_weight set to: {scale_pos_weight:.2f}")

    # --- Start MLflow Run ---
    with mlflow.start_run() as run:
        print(f"🚀 [5/8] Starting MLflow Run ID: {run.info.run_id}")

        # --- 5. Set Parameters ---
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "aucpr"],
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "tree_method": "hist",
            "enable_categorical": True,  # Tell XGBoost about categories
            "scale_pos_weight": scale_pos_weight,
        }
        mlflow.log_params(params)

        # --- 6. Train Model ---
        print("🧠 [6/8] Training XGBoost model...")

        # Tell XGBoost which columns are categorical
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
            input_example=X_train.head(5),
        )

        print(f"\n✅ Successfully completed run {run.info.run_id}")


if __name__ == "__main__":
    train()
