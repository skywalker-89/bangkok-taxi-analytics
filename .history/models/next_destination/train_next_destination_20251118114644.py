import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
import os
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

# import matplotlib.pyplot as plt  <-- REMOVED

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:8882"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "bangkok_taxi_next_destination"
mlflow.set_experiment(EXPERIMENT_NAME)
REGISTERED_MODEL_NAME = "next_destination_model"

# --- Database & Feature Config ---
DB_URL = "postgresql+psycopg2://postgres:mypassword@localhost:5432/bangkok_taxi_db"
FEATURE_TABLE_NAME = "model_destination_features"

# --- Model Parameters ---
N_TOP_DESTINATIONS = 75
TIME_BASED_TEST_SPLIT = 0.2
ENCODER_FILENAME = "destination_label_encoders.pkl"


def load_data():
    """Loads features from the database."""
    print("📚 [1/8] Loading features from database...")
    engine = create_engine(DB_URL)
    query = f"SELECT * FROM {FEATURE_TABLE_NAME}"
    df = pd.read_sql(query, engine)
    df["pickup_time"] = pd.to_datetime(df["pickup_time"])
    df = df.sort_values(by="pickup_time")  # Critical for time-split
    print(f"✓ Loaded {len(df)} samples.")
    return df


def handle_class_imbalance(df, n_top):
    """
    Keeps the top N most frequent destinations
    and groups the rest into an 'OTHER' category.
    """
    print(f"📊 [2/8] Handling class imbalance (Top {n_top})...")
    top_n_destinations = df["h3_end"].value_counts().head(n_top).index.tolist()

    df["h3_end_processed"] = df["h3_end"].apply(
        lambda x: x if x in top_n_destinations else "OTHER"
    )
    n_classes = len(df["h3_end_processed"].unique())
    print(f"✓ Final number of classes: {n_classes}")
    return df, n_classes


def encode_features(df):
    """
    LabelEncodes categorical features (h3_start) and target (h3_end_processed).
    Saves encoders for inference.
    """
    print("🔑 [3/8] Encoding features and target...")

    start_encoder = LabelEncoder()
    target_encoder = LabelEncoder()

    df["h3_start_encoded"] = start_encoder.fit_transform(df["h3_start"])
    df["h3_end_encoded"] = target_encoder.fit_transform(df["h3_end_processed"])

    encoders = {"h3_start_encoder": start_encoder, "h3_end_encoder": target_encoder}
    joblib.dump(encoders, ENCODER_FILENAME)
    print(f"✓ Encoders saved to {ENCODER_FILENAME}")
    return df, encoders


def train():
    """
    Main training function with MLflow tracking.
    """
    df = load_data()
    df, num_classes = handle_class_imbalance(df, N_TOP_DESTINATIONS)
    df, encoders = encode_features(df)

    features = [
        "h3_start_encoded",
        "pickup_hour",
        "pickup_dayofweek",
        "is_weekend",
        "pickup_month",
        "pickup_day",
        "start_dist_from_center",
        "od_pair_historical_count",
        "origin_historical_count",
        "origin_to_dest_popularity",
    ]
    target = "h3_end_encoded"

    X = df[features]
    y = df[target]

    print("⏰ [4/8] Creating time-based split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TIME_BASED_TEST_SPLIT, shuffle=False
    )
    print(f"✓ Train set: {len(X_train)}, Test set: {len(X_test)}")

    with mlflow.start_run() as run:
        print(f"🚀 [5/8] Starting MLflow Run ID: {run.info.run_id}")

        params = {
            "objective": "multi:softmax",
            "num_class": num_classes,
            "max_depth": 10,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "n_estimators": 500,
            "early_stopping_rounds": 20,
            "tree_method": "hist",
            "enable_categorical": True,
        }

        mlflow.log_params(
            {
                "n_top_destinations": N_TOP_DESTINATIONS,
                "num_classes_final": num_classes,
                "time_split_test_size": TIME_BASED_TEST_SPLIT,
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "eta": params["eta"],
            }
        )

        print("🧠 [6/8] Training XGBoost model...")
        model = xgb.XGBClassifier(**params)

        X_train["h3_start_encoded"] = X_train["h3_start_encoded"].astype("category")
        X_test["h3_start_encoded"] = X_test["h3_start_encoded"].astype("category")

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

        print("📈 [7/8] Evaluating model and logging metrics...")

        y_pred_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        test_acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        top3_acc = top_k_accuracy_score(y_test, y_pred_proba, k=3)
        top5_acc = top_k_accuracy_score(y_test, y_pred_proba, k=5)

        print(f"\n--- Test Metrics ---")
        print(f"  Accuracy:       {test_acc:.4f}")
        print(f"  Top-3 Accuracy: {top3_acc:.4f}")
        print(f"  Top-5 Accuracy: {top5_acc:.4f}")
        print(f"  F1 (Weighted):  {f1_weighted:.4f}")

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_weighted", f1_weighted)
        mlflow.log_metric("test_top3_accuracy", top3_acc)
        mlflow.log_metric("test_top5_accuracy", top5_acc)

        print("💾 [8/8] Logging artifacts and registering model...")

        # --- Feature Importance Plot ---
        # (REMOVED to speed up the script)

        # Log the encoders
        mlflow.log_artifact(ENCODER_FILENAME)

        # --- Model Logging & Registration ---
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(5),
        )

        print(f"\n✅ Successfully completed run {run.info.run_id}")
        print(f"Model registered as '{REGISTERED_MODEL_NAME}'")


if __name__ == "__main__":
    train()
