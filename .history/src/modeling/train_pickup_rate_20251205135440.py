import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from src.utils.db import get_engine

# --- Config ---
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "bangkok_taxi_pickup_rate"
mlflow.set_experiment(EXPERIMENT_NAME)
REGISTERED_MODEL_NAME = "pickup_rate_model"

FEATURE_TABLE_NAME = "features_pickup_rate"
TARGET_COL = "is_high_demand"

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "pickup_rate_location_encoder.pkl")


def load_data():
    print(f"📡 Fetching data from {FEATURE_TABLE_NAME}...")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {FEATURE_TABLE_NAME}", engine)
    return df


def train():
    df = load_data()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Time-based split (ideal) or random split
    # For simplicity here, random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🚀 Training Classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="auc",
        use_label_encoder=False,
    )

    with mlflow.start_run():
        mlflow.set_tag("training_data", FEATURE_TABLE_NAME)
        mlflow.set_tag("model_type", "xgboost_classifier")

        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Test ROC AUC:  {auc:.4f}")

        mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})

        # Log Artifacts (Encoder)
        if os.path.exists(ENCODER_PATH):
            print("💾 Logging location encoder...")
            mlflow.log_artifact(ENCODER_PATH, artifact_path="encoders")
        else:
            print(f"⚠️ Warning: {ENCODER_PATH} not found.")

        # Log Model
        signature = infer_signature(X_test, y_pred)
        mlflow.xgboost.log_model(
            clf,
            "model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
        )

        print("🎉 Training Complete!")


if __name__ == "__main__":
    train()
