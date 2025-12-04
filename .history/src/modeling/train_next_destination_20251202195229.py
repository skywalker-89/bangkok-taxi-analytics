import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from src.utils.db import get_engine

load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "bangkok_taxi_next_destination"
mlflow.set_experiment(EXPERIMENT_NAME)
REGISTERED_MODEL_NAME = "next_destination_model"

FEATURE_TABLE_NAME = "model_destination_features"

# We now save TWO encoders: one for inputs (start) and one for targets (end)
ENCODER_FEATURE_FILE = "dest_feature_encoder.pkl"  # For h3_start
ENCODER_TARGET_FILE = "dest_target_encoder.pkl"  # For h3_end (Target)


def load_data():
    print(f"📡 Fetching data from {FEATURE_TABLE_NAME}...")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {FEATURE_TABLE_NAME}", engine)
    return df


def train():
    df = load_data()

    # --- 1. Filter Rare Targets FIRST (Before Encoding) ---
    # This prevents gaps in the label sequence later.
    print("✂️ Filtering rare destinations (must have >1 sample)...")
    target_counts = df["h3_end"].value_counts()
    valid_targets = target_counts[target_counts > 1].index

    initial_len = len(df)
    df = df[df["h3_end"].isin(valid_targets)].copy()
    print(f"   Removed {initial_len - len(df)} rare samples.")

    # --- 2. Encode Features (h3_start) ---
    print("🔑 Encoding Start Locations (Features)...")
    # We fit this on ALL zones (start + end) to ensure we cover the whole map map for features
    # Note: It's okay if feature inputs have gaps/unused IDs.
    all_zones = pd.concat([df["h3_start"], df["h3_end"]]).unique()

    le_features = LabelEncoder()
    le_features.fit(all_zones)
    df["h3_start_idx"] = le_features.transform(df["h3_start"])

    # Save Feature Encoder
    joblib.dump(le_features, ENCODER_FEATURE_FILE)

    # --- 3. Encode Targets (h3_end) ---
    print("🔑 Encoding Destinations (Targets)...")
    # NOW we encode. Since we already filtered, this will produce
    # strict consecutive integers 0..N-1 with no gaps.
    le_target = LabelEncoder()
    df["h3_end_idx"] = le_target.fit_transform(df["h3_end"])

    # Save Target Encoder
    joblib.dump(le_target, ENCODER_TARGET_FILE)

    num_classes = len(le_target.classes_)
    print(f"ℹ️ Model will predict {num_classes} unique destination zones.")

    # --- 4. Select Features & Split ---
    feature_cols = [
        "h3_start_idx",
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
    target_col = "h3_end_idx"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 5. Train XGBoost Classifier ---
    print(f"🚀 Training Multi-Class Classifier...")

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,  # Explicitly tell XGB how many classes
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        tree_method="hist",
        random_state=42,
    )

    with mlflow.start_run():
        mlflow.set_tag("training_data", FEATURE_TABLE_NAME)
        mlflow.set_tag("model_type", "xgboost_classifier")

        clf.fit(X_train, y_train)

        # --- 6. Evaluate ---
        print("Evaluating...")
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Calculate Top-K
        all_labels = range(num_classes)
        top3 = top_k_accuracy_score(y_test, y_proba, k=3, labels=all_labels)
        top5 = top_k_accuracy_score(y_test, y_proba, k=5, labels=all_labels)

        print(f"  Accuracy: {acc:.4f}")
        print(f"  Top-3:    {top3:.4f}")
        print(f"  Top-5:    {top5:.4f}")

        mlflow.log_metrics(
            {
                "accuracy": acc,
                "f1_weighted": f1,
                "top3_accuracy": top3,
                "top5_accuracy": top5,
            }
        )

        # Log Both Encoders
        mlflow.log_artifact(ENCODER_FEATURE_FILE)
        mlflow.log_artifact(ENCODER_TARGET_FILE)

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
