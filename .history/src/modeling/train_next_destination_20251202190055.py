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
ENCODER_FILENAME = "destination_label_encoders.pkl"


def load_data():
    print(f"📡 Fetching data from {FEATURE_TABLE_NAME}...")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {FEATURE_TABLE_NAME}", engine)

    # We need to encode 'h3_start' (Feature) and 'h3_end' (Target)
    # They are currently strings. XGBoost needs numeric input.
    return df


def train():
    df = load_data()

    # 1. Label Encode Zones
    print("🔑 Encoding H3 Zones...")
    le_h3 = LabelEncoder()

    # Fit on BOTH start and end to ensure we cover the whole map
    all_zones = pd.concat([df["h3_start"], df["h3_end"]]).unique()
    le_h3.fit(all_zones)

    df["h3_start_idx"] = le_h3.transform(df["h3_start"])
    df["h3_end_idx"] = le_h3.transform(df["h3_end"])

    # Save Encoder (Crucial for decoding predictions back to H3)
    joblib.dump(le_h3, ENCODER_FILENAME)

    # 2. Select Features
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

    # 3. Stratified Split (Important for classification)
    # Filter out classes with only 1 sample to avoid split error
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts > 1].index
    mask = y.isin(valid_classes)

    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train XGBoost Classifier
    # Limit output classes to prevent explosion? XGBoost handles multi-class but it can be slow if 1000s of classes.
    num_classes = len(le_h3.classes_)
    print(f"🚀 Training Multi-Class Classifier on {num_classes} Zones...")

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        tree_method="hist",  # Faster for large datasets
        random_state=42,
    )

    with mlflow.start_run():
        mlflow.set_tag("training_data", FEATURE_TABLE_NAME)
        mlflow.set_tag("model_type", "xgboost_classifier")

        clf.fit(X_train, y_train)

        # 5. Evaluate
        print("Evaluating...")
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Top-K Accuracy
        # (Handling case where we might have fewer classes in test than total)
        top3 = top_k_accuracy_score(y_test, y_proba, k=3, labels=range(num_classes))
        top5 = top_k_accuracy_score(y_test, y_proba, k=5, labels=range(num_classes))

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

        # Log Artifacts
        mlflow.log_artifact(ENCODER_FILENAME)

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
