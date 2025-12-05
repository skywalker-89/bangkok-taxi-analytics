import pandas as pd
import numpy as np
import h3
import sys
import gc
import joblib
import os
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from src.utils.db import get_engine, get_db_connection

# --- Configuration ---
FEATURE_TABLE_NAME = "features_pickup_rate"
H3_RESOLUTION = 8
TIME_BIN_MINUTES = 60  # 1-hour bins for rate prediction
WINDOW_SIZE_HOURS = 24  # Look back 24 hours
FORECAST_HORIZON_HOURS = 1  # Predict next 1 hour

# Encoder path
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "pickup_rate_location_encoder.pkl")


def setup_database_indexes():
    """Optimizes the raw table for time-series queries."""
    print("🔧 Checking database indexes...")
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    try:
        # We need a composite index on (timestamp, latitude, longitude) or similar for speed
        # But (vehicle_id, timestamp) is standard.
        # Let's ensure a timestamp index exists for fast range queries.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp_raw ON taxi_probe_raw(timestamp);"
        )
        print("✅ Database indexes ready.")
    finally:
        cur.close()
        conn.close()


def prepare_pickup_rate_data():
    try:
        setup_database_indexes()
        engine = get_engine()
        print(f"📚 [1/6] Reading raw data...")

        # 1. Fetch Data (Only 'For Hire' transitions = Pickups)
        # We need to find points where light goes 1 -> 0 (Pickup)
        query = """
        SELECT timestamp, latitude, longitude, for_hire_light, vehicle_id
        FROM taxi_probe_raw
        ORDER BY vehicle_id, timestamp
        """
        # Load in chunks if needed, but for "pickups only" the dataset is smaller.
        # Let's assume we can load the raw logic or process chunks.
        # For simplicity/robustness on M2, let's load specific columns.

        # NOTE: Calculating "Pickups" from raw light status is complex in SQL.
        # We will load raw frames and process in pandas for flexibility.
        df_iter = pd.read_sql(query, engine, chunksize=1_000_000)

        pickups_list = []
        print("🔄 Processing chunks to identify pickup events...")

        for chunk in df_iter:
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
            chunk["prev_light"] = chunk.groupby("vehicle_id")["for_hire_light"].shift(1)

            # Pickup = Light goes Free(1) -> Occupied(0)
            is_pickup = (chunk["for_hire_light"] == 0) & (chunk["prev_light"] == 1)
            pickups = chunk[is_pickup].copy()

            if not pickups.empty:
                pickups_list.append(pickups[["timestamp", "latitude", "longitude"]])

        if not pickups_list:
            print("❌ No pickups found.")
            return

        df_pickups = pd.concat(pickups_list)
        print(f"✅ Found {len(df_pickups)} total pickup events.")

        # --- Feature Engineering ---
        print("🛠️ [2/6] Binning Data (Spatial & Temporal)...")

        # 1. Spatial Binning (H3)
        df_pickups["h3_cell"] = df_pickups.apply(
            lambda x: h3.latlng_to_cell(x["latitude"], x["longitude"], H3_RESOLUTION),
            axis=1,
        )

        # 2. Temporal Binning
        # Round down to nearest hour
        df_pickups["time_bin"] = df_pickups["timestamp"].dt.floor(
            f"{TIME_BIN_MINUTES}min"
        )

        # 3. Aggregation: Count pickups per (Cell, Hour)
        df_counts = (
            df_pickups.groupby(["time_bin", "h3_cell"])
            .size()
            .reset_index(name="pickup_count")
        )

        # Fill missing hours/cells?
        # In a pro system, we create a full grid (Cartesian Product of All Time x All Cells)
        # filling 0s. For this script, we'll stick to sparse data (rows exist only if pickups > 0)
        # to save memory, but this is a model decision.

        # --- Lag Features (Sliding Window) ---
        print("📊 [3/6] Generating Lag Features...")
        df_counts = df_counts.sort_values(["h3_cell", "time_bin"])

        # Lag 1: Count 1 hour ago
        df_counts["lag_1h"] = df_counts.groupby("h3_cell")["pickup_count"].shift(1)
        # Lag 24: Count 24 hours ago (Same time yesterday)
        df_counts["lag_24h"] = df_counts.groupby("h3_cell")["pickup_count"].shift(24)

        # Rolling Mean (past 3 hours)
        df_counts["rolling_mean_3h"] = df_counts.groupby("h3_cell")[
            "pickup_count"
        ].transform(lambda x: x.rolling(window=3).mean())

        df_counts.dropna(inplace=True)  # Drop rows with NaN lags

        # --- Time Features ---
        df_counts["hour"] = df_counts["time_bin"].dt.hour
        df_counts["dayofweek"] = df_counts["time_bin"].dt.dayofweek
        df_counts["is_weekend"] = (df_counts["dayofweek"] >= 5).astype(int)

        # --- Encoding ---
        print("🔑 [4/6] Encoding H3 Locations...")
        le = LabelEncoder()
        df_counts["location_id"] = le.fit_transform(df_counts["h3_cell"])

        joblib.dump(le, ENCODER_PATH)
        print(f"✓ Encoder saved to {ENCODER_PATH}")

        # --- Save to SQL ---
        print(f"💾 [5/6] Saving features to SQL Table: {FEATURE_TABLE_NAME}...")

        # Define Target: "High Demand" (Binary Classification)
        # Example: High Demand if > 75th percentile of pickups?
        # Or simply predict the raw count (Regression).
        # Your training script uses Classifier, so let's bin the target.
        threshold = df_counts["pickup_count"].median()
        df_counts["is_high_demand"] = (df_counts["pickup_count"] > threshold).astype(
            int
        )

        features_to_save = [
            "location_id",
            "hour",
            "dayofweek",
            "is_weekend",
            "lag_1h",
            "lag_24h",
            "rolling_mean_3h",
            "is_high_demand",  # TARGET
        ]

        # Write to DB
        df_counts[features_to_save].to_sql(
            FEATURE_TABLE_NAME,
            engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=5000,
        )

        print(f"✅ Data preparation complete. Saved {len(df_counts)} rows.")

    except Exception as e:
        print(f"❌ Error: {e}")
        # import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    prepare_pickup_rate_data()
