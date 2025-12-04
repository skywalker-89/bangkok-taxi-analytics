import pandas as pd
import numpy as np
import h3
import sys
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from src.utils.db import get_engine

# --- Configuration ---
FEATURE_TABLE_NAME = "features_trip_duration"
CITY_CENTER = (13.7563, 100.5018)
H3_RESOLUTION = 8

# Artifacts (Encoders needed for Training/Inference)
START_ENCODER_PATH = "duration_start_zone_encoder.pkl"
END_ENCODER_PATH = "duration_end_zone_encoder.pkl"


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points on Earth."""
    R = 6371
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def prepare_duration_data():
    try:
        engine = get_engine()
        print(f"📚 [1/6] Connecting to database...")

        # We read ALL raw data, but in chunks to respect memory
        query = """
        SELECT vehicle_id, timestamp, latitude, longitude, for_hire_light, speed 
        FROM taxi_probe_raw 
        ORDER BY vehicle_id, timestamp
        """

        # Lower chunk size for safety (500k is usually safe for laptops)
        chunk_size = 500_000
        df_chunks = pd.read_sql(query, engine, chunksize=chunk_size)

        print("Processing raw GPS data in chunks to reconstruct trips...")
        processed_trips = []

        for i, df in enumerate(df_chunks):
            print(f"  --- Processing Chunk {i+1} ---")

            # 1. Convert timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # 2. Identify Trip Boundaries (0 -> 1 transitions)
            # Logic: If for_hire_light goes from 0 (Occupied) -> 1 (Free), a trip just ended?
            # Actually, usually 1=Free (For Hire), 0=Occupied (Hired).
            # Trip starts when light goes 1 -> 0. Trip ends when 0 -> 1.
            # Your logic: df["for_hire_light"] == 1 (Free?).
            # NOTE: I will preserve YOUR logic from the snippet, assuming 1=Hired in your specific dataset context.
            # If 1=Free, this logic extracts "Free Roaming". If 1=Hired, it extracts "Passenger Trips".

            df["prev_light"] = df.groupby("vehicle_id")["for_hire_light"].shift(1)
            df["is_new_trip"] = (df["for_hire_light"] == 1) & (df["prev_light"] == 0)
            df["trip_id"] = df.groupby("vehicle_id")["is_new_trip"].cumsum()

            # 3. Filter: Keep only the "Hired" (or whatever 1 represents) frames
            df_trips = df[df["for_hire_light"] == 1].copy()
            if df_trips.empty:
                continue

            # 4. Aggregate Points into Trips
            # This compresses 1000 GPS points into 1 row per trip
            # We explicitly handle the aggregations
            trips_agg = (
                df_trips.groupby(["vehicle_id", "trip_id"])
                .agg(
                    start_lat=("latitude", "first"),
                    start_lon=("longitude", "first"),
                    end_lat=("latitude", "last"),
                    end_lon=("longitude", "last"),
                    pickup_time=("timestamp", "first"),
                    dropoff_time=("timestamp", "last"),
                    average_speed=("speed", "mean"),
                    max_speed=("speed", "max"),
                )
                .reset_index()
            )

            # 5. Filter Invalid Trips (Zero duration or stationary)
            trips_agg["duration_minutes"] = (
                trips_agg["dropoff_time"] - trips_agg["pickup_time"]
            ).dt.total_seconds() / 60

            # Valid trip rules: > 1 min duration, coordinates actually changed
            trips_agg = trips_agg[trips_agg["duration_minutes"] > 1]
            trips_agg["dist_check"] = haversine(
                trips_agg["start_lat"],
                trips_agg["start_lon"],
                trips_agg["end_lat"],
                trips_agg["end_lon"],
            )
            trips_agg = trips_agg[trips_agg["dist_check"] > 0.1]

            processed_trips.append(trips_agg)

        print("\n--- Final Assembly ---")
        if not processed_trips:
            print("❌ No valid trips found. Check your 'for_hire_light' logic.")
            return

        final_df = pd.concat(processed_trips, ignore_index=True)
        print(f"✅ Assembled {len(final_df)} unique trips.")

        # --- 7. Feature Engineering ---
        print("🛠️ [2/6] Engineering time features...")
        final_df["pickup_hour"] = final_df["pickup_time"].dt.hour
        final_df["pickup_dayofweek"] = final_df["pickup_time"].dt.dayofweek
        final_df["is_weekend"] = (final_df["pickup_dayofweek"] >= 5).astype(int)

        print(f"🗺️ [3/6] Engineering H3 (Res {H3_RESOLUTION}) features...")
        # Use apply for simplicity, or vectorized h3 if available
        final_df["start_h3_zone"] = final_df.apply(
            lambda r: h3.latlng_to_cell(r["start_lat"], r["start_lon"], H3_RESOLUTION),
            axis=1,
        )
        final_df["end_h3_zone"] = final_df.apply(
            lambda r: h3.latlng_to_cell(r["end_lat"], r["end_lon"], H3_RESOLUTION),
            axis=1,
        )

        print("📍 [4/6] Engineering distance features...")
        final_df["haversine_distance"] = haversine(
            final_df["start_lat"],
            final_df["start_lon"],
            final_df["end_lat"],
            final_df["end_lon"],
        )
        final_df["dist_from_center"] = haversine(
            final_df["start_lat"], final_df["start_lon"], CITY_CENTER[0], CITY_CENTER[1]
        )

        # --- 8. Label Encoding & Artifact Saving ---
        print("🔑 [5/6] Label encoding H3 zones...")
        le_start = LabelEncoder()
        le_end = LabelEncoder()

        # Fit encoders on the strings (e.g., '883...'), convert to int (0, 1, 2...)
        final_df["start_h3_zone_idx"] = le_start.fit_transform(
            final_df["start_h3_zone"]
        )
        final_df["end_h3_zone_idx"] = le_end.fit_transform(final_df["end_h3_zone"])

        # Save artifacts so training/inference can understand these integers
        joblib.dump(le_start, START_ENCODER_PATH)
        joblib.dump(le_end, END_ENCODER_PATH)
        print(f"✓ Encoders saved to: {START_ENCODER_PATH}, {END_ENCODER_PATH}")

        # --- 9. Save to SQL ---
        print(f"💾 [6/6] Saving features to SQL Table: {FEATURE_TABLE_NAME}...")

        # Define exact schema for the model
        columns_to_save = [
            "start_h3_zone_idx",
            "end_h3_zone_idx",
            "pickup_hour",
            "pickup_dayofweek",
            "is_weekend",
            "haversine_distance",
            "dist_from_center",
            "average_speed",
            "max_speed",
            "duration_minutes",  # TARGET
        ]

        # Verify columns exist
        missing = [c for c in columns_to_save if c not in final_df.columns]
        if missing:
            print(f"❌ Critical Error: Missing columns {missing}")
            sys.exit(1)

        # Write to DB
        final_df[columns_to_save].to_sql(
            FEATURE_TABLE_NAME,
            engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=5000,
        )

        print(f"✅ Data preparation complete! Saved {len(final_df)} rows to DB.")

    except ImportError:
        print("❌ Error: 'h3' library not found. pip install h3")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    prepare_duration_data()
