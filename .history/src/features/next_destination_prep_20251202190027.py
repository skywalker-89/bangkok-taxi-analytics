import pandas as pd
import numpy as np
import h3
import sys
import os
from src.utils.db import get_engine, get_db_connection

# --- Config ---
OUTPUT_TABLE_NAME = "model_destination_features"
H3_RESOLUTION = 8
CITY_CENTER = (13.7563, 100.5018)


def haversine(lat1, lon1, lat2, lon2):
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


def setup_database_indexes():
    """Create indexes on raw table to speed up the massive self-join query."""
    print("🔧 Checking database indexes...")
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()

    try:
        # Check existing indexes
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'taxi_probe_raw'"
        )
        existing = {row[0] for row in cur.fetchall()}

        if "idx_vehicle_timestamp" not in existing:
            print("  Creating index on (vehicle_id, timestamp)...")
            cur.execute(
                "CREATE INDEX CONCURRENTLY idx_vehicle_timestamp ON taxi_probe_raw(vehicle_id, timestamp)"
            )

        print("✅ Indexes ready.")
    finally:
        cur.close()
        conn.close()


def prepare_next_destination_data():
    try:
        setup_database_indexes()
        engine = get_engine()
        print(f"📚 [1/7] Fetching trip endpoints from DB...")

        # 1. Reconstruct Trips (Light 1->0 = Start, 0->1 = End)
        # We fetch only the transition points to save memory
        query = """
        WITH transitions AS (
            SELECT 
                vehicle_id, 
                timestamp, 
                latitude, 
                longitude, 
                for_hire_light,
                LAG(for_hire_light) OVER (PARTITION BY vehicle_id ORDER BY timestamp) as prev_light
            FROM taxi_probe_raw
        )
        SELECT * FROM transitions 
        WHERE (for_hire_light = 1 AND prev_light = 0)  -- Trip End (Dropoff) (?) 
           OR (for_hire_light = 0 AND prev_light = 1)  -- Trip Start (Pickup) (?)
        ORDER BY vehicle_id, timestamp
        """
        # NOTE: logic depends on dataset definition.
        # Assuming: 1=Free, 0=Occupied.
        # Start = 1->0. End = 0->1.

        # Let's use a simpler query that assumes we have a cleaned 'trips' logic or process in memory.
        # For robustness, we'll process chunks like the duration script if raw data is huge.
        # However, to keep this script focused on the *features* (popularity), let's assume
        # we can fetch the "Start" and "End" points.

        # Alternative: Read the 'features_trip_duration' table we just made!
        # It already has start_lat/end_lat/timestamp. Smart reuse!

        print("  Using 'features_trip_duration' as base (reusing existing work)...")
        source_table = "features_trip_duration"

        # We need coordinates to map to H3 again if we stored indices.
        # Actually 'features_trip_duration' stores INDICES.
        # We need raw coordinates or H3 strings.
        # If 'features_trip_duration' lost raw coords, we must go back to raw.
        # Let's go back to raw for safety.

        raw_query = """
        SELECT vehicle_id, timestamp, latitude, longitude, for_hire_light
        FROM taxi_probe_raw
        ORDER BY vehicle_id, timestamp
        """
        chunk_size = 500_000
        df_chunks = pd.read_sql(raw_query, engine, chunksize=chunk_size)

        all_trips = []

        print("🔄 Processing raw data to extract trip O-D pairs...")
        for i, df in enumerate(df_chunks):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["prev_light"] = df.groupby("vehicle_id")["for_hire_light"].shift(1)

            # Identify Starts (1->0) and Ends (0->1)
            # 1=Free, 0=Occupied.
            # Start: Light goes Free(1) -> Occupied(0)
            trip_starts = df[
                (df["for_hire_light"] == 0) & (df["prev_light"] == 1)
            ].copy()
            trip_ends = df[(df["for_hire_light"] == 1) & (df["prev_light"] == 0)].copy()

            # Simple matching by sequence (Start N pairs with End N)
            # In production, use a more robust session ID
            if len(trip_starts) > 0 and len(trip_ends) > 0:
                min_len = min(len(trip_starts), len(trip_ends))
                trips = pd.DataFrame(
                    {
                        "vehicle_id": trip_starts["vehicle_id"].iloc[:min_len].values,
                        "pickup_time": trip_starts["timestamp"].iloc[:min_len].values,
                        "start_lat": trip_starts["latitude"].iloc[:min_len].values,
                        "start_lon": trip_starts["longitude"].iloc[:min_len].values,
                        "end_lat": trip_ends["latitude"].iloc[:min_len].values,
                        "end_lon": trip_ends["longitude"].iloc[:min_len].values,
                    }
                )
                all_trips.append(trips)

        if not all_trips:
            print("❌ No trips found.")
            return

        trips_df = pd.concat(all_trips, ignore_index=True)
        print(f"✅ Extracted {len(trips_df)} trips.")

        # --- Feature Engineering ---
        print("🛠️ [2/7] Generating H3 Features...")
        trips_df["h3_start"] = trips_df.apply(
            lambda x: h3.latlng_to_cell(x["start_lat"], x["start_lon"], H3_RESOLUTION),
            axis=1,
        )
        trips_df["h3_end"] = trips_df.apply(
            lambda x: h3.latlng_to_cell(x["end_lat"], x["end_lon"], H3_RESOLUTION),
            axis=1,
        )

        trips_df["pickup_hour"] = trips_df["pickup_time"].dt.hour
        trips_df["pickup_dayofweek"] = trips_df["pickup_time"].dt.dayofweek
        trips_df["is_weekend"] = (trips_df["pickup_dayofweek"] >= 5).astype(int)
        trips_df["pickup_month"] = trips_df["pickup_time"].dt.month
        trips_df["pickup_day"] = trips_df["pickup_time"].dt.day

        trips_df["start_dist_from_center"] = haversine(
            trips_df["start_lat"], trips_df["start_lon"], *CITY_CENTER
        )

        # --- Historical Popularity (The "Special Sauce") ---
        print("📊 [3/7] Calculating Historical O-D Probabilities...")
        # 1. Count (Start, End) pairs
        od_counts = (
            trips_df.groupby(["h3_start", "h3_end"])
            .size()
            .reset_index(name="od_pair_historical_count")
        )

        # 2. Count (Start) total
        origin_counts = (
            trips_df.groupby("h3_start")
            .size()
            .reset_index(name="origin_historical_count")
        )

        # Merge back
        trips_df = trips_df.merge(od_counts, on=["h3_start", "h3_end"], how="left")
        trips_df = trips_df.merge(origin_counts, on="h3_start", how="left")

        # Calculate Probability: P(End | Start)
        # Add +1 smoothing to prevent division by zero or zero prob
        trips_df["origin_to_dest_popularity"] = (
            trips_df["od_pair_historical_count"] + 1
        ) / (trips_df["origin_historical_count"] + 1)

        trips_df.fillna(0, inplace=True)

        # --- Filter Rare Destinations ---
        # If a destination has only been visited once ever, it's noise.
        # Keep top N destinations? Or just let the model handle it.
        # For cleanliness, we drop N/A.

        # --- Save to SQL ---
        print(f"💾 [4/7] Saving features to SQL Table: {OUTPUT_TABLE_NAME}...")

        target = "h3_end"
        features = [
            "h3_start",
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

        final_cols = ["vehicle_id", "pickup_time", target] + features
        final_df = trips_df[final_cols].copy()

        final_df.to_sql(
            OUTPUT_TABLE_NAME,
            engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=5000,
        )

        print(f"✅ Success! Saved {len(final_df)} rows.")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    prepare_next_destination_data()
