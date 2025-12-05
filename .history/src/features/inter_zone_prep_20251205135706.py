import pandas as pd
import numpy as np
import h3
import sys
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from src.utils.db import get_engine

# --- Configuration ---
FEATURE_TABLE_NAME = "features_inter_zone"
CITY_CENTER = (13.7563, 100.5018)
H3_RESOLUTION = 8

# Artifact paths (Encoders needed for Training/Inference)
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
START_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "inter_zone_start_encoder.pkl")
END_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "inter_zone_end_encoder.pkl")


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


def bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two points."""
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - (
        np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    )
    initial_bearing = np.arctan2(x, y)
    return initial_bearing


def prepare_inter_zone_data():
    try:
        engine = get_engine()
        print(f"📚 [1/7] Connecting to database...")

        # 1. Fetch Raw Data to reconstruct trips
        query = """
        SELECT vehicle_id, timestamp, latitude, longitude, for_hire_light 
        FROM taxi_probe_raw 
        ORDER BY vehicle_id, timestamp
        """
        # Read in chunks to prevent OOM
        chunk_size = 500_000
        df_chunks = pd.read_sql(query, engine, chunksize=chunk_size)

        print("🔄 Processing chunks to find Empty Trips (Dropoff -> Next Pickup)...")
        all_empty_trips = []

        for i, df in enumerate(df_chunks):
            # Convert time
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Logic: An "Empty Trip" is the movement between a Dropoff (End of Trip A)
            # and a Pickup (Start of Trip B).
            # We identify "Passenger Trips" first, then shift to find the gap.

            # Identify transitions
            df["prev_light"] = df.groupby("vehicle_id")["for_hire_light"].shift(1)

            # Start of Passenger Trip: Light 1 -> 0 (or whatever logic 1=Free/Hired implies)
            # Assuming logic: 1=Hired, 0=Free (based on previous files)
            df["is_trip_start"] = (df["for_hire_light"] == 1) & (df["prev_light"] == 0)
            df["is_trip_end"] = (df["for_hire_light"] == 0) & (df["prev_light"] == 1)

            # Extract Start Points and End Points
            trip_starts = (
                df[df["is_trip_start"]]
                .copy()
                .rename(
                    columns={
                        "latitude": "next_pickup_lat",
                        "longitude": "next_pickup_lon",
                        "timestamp": "next_pickup_time",
                    }
                )
            )
            trip_ends = (
                df[df["is_trip_end"]]
                .copy()
                .rename(
                    columns={
                        "latitude": "dropoff_lat",
                        "longitude": "dropoff_lon",
                        "timestamp": "dropoff_time",
                    }
                )
            )

            # Merge to find the "Gap"
            # We align the 'dropoff' of Trip N with 'pickup' of Trip N+1
            # Note: In a chunked approach, this might lose connections at chunk boundaries.
            # A pro production system handles this with window functions in SQL.
            # For this script, we assume sufficient data density per chunk.

            trip_ends["join_key"] = range(len(trip_ends))
            trip_starts["join_key"] = range(len(trip_starts))

            # Shift starts back by 1 to align: End(Trip 1) -> Start(Trip 2)
            trip_starts["join_key"] -= 1

            empty_trips = pd.merge(
                trip_ends, trip_starts, on=["vehicle_id", "join_key"], how="inner"
            )

            # Filter valid Empty Trips
            empty_trips["travel_time_minutes"] = (
                empty_trips["next_pickup_time"] - empty_trips["dropoff_time"]
            ).dt.total_seconds() / 60
            empty_trips = empty_trips[
                (empty_trips["travel_time_minutes"] > 0)
                & (empty_trips["travel_time_minutes"] < 120)
            ]  # reasonable limits

            all_empty_trips.append(empty_trips)

        print("🔗 Assembling empty trips...")
        if not all_empty_trips:
            print("❌ No empty trips found. Exiting.")
            return

        final_df = pd.concat(all_empty_trips, ignore_index=True)
        print(f"✅ Found {len(final_df)} empty trips (Inter-Zone movement).")

        # --- Feature Engineering ---
        print("🛠️ [2/7] Calculating Geo Features...")

        # Coordinates
        final_df["start_lat"] = final_df["dropoff_lat"]
        final_df["start_lon"] = final_df["dropoff_lon"]
        final_df["end_lat"] = final_df["next_pickup_lat"]
        final_df["end_lon"] = final_df["next_pickup_lon"]

        # Distance & Bearing
        final_df["direct_distance_km"] = haversine(
            final_df["start_lat"],
            final_df["start_lon"],
            final_df["end_lat"],
            final_df["end_lon"],
        )
        b = bearing(
            final_df["start_lat"],
            final_df["start_lon"],
            final_df["end_lat"],
            final_df["end_lon"],
        )
        final_df["bearing_sin"] = np.sin(b)
        final_df["bearing_cos"] = np.cos(b)

        # Center Distance
        final_df["start_dist_from_center"] = haversine(
            final_df["start_lat"], final_df["start_lon"], *CITY_CENTER
        )
        final_df["end_dist_from_center"] = haversine(
            final_df["end_lat"], final_df["end_lon"], *CITY_CENTER
        )

        # Time Features
        final_df["start_hour"] = final_df["dropoff_time"].dt.hour
        final_df["start_dayofweek"] = final_df["dropoff_time"].dt.dayofweek
        final_df["is_weekend"] = (final_df["start_dayofweek"] >= 5).astype(int)

        # H3 Zones
        print(f"🗺️ [3/7] Mapping H3 Zones (Res {H3_RESOLUTION})...")
        final_df["start_h3"] = final_df.apply(
            lambda x: h3.latlng_to_cell(x["start_lat"], x["start_lon"], H3_RESOLUTION),
            axis=1,
        )
        final_df["end_h3"] = final_df.apply(
            lambda x: h3.latlng_to_cell(x["end_lat"], x["end_lon"], H3_RESOLUTION),
            axis=1,
        )

        # --- Aggregate Statistics (Popularity) ---
        print("📊 [4/7] Calculating Zone Popularity stats...")
        # 1. Zone Pair Count
        pair_counts = (
            final_df.groupby(["start_h3", "end_h3"])
            .size()
            .reset_index(name="zone_pair_count")
        )
        final_df = final_df.merge(pair_counts, on=["start_h3", "end_h3"], how="left")

        # 2. Origin Count
        origin_counts = (
            final_df.groupby("start_h3").size().reset_index(name="origin_zone_count")
        )
        final_df = final_df.merge(origin_counts, on="start_h3", how="left")

        # 3. Ratio
        final_df["origin_to_dest_popularity"] = final_df["zone_pair_count"] / (
            final_df["origin_zone_count"] + 1
        )

        # 4. Avg Travel Time from Zone
        avg_time = (
            final_df.groupby("start_h3")["travel_time_minutes"]
            .mean()
            .reset_index(name="avg_travel_time_from_zone")
        )
        final_df = final_df.merge(avg_time, on="start_h3", how="left")

        # 5. Avg Dist from Zone
        avg_dist = (
            final_df.groupby("start_h3")["direct_distance_km"]
            .mean()
            .reset_index(name="avg_dist_from_zone")
        )
        final_df = final_df.merge(avg_dist, on="start_h3", how="left")

        # Fill NaNs from merges
        final_df.fillna(0, inplace=True)

        # --- Encoding ---
        print("🔑 [5/7] Label Encoding Zones...")
        le_start = LabelEncoder()
        le_end = LabelEncoder()

        final_df["start_h3_idx"] = le_start.fit_transform(final_df["start_h3"])
        final_df["end_h3_idx"] = le_end.fit_transform(final_df["end_h3"])

        # Save Artifacts
        joblib.dump(le_start, START_ENCODER_PATH)
        joblib.dump(le_end, END_ENCODER_PATH)
        print(f"✓ Encoders saved to {START_ENCODER_PATH}, {END_ENCODER_PATH}")

        # --- Save to SQL ---
        print(f"💾 [6/7] Saving features to SQL Table: {FEATURE_TABLE_NAME}...")

        features_to_save = [
            "start_h3_idx",
            "end_h3_idx",
            "direct_distance_km",
            "bearing_sin",
            "bearing_cos",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "start_hour",
            "start_dayofweek",
            "is_weekend",
            "start_dist_from_center",
            "end_dist_from_center",
            "zone_pair_count",
            "origin_zone_count",
            "origin_to_dest_popularity",
            "avg_travel_time_from_zone",
            "avg_dist_from_zone",
            "travel_time_minutes",  # TARGET
        ]

        # Verify columns exist
        final_df[features_to_save].to_sql(
            FEATURE_TABLE_NAME,
            engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=5000,
        )

        print(f"✅ Data preparation complete! {len(final_df)} rows saved.")

    except Exception as e:
        print(f"❌ Error: {e}")
        # import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    prepare_inter_zone_data()
