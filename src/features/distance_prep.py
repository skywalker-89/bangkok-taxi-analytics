import pandas as pd
import numpy as np
import h3
from sqlalchemy import create_engine
import sys
from src.utils.db import get_engine

# --- Database Connection ---
FEATURE_TABLE_NAME = "features_trip_distance"


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth
    using the Haversine formula.
    """
    R = 6371  # Earth radius in kilometers

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def prepare_distance_data():
    """
    Main function to load, process, and save
    the feature data for the distance model.
    """
    try:
        engine = get_engine()
        print(f"📚 Connecting to database...")
        # NOTE: This loads the ENTIRE table.
        # Your notebook had a LIMIT of 25M.
        # Add a LIMIT or WHERE clause if this is too large for your RAM.
        # Example: query = "SELECT * FROM taxi_probe_raw LIMIT 25000000"
        query = "SELECT * FROM taxi_probe_raw"

        # Use chunking for memory efficiency
        chunk_size = 5_000_000
        df_chunks = pd.read_sql(query, engine, chunksize=chunk_size)

        print("Starting data processing in chunks...")
        processed_trips = []

        for i, df in enumerate(df_chunks):
            print(f"--- Processing Chunk {i+1} ---")

            # 1. Convert timestamp and sort
            print("Sorting and converting timestamps...")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(by=["vehicle_id", "timestamp"])

            # 2. Identify trips
            print("Identifying trip IDs...")
            df["is_new_trip"] = (df["for_hire_light"] == 1) & (
                df["for_hire_light"].shift(1) == 0
            )
            df["trip_id"] = df["is_new_trip"].cumsum()

            # 3. Filter for 'for_hire' data only
            df = df[df["for_hire_light"] == 1].copy()
            if df.empty:
                print("No 'for_hire' data in this chunk.")
                continue

            # 4. Calculate point-to-point distance and time diff
            print("Calculating time/distance diffs...")
            df = df.sort_values(by=["vehicle_id", "trip_id", "timestamp"])
            df["time_diff"] = (
                df.groupby(["vehicle_id", "trip_id"])["timestamp"]
                .diff()
                .dt.total_seconds()
                .fillna(0)
            )

            df["prev_lat"] = df.groupby(["vehicle_id", "trip_id"])["latitude"].shift(1)
            df["prev_lon"] = df.groupby(["vehicle_id", "trip_id"])["longitude"].shift(1)

            # Calculate distance diff
            df["distance_diff_km"] = haversine(
                df["prev_lat"], df["prev_lon"], df["latitude"], df["longitude"]
            )
            # Fill first point of each trip with 0
            df["distance_diff_km"] = df["distance_diff_km"].fillna(0)

            # 5. Calculate idle time (within this chunk's data)
            print("Calculating idle time...")
            idle_df = df[(df["speed"] == 0) & (df["for_hire_light"] == 1)].copy()
            idle_time_per_trip = (
                idle_df.groupby(["vehicle_id", "trip_id"])["time_diff"]
                .sum()
                .reset_index()
            )
            idle_time_per_trip.rename(
                columns={"time_diff": "total_idle_seconds"}, inplace=True
            )

            # 6. Aggregate trips
            print("Aggregating trips...")
            trips_df = df.groupby(["vehicle_id", "trip_id"]).agg(
                start_lat=("latitude", "first"),
                start_lon=("longitude", "first"),
                end_lat=("latitude", "last"),
                end_lon=("longitude", "last"),
                total_trip_distance_km=("distance_diff_km", "sum"),
                pickup_time=("timestamp", "first"),
                dropoff_time=("timestamp", "last"),
                average_speed=("speed", "mean"),
            )

            # 7. Calculate duration and filter
            trips_df["duration_minutes"] = (
                trips_df["dropoff_time"] - trips_df["pickup_time"]
            ).dt.total_seconds() / 60
            trips_df = trips_df[
                (trips_df["duration_minutes"] > 1)
                & (trips_df["total_trip_distance_km"] > 0.1)
            ]

            # 8. Merge idle time
            trips_df = trips_df.reset_index()
            trips_df = pd.merge(
                trips_df, idle_time_per_trip, on=["vehicle_id", "trip_id"], how="left"
            )
            trips_df["total_idle_minutes"] = (
                trips_df["total_idle_seconds"].fillna(0) / 60
            )

            processed_trips.append(trips_df)

        print("\n--- Final Assembly ---")
        if not processed_trips:
            print("❌ No data processed. Exiting.")
            return

        # Combine all processed trip chunks
        final_df = pd.concat(processed_trips, ignore_index=True)
        print(f"Assembled {len(final_df)} trips total.")

        # --- Feature Engineering on Assembled Trips ---

        # 1. Time features
        print("Creating time features...")
        final_df["pickup_hour"] = final_df["pickup_time"].dt.hour
        final_df["pickup_dayofweek"] = final_df["pickup_time"].dt.dayofweek
        final_df["pickup_month"] = final_df["pickup_time"].dt.month
        final_df["pickup_day"] = final_df["pickup_time"].dt.day
        final_df["pickup_weekofyear"] = final_df["pickup_time"].dt.isocalendar().week

        # 2. H3 features
        print("Creating H3 features (Resolution 9)...")
        # Note: h3.latlng_to_cell is not vectorized, so we use .apply()
        h3_res = 9
        final_df["h3_start"] = final_df.apply(
            lambda row: h3.latlng_to_cell(row["start_lat"], row["start_lon"], h3_res),
            axis=1,
        )
        final_df["h3_end"] = final_df.apply(
            lambda row: h3.latlng_to_cell(row["end_lat"], row["end_lon"], h3_res),
            axis=1,
        )

        # 3. Create final X and y
        print("Creating final X and y DataFrames...")
        # (Renamed VehicleID to vehicle_id to match db)
        distance_features = [
            "vehicle_id",
            "trip_id",
            "duration_minutes",
            "pickup_hour",
            "pickup_dayofweek",
            "pickup_month",
            "pickup_day",
            "pickup_weekofyear",
            "average_speed",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "h3_start",
            "h3_end",
            "total_idle_minutes",
        ]

        # Ensure columns exist before selection
        final_df.rename(
            columns={"vehicle_id": "VehicleID"}, inplace=True
        )  # Match notebook's final CSV
        distance_features[0] = "VehicleID"  # Change back for selection

        # Select only features that are actually in the final_df
        available_features = [
            col for col in distance_features if col in final_df.columns
        ]
        missing_features = set(distance_features) - set(available_features)
        if missing_features:
            print(f"Warning: Missing expected features: {missing_features}")

        X = final_df[available_features].drop_duplicates(
            subset=["VehicleID", "trip_id"]
        )
        y = final_df[
            ["VehicleID", "trip_id", "total_trip_distance_km"]
        ].drop_duplicates(subset=["VehicleID", "trip_id"])

        # 4. Save to CSV
        # --- 🔍 DEBUGGING: Print what columns you actually have ---
        print(f"ℹ️ Actual columns in final_df: {list(final_df.columns)}")

        # --- 🛠️ FIX: Calculate Missing Features 🛠️ ---
        print("🛠️ Calculating missing features...")

        # 1. Ensure Timestamp is datetime
        if "pickup_time" in final_df.columns:
            final_df["pickup_time"] = pd.to_datetime(final_df["pickup_time"])
            final_df["pickup_hour"] = final_df["pickup_time"].dt.hour
            final_df["pickup_dayofweek"] = final_df["pickup_time"].dt.dayofweek
            final_df["is_weekend"] = (final_df["pickup_dayofweek"] >= 5).astype(int)

        # 2. Calculate Haversine Distance (if coordinates exist)
        if all(
            c in final_df.columns
            for c in ["start_lat", "start_lon", "end_lat", "end_lon"]
        ):
            final_df["haversine_distance"] = haversine(
                final_df["start_lat"],
                final_df["start_lon"],
                final_df["end_lat"],
                final_df["end_lon"],
            )

        # 3. Calculate Distance from Center
        # Default Bangkok Center if CITY_CENTER is missing
        center_lat, center_lon = (13.7563, 100.5018)
        if all(c in final_df.columns for c in ["start_lat", "start_lon"]):
            final_df["dist_from_center"] = haversine(
                final_df["start_lat"], final_df["start_lon"], center_lat, center_lon
            )

        # Define the columns we WANT
        desired_features = [
            "pickup_hour",
            "pickup_dayofweek",
            "is_weekend",
            "haversine_distance",
            "dist_from_center",
        ]
        target_col = "total_trip_distance_km"

        # Check if target exists (Critical)
        if target_col not in final_df.columns:
            print(f"❌ CRITICAL ERROR: Target column '{target_col}' is missing!")
            print("Check your feature engineering logic.")
            sys.exit(1)

        # Filter features: Only keep ones that exist
        valid_features = [c for c in desired_features if c in final_df.columns]
        missing_features = set(desired_features) - set(valid_features)

        if missing_features:
            print(
                f"⚠️ WARNING: The following columns are missing and will be skipped: {missing_features}"
            )

        # Combine valid features + target
        columns_to_save = valid_features + [target_col]

        print(
            f"💾 Saving {len(columns_to_save)} columns to SQL Table: {FEATURE_TABLE_NAME}..."
        )

        df_to_save = final_df[columns_to_save].copy()

        df_to_save.to_sql(
            FEATURE_TABLE_NAME,
            engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=5000,
        )

        print(f"✅ Success! Data stored in table '{FEATURE_TABLE_NAME}'")

    except Exception as e:
        print(f"❌ Error: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment to see full error details
        sys.exit(1)


if __name__ == "__main__":
    prepare_distance_data()
