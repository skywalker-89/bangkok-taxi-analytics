import pandas as pd
import numpy as np
import h3
from sqlalchemy import create_engine
import psycopg2
import sys

# --- Config ---
from src.utils.db import get_engine

H3_RESOLUTION = 8
OUTPUT_TABLE_NAME = "model_destination_features"
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


def latlng_to_h3_vectorized(lats, lngs, resolution):
    """Vectorized H3 conversion"""
    return [h3.latlng_to_cell(lat, lng, resolution) for lat, lng in zip(lats, lngs)]


def setup_database_indexes():
    """Create indexes for faster queries."""
    print("🔧 Checking database indexes...")

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="bangkok_taxi_db",
        user="postgres",
        password="mypassword",
    )
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'taxi_probe_raw' 
            AND indexname = 'idx_vehicle_timestamp'
        """
        )

        if not cur.fetchone():
            print("  Creating index (this may take 5-10 minutes)...")
            cur.execute(
                """
                CREATE INDEX CONCURRENTLY idx_vehicle_timestamp 
                ON taxi_probe_raw(vehicle_id, timestamp)
            """
            )
            print("  ✓ Index created")
        else:
            print("  ✓ Index already exists")

    except Exception as e:
        print(f"  ⚠️ Index setup: {e}")
    finally:
        cur.close()
        conn.close()


def create_feature_table():
    """
    Loads raw data in chunks, segments into trips, and engineers features.
    """
    try:
        setup_database_indexes()

        engine = get_engine()
        print("📚 [1/7] Loading probe data in chunks...")

        # --- FIX ---
        # REMOVED "WHERE for_hire_light = 1" to load ALL data
        # This is required for the segmentation logic to work
        query = """
            SELECT vehicle_id, latitude, longitude, timestamp, for_hire_light 
            FROM taxi_probe_raw 
            ORDER BY vehicle_id, timestamp
        """

        chunk_size = 5_000_000
        df_chunks = pd.read_sql(query, engine, chunksize=chunk_size)

        all_trips = []

        for i, df in enumerate(df_chunks):
            print(f"  Processing chunk {i+1}...")

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(by=["vehicle_id", "timestamp"])

            # Segment trips: Find the 0 -> 1 transition
            df["is_new_trip"] = (df["for_hire_light"] == 1) & (
                df["for_hire_light"].shift(1) == 0
            )
            df["trip_id"] = df["is_new_trip"].cumsum()

            # --- FIX ---
            # Now, filter for *only* rows that are part of a trip
            df_in_trip = df[df["for_hire_light"] == 1].copy()

            if df_in_trip.empty:
                print("  No 'for_hire_light == 1' rows in this chunk, skipping.")
                continue

            # Aggregate trips
            trips = df_in_trip.groupby(["vehicle_id", "trip_id"]).agg(
                pickup_time=("timestamp", "first"),
                dropoff_time=("timestamp", "last"),
                start_lat=("latitude", "first"),
                start_lon=("longitude", "first"),
                end_lat=("latitude", "last"),
                end_lon=("longitude", "last"),
            )

            # Drop trip_id 0, which is "not a trip" (data before first trip)
            if 0 in trips.index.get_level_values("trip_id"):
                trips = trips.drop(0, level="trip_id")
            # --- END FIX ---

            all_trips.append(trips)

        print("\n🚗 [2/7] Assembling all trips...")
        if not all_trips:
            print("❌ No trips found. Exiting.")
            return

        trips_df = pd.concat(all_trips).reset_index()
        print(f"Found {len(trips_df)} trips total.")

        # Calculate duration
        trips_df["duration_minutes"] = (
            trips_df["dropoff_time"] - trips_df["pickup_time"]
        ).dt.total_seconds() / 60

        # Filter noise
        trips_df = trips_df[
            (trips_df["duration_minutes"] > 1) & (trips_df["duration_minutes"] < 180)
        ].copy()

        print(f"After filtering: {len(trips_df)} valid trips.")

        if len(trips_df) < 100:
            print("⚠️  Warning: Very few trips found. Check raw data.")
            if len(trips_df) == 0:
                print("❌ No valid trips after filtering. Exiting.")
                return

        # --- 3. H3 & Core Features ---
        print(f"🗺️ [3/7] Engineering H3 (Res {H3_RESOLUTION}) features...")

        trips_df["h3_start"] = latlng_to_h3_vectorized(
            trips_df["start_lat"].values, trips_df["start_lon"].values, H3_RESOLUTION
        )
        trips_df["h3_end"] = latlng_to_h3_vectorized(
            trips_df["end_lat"].values, trips_df["end_lon"].values, H3_RESOLUTION
        )

        print("⏰ [4/7] Engineering time features...")
        trips_df["pickup_hour"] = trips_df["pickup_time"].dt.hour
        trips_df["pickup_dayofweek"] = trips_df["pickup_time"].dt.dayofweek
        trips_df["pickup_month"] = trips_df["pickup_time"].dt.month
        trips_df["pickup_day"] = trips_df["pickup_time"].dt.day
        trips_df["is_weekend"] = (trips_df["pickup_dayofweek"] >= 5).astype(int)

        print("📍 [5/7] Engineering geographic features...")
        trips_df["haversine_distance"] = haversine(
            trips_df["start_lat"],
            trips_df["start_lon"],
            trips_df["end_lat"],
            trips_df["end_lon"],
        )
        trips_df["start_dist_from_center"] = haversine(
            trips_df["start_lat"], trips_df["start_lon"], CITY_CENTER[0], CITY_CENTER[1]
        )

        # Filter nonsensical trips
        trips_df = trips_df[trips_df["haversine_distance"] > 0.1].copy()
        trips_df = trips_df.sort_values(by="pickup_time").reset_index(drop=True)

        # --- 4. Historical Features ---
        print("📈 [6/7] Engineering historical (expanding window) features...")

        trips_df["od_pair"] = trips_df["h3_start"] + "_" + trips_df["h3_end"]
        trips_df["od_pair_historical_count"] = trips_df.groupby("od_pair").cumcount()
        trips_df["origin_historical_count"] = trips_df.groupby("h3_start").cumcount()
        trips_df["origin_to_dest_popularity"] = (
            trips_df["od_pair_historical_count"] + 1
        ) / (trips_df["origin_historical_count"] + 1)

        # --- 5. Finalize ---
        print("🎯 [7/7] Finalizing and saving to database...")

        target = "h3_end"
        feature_columns = [
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

        final_columns = ["vehicle_id", "pickup_time", target] + feature_columns
        final_df = trips_df[final_columns].copy()
        final_df.dropna(inplace=True)

        print(f"Saving {len(final_df)} rows to '{OUTPUT_TABLE_NAME}'...")
        final_df.to_sql(
            OUTPUT_TABLE_NAME,
            engine,
            if_exists="replace",
            index=False,
            method="multi",
        )

        print("\n✅ Data preparation complete!")
        print(f"Table '{OUTPUT_TABLE_NAME}' is ready for model training.")

    except ImportError:
        print("❌ Error: 'h3' library not found. Please install: pip install h3")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    create_feature_table()
