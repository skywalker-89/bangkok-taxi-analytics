import pandas as pd
import numpy as np
import h3
from sqlalchemy import create_engine
import sys

# --- Config ---
DB_URL = "postgresql+psycopg2://postgres:mypassword@localhost:5432/bangkok_taxi_db"
H3_RESOLUTION = 8  # As per your notebook
CITY_CENTER = (13.7563, 100.5018)
X_FILE = "X_features_duration.csv"
Y_FILE = "y_target_duration.csv"


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
    """
    Main function to load, process, and save
    the feature data for the duration model.
    """
    try:
        engine = create_engine(DB_URL)
        print(f"📚 [1/6] Connecting to database...")

        query = "SELECT vehicle_id, latitude, longitude, timestamp, speed, for_hire_light FROM taxi_probe_raw"

        chunk_size = 5_000_000
        df_chunks = pd.read_sql(query, engine, chunksize=chunk_size)

        print("Processing data in chunks...")
        processed_trips = []

        for i, df in enumerate(df_chunks):
            print(f"--- Processing Chunk {i+1} ---")

            # 1. Convert timestamp and sort
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(by=["vehicle_id", "timestamp"])

            # 2. Identify trips
            df["is_new_trip"] = (df["for_hire_light"] == 1) & (
                df["for_hire_light"].shift(1) == 0
            )
            df["trip_id"] = df["is_new_trip"].cumsum()

            # 3. Filter for 'for_hire' data
            df = df[df["for_hire_light"] == 1].copy()
            if df.empty:
                continue

            # 4. Calculate point-to-point distance for trip length
            df["time_diff"] = (
                df.groupby(["vehicle_id", "trip_id"])["timestamp"]
                .diff()
                .dt.total_seconds()
                .fillna(0)
            )
            df["prev_lat"] = df.groupby(["vehicle_id", "trip_id"])["latitude"].shift(1)
            df["prev_lon"] = df.groupby(["vehicle_id", "trip_id"])["longitude"].shift(1)

            df["distance_diff_km"] = haversine(
                df["prev_lat"], df["prev_lon"], df["latitude"], df["longitude"]
            )
            df["distance_diff_km"] = df["distance_diff_km"].fillna(0)

            # 5. Aggregate trips
            trips_df = df.groupby(["vehicle_id", "trip_id"]).agg(
                start_lat=("latitude", "first"),
                start_lon=("longitude", "first"),
                end_lat=("latitude", "last"),
                end_lon=("longitude", "last"),
                pickup_time=("timestamp", "first"),
                dropoff_time=("timestamp", "last"),
                average_speed=("speed", "mean"),
                max_speed=("speed", "max"),
                total_distance_km=("distance_diff_km", "sum"),
            )

            # 6. Calculate duration (TARGET) and filter
            trips_df["duration_minutes"] = (
                trips_df["dropoff_time"] - trips_df["pickup_time"]
            ).dt.total_seconds() / 60
            trips_df = trips_df[
                (trips_df["duration_minutes"] > 1)
                & (trips_df["total_distance_km"] > 0.1)
            ]

            processed_trips.append(trips_df)

        print("\n--- Final Assembly ---")
        if not processed_trips:
            print("❌ No data processed. Exiting.")
            return

        final_df = pd.concat(processed_trips, ignore_index=True).reset_index(drop=True)
        print(f"Assembled {len(final_df)} trips total.")

        # --- 7. Feature Engineering on Assembled Trips ---
        print("🛠️ [2/6] Engineering time features...")
        final_df["pickup_hour"] = final_df["pickup_time"].dt.hour
        final_df["pickup_dayofweek"] = final_df["pickup_time"].dt.dayofweek
        final_df["is_weekend"] = (final_df["pickup_dayofweek"] >= 5).astype(int)

        print(f"🗺️ [3/6] Engineering H3 (Res {H3_RESOLUTION}) features...")
        final_df["start_h3_zone"] = final_df.apply(
            lambda r: h3.latlng_to_cell(r["start_lat"], r["start_lon"], H3_RESOLUTION),
            axis=1,
        )
        final_df["end_h3_zone"] = final_df.apply(
            lambda r: h3.latlng_to_cell(r["end_lat"], r["end_lon"], H3_RESOLUTION),
            axis=1,
        )

        print("📍 [4/6] Engineering geo features...")
        final_df["haversine_distance"] = haversine(
            final_df["start_lat"],
            final_df["start_lon"],
            final_df["end_lat"],
            final_df["end_lon"],
        )
        final_df["dist_from_center"] = haversine(
            final_df["start_lat"], final_df["start_lon"], CITY_CENTER[0], CITY_CENTER[1]
        )

        # --- 8. Create final X and y ---
        print(" separating features (X) and target (y)...")

        feature_columns = [
            "start_h3_zone",
            "end_h3_zone",
            "pickup_hour",
            "pickup_dayofweek",
            "is_weekend",
            "haversine_distance",
            "dist_from_center",
            "average_speed",
            "max_speed",
            "total_distance_km",
        ]

        X = final_df[feature_columns].copy()
        y = final_df[["duration_minutes"]]

        # 9. Label Encode H3 zones (as in notebook)
        print("🔑 [5/6] Label encoding H3 zones...")
        from sklearn.preprocessing import LabelEncoder

        # We need to save these encoders for prediction
        le_start = LabelEncoder()
        le_end = LabelEncoder()

        X["start_h3_zone"] = le_start.fit_transform(X["start_h3_zone"])
        X["end_h3_zone"] = le_end.fit_transform(X["end_h3_zone"])

        # Save encoders
        import joblib

        joblib.dump(le_start, "duration_start_zone_encoder.pkl")
        joblib.dump(le_end, "duration_end_zone_encoder.pkl")
        print("✓ Encoders saved.")

        # 10. Save to CSV
        print(f"💾 [6/6] Saving to {X_FILE} and {Y_FILE}...")
        X.to_csv(X_FILE, index=False)
        y.to_csv(Y_FILE, index=False)

        print("\n✅ Data preparation complete!")
        print(f"Saved {len(X)} feature rows.")

    except ImportError:
        print("❌ Error: 'h3' library not found. Please install: pip install h3")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    prepare_duration_data()
