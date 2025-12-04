import pandas as pd
import numpy as np
import h3
from sqlalchemy import create_engine
import sys

# --- Config ---
DB_URL = "postgresql+psycopg2://postgres:mypassword@localhost:5432/bangkok_taxi_db"
H3_RESOLUTION = 8  # As per your notebook
OUTPUT_TABLE_NAME = "model_destination_features"

# Bangkok city center (approx.) for distance features
CITY_CENTER = (13.7563, 100.5018)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def create_feature_table():
    """
    Loads raw data, segments into trips, and engineers advanced
    features for next-destination prediction.
    Saves the final features to a new database table.
    """
    try:
        engine = create_engine(DB_URL)
        print("📚 [1/7] Connecting to database and loading all probe data...")
        # WARNING: This loads the entire table. This is necessary for
        # the historical, expanding-window features.
        query = "SELECT * FROM taxi_probe_raw"
        df = pd.read_sql(query, engine)

        if df.empty:
            print("❌ No data found in taxi_probe_raw table. Exiting.")
            return

        print(f"Loaded {len(df)} raw probe points.")

        # --- 1. Basic Prep & Trip Segmentation ---
        print("🛠️ [2/7] Segmenting trips...")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by=["vehicle_id", "timestamp"])

        df["is_new_trip"] = (df["for_hire_light"] == 1) & (
            df["for_hire_light"].shift(1) == 0
        )
        df["trip_id"] = df["is_new_trip"].cumsum()

        df_hired = df[df["for_hire_light"] == 1].copy()

        # --- 2. Create Trip-Level Data ---
        print("🚗 [3/7] Aggregating trip start/end points...")
        trips_df = df_hired.groupby(["vehicle_id", "trip_id"]).agg(
            pickup_time=("timestamp", "first"),
            dropoff_time=("timestamp", "last"),
            start_lat=("latitude", "first"),
            start_lon=("longitude", "first"),
            end_lat=("latitude", "last"),
            end_lon=("longitude", "last"),
        )

        trips_df["duration_minutes"] = (
            trips_df["dropoff_time"] - trips_df["pickup_time"]
        ).dt.total_seconds() / 60

        # Filter out noise (trips < 1 min or > 3 hours)
        trips_df = trips_df[
            (trips_df["duration_minutes"] > 1) & (trips_df["duration_minutes"] < 180)
        ].copy()

        # --- 3. H3 & Core Features ---
        print(f"🗺️ [4/7] Engineering H3 (Res {H3_RESOLUTION}) and time features...")

        # Add H3 hex ID for start and end
        trips_df["h3_start"] = trips_df.apply(
            lambda r: h3.latlng_to_cell(r["start_lat"], r["start_lon"], H3_RESOLUTION),
            axis=1,
        )
        trips_df["h3_end"] = trips_df.apply(
            lambda r: h3.latlng_to_cell(r["end_lat"], r["end_lon"], H3_RESOLUTION),
            axis=1,
        )

        # Time features
        trips_df["pickup_hour"] = trips_df["pickup_time"].dt.hour
        trips_df["pickup_dayofweek"] = trips_df["pickup_time"].dt.dayofweek
        trips_df["pickup_month"] = trips_df["pickup_time"].dt.month
        trips_df["pickup_day"] = trips_df["pickup_time"].dt.day
        trips_df["is_weekend"] = (trips_df["pickup_dayofweek"] >= 5).astype(int)

        # Geographic features
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
        trips_df.sort_values(by="pickup_time", inplace=True)
        trips_df.reset_index(inplace=True)  # Now in chronological order

        # --- 4. Advanced Historical Features (Time-Travel-Safe) ---
        print("📈 [5/7] Engineering historical (expanding window) features...")

        # This is the "expanding window" from your notebook.
        # It calculates counts *up to that point in time* to prevent data leakage.

        # a. Historical OD Pair Counts
        trips_df["od_pair"] = trips_df["h3_start"] + "_" + trips_df["h3_end"]
        # .cumcount() groups by 'od_pair' and counts occurrences, ordered by time (since we sorted)
        trips_df["od_pair_historical_count"] = trips_df.groupby("od_pair").cumcount()

        # b. Origin Popularity (how many trips start here)
        trips_df["origin_historical_count"] = trips_df.groupby("h3_start").cumcount()

        # c. Origin-to-Destination Popularity (what % of trips from this origin go to this destination)
        # We add +1 to avoid division by zero (Laplace smoothing)
        trips_df["origin_to_dest_popularity"] = (
            trips_df["od_pair_historical_count"] + 1
        ) / (trips_df["origin_historical_count"] + 1)

        # --- 5. Define Feature List & Target ---
        print("🎯 [6/7] Finalizing feature list...")

        # Target variable
        target = "h3_end"

        # Feature columns
        feature_columns = [
            "h3_start",  # Categorical feature
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

        # Add vehicle_id and pickup_time for potential time-series splitting
        final_columns = ["vehicle_id", "pickup_time", target] + feature_columns

        # Create the final DataFrame
        final_df = trips_df[final_columns].copy()
        final_df.dropna(inplace=True)

        # --- 6. Save to Database ---
        print(
            f"💾 [7/7] Saving {len(final_df)} feature rows to table '{OUTPUT_TABLE_NAME}'..."
        )

        final_df.to_sql(
            OUTPUT_TABLE_NAME,
            engine,
            if_exists="replace",  # Overwrite table
            index=False,
            method="multi",  # Fast insertion
        )

        print("\n✅ Data preparation complete!")
        print(f"Table '{OUTPUT_TABLE_NAME}' is ready for model training.")

    except ImportError:
        print("❌ Error: 'h3' library not found. Please install it: pip install h3")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    create_feature_table()
