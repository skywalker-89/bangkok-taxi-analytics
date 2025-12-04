import pandas as pd
import numpy as np
import h3
from sqlalchemy import create_engine
import sys

# --- Config ---
DB_URL = "postgresql+psycopg2://postgres:mypassword@localhost:5432/bangkok_taxi_db"
H3_RESOLUTION = 8
CITY_CENTER = (13.7563, 100.5018)
X_FILE = "X_features_inter_zone.csv"
Y_FILE = "y_target_inter_zone.csv"


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points on Earth."""
    R = 6371

    # --- FIX: Explicitly use .values to get 1D numpy arrays ---
    lat1_rad = np.radians(lat1.values)
    lon1_rad = np.radians(lon1.values)
    lat2_rad = np.radians(lat2.values)
    lon2_rad = np.radians(lon2.values)
    # --- End of Fix ---

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

    # --- FIX: Convert to numpy arrays to bypass pandas index alignment ---
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])

    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    x = np.cos(lat2_rad) * np.sin(dlon)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
        lat2_rad
    ) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))


def prepare_inter_zone_data():
    """
    Main function to build the inter-zone (empty trip) dataset.
    """
    try:
        engine = create_engine(DB_URL)
        print("📚 [1/7] Connecting to database and loading all probe data...")

        query = "SELECT vehicle_id, latitude, longitude, timestamp, for_hire_light FROM taxi_probe_raw"

        chunk_size = 5_000_000
        df_chunks = pd.read_sql(query, engine, chunksize=chunk_size)

        print("Processing data in chunks to find trips...")
        all_trips_df = []

        for i, df in enumerate(df_chunks):
            print(f"--- Processing Chunk {i+1} ---")

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(by=["vehicle_id", "timestamp"])

            df["is_new_trip"] = (df["for_hire_light"] == 1) & (
                df["for_hire_light"].shift(1) == 0
            )
            df["trip_id"] = df["is_new_trip"].cumsum()
            df_hired = df[df["for_hire_light"] == 1].copy()

            if df_hired.empty:
                continue

            trips_df = df_hired.groupby(["vehicle_id", "trip_id"]).agg(
                pickup_time=("timestamp", "first"),
                dropoff_time=("timestamp", "last"),
                start_lat=("latitude", "first"),
                start_lon=("longitude", "first"),
                end_lat=("latitude", "last"),
                end_lon=("longitude", "last"),
            )
            all_trips_df.append(trips_df)

        print("\n--- Final Assembly of Trips ---")
        if not all_trips_df:
            print("❌ No data processed. Exiting.")
            return

        combined_df = pd.concat(all_trips_df).reset_index()
        combined_df = combined_df.sort_values(by=["vehicle_id", "pickup_time"])
        print(f"Assembled {len(combined_df)} hired trips total.")

        # --- 2. Create Empty Trip Dataset (Core Logic) ---
        print("🚗 [2/7] Creating 'empty trip' (inter-zone) dataset...")

        # Shift data to find time/location between trips
        combined_df["prev_dropoff_time"] = combined_df.groupby("vehicle_id")[
            "dropoff_time"
        ].shift(1)
        combined_df["prev_end_lat"] = combined_df.groupby("vehicle_id")[
            "end_lat"
        ].shift(1)
        combined_df["prev_end_lon"] = combined_df.groupby("vehicle_id")[
            "end_lon"
        ].shift(1)

        # Create the "empty trips" dataframe
        empty_trips = combined_df.dropna(subset=["prev_dropoff_time"]).copy()

        # RENAME columns to be logical for the "empty" trip
        empty_trips.rename(
            columns={
                "prev_dropoff_time": "start_time",
                "prev_end_lat": "start_lat",
                "prev_end_lon": "start_lon",
                "pickup_time": "end_time",
                "start_lat": "end_lat",
                "start_lon": "end_lon",
            },
            inplace=True,
        )

        # --- 3. Engineer Target and Core Features ---
        print("🎯 [3/7] Engineering target (travel_time) and core features...")

        # Create Target (Travel Time in minutes)
        empty_trips["travel_time_minutes"] = (
            empty_trips["end_time"] - empty_trips["start_time"]
        ).dt.total_seconds() / 60

        # Direct distance
        empty_trips["direct_distance_km"] = haversine(
            empty_trips["start_lat"],
            empty_trips["start_lon"],
            empty_trips["end_lat"],
            empty_trips["end_lon"],
        )

        # Filter out nonsensical data
        empty_trips = empty_trips[
            (empty_trips["travel_time_minutes"] > 1)
            & (empty_trips["travel_time_minutes"] < 120)
            & (empty_trips["direct_distance_km"] > 0.1)
        ].copy()

        # --- 4. Time, Geo, and H3 Features ---
        print("🛠️ [4/7] Engineering time, geo, and H3 features...")

        # Time features
        empty_trips["start_hour"] = empty_trips["start_time"].dt.hour
        empty_trips["start_dayofweek"] = empty_trips["start_time"].dt.dayofweek
        empty_trips["is_weekend"] = (empty_trips["start_dayofweek"] >= 5).astype(int)

        # Bearing features
        bear = bearing(
            empty_trips["start_lat"],
            empty_trips["start_lon"],
            empty_trips["end_lat"],
            empty_trips["end_lon"],
        )
        empty_trips["bearing_sin"] = np.sin(np.radians(bear))
        empty_trips["bearing_cos"] = np.cos(np.radians(bear))

        # H3 features
        empty_trips["start_h3"] = empty_trips.apply(
            lambda r: h3.latlng_to_cell(r["start_lat"], r["start_lon"], H3_RESOLUTION),
            axis=1,
        )
        empty_trips["end_h3"] = empty_trips.apply(
            lambda r: h3.latlng_to_cell(r["end_lat"], r["end_lon"], H3_RESOLUTION),
            axis=1,
        )

        # Distance from center
        empty_trips["start_dist_from_center"] = haversine(
            empty_trips["start_lat"],
            empty_trips["start_lon"],
            CITY_CENTER[0],
            CITY_CENTER[1],
        )
        empty_trips["end_dist_from_center"] = haversine(
            empty_trips["end_lat"],
            empty_trips["end_lon"],
            CITY_CENTER[0],
            CITY_CENTER[1],
        )

        # --- 5. Historical (Expanding Window) Features ---
        print("📈 [5/7] Engineering historical (expanding window) features...")

        empty_trips = empty_trips.sort_values(by="start_time")

        # a. Historical OD Pair Counts
        empty_trips["zone_pair"] = empty_trips["start_h3"] + "_" + empty_trips["end_h3"]
        empty_trips["zone_pair_count"] = empty_trips.groupby("zone_pair").cumcount()

        # b. Origin Popularity
        empty_trips["origin_zone_count"] = empty_trips.groupby("start_h3").cumcount()

        # c. Origin-to-Destination Popularity
        empty_trips["origin_to_dest_popularity"] = (
            empty_trips["zone_pair_count"] + 1
        ) / (empty_trips["origin_zone_count"] + 1)

        # --- 6. Create final X and y ---
        print(" separating features (X) and target (y)...")

        target = "travel_time_minutes"
        features = [
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
            "start_h3",
            "end_h3",
            "start_dist_from_center",
            "end_dist_from_center",
            "zone_pair_count",
            "origin_zone_count",
            "origin_to_dest_popularity",
        ]

        # (From your notebook) Aggregate stats for start_h3
        print("📈 [6/7] Engineering H3-zone statistics...")

        zone_stats = (
            empty_trips.groupby("start_h3")
            .agg(
                avg_travel_time_from_zone=("travel_time_minutes", "mean"),
                avg_dist_from_zone=("direct_distance_km", "mean"),
            )
            .reset_index()
        )

        empty_trips = pd.merge(empty_trips, zone_stats, on="start_h3", how="left")

        # Update feature list
        features.extend(["avg_travel_time_from_zone", "avg_dist_from_zone"])

        X = empty_trips[features].copy()
        y = empty_trips[target].copy()

        # FillNa from merge
        X.fillna(0, inplace=True)

        # --- 7. Save to CSV ---
        print(f"💾 [7/7] Saving to {X_FILE} and {Y_FILE}...")
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
    prepare_inter_zone_data()
