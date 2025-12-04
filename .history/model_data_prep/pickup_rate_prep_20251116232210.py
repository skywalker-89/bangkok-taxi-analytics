import pandas as pd
import numpy as np
import h3
from sqlalchemy import create_engine
import psycopg2
import sys
from datetime import timedelta
from multiprocessing import Pool, cpu_count

# --- Config ---
DB_URL = "postgresql+psycopg2://postgres:mypassword@localhost:5432/bangkok_taxi_db"
H3_RESOLUTION = 9
TIME_BIN_MINUTES = 5

# --- Sliding Window Config ---
WINDOW_LENGTH = "14D"
WINDOW_STEP = "1D"
FORECAST_HORIZON = "1h"

# --- Output File Names ---
X_FILE = "X_features_5min.csv"
Y_FILE = "y_target_5min.csv"
META_FILE = "meta_5min.csv"

N_WORKERS = max(1, cpu_count() - 1)


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
            AND indexname IN ('idx_timestamp', 'idx_vehicle_timestamp')
        """
        )
        existing = {row[0] for row in cur.fetchall()}

        if "idx_vehicle_timestamp" not in existing:
            print("  Creating vehicle+timestamp index (5-10 min)...")
            cur.execute(
                """
                CREATE INDEX CONCURRENTLY idx_vehicle_timestamp 
                ON taxi_probe_raw(vehicle_id, timestamp)
            """
            )
            print("  ✓ Vehicle index created")
        else:
            print("  ✓ Indexes already exist")

    except Exception as e:
        print(f"  ⚠️ Index setup: {e}")
    finally:
        cur.close()
        conn.close()


def vectorized_h3(lats, lons, resolution):
    """Vectorized H3 conversion"""
    return np.array(
        [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(lats, lons)]
    )


def add_time_features(df):
    """Optimized time features"""
    df["dow"] = df["time_bin"].dt.dayofweek.astype("int16")
    df["hour"] = df["time_bin"].dt.hour.astype("int16")
    df["minute_of_day"] = (df["hour"] * 60 + df["time_bin"].dt.minute).astype("int16")
    return df


def get_lagged_features(df, lags):
    """Optimized lagged features"""
    df = df.sort_values(["h3_cell", "time_bin"]).set_index(["h3_cell", "time_bin"])
    lagged_cols = {}
    grouped = df.groupby(level="h3_cell")["pickup_count"]

    for lag in lags:
        lagged_cols[f"pickup_count_lag_{lag}"] = grouped.shift(lag)

    df_lags = pd.DataFrame(lagged_cols, index=df.index).reset_index()
    return df_lags


def get_rolling_features(df, windows):
    """Optimized rolling features"""
    df = df.sort_values(["h3_cell", "time_bin"]).set_index(["h3_cell", "time_bin"])
    rolling_cols = {}
    grouped = df.groupby(level="h3_cell")["pickup_count"]

    for window, min_p in windows:
        rolling_cols[f"pickup_count_roll_avg_{window}"] = (
            grouped.rolling(window, min_periods=min_p).mean().droplevel(0)
        )

    df_rolls = pd.DataFrame(rolling_cols, index=df.index).reset_index()
    return df_rolls


def process_window(args):
    """
    Process a single window using PRE-LOADED data.
    This is MUCH faster than querying the database each time.
    """
    window_start, window_config, pickups_df = args

    w_len = pd.to_timedelta(window_config["WINDOW_LENGTH"])
    f_hor = pd.to_timedelta(window_config["FORECAST_HORIZON"])

    train_start = window_start
    train_end = window_start + w_len
    forecast_start = train_end
    forecast_end = train_end + f_hor
    query_end = forecast_end

    # Filter pre-loaded data for this window
    window_pickups = pickups_df[
        (pickups_df["time_bin"] >= train_start) & (pickups_df["time_bin"] < query_end)
    ].copy()

    if len(window_pickups) == 0:
        return None, None, None

    # Create base grid
    all_bins = pd.date_range(
        start=train_start,
        end=query_end,
        freq=f"{TIME_BIN_MINUTES}min",
        inclusive="left",
    )
    all_cells = window_pickups["h3_cell"].unique()

    if len(all_cells) == 0:
        return None, None, None

    # Efficient grid creation
    base_df = pd.DataFrame(
        {
            "h3_cell": np.repeat(all_cells, len(all_bins)),
            "time_bin": np.tile(all_bins, len(all_cells)),
        }
    )

    # Get pickup counts
    pickup_counts = (
        window_pickups.groupby(["h3_cell", "time_bin"], observed=True)
        .size()
        .rename("pickup_count")
    )

    base_df = base_df.merge(pickup_counts, on=["h3_cell", "time_bin"], how="left")
    base_df["pickup_count"] = base_df["pickup_count"].fillna(0).astype("int16")

    # Engineer Features
    base_df = add_time_features(base_df)

    bins_per_hour = 60 // TIME_BIN_MINUTES
    bins_per_day = 24 * bins_per_hour

    lags = [bins_per_hour, 2 * bins_per_hour, bins_per_day]
    lags_df = get_lagged_features(base_df, lags)

    windows = [(bins_per_hour, 2), (6 * bins_per_hour, 6), (bins_per_day, 12)]
    rolls_df = get_rolling_features(base_df, windows)

    # Combine features
    features_df = lags_df.merge(rolls_df, on=["h3_cell", "time_bin"])
    features_df = features_df.merge(
        base_df[["h3_cell", "time_bin", "dow", "hour", "minute_of_day"]],
        on=["h3_cell", "time_bin"],
    )

    # Create Target
    forecast_pickups = window_pickups[
        (window_pickups["time_bin"] >= forecast_start)
        & (window_pickups["time_bin"] < forecast_end)
    ]

    target_df = (
        forecast_pickups.groupby("h3_cell", observed=True)
        .size()
        .ge(1)
        .rename("y")
        .reset_index()
    )

    # Align X and y
    X = features_df[features_df["time_bin"] == train_end].copy()
    X_with_target = X.merge(target_df, on="h3_cell", how="left")
    X_with_target["y"] = X_with_target["y"].fillna(0).astype("int8")

    y = X_with_target["y"]
    X = X_with_target.drop(columns=["y"])

    meta = X[["h3_cell", "time_bin"]]
    X = X.drop(columns=["h3_cell", "time_bin"])

    return X, y, meta


def create_pickup_features():
    """Main function - LOAD DATA ONCE, then process in parallel"""
    try:
        # Setup indexes first
        setup_database_indexes()

        engine = create_engine(DB_URL)

        # Get date range
        print("📚 [1/5] Getting date range...")
        min_date, max_date = pd.read_sql(
            "SELECT MIN(timestamp), MAX(timestamp) FROM taxi_probe_raw", engine
        ).iloc[0]

        # CRITICAL OPTIMIZATION: Load ALL data ONCE
        print("📥 [2/5] Loading ALL pickup data (this will take 2-5 minutes)...")
        query = """
            SELECT vehicle_id, latitude, longitude, timestamp, for_hire_light
            FROM taxi_probe_raw
            ORDER BY vehicle_id, timestamp
        """

        # Load in chunks to show progress
        chunk_size = 5_000_000
        all_chunks = []

        for i, chunk in enumerate(pd.read_sql(query, engine, chunksize=chunk_size)):
            print(f"  Loading chunk {i+1}...")
            all_chunks.append(chunk)

        df = pd.concat(all_chunks, ignore_index=True)
        engine.dispose()

        print(f"  Loaded {len(df):,} probe points")

        # Detect pickups
        print("🚗 [3/5] Detecting pickups...")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by=["vehicle_id", "timestamp"])
        df["prev_light"] = df.groupby("vehicle_id")["for_hire_light"].shift(1)
        df["is_new_trip"] = (df["for_hire_light"] == 1) & (df["prev_light"] == 0)
        pickups_df = df[df["is_new_trip"]].copy()

        print(f"  Found {len(pickups_df):,} pickups")

        # Convert to H3
        print("🗺️  Converting to H3 hexagons...")
        pickups_df["h3_cell"] = vectorized_h3(
            pickups_df["latitude"].values, pickups_df["longitude"].values, H3_RESOLUTION
        )

        # Create time bins
        pickups_df["time_bin"] = pickups_df["timestamp"].dt.floor(
            f"{TIME_BIN_MINUTES}min"
        )

        # Free memory
        del df

        # Calculate windows
        window_starts = pd.date_range(
            start=min_date,
            end=max_date
            - pd.to_timedelta(WINDOW_LENGTH)
            - pd.to_timedelta(FORECAST_HORIZON),
            freq=WINDOW_STEP,
        )

        print(
            f"\n🔄 [4/5] Processing {len(window_starts)} windows with {N_WORKERS} workers..."
        )
        print(f"   Estimated time: 15-45 minutes (much faster with pre-loaded data!)\n")

        window_config = {
            "WINDOW_LENGTH": WINDOW_LENGTH,
            "FORECAST_HORIZON": FORECAST_HORIZON,
        }

        # Parallel processing with pre-loaded data
        args_list = [(start, window_config, pickups_df) for start in window_starts]

        with Pool(N_WORKERS) as pool:
            results = []
            for i, result in enumerate(pool.imap(process_window, args_list), 1):
                if result[0] is not None:
                    results.append(result)
                if i % 100 == 0:
                    print(
                        f"  Processed {i}/{len(window_starts)} windows ({i*100//len(window_starts)}%)..."
                    )

        if not results:
            print("❌ No data processed. Exiting.")
            return

        print(f"\n✅ [5/5] Processed {len(results)} windows successfully")
        print("📦 Combining and saving results...")

        all_X, all_y, all_meta = zip(*results)

        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        meta_combined = pd.concat(all_meta, ignore_index=True)

        # Optimize dtypes
        for col in X_combined.select_dtypes(include=["float64"]).columns:
            X_combined[col] = X_combined[col].astype("float32")

        print(f"💾 Saving {len(X_combined):,} rows to CSVs...")
        X_combined.to_csv(X_FILE, index=False)
        y_combined.to_csv(Y_FILE, index=False)
        meta_combined.to_csv(META_FILE, index=False)

        print("\n✅ Data preparation complete!")
        print(
            f"  - {X_FILE} ({X_combined.memory_usage(deep=True).sum() / 1024**2:.1f} MB)"
        )
        print(f"  - {Y_FILE}")
        print(f"  - {META_FILE}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    create_pickup_features()
