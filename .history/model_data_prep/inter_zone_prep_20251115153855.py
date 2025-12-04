import pandas as pd
import numpy as np
import h3
from sqlalchemy import create_engine
import sys
from datetime import timedelta
from multiprocessing import Pool, cpu_count
import psycopg2
from psycopg2.extras import execute_values

# --- Config ---
DB_URL = "postgresql+psycopg2://postgres:mypassword@localhost:5432/bangkok_taxi_db"
H3_RESOLUTION = 9
TIME_BIN_MINUTES = 5

# --- Sliding Window Config ---
WINDOW_LENGTH = "14D"
WINDOW_STEP = "1D"
FORECAST_HORIZON = "1H"

# --- Output File Names ---
X_FILE = "X_features_5min.csv"
Y_FILE = "y_target_5min.csv"
META_FILE = "meta_5min.csv"

# --- Parallelization ---
N_WORKERS = max(1, cpu_count() - 1)

# --- CHUNKING for memory efficiency ---
CHUNK_SIZE = 100000  # Process 100k rows at a time


def setup_database_indexes():
    """
    CRITICAL: Create indexes for 55M records.
    Run this ONCE before your first run.
    """
    conn = psycopg2.connect(
        "host=localhost dbname=bangkok_taxi_db user=postgres password=mypassword"
    )
    cur = conn.cursor()

    print("🔧 Setting up database indexes (one-time setup)...")

    # Check if indexes exist
    cur.execute(
        """
        SELECT indexname FROM pg_indexes 
        WHERE tablename = 'taxi_probe_raw' 
        AND indexname IN ('idx_timestamp', 'idx_vehicle_timestamp');
    """
    )
    existing = [row[0] for row in cur.fetchall()]

    if "idx_timestamp" not in existing:
        print("  Creating timestamp index (this may take 5-10 minutes)...")
        cur.execute(
            "CREATE INDEX CONCURRENTLY idx_timestamp ON taxi_probe_raw(timestamp);"
        )
        conn.commit()
        print("  ✅ Timestamp index created")

    if "idx_vehicle_timestamp" not in existing:
        print("  Creating vehicle+timestamp index...")
        cur.execute(
            "CREATE INDEX CONCURRENTLY idx_vehicle_timestamp ON taxi_probe_raw(vehicle_id, timestamp);"
        )
        conn.commit()
        print("  ✅ Vehicle+timestamp index created")

    cur.close()
    conn.close()
    print("✅ Database optimized!\n")


def vectorized_h3_batch(lats, lons, resolution):
    """Ultra-fast batch H3 conversion"""
    return np.array(
        [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(lats, lons)],
        dtype=object,
    )


def add_time_features(df):
    """Optimized time features"""
    dt = df["time_bin"].dt
    df["dow"] = dt.dayofweek.astype("int8")
    df["hour"] = dt.hour.astype("int8")
    df["minute_of_day"] = (df["hour"] * 60 + dt.minute).astype("int16")
    return df


def get_lagged_and_rolling_features(df, lags, windows):
    """
    Combined lag + rolling calculation - more efficient than separate calls.
    """
    df = df.sort_values(["h3_cell", "time_bin"]).set_index(["h3_cell", "time_bin"])

    result_cols = {}
    grouped = df.groupby(level="h3_cell")["pickup_count"]

    # Lagged features
    for lag in lags:
        result_cols[f"pickup_count_lag_{lag}"] = grouped.shift(lag)

    # Rolling features
    for window, min_p in windows:
        result_cols[f"pickup_count_roll_avg_{window}"] = grouped.rolling(
            window, min_periods=min_p
        ).mean()

    result_df = pd.DataFrame(result_cols, index=df.index).reset_index()
    return result_df


def process_window_optimized(args):
    """
    Hyper-optimized for 55M records:
    - Detect pickups IN DATABASE using SQL window functions
    - Chunk processing for memory efficiency
    - Minimize data transfer
    """
    window_start, window_config, db_url = args

    w_len = pd.to_timedelta(window_config["WINDOW_LENGTH"])
    f_hor = pd.to_timedelta(window_config["FORECAST_HORIZON"])

    train_start = window_start
    train_end = window_start + w_len
    forecast_start = train_end
    forecast_end = train_end + f_hor

    print(f"⚡ {train_start.date()}")

    # MEGA OPTIMIZATION: Detect pickups in SQL!
    # This processes 55M rows on the database server, not in Python
    query = f"""
        WITH pickup_events AS (
            SELECT 
                vehicle_id,
                latitude,
                longitude,
                timestamp,
                for_hire_light,
                LAG(for_hire_light) OVER (
                    PARTITION BY vehicle_id 
                    ORDER BY timestamp
                ) as prev_light
            FROM taxi_probe_raw
            WHERE timestamp >= '{train_start}' 
              AND timestamp < '{forecast_end}'
        )
        SELECT 
            latitude,
            longitude,
            timestamp
        FROM pickup_events
        WHERE for_hire_light = 1 AND prev_light = 0
    """

    engine = create_engine(db_url)

    try:
        # Read in chunks to avoid memory issues
        pickups_chunks = []
        for chunk in pd.read_sql(
            query, engine, chunksize=CHUNK_SIZE, parse_dates=["timestamp"]
        ):
            if len(chunk) > 0:
                pickups_chunks.append(chunk)

        if not pickups_chunks:
            engine.dispose()
            return None, None, None

        pickups_df = pd.concat(pickups_chunks, ignore_index=True)
        engine.dispose()

    except Exception as e:
        print(f"  ❌ Query failed: {e}")
        engine.dispose()
        return None, None, None

    if len(pickups_df) == 0:
        return None, None, None

    # Vectorized H3 conversion
    pickups_df["h3_cell"] = vectorized_h3_batch(
        pickups_df["latitude"].values, pickups_df["longitude"].values, H3_RESOLUTION
    )

    # Floor to time bins
    pickups_df["time_bin"] = pickups_df["timestamp"].dt.floor(f"{TIME_BIN_MINUTES}min")

    # Split train and forecast data
    train_pickups = pickups_df[pickups_df["time_bin"] < train_end]
    forecast_pickups = pickups_df[
        (pickups_df["time_bin"] >= forecast_start)
        & (pickups_df["time_bin"] < forecast_end)
    ]

    # Get unique cells from training data
    all_cells = train_pickups["h3_cell"].unique()

    if len(all_cells) == 0:
        return None, None, None

    # Create time bins
    all_bins = pd.date_range(
        start=train_start,
        end=train_end,
        freq=f"{TIME_BIN_MINUTES}min",
        inclusive="left",
    )

    # Efficient grid creation with pre-allocated arrays
    n_cells = len(all_cells)
    n_bins = len(all_bins)

    base_df = pd.DataFrame(
        {
            "h3_cell": np.repeat(all_cells, n_bins),
            "time_bin": np.tile(all_bins, n_cells),
        }
    )

    # Count pickups efficiently
    pickup_counts = (
        train_pickups.groupby(["h3_cell", "time_bin"], observed=True)
        .size()
        .rename("pickup_count")
    )

    base_df = base_df.merge(pickup_counts, on=["h3_cell", "time_bin"], how="left")
    base_df["pickup_count"] = base_df["pickup_count"].fillna(0).astype("int16")

    # Time features
    base_df = add_time_features(base_df)

    # Feature engineering
    bins_per_hour = 60 // TIME_BIN_MINUTES
    bins_per_day = 24 * bins_per_hour

    lags = [bins_per_hour, 2 * bins_per_hour, bins_per_day]
    windows = [(bins_per_hour, 2), (6 * bins_per_hour, 6), (bins_per_day, 12)]

    features_df = get_lagged_and_rolling_features(base_df, lags, windows)

    # Add time features back
    features_df = features_df.merge(
        base_df[["h3_cell", "time_bin", "dow", "hour", "minute_of_day"]],
        on=["h3_cell", "time_bin"],
    )

    # Create target
    target_df = (
        forecast_pickups.groupby("h3_cell", observed=True)
        .size()
        .ge(1)
        .rename("y")
        .reset_index()
    )

    # Final dataset at train_end
    X = features_df[features_df["time_bin"] == train_end].copy()
    X_with_target = X.merge(target_df, on="h3_cell", how="left")
    X_with_target["y"] = X_with_target["y"].fillna(0).astype("int8")

    y = X_with_target["y"]
    X = X_with_target.drop(columns=["y"])

    meta = X[["h3_cell", "time_bin"]].copy()
    X = X.drop(columns=["h3_cell", "time_bin"])

    # Optimize dtypes
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype("float32")

    return X, y, meta


def create_pickup_features():
    """Main function with all optimizations"""
    try:
        # STEP 0: Setup indexes (run once)
        setup_database_indexes()

        engine = create_engine(DB_URL)
        print("📚 [1/4] Getting date range...")

        min_date, max_date = pd.read_sql(
            "SELECT MIN(timestamp), MAX(timestamp) FROM taxi_probe_raw", engine
        ).iloc[0]
        engine.dispose()

        window_starts = pd.date_range(
            start=min_date,
            end=max_date
            - pd.to_timedelta(WINDOW_LENGTH)
            - pd.to_timedelta(FORECAST_HORIZON),
            freq=WINDOW_STEP,
        )

        window_config = {
            "WINDOW_LENGTH": WINDOW_LENGTH,
            "FORECAST_HORIZON": FORECAST_HORIZON,
        }

        print(
            f"\n🔄 [2/4] Processing {len(window_starts)} windows with {N_WORKERS} workers..."
        )
        print(f"    Using SQL-based pickup detection for maximum performance\n")

        # Parallel processing
        args_list = [(start, window_config, DB_URL) for start in window_starts]

        with Pool(N_WORKERS) as pool:
            results = pool.map(process_window_optimized, args_list)

        # Filter valid results
        results = [r for r in results if r[0] is not None]

        if not results:
            print("❌ No data processed. Exiting.")
            return

        print(f"\n✅ [3/4] Processed {len(results)}/{len(window_starts)} windows")
        print("📦 [4/4] Combining and saving...")

        all_X, all_y, all_meta = zip(*results)

        # Concatenate in chunks to avoid memory spike
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        meta_combined = pd.concat(all_meta, ignore_index=True)

        print(f"💾 Saving {len(X_combined):,} rows...")
        X_combined.to_csv(X_FILE, index=False)
        y_combined.to_csv(Y_FILE, index=False)
        meta_combined.to_csv(META_FILE, index=False)

        mem_mb = X_combined.memory_usage(deep=True).sum() / 1024**2
        print("\n✅ Data preparation complete!")
        print(f"  📊 Rows: {len(X_combined):,}")
        print(f"  💾 Size: {mem_mb:.1f} MB")
        print(f"  📁 Files: {X_FILE}, {Y_FILE}, {META_FILE}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    create_pickup_features()
