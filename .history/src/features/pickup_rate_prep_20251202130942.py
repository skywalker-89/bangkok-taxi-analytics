import pandas as pd
import numpy as np
import h3
from sqlalchemy import create_engine
import psycopg2
import sys
from datetime import timedelta
import gc

# --- Config ---
from src.utils.db import get_engine, get_db_connection

H3_RESOLUTION = 8  # Lower resolution = fewer cells = less memory
TIME_BIN_MINUTES = 5

# --- Sliding Window Config ---
WINDOW_LENGTH = "7D"
WINDOW_STEP = "2D"
FORECAST_HORIZON = "1h"

# --- Output File Names ---
X_FILE = "X_features_5min.csv"
Y_FILE = "y_target_5min.csv"
META_FILE = "meta_5min.csv"

# --- Memory optimization ---
MAX_CELLS_PER_WINDOW = 5000  # Limit cells to avoid memory explosion


def setup_database_indexes():
    """Create indexes for faster queries."""
    print("🔧 Checking database indexes...")

    conn = get_db_connection()
    conn.autocommit = True  # Alternative to set_isolation_level
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
            print("  Creating vehicle+timestamp index (5-10 min one-time wait)...")
            cur.execute(
                """
                CREATE INDEX CONCURRENTLY idx_vehicle_timestamp 
                ON taxi_probe_raw(vehicle_id, timestamp)
            """
            )
            print("  ✓ Index created")
        else:
            print("  ✓ Indexes already exist")

    except Exception as e:
        print(f"  ⚠️ Index setup: {e}")
    finally:
        cur.close()
        conn.close()


def vectorized_h3(lats, lons, resolution):
    """Vectorized H3 conversion"""
    return [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(lats, lons)]


def add_time_features(df):
    """Optimized time features"""
    # --- FIX: Create columns with the correct names ---
    df["day_of_week"] = df["time_bin"].dt.dayofweek.astype("int8")
    df["hour_of_day"] = df["time_bin"].dt.hour.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["minute_of_day"] = (df["hour_of_day"] * 60 + df["time_bin"].dt.minute).astype(
        "int16"
    )
    # --- END FIX ---
    return df


def get_lagged_features(df, lags):
    """Memory-efficient lagged features"""
    df = df.sort_values(["h3_cell", "time_bin"]).set_index(["h3_cell", "time_bin"])

    lagged_cols = {}
    for lag in lags:
        lagged_cols[f"pickup_count_lag_{lag}"] = df.groupby(level="h3_cell")[
            "pickup_count"
        ].shift(lag)

    result = pd.DataFrame(lagged_cols, index=df.index).reset_index()
    return result


def get_rolling_features(df, windows):
    """Memory-efficient rolling features"""
    df = df.sort_values(["h3_cell", "time_bin"]).set_index(["h3_cell", "time_bin"])

    rolling_cols = {}
    for window, min_p in windows:
        rolling_cols[f"pickup_count_roll_avg_{window}"] = (
            df.groupby(level="h3_cell")["pickup_count"]
            .rolling(window, min_periods=min_p)
            .mean()
            .droplevel(0)
        )

    result = pd.DataFrame(rolling_cols, index=df.index).reset_index()
    return result


def process_window_streaming(window_start, window_config, engine):
    """
    Process window with aggressive memory management
    Returns data or None
    """
    w_len = pd.to_timedelta(window_config["WINDOW_LENGTH"])
    f_hor = pd.to_timedelta(window_config["FORECAST_HORIZON"])

    train_start = window_start
    train_end = window_start + w_len
    forecast_start = train_end
    forecast_end = train_end + f_hor
    query_end = forecast_end

    query = f"""
        SELECT vehicle_id, latitude, longitude, timestamp, for_hire_light
        FROM taxi_probe_raw
        WHERE timestamp >= '{train_start}' 
        AND timestamp < '{query_end}'
        ORDER BY vehicle_id, timestamp
    """

    try:
        df = pd.read_sql(query, engine, parse_dates=["timestamp"])
    except Exception as e:
        print(f"  ⚠️ Query error: {e}")
        return None

    if df.empty or len(df) < 10:
        return None

    df["prev_light"] = df.groupby("vehicle_id")["for_hire_light"].shift(1)
    df["is_new_trip"] = (df["for_hire_light"] == 1) & (df["prev_light"] == 0)
    pickups_df = df[df["is_new_trip"]][["latitude", "longitude", "timestamp"]].copy()

    del df
    gc.collect()

    if len(pickups_df) == 0:
        return None

    pickups_df["h3_cell"] = vectorized_h3(
        pickups_df["latitude"].values,
        pickups_df["longitude"].values,
        H3_RESOLUTION,
    )
    pickups_df["time_bin"] = pickups_df["timestamp"].dt.floor(f"{TIME_BIN_MINUTES}min")

    all_cells = pickups_df["h3_cell"].unique()
    if len(all_cells) > MAX_CELLS_PER_WINDOW:
        cell_counts = pickups_df["h3_cell"].value_counts()
        top_cells = cell_counts.head(MAX_CELLS_PER_WINDOW).index
        pickups_df = pickups_df[pickups_df["h3_cell"].isin(top_cells)]
        all_cells = top_cells.values

    if len(all_cells) == 0:
        return None

    all_bins = pd.date_range(
        start=train_start,
        end=query_end,
        freq=f"{TIME_BIN_MINUTES}min",
        inclusive="left",
    )
    base_df = pd.DataFrame(
        {
            "h3_cell": np.repeat(all_cells, len(all_bins)),
            "time_bin": np.tile(all_bins, len(all_cells)),
        }
    )

    pickup_counts = (
        pickups_df.groupby(["h3_cell", "time_bin"], observed=True)
        .size()
        .rename("pickup_count")
    )
    base_df = base_df.merge(pickup_counts, on=["h3_cell", "time_bin"], how="left")
    base_df["pickup_count"] = base_df["pickup_count"].fillna(0).astype("int16")

    base_df = add_time_features(base_df)

    bins_per_hour = 60 // TIME_BIN_MINUTES
    bins_per_day = 24 * bins_per_hour
    lags = [bins_per_hour, 2 * bins_per_hour, bins_per_day]
    lags_df = get_lagged_features(
        base_df[["h3_cell", "time_bin", "pickup_count"]], lags
    )
    windows = [(bins_per_hour, 2), (6 * bins_per_hour, 6), (bins_per_day, 12)]
    rolls_df = get_rolling_features(
        base_df[["h3_cell", "time_bin", "pickup_count"]], windows
    )

    features_df = lags_df.merge(rolls_df, on=["h3_cell", "time_bin"])

    # --- FIX: Merge the correct column names ---
    features_df = features_df.merge(
        base_df[
            [
                "h3_cell",
                "time_bin",
                "day_of_week",
                "hour_of_day",
                "is_weekend",
                "minute_of_day",
            ]
        ],
        on=["h3_cell", "time_bin"],
    )
    # --- END FIX ---

    forecast_pickups = pickups_df[
        (pickups_df["time_bin"] >= forecast_start)
        & (pickups_df["time_bin"] < forecast_end)
    ]
    target_df = (
        forecast_pickups.groupby("h3_cell", observed=True)
        .size()
        .ge(1)
        .astype("int8")
        .rename("y")
        .reset_index()
    )

    X = features_df[features_df["time_bin"] == train_end].copy()
    if len(X) == 0:
        return None

    X_with_target = X.merge(target_df, on="h3_cell", how="left")
    X_with_target["y"] = X_with_target["y"].fillna(0).astype("int8")

    y = X_with_target[["y"]]

    # --- FIX: Save time features to META file ---
    meta_cols = ["h3_cell", "time_bin", "hour_of_day", "day_of_week", "is_weekend"]
    meta = X_with_target[meta_cols]

    # Drop all meta columns and target from X
    X = X_with_target.drop(columns=["y"] + meta_cols)
    # --- END FIX ---

    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype("float32")

    del base_df, lags_df, rolls_df, features_df, pickups_df
    gc.collect()

    return X, y, meta


def create_pickup_features():
    """
    STREAMING approach: Process one window at a time, write immediately
    """
    try:
        setup_database_indexes()
        engine = create_engine(DB_URL, pool_pre_ping=True)

        print("📚 [1/3] Getting date range...")
        min_date, max_date = pd.read_sql(
            "SELECT MIN(timestamp), MAX(timestamp) FROM taxi_probe_raw", engine
        ).iloc[0]

        window_starts = pd.date_range(
            start=min_date,
            end=max_date
            - pd.to_timedelta(WINDOW_LENGTH)
            - pd.to_timedelta(FORECAST_HORIZON),
            freq=WINDOW_STEP,
        )

        total_windows = len(window_starts)
        print(f"\n🔄 [2/3] Processing {total_windows} windows SEQUENTIALLY...")
        print(f"   Strategy: One window at a time (memory-safe)")

        window_config = {
            "WINDOW_LENGTH": WINDOW_LENGTH,
            "FORECAST_HORIZON": FORECAST_HORIZON,
        }

        first_write = True
        windows_processed = 0

        for i, window_start in enumerate(window_starts, 1):
            print(f"  [{i}/{total_windows}] {window_start.date()}...", end=" ")
            result = process_window_streaming(window_start, window_config, engine)

            if result is None:
                print("❌ No data")
                continue

            X, y, meta = result

            if first_write:
                X.to_csv(X_FILE, mode="w", index=False)
                y.to_csv(Y_FILE, mode="w", index=False)
                meta.to_csv(META_FILE, mode="w", index=False)
                first_write = False
            else:
                X.to_csv(X_FILE, mode="a", index=False, header=False)
                y.to_csv(Y_FILE, mode="a", index=False, header=False)
                meta.to_csv(META_FILE, mode="a", index=False, header=False)

            windows_processed += 1
            print(f"✓ ({len(X)} rows)")

            del X, y, meta, result
            gc.collect()

            if i % 10 == 0:
                print(f"  Progress: {i}/{total_windows} ({i*100//total_windows}%)")

        engine.dispose()

        if windows_processed == 0:
            print("❌ No data processed. Exiting.")
            return

        print(
            f"\n✅ [3/3] Successfully processed {windows_processed}/{total_windows} windows"
        )
        print(f"📦 Files saved:")
        print(f"  - {X_FILE}")
        print(f"  - {Y_FILE}")
        print(f"  - {META_FILE}")

        import os

        for fname in [X_FILE, Y_FILE, META_FILE]:
            if os.path.exists(fname):
                size_mb = os.path.getsize(fname) / 1024 / 1024
                print(f"    {fname}: {size_mb:.1f} MB")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("🍎 M2 MacBook Air Optimized Mode")
    print("=" * 50)
    create_pickup_features()
