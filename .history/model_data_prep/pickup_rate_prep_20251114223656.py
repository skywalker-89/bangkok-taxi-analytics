import pandas as pd
import numpy as np
import h3
from sqlalchemy import create_engine
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
FORECAST_HORIZON = "1H"

# --- Output File Names ---
X_FILE = "X_features_5min.csv"
Y_FILE = "y_target_5min.csv"
META_FILE = "meta_5min.csv"

# --- NEW: Parallelization ---
N_WORKERS = max(1, cpu_count() - 1)  # Leave 1 core free


def vectorized_h3(lats, lons, resolution):
    """Vectorized H3 conversion - MUCH faster than apply()"""
    return [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(lats, lons)]


def add_time_features(df):
    """Optimized time features - vectorized operations"""
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
        rolling_cols[f"pickup_count_roll_avg_{window}"] = grouped.rolling(
            window, min_periods=min_p
        ).mean()

    df_rolls = pd.DataFrame(rolling_cols, index=df.index).reset_index()
    return df_rolls


def process_window(args):
    """
    Process a single window - refactored for multiprocessing.
    Args is a tuple: (window_start, window_config, db_url)
    """
    window_start, window_config, db_url = args

    # Each process needs its own engine
    engine = create_engine(db_url)

    w_len = pd.to_timedelta(window_config["WINDOW_LENGTH"])
    f_hor = pd.to_timedelta(window_config["FORECAST_HORIZON"])

    train_start = window_start
    train_end = window_start + w_len
    forecast_start = train_end
    forecast_end = train_end + f_hor
    query_end = forecast_end

    print(f"Processing: {train_start.date()}")

    # OPTIMIZATION 1: Query only necessary columns
    query = f"""
        SELECT vehicle_id, latitude, longitude, timestamp, for_hire_light
        FROM taxi_probe_raw
        WHERE timestamp >= '{train_start}' AND timestamp < '{query_end}'
    """
    df = pd.read_sql(query, engine, parse_dates=["timestamp"])

    engine.dispose()  # Close connection

    if df.empty:
        return None, None, None

    # OPTIMIZATION 2: Vectorized operations for pickup detection
    df = df.sort_values(by=["vehicle_id", "timestamp"])
    df["prev_light"] = df.groupby("vehicle_id")["for_hire_light"].shift(1)
    df["is_new_trip"] = (df["for_hire_light"] == 1) & (df["prev_light"] == 0)
    pickups_df = df[df["is_new_trip"]].copy()

    if len(pickups_df) == 0:
        return None, None, None

    # OPTIMIZATION 3: Vectorized H3 conversion
    pickups_df["h3_cell"] = vectorized_h3(
        pickups_df["latitude"].values, pickups_df["longitude"].values, H3_RESOLUTION
    )

    # OPTIMIZATION 4: Use floor division for binning (faster)
    pickups_df["time_bin"] = pickups_df["timestamp"].dt.floor(f"{TIME_BIN_MINUTES}min")

    # Create base grid
    all_bins = pd.date_range(
        start=train_start,
        end=query_end,
        freq=f"{TIME_BIN_MINUTES}min",
        inclusive="left",
    )
    all_cells = pickups_df["h3_cell"].unique()

    if len(all_cells) == 0:
        return None, None, None

    # OPTIMIZATION 5: More efficient grid creation
    base_df = pd.DataFrame(
        {
            "h3_cell": np.repeat(all_cells, len(all_bins)),
            "time_bin": np.tile(all_bins, len(all_cells)),
        }
    )

    # Get pickup counts
    pickup_counts = (
        pickups_df.groupby(["h3_cell", "time_bin"], observed=True)
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
    forecast_pickups = pickups_df[
        (pickups_df["time_bin"] >= forecast_start)
        & (pickups_df["time_bin"] < forecast_end)
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
    """Main function with parallel processing"""
    try:
        engine = create_engine(DB_URL)
        print("📚 [1/4] Connected to database.")

        # OPTIMIZATION 6: Get date range efficiently
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
            f"🔄 [2/4] Processing {len(window_starts)} windows with {N_WORKERS} workers..."
        )

        # OPTIMIZATION 7: Parallel processing
        args_list = [(start, window_config, DB_URL) for start in window_starts]

        with Pool(N_WORKERS) as pool:
            results = pool.map(process_window, args_list)

        # Filter out None results
        results = [r for r in results if r[0] is not None]

        if not results:
            print("❌ No data processed. Exiting.")
            return

        print(f"\n✅ [3/4] Processed {len(results)} windows successfully")
        print("📦 [4/4] Combining and saving results...")

        all_X, all_y, all_meta = zip(*results)

        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        meta_combined = pd.concat(all_meta, ignore_index=True)

        # OPTIMIZATION 8: Optimize dtypes before saving
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
