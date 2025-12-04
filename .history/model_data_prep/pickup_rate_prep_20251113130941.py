import pandas as pd
import numpy as np
import h3
from sqlalchemy import create_engine
import sys
from datetime import timedelta

# --- Config ---
DB_URL = "postgresql+psycopg2://postgres:mypassword@localhost:5432/bangkok_taxi_db"
H3_RESOLUTION = 9  # From your notebook (Res 9)
TIME_BIN_MINUTES = 5  # From your notebook

# --- Sliding Window Config ---
WINDOW_LENGTH = "14D"  # Use 14 days of data...
WINDOW_STEP = "1D"  # ...stepping forward 1 day at a time...
FORECAST_HORIZON = "1H"  # ...to predict 1 hour in the future.

# --- Output File Names ---
X_FILE = "X_features_5min.csv"
Y_FILE = "y_target_5min.csv"
META_FILE = "meta_5min.csv"


def add_time_features(df):
    """Adds dow, hour, minute_of_day features from time_bin."""
    df["dow"] = df["time_bin"].dt.dayofweek
    df["hour"] = df["time_bin"].dt.hour
    df["minute_of_day"] = df["hour"] * 60 + df["time_bin"].dt.minute
    return df


def get_lagged_features(df, lags):
    """Creates lagged pickup_count features."""
    df = df.set_index(["h3_cell", "time_bin"])
    lagged_dfs = []
    for lag in lags:
        lag_col = f"pickup_count_lag_{lag}"
        lagged = df.groupby(level="h3_cell")["pickup_count"].shift(lag).rename(lag_col)
        lagged_dfs.append(lagged)

    df_lags = pd.concat(lagged_dfs, axis=1)
    df_lags = df_lags.reset_index()
    return df_lags


def get_rolling_features(df, windows):
    """Creates rolling avg pickup_count features."""
    df = df.set_index(["h3_cell", "time_bin"]).sort_index()
    rolling_dfs = []
    for window, min_p in windows:
        roll_col = f"pickup_count_roll_avg_{window}"
        # Group by h3_cell, then apply rolling
        grouped = df.groupby(level="h3_cell")["pickup_count"]
        rolled = grouped.rolling(window, min_periods=min_p).mean().rename(roll_col)
        # Drop h3_cell index to re-merge
        rolled = rolled.reset_index(level=0, drop=True)
        rolling_dfs.append(rolled)

    df_rolls = pd.concat(rolling_dfs, axis=1)
    df_rolls = df_rolls.reset_index()
    return df_rolls


def process_window(window_start, window_config, engine):
    """
    Processes a single time window.
    Queries data, finds pickups, creates base grid,
    engineers features, and builds target.
    """
    # 1. Define time slices
    w_len = pd.to_timedelta(window_config["WINDOW_LENGTH"])
    f_hor = pd.to_timedelta(window_config["FORECAST_HORIZON"])

    train_start = window_start
    train_end = window_start + w_len
    forecast_start = train_end
    forecast_end = train_end + f_hor

    query_end = forecast_end

    print(f"--- Processing window: {train_start} -> {train_end} ---")

    # 2. Query DB for this window + forecast
    query = f"""
        SELECT vehicle_id, latitude, longitude, timestamp, for_hire_light
        FROM taxi_probe_raw
        WHERE timestamp >= '{train_start}' AND timestamp < '{query_end}'
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        print("  No data in window.")
        return None, None, None

    # 3. Find pickup events
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["vehicle_id", "timestamp"])
    df["is_new_trip"] = (df["for_hire_light"] == 1) & (
        df["for_hire_light"].shift(1) == 0
    )
    pickups_df = df[df["is_new_trip"]].copy()

    # 4. Grid data into H3 cells and 5-min time bins
    pickups_df["h3_cell"] = pickups_df.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], H3_RESOLUTION),
        axis=1,
    )
    pickups_df["time_bin"] = pickups_df["timestamp"].dt.floor(f"{TIME_BIN_MINUTES}min")

    # 5. Create base grid (all h3 cells and time bins)
    all_bins = pd.date_range(
        start=train_start,
        end=query_end,
        freq=f"{TIME_BIN_MINUTES}min",
        inclusive="left",
    )
    all_cells = pickups_df["h3_cell"].unique()

    if len(all_cells) == 0:
        print("  No pickups in window.")
        return None, None, None

    base_grid = pd.MultiIndex.from_product(
        [all_cells, all_bins], names=["h3_cell", "time_bin"]
    )
    base_df = pd.DataFrame(index=base_grid).reset_index()

    # 6. Get pickup counts per cell/bin
    pickup_counts = (
        pickups_df.groupby(["h3_cell", "time_bin"]).size().rename("pickup_count")
    )
    base_df = base_df.merge(
        pickup_counts, on=["h3_cell", "time_bin"], how="left"
    ).fillna(0)

    # 7. Engineer Features

    # a. Time features
    base_df = add_time_features(base_df)

    # b. Lagged features (12 = 1hr, 24 = 2hr, 288 = 1 day)
    bins_per_hour = 60 // TIME_BIN_MINUTES
    bins_per_day = 24 * bins_per_hour
    lags = [bins_per_hour, 2 * bins_per_hour, bins_per_day]
    lags_df = get_lagged_features(base_df, lags)

    # c. Rolling features (1hr, 6hr, 1day)
    windows = [(bins_per_hour, 2), (6 * bins_per_hour, 6), (bins_per_day, 12)]
    rolls_df = get_rolling_features(base_df, windows)

    # 8. Combine all features
    features_df = lags_df.merge(rolls_df, on=["h3_cell", "time_bin"])
    features_df = features_df.merge(
        base_df[["h3_cell", "time_bin", "dow", "hour", "minute_of_day"]],
        on=["h3_cell", "time_bin"],
    )

    # 9. Create Target (y)
    # Get all pickups in the *forecast* window
    forecast_pickups = pickups_df[
        (pickups_df["time_bin"] >= forecast_start)
        & (pickups_df["time_bin"] < forecast_end)
    ]

    # Target = was there at least 1 pickup in this cell during the forecast window?
    target_df = (
        forecast_pickups.groupby("h3_cell").size().ge(1).rename("y").reset_index()
    )

    # 10. Align X and y
    # We want features from *exactly* train_end, predicting for forecast_start -> forecast_end
    X = features_df[features_df["time_bin"] == train_end].copy()

    # Merge target
    X_with_target = X.merge(target_df, on="h3_cell", how="left").fillna({"y": 0})

    y = X_with_target["y"].astype(int)
    X = X_with_target.drop(columns=["y"])

    # Meta
    meta = X[["h3_cell", "time_bin"]]
    X = X.drop(columns=["h3_cell", "time_bin"])

    return X, y, meta


def create_pickup_features():
    """
    Main function to run the sliding window feature engineering.
    """
    try:
        engine = create_engine(DB_URL)
        print("📚 [1/3] Connected to database.")

        # Get overall date range
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

        all_X, all_y, all_meta = [], [], []

        window_config = {
            "WINDOW_LENGTH": WINDOW_LENGTH,
            "FORECAST_HORIZON": FORECAST_HORIZON,
        }

        print(f"Starting feature engineering for {len(window_starts)} windows...")
        for start_time in window_starts:
            X, y, meta = process_window(start_time, window_config, engine)
            if X is not None:
                all_X.append(X)
                all_y.append(y)
                all_meta.append(meta)

        if not all_X:
            print("❌ No data processed. Exiting.")
            return

        print("\n--- Combining all windows ---")
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        meta_combined = pd.concat(all_meta, ignore_index=True)

        print(f"💾 [3/3] Saving {len(X_combined)} rows to CSVs...")
        X_combined.to_csv(X_FILE, index=False)
        y_combined.to_csv(Y_FILE, index=False)
        meta_combined.to_csv(META_FILE, index=False)

        print("\n✅ Data preparation complete!")
        print(f"  - {X_FILE}")
        print(f"  - {Y_FILE}")
        print(f"  - {META_FILE}")

    except ImportError:
        print("❌ Error: 'h3' library not found. Please install: pip install h3")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    create_pickup_features()
