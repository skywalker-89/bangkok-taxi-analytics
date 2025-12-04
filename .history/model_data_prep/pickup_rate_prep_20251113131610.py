# model_data_prep/pickup_rate_prep.py (NEW FAST VERSION)

import pandas as pd
from sqlalchemy import create_engine
import sys

# --- Config ---
DB_URL = "postgresql+psycopg2://postgres:mypassword@localhost:5432/bangkok_taxi_db"

# --- Output File Names ---
X_FILE = "X_features_5min.csv"
Y_FILE = "y_target_5min.csv"
META_FILE = "meta_5min.csv"

# --- Feature Config (bins) ---
BINS_PER_HOUR = 12  # 60 min / 5 min bins
BINS_PER_DAY = 288  # 24 hours * 12 bins/hr


def create_features_from_sql():
    """
    Builds the entire feature/target set using a single,
    powerful SQL query. No more Python loops.
    """
    try:
        engine = create_engine(DB_URL)
        print("📚 [1/3] Connecting to database...")

        # This SQL query does ALL the work that the old script did.
        # It builds lags, rolling averages, time features, and the target.
        feature_query = f"""
        WITH 
        base_grid AS (
            -- Create the full grid of all cells and all bins
            SELECT *
            FROM (SELECT DISTINCT h3_cell FROM pickups_binned_5min) AS cells
            CROSS JOIN (SELECT DISTINCT time_bin FROM pickups_binned_5min) AS bins
        ),
        binned_pickups_filled AS (
            -- Fill the grid with 0s for non-pickup times
            SELECT
                b.h3_cell,
                b.time_bin,
                COALESCE(p.pickup_count, 0) AS pickup_count
            FROM base_grid b
            LEFT JOIN pickups_binned_5min p 
              ON b.h3_cell = p.h3_cell AND b.time_bin = p.time_bin
        ),
        features_and_target AS (
            -- This is the magic.
            -- Engineer all features and the target in one pass.
            SELECT
                h3_cell,
                time_bin,
                
                -- Time Features
                EXTRACT(dow FROM time_bin) AS dow,
                EXTRACT(hour FROM time_bin) AS hour,
                (EXTRACT(hour FROM time_bin) * 60 + 
                 EXTRACT(minute FROM time_bin)) AS minute_of_day,
                
                -- Lag Features
                LAG(pickup_count, {BINS_PER_HOUR}) OVER w AS pickup_count_lag_1hr,
                LAG(pickup_count, {2 * BINS_PER_HOUR}) OVER w AS pickup_count_lag_2hr,
                LAG(pickup_count, {BINS_PER_DAY}) OVER w AS pickup_count_lag_1day,
                
                -- Rolling Features
                AVG(pickup_count) OVER (w ROWS BETWEEN {BINS_PER_HOUR - 1} PRECEDING AND CURRENT ROW) 
                  AS pickup_count_roll_avg_1hr,
                AVG(pickup_count) OVER (w ROWS BETWEEN {6 * BINS_PER_HOUR - 1} PRECEDING AND CURRENT ROW) 
                  AS pickup_count_roll_avg_6hr,
                AVG(pickup_count) OVER (w ROWS BETWEEN {BINS_PER_DAY - 1} PRECEDING AND CURRENT ROW) 
                  AS pickup_count_roll_avg_1day,
                
                -- Target (y)
                -- Was there at least 1 pickup in the *next* hour?
                (SUM(pickup_count) OVER (w ROWS BETWEEN 1 FOLLOWING AND {BINS_PER_HOUR} FOLLOWING) > 0)::int 
                  AS y
            
            FROM binned_pickups_filled
            -- Define the 'window' for all window functions
            WINDOW w AS (PARTITION BY h3_cell ORDER BY time_bin)
        )
        -- Final select, dropping rows where lags/rolls are null
        SELECT *
        FROM features_and_target
        WHERE
            pickup_count_lag_1day IS NOT NULL
            AND pickup_count_roll_avg_1day IS NOT NULL
        """

        print("🚀 [2/3] Executing SQL feature engineering query...")
        print("   (This will take a few minutes, but it's *much* faster than Python)")

        df = pd.read_sql(feature_query, engine)

        print(f"✅ Query complete! Generated {len(df)} feature rows.")

        print(f"💾 [3/3] Saving to {X_FILE}, {Y_FILE}, {META_FILE}...")

        # Split into X, y, and meta
        meta_cols = ["h3_cell", "time_bin"]
        target_col = "y"

        meta_df = df[meta_cols]
        y_df = df[target_col]

        # X is everything else
        X_df = df.drop(columns=meta_cols + [target_col])

        # Save to files
        X_df.to_csv(X_FILE, index=False)
        y_df.to_csv(Y_FILE, index=False)
        meta_df.to_csv(META_FILE, index=False)

        print("\n✅ Data preparation complete!")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    create_features_from_sql()
