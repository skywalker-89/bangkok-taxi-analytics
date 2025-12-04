import pandas as pd
import numpy as np
import h3
import sys
import gc
from datetime import timedelta
from src.utils.db import get_engine, get_db_connection

# --- Config ---
FEATURE_TABLE_NAME = "features_pickup_rate"
H3_RESOLUTION = 8
TIME_BIN_MINUTES = 5
WINDOW_LENGTH = "7D"
WINDOW_STEP = "2D"


def prepare_pickup_rate_data():
    engine = get_engine()

    # 1. Clear old table if exists
    print(f"🧹 Clearing old table {FEATURE_TABLE_NAME}...")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {FEATURE_TABLE_NAME}")
            conn.commit()

    try:
        # 2. Sliding Window Loop (Simplified)
        # In reality, you'd loop through your date range here

        # Example Loop Structure
        start_date = pd.Timestamp("2019-01-01")  # Replace with actual start
        end_date = pd.Timestamp("2019-01-30")  # Replace with actual end
        current_date = start_date

        while current_date < end_date:
            window_end = current_date + pd.Timedelta(WINDOW_LENGTH)
            print(f"🔄 Processing window: {current_date} to {window_end}")

            # ... (Your logic to fetch data, bin it, and create lag features) ...
            # ... (Let's call the result 'df_batch') ...

            # Example Placeholder DataFrame
            # df_batch = ...

            # 3. Save Batch to SQL
            if "df_batch" in locals():
                print(f"  💾 Appending {len(df_batch)} rows...")
                df_batch.to_sql(
                    FEATURE_TABLE_NAME,
                    engine,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=2000,
                )

            current_date += pd.Timedelta(WINDOW_STEP)
            gc.collect()

        print(f"✅ Success! Data stored in {FEATURE_TABLE_NAME}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    prepare_pickup_rate_data()
