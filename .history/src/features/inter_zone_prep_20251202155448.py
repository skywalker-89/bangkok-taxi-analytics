import pandas as pd
import numpy as np
import h3
import sys
from src.utils.db import get_engine

FEATURE_TABLE_NAME = "features_inter_zone"
CITY_CENTER = (13.7563, 100.5018)


def haversine(lat1, lon1, lat2, lon2):
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


def prepare_inter_zone_data():
    try:
        engine = get_engine()
        print("📚 Loading raw trip data...")

        # Simplified query to fetch pre-calculated trips if you have a trips table,
        # otherwise replicate the trip generation logic from raw probes.
        # Assuming we can reuse the logic or read from a 'trips' table if it existed.
        # For this refactor, I will assume we read raw and aggregate fast.

        # NOTE: For brevity in this refactor, ensure this matches your logic for
        # identifying 'Empty Trips' (between dropoff and next pickup).

        # Placeholder for the complex 'Empty Trip' logic from your original file:
        # [ ... Insert Empty Trip Calculation Logic Here ... ]
        # Let's assume 'empty_trips' is the resulting DataFrame.

        # If running purely on the provided file, you need the full logic block here.
        # Since I cannot see the full logic in the snippet history for inter-zone specifically,
        # I will set up the structure. *You must copy your empty trip calculation logic here.*

        print("⚠️ NOTE: Ensure you pasted your 'Empty Trip' logic inside this function.")

        # ... (After Calculating features) ...
        # features = ["direct_distance_km", "bearing_sin", "bearing_cos", ...]
        # target = "travel_time_minutes"

        # For now, let's assume 'final_df' is ready to save
        # final_df.to_sql(FEATURE_TABLE_NAME, engine, if_exists="replace", ...)

        print("✅ (Inter-zone structure ready - Insert logic body)")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    prepare_inter_zone_data()
