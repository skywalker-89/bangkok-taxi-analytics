import pandas as pd


def transform_probe_data(df):
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Remove invalid GPS fixes or timestamps
    df = df[df["gps_valid"] == 1]
    df = df.dropna(subset=["latitude", "longitude", "timestamp"])

    # Sort by time for each vehicle
    df = df.sort_values(by=["vehicle_id", "timestamp"])

    # Optional: remove duplicates
    df = df.drop_duplicates()

    print(f"✅ Cleaned data: {len(df)} valid rows remaining")
    return df
