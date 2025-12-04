import pandas as pd
import os


def extract_all_raw():
    raw_dir = "data/raw"
    all_dfs = []

    for file in os.listdir(raw_dir):
        if file.endswith(".csv"):
            path = os.path.join(raw_dir, file)
            print(f"📥 Extracting {path}")
            df = pd.read_csv(path, header=None)
            df.columns = [
                "vehicle_id",
                "gps_valid",
                "latitude",
                "longitude",
                "timestamp",
                "speed",
                "heading",
                "for_hire_light",
                "engine_acc",
            ]
            all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Extracted total {len(full_df)} rows from {len(all_dfs)} files")
    return full_df
