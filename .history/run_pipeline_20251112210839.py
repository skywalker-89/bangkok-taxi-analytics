from etl_pipeline.extract import extract_all_raw
from etl_pipeline.transform import transform_probe_data
from etl_pipeline.load import load_probe_data

if __name__ == "__main__":
    df_raw = extract_all_raw()
    df_clean = transform_probe_data(df_raw)
    load_probe_data(df_clean)
    print("✅ Pipeline completed successfully!")
