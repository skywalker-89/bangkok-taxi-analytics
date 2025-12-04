from sqlalchemy import create_engine


def load_probe_data(df):
    engine = create_engine(
        "postgresql+psycopg2://postgres:mypassword@localhost:5432/bangkok_taxi_db"
    )
    df.to_sql("taxi_probe_raw", engine, if_exists="append", index=False, chunksize=100000,  # <-- THIS IS THE MAGIC (100k rows per chunk)
        method='multi'     # <-- This makes the inserts much faster)
    print(f"🚀 Loaded {len(df)} rows into PostgreSQL")
 