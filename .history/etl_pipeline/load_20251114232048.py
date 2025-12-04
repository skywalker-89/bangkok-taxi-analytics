import io
from sqlalchemy import create_engine


def load_probe_data(df):
    engine = create_engine(
        "postgresql+psycopg2://jul:mypassword@localhost:5432/bangkok_taxi_db"
    )

    print(f"🚀 Preparing {len(df)} rows for high-speed COPY...")

    # Convert df → temp text buffer
    buffer = io.StringIO()
    df.to_csv(buffer, sep="\t", header=False, index=False)
    buffer.seek(0)

    conn = engine.raw_connection()

    try:
        with conn.cursor() as cursor:
            print("⚡ Truncating table to avoid duplicates...")
            cursor.execute("TRUNCATE TABLE taxi_probe_raw;")

            print("⚡ Executing PostgreSQL COPY...")
            cursor.copy_expert(
                sql=f"""
                    COPY taxi_probe_raw ({",".join(df.columns)})
                    FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')
                """,
                file=buffer,
            )

        conn.commit()
        print(f"✅ Successfully loaded {len(df)} fresh rows into taxi_probe_raw!")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error during COPY: {e}")
        raise

    finally:
        conn.close()
