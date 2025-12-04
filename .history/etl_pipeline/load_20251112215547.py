import io
from sqlalchemy import create_engine


def load_probe_data(df):
    engine = create_engine(
        "postgresql+psycopg2://postgres:mypassword@localhost:5432/bangkok_taxi_db"
    )

    print(f"🚀 Preparing {len(df)} rows for high-speed COPY...")

    # Create an in-memory "file"
    buffer = io.StringIO()
    df.to_csv(buffer, sep="\t", header=False, index=False)
    buffer.seek(0)

    # Get the low-level psycopg2 connection.
    # We CANNOT use a 'with' statement here, so we use try/finally.
    conn = engine.raw_connection()

    try:
        # The cursor *can* be used with a 'with' statement
        with conn.cursor() as cursor:
            print("⚡ Executing native PostgreSQL COPY command...")

            cursor.copy_expert(
                sql=f"COPY taxi_probe_raw ({','.join(df.columns)}) FROM STDIN WITH (FORMAT 'text', DELIMITER E'\\t')",
                file=buffer,
            )

        # If no errors, commit the transaction
        conn.commit()
        print(f"✅ Successfully loaded {len(df)} rows into PostgreSQL!")

    except Exception as e:
        # If an error occurs, roll back the transaction
        conn.rollback()
        print(f"❌ Error during COPY: {e}")
        raise

    finally:
        # ALWAYS close the connection to return it to the pool
        conn.close()
