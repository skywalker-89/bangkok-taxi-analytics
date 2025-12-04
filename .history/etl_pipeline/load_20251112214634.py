import io
from sqlalchemy import create_engine


def load_probe_data(df):
    engine = create_engine(
        "postgresql+psycopg2://postgres:mypassword@localhost:5432/bangkok_taxi_db"
    )

    print(f"🚀 Preparing {len(df)} rows for high-speed COPY...")

    # Create an in-memory "file"
    buffer = io.StringIO()

    # Write the DataFrame to the in-memory file as a CSV
    # We use a tab separator for safety, and no header/index
    df.to_csv(buffer, sep="\t", header=False, index=False)

    # "Rewind" the in-memory file to the beginning
    buffer.seek(0)

    # Get the low-level psycopg2 connection from the SQLAlchemy engine
    with engine.raw_connection() as conn:
        try:
            with conn.cursor() as cursor:
                print("⚡ Executing native PostgreSQL COPY command...")

                # Use copy_expert to stream the data
                # We specify the table, the delimiter (tab), and the columns
                cursor.copy_expert(
                    sql=f"COPY taxi_probe_raw ({','.join(df.columns)}) FROM STDIN WITH (FORMAT 'text', DELIMITER E'\\t')",
                    file=buffer,
                )

            # Commit the transaction
            conn.commit()
            print(f"✅ Successfully loaded {len(df)} rows into PostgreSQL!")

        except Exception as e:
            conn.rollback()
            print(f"❌ Error during COPY: {e}")
            raise
