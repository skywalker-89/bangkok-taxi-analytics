import io
from sqlalchemy import create_engine
import psycopg2


def load_probe_data(df):
    """Ultra-fast PostgreSQL loading with optimizations"""

    print(f"🚀 Preparing {len(df):,} rows for high-speed COPY...")

    # Direct psycopg2 connection (faster than SQLAlchemy)
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="bangkok_taxi_db",
        user="jul",
        password="mypassword",
    )

    try:
        cursor = conn.cursor()

        # Optimize PostgreSQL settings for bulk load
        print("⚙️ Optimizing PostgreSQL for bulk insert...")
        cursor.execute("SET synchronous_commit = OFF;")  # Huge speed boost
        cursor.execute("SET maintenance_work_mem = '256MB';")

        print("⚡ Truncating table...")
        cursor.execute("TRUNCATE TABLE taxi_probe_raw;")

        # Drop indexes before loading (recreate after)
        print("🔓 Dropping indexes temporarily...")
        cursor.execute(
            """
            DO $$ 
            DECLARE r RECORD;
            BEGIN
                FOR r IN (
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'taxi_probe_raw' 
                    AND indexname != 'taxi_probe_raw_pkey'
                ) LOOP
                    EXECUTE 'DROP INDEX IF EXISTS ' || r.indexname;
                END LOOP;
            END $$;
        """
        )

        print("⚡ Executing PostgreSQL COPY (this is the fastest method)...")

        # Convert to CSV buffer (optimized)
        buffer = io.StringIO()
        df.to_csv(
            buffer,
            sep="\t",
            header=False,
            index=False,
            na_rep="\\N",  # PostgreSQL NULL representation
            escapechar="\\",
        )
        buffer.seek(0)

        # COPY from buffer
        cursor.copy_expert(
            sql=f"""
                COPY taxi_probe_raw ({",".join(df.columns)})
                FROM STDIN WITH (FORMAT text, DELIMITER E'\\t', NULL '\\N')
            """,
            file=buffer,
        )

        conn.commit()
        print(f"✅ Loaded {len(df):,} rows!")

        # Recreate indexes
        print("🔨 Recreating indexes (this may take 5-10 minutes)...")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_vehicle_timestamp 
            ON taxi_probe_raw(vehicle_id, timestamp);
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON taxi_probe_raw(timestamp);
        """
        )

        conn.commit()
        print("✅ Indexes recreated!")

        # Reset settings
        cursor.execute("SET synchronous_commit = ON;")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error during COPY: {e}")
        raise

    finally:
        cursor.close()
        conn.close()


def load_probe_data_chunked(df, chunk_size=500_000):
    """
    Alternative: Load in chunks to avoid memory issues with giant DataFrames.
    Use this if df is > 5 million rows.
    """

    print(f"🚀 Preparing {len(df):,} rows for chunked high-speed COPY...")

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="bangkok_taxi_db",
        user="jul",
        password="mypassword",
    )

    try:
        cursor = conn.cursor()

        # Optimize settings
        cursor.execute("SET synchronous_commit = OFF;")
        cursor.execute("SET maintenance_work_mem = '256MB';")

        print("⚡ Truncating table...")
        cursor.execute("TRUNCATE TABLE taxi_probe_raw;")

        # Drop indexes
        print("🔓 Dropping indexes temporarily...")
        cursor.execute(
            """
            DO $$ 
            DECLARE r RECORD;
            BEGIN
                FOR r IN (
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'taxi_probe_raw' 
                    AND indexname != 'taxi_probe_raw_pkey'
                ) LOOP
                    EXECUTE 'DROP INDEX IF EXISTS ' || r.indexname;
                END LOOP;
            END $$;
        """
        )

        # Load in chunks
        n_chunks = len(df) // chunk_size + 1
        print(f"⚡ Loading {n_chunks} chunks...")

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size]

            buffer = io.StringIO()
            chunk.to_csv(
                buffer,
                sep="\t",
                header=False,
                index=False,
                na_rep="\\N",
                escapechar="\\",
            )
            buffer.seek(0)

            cursor.copy_expert(
                sql=f"""
                    COPY taxi_probe_raw ({",".join(df.columns)})
                    FROM STDIN WITH (FORMAT text, DELIMITER E'\\t', NULL '\\N')
                """,
                file=buffer,
            )

            chunk_num = i // chunk_size + 1
            if chunk_num % 5 == 0:
                print(f"  Loaded chunk {chunk_num}/{n_chunks} ({i:,}/{len(df):,} rows)")

        conn.commit()
        print(f"✅ Loaded {len(df):,} rows!")

        # Recreate indexes
        print("🔨 Recreating indexes (this may take 5-10 minutes)...")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_vehicle_timestamp 
            ON taxi_probe_raw(vehicle_id, timestamp);
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON taxi_probe_raw(timestamp);
        """
        )

        conn.commit()
        print("✅ Indexes recreated!")

        cursor.execute("SET synchronous_commit = ON;")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error during COPY: {e}")
        raise

    finally:
        cursor.close()
        conn.close()
