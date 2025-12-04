# src/utils/db.py
import os
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load secrets from .env automatically
load_dotenv()


def get_db_url():
    """Returns the SQLAlchemy connection string."""
    return f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"


def get_db_connection():
    """Returns a raw psycopg2 connection object."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
    )


def get_engine():
    """Returns a SQLAlchemy engine."""
    return create_engine(get_db_url())
