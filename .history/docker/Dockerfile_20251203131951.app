FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# ✅ FIX: Use MLflow 2.12.1
RUN pip install --default-timeout=1000 --no-cache-dir \
    pandas \
    numpy \
    xgboost \
    mlflow==2.12.1 \
    flask \
    fastapi \
    uvicorn \
    python-dotenv \
    joblib \
    scikit-learn \
    psycopg2-binary \
    pydantic

COPY src/ src/
COPY flows/ flows/

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]