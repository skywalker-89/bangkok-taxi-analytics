# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps & wait-for-it to wait for services
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl gnupg && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Entrypoint for running the FastAPI app via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
