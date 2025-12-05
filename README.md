# Bangkok Taxi Route Optimization Platform 🚖

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-0194E2?logo=mlflow&logoColor=white)
![Prefect](https://img.shields.io/badge/Prefect-Orchestration-white?logo=prefect)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Data-336791?logo=postgresql&logoColor=white)

An End-to-End MLOps Platform designed to optimize taxi routes in Bangkok. This system leverages historical GPS probe data, XGBoost predictive modeling, and Monte Carlo simulations to recommend optimal driving routes that maximize revenue based on predicted passenger demand and traffic conditions.

## 📖 Project Overview

Taxi drivers often face the **"Empty Leg" problem**—driving without passengers. This platform solves that by providing a decision-support system that predicts:

1.  **Next Destination:** Where a passenger is likely to want to go from the current location.
2.  **Trip Duration:** Accurate travel times accounting for Bangkok's traffic.
3.  **Revenue:** Estimated fare calculation based on distance and time.

The system serves these predictions via a Dockerized Web Application featuring an interactive map, allowing drivers to simulate shifts and identify high-value routes.

---

## 🏗 System Architecture

The project follows a modern microservices architecture, fully containerized for reproducibility:

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Data Layer** | **PostgreSQL** | Stores millions of raw GPS probe points and engineered feature tables. |
| **Orchestration** | **Prefect** | Automated pipeline that extracts raw data, cleans it, and executes feature engineering tasks. |
| **Spatial Indexing** | **Uber H3** | Uses Resolution 8 hexagonal indexing to discretize GPS coordinates into spatial zones. |
| **Model Registry** | **MLflow** | Tracks experiments, logs metrics (MAE, Accuracy), and stores artifacts (encoders, models) for deployment. |
| **Serving** | **Flask API** | Loads production models from MLflow and serves endpoints. |
| **Frontend** | **Leaflet.js** | Interactive map with **OSRM** (Open Source Routing Machine) routing. |

### 🤖 The Models (XGBoost)
* **Next Destination:** Multi-class classification model (Top-100 high-density zones).
* **Trip Duration:** Regression model predicting travel time in minutes.
* **Trip Distance:** Regression model correcting Haversine distance to actual road distance.

---

## 🌟 Key Features

* **🎲 Monte Carlo Route Simulation**
    Runs hundreds of probabilistic simulations per request to find the "Golden Route" with the highest expected revenue.

* **🔄 Automated Retraining Pipeline**
    A CI/CD workflow (via GitHub Actions) checks for new monthly data, retrains models, and promotes them to production only if they beat the current champion.

* **🛠 Production-Grade MLOps**
    Implements best practices like artifact versioning, experiment tracking, and containerization.

* **🗺 Real-World Routing**
    Integrates OSRM to visualize actual driving paths on the map rather than simple straight lines.

---

## 🚀 Getting Started

### Prerequisites
* Docker & Docker Compose
* Python 3.11+
* Poetry (for local dependency management)

### Installation & Run

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/skywalker-89/bangkok-taxi-analytics.git](https://github.com/skywalker-89/bangkok-taxi-analytics.git)
    cd bangkok-taxi-analytics
    ```

2.  **Start Infrastructure**
    Spin up the Database, MLflow Server, API, and Web App using Docker Compose.
    ```bash
    docker-compose up -d --build
    ```

3.  **Run the Data Pipeline**
    Execute the ETL and Training workflows to populate the database and train the initial models.
    ```bash
    # Install dependencies
    pip install poetry && poetry install

    # Run the Prefect flow
    poetry run python -m flows.main_flow
    ```

---
