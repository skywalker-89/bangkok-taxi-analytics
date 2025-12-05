# 🚖 Bangkok Taxi Route Optimization Platform

An end-to-end **MLOps-driven decision support system** for optimizing taxi routes in Bangkok.  
This platform processes large-scale GPS probe data, trains predictive ML models, and serves real-time recommendations that help taxi drivers **maximize revenue** by reducing empty driving time.

---

## 📖 Overview

Taxi drivers commonly face the **"Empty Leg"** problem — driving without passengers.  
This project solves that by predicting:

- **Next Destination** — Likely passenger destinations based on current location
- **Trip Duration** — Travel time adjusted for real Bangkok traffic
- **Trip Distance** — More accurate road distance instead of raw Haversine
- **Estimated Revenue** — Based on trip distance, duration, and fare models

A fully Dockerized web application exposes these predictions using an **interactive Leaflet map** and a **Monte Carlo-based route optimizer** that simulates multiple potential routes to identify the highest expected revenue.

---

## 🏗 System Architecture

This project follows a **modern microservices architecture**, fully containerized for reproducibility and deployment.

### **1. Data Layer — PostgreSQL**
- Stores millions of historical GPS probe points  
- Feature tables indexed using **Uber H3 (Resolution 8)**

### **2. ETL & Orchestration — Prefect**
- Automated data cleaning  
- Spatial feature engineering  
- Monthly scheduled retraining triggers

### **3. Machine Learning Models — XGBoost**
- **Next Destination:** Multi-class classifier (Top 100 H3 zones)  
- **Trip Duration:** Regression model predicting travel time  
- **Trip Distance:** Regression model correcting Haversine distances

### **4. Model Registry — MLflow**
- Tracks experiments & metrics  
- Stores:  
  - Production-ready models  
  - Encoders  
  - Spatial metadata

### **5. Serving Layer — Flask API**
- Loads production models from MLflow  
- Exposes prediction and simulation endpoints  
- Performs Monte Carlo route search

### **6. Frontend — Leaflet.js + OSRM**
- Interactive map UI  
- Real-world road routing (not straight lines)  
- Visualizes optimal "Golden Routes" for drivers

---

## 🌟 Key Features

### **🔁 Automated Retraining Workflow**
- GitHub Actions checks for new monthly data  
- Retrains models automatically  
- Promotes new models only if they outperform the current champion

### **🎲 Monte Carlo Route Simulation**
- Hundreds of probabilistic route simulations per request  
- Computes revenue distributions  
- Recommends the route with the **highest expected value**

### **📦 Production-Grade MLOps**
- Fully containerized infrastructure  
- Artifact versioning via MLflow  
- Reproducible ETL pipelines  
- Seamless dev → staging → production workflow

### **🗺 Real-World Routing**
- Integrates **OSRM** for driving paths  
- Visual route overlays on Leaflet map  
- Accurate travel time estimates

---

## 🚀 Getting Started

### **Prerequisites**
- Docker & Docker Compose  
- Python 3.11+  
- Poetry (optional, for local development)

---

## 🛠 Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/skywalker-89/bangkok-taxi-analytics.git
cd bangkok-taxi-analytics
