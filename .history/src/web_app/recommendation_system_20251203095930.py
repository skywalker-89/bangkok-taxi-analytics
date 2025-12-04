import pandas as pd
import numpy as np
import h3
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
from datetime import datetime, timedelta
import os
from mlflow.tracking import MlflowClient


class BangkokTaxiOptimizer:
    """
    Route optimizer that loads trained models directly from MLflow.
    """

    def __init__(self):
        print("⏳ Connecting to MLflow to load models...")

        # 1. Setup MLflow Connection
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(self.mlflow_uri)
        client = MlflowClient()

        # 2. Load Models (Dynamic Fetch from Registry)
        try:
            self.dest_model = mlflow.xgboost.load_model(
                "models:/next_destination_model/None"
            )
            self.duration_model = mlflow.xgboost.load_model(
                "models:/trip_duration_model/None"
            )
            # self.distance_model = mlflow.xgboost.load_model("models:/trip_distance_model/None")
            print("✅ Models loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e

        # 3. Download & Load Encoders (Artifacts)
        print("⏳ Downloading feature encoders...")
        try:
            # Find the run ID of the latest version of next_destination_model
            # Note: get_latest_versions is deprecated in new MLflow, using search_model_versions is safer
            # or just grabbing the run_id from the loaded model metadata if possible.
            # For robustness, let's just grab the latest run for the experiment.

            exp = client.get_experiment_by_name("bangkok_taxi_next_destination")
            runs = client.search_runs(
                exp.experiment_id, order_by=["start_time DESC"], max_results=1
            )
            run_id = runs[0].info.run_id

            # Download artifacts locally
            client.download_artifacts(run_id, "dest_feature_encoder.pkl", ".")
            client.download_artifacts(run_id, "dest_target_encoder.pkl", ".")

            self.le_start = joblib.load("dest_feature_encoder.pkl")
            self.le_end = joblib.load("dest_target_encoder.pkl")
            print("✅ Encoders loaded.")

        except Exception as e:
            print(f"❌ Error loading encoders: {e}")
            # Fallback or crash
            raise e

    def h3_to_latlng(self, h3_addr):
        return h3.cell_to_latlng(h3_addr)

    def latlng_to_h3(self, lat, lng):
        return h3.latlng_to_cell(lat, lng, 8)  # Resolution 8 matches your training

    def predict_next_step(self, current_h3, current_time):
        """
        Predicts top 3 likely destinations from current location using the trained model.
        """
        # Prepare Input DataFrame (Must match training signature exactly)
        # We need to convert h3 string -> integer index using the encoder
        try:
            # Handle unseen zones safely
            if current_h3 not in self.le_start.classes_:
                return []
            start_idx = self.le_start.transform([current_h3])[0]
        except ValueError:
            return []

        # ✅ FIX: Use .weekday() instead of .dayofweek for datetime objects
        day_of_week = current_time.weekday()  # 0=Monday, 6=Sunday

        input_df = pd.DataFrame(
            [
                {
                    "h3_start_idx": start_idx,
                    "pickup_hour": current_time.hour,
                    "pickup_dayofweek": day_of_week,
                    "is_weekend": 1 if day_of_week >= 5 else 0,
                    "pickup_month": current_time.month,
                    "pickup_day": current_time.day,
                    # Add other features like distance_from_center if needed by model
                    # For inference, we use defaults or pre-calculate relative to center
                    "start_dist_from_center": 0.0,
                    "od_pair_historical_count": 0,
                    "origin_historical_count": 0,
                    "origin_to_dest_popularity": 0,
                }
            ]
        )

        # Predict Probabilities
        try:
            probas = self.dest_model.predict_proba(input_df)[0]

            # Get Top 3 Indices
            top3_indices = np.argsort(probas)[-3:][::-1]

            results = []
            for idx in top3_indices:
                # Decode index back to H3 string
                dest_h3 = self.le_end.inverse_transform([idx])[0]
                prob = probas[idx]
                results.append({"h3": dest_h3, "probability": float(prob)})

            return results
        except Exception as e:
            print(f"Prediction Error: {e}")
            return []

    def optimize_route(
        self,
        start_location,
        end_location,
        start_time,
        end_time,
        n_simulations=10,
        top_n=3,
    ):
        """
        Simplified Monte Carlo Simulation to find best routes.
        """
        start_h3 = self.latlng_to_h3(*start_location)
        # end_h3 = self.latlng_to_h3(*end_location) # Unused in this simulation logic for now

        routes = []

        print(f"🔄 Simulating {n_simulations} routes...")

        for i in range(n_simulations):
            current_h3 = start_h3
            current_time = start_time
            path = []
            total_revenue = 0

            # Simulate a sequence of 3 trips
            for _ in range(3):
                # 1. Predict Next Destination
                next_options = self.predict_next_step(current_h3, current_time)
                if not next_options:
                    break

                # Pick one weighted by probability (Monte Carlo)
                probs = [x["probability"] for x in next_options]
                # Normalize probs to sum to 1 (fix floating point errors)
                probs = np.array(probs)
                probs /= probs.sum()

                choice = np.random.choice(range(len(next_options)), p=probs)
                dest_h3 = next_options[choice]["h3"]

                # 2. Estimate Fare (Placeholder logic, replace with price model if available)
                estimated_fare = np.random.randint(50, 300)

                step = {
                    "pickup_zone": current_h3,
                    "dropoff_zone": dest_h3,
                    "fare": estimated_fare,
                }
                path.append(step)
                total_revenue += estimated_fare

                current_h3 = dest_h3
                current_time += timedelta(minutes=30)  # Assume 30 mins per trip

            if path:
                routes.append(
                    {
                        "path": path,
                        "total_estimated_revenue": total_revenue,
                        "duration_minutes": (current_time - start_time).seconds / 60,
                    }
                )

        # Sort by revenue and return top N
        routes.sort(key=lambda x: x["total_estimated_revenue"], reverse=True)
        return routes[:top_n]
