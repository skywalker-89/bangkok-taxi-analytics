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
            self.distance_model = mlflow.xgboost.load_model(
                "models:/trip_distance_model/None"
            )
            print("✅ All Models loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e

        # 3. Download & Load Encoders (Artifacts)
        print("⏳ Downloading feature encoders...")
        try:
            # ✅ FIX: Get Run ID directly from the Registered Model
            # Using 'None' stage means we get the latest version
            latest_versions = client.get_latest_versions(
                "next_destination_model", stages=None
            )

            if not latest_versions:
                raise Exception(
                    "No registered version found for 'next_destination_model'"
                )

            # Sort by version number (convert string to int to avoid "10" < "2" bug)
            latest_version = sorted(
                latest_versions, key=lambda x: int(x.version), reverse=True
            )[0]
            run_id = latest_version.run_id

            print(
                f"   Fetching artifacts from Run ID: {run_id} (Version {latest_version.version})"
            )

            # Download artifacts to current directory
            client.download_artifacts(run_id, "dest_feature_encoder.pkl", ".")
            client.download_artifacts(run_id, "dest_target_encoder.pkl", ".")

            self.le_start = joblib.load("dest_feature_encoder.pkl")
            self.le_end = joblib.load("dest_target_encoder.pkl")

            print("✅ Encoders loaded.")

        except Exception as e:
            print(f"❌ Error loading encoders: {e}")
            print("⚠️ Warning: Encoders failed to load. Predictions may fail.")

    def h3_to_latlng(self, h3_addr):
        return h3.cell_to_latlng(h3_addr)

    def latlng_to_h3(self, lat, lng):
        return h3.latlng_to_cell(lat, lng, 8)

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def predict_trip_metrics(self, start_h3, end_h3, current_time):
        """
        Uses XGBoost models to predict realistic Distance and Duration.
        """
        slat, slon = self.h3_to_latlng(start_h3)
        elat, elon = self.h3_to_latlng(end_h3)

        hav_dist = self.haversine(slat, slon, elat, elon)
        center_lat, center_lon = 13.7563, 100.5018
        dist_center = self.haversine(slat, slon, center_lat, center_lon)

        day_of_week = current_time.weekday()

        # Prepare Input DataFrame
        try:
            # Safe transform with fallback
            if hasattr(self, "le_start"):
                s_idx = (
                    self.le_start.transform([start_h3])[0]
                    if start_h3 in self.le_start.classes_
                    else 0
                )
                e_idx = (
                    self.le_start.transform([end_h3])[0]
                    if end_h3 in self.le_start.classes_
                    else 0
                )
            else:
                s_idx, e_idx = 0, 0
        except:
            s_idx, e_idx = 0, 0

        input_df = pd.DataFrame(
            [
                {
                    "start_h3_zone_idx": s_idx,
                    "end_h3_zone_idx": e_idx,
                    "pickup_hour": current_time.hour,
                    "pickup_dayofweek": day_of_week,
                    "is_weekend": 1 if day_of_week >= 5 else 0,
                    "haversine_distance": hav_dist,
                    "dist_from_center": dist_center,
                    "average_speed": 30,
                    "max_speed": 50,
                    "total_trip_distance_km": 0,
                }
            ]
        )

        # 1. Predict Real Distance
        try:
            dist_features = [
                "pickup_hour",
                "pickup_dayofweek",
                "is_weekend",
                "haversine_distance",
                "dist_from_center",
            ]
            pred_distance = self.distance_model.predict(input_df[dist_features])[0]
            pred_distance = max(0.5, float(pred_distance))
        except:
            pred_distance = hav_dist * 1.3  # Fallback

        # 2. Predict Duration
        try:
            pred_duration = self.duration_model.predict(input_df)[0]
        except:
            pred_duration = pred_distance * 3

        pred_duration = max(1.0, float(pred_duration))

        return pred_distance, pred_duration

    def predict_next_step(self, current_h3, current_time):
        if not hasattr(self, "le_start") or not hasattr(self, "le_end"):
            return []

        try:
            if current_h3 not in self.le_start.classes_:
                return []
            start_idx = self.le_start.transform([current_h3])[0]
        except ValueError:
            return []

        day_of_week = current_time.weekday()

        input_df = pd.DataFrame(
            [
                {
                    "h3_start_idx": start_idx,
                    "pickup_hour": current_time.hour,
                    "pickup_dayofweek": day_of_week,
                    "is_weekend": 1 if day_of_week >= 5 else 0,
                    "pickup_month": current_time.month,
                    "pickup_day": current_time.day,
                    "start_dist_from_center": 0.0,
                    "od_pair_historical_count": 0,
                    "origin_historical_count": 0,
                    "origin_to_dest_popularity": 0,
                }
            ]
        )

        try:
            probas = self.dest_model.predict_proba(input_df)[0]
            top3_indices = np.argsort(probas)[-3:][::-1]

            results = []
            for idx in top3_indices:
                dest_h3 = self.le_end.inverse_transform([idx])[0]
                prob = probas[idx]
                results.append({"h3": dest_h3, "probability": float(prob)})
            return results
        except:
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
        start_h3 = self.latlng_to_h3(*start_location)
        routes = []

        print(f"🔄 Simulating {n_simulations} routes with AI pricing...")

        for i in range(n_simulations):
            current_h3 = start_h3
            current_time = start_time
            path = []
            total_revenue = 0

            for _ in range(3):
                next_options = self.predict_next_step(current_h3, current_time)
                if not next_options:
                    break

                probs = [x["probability"] for x in next_options]
                probs = np.array(probs) / np.sum(probs)
                choice = np.random.choice(range(len(next_options)), p=probs)
                dest_h3 = next_options[choice]["h3"]

                pred_dist, pred_dur = self.predict_trip_metrics(
                    current_h3, dest_h3, current_time
                )

                # Bangkok Taxi Fare Logic
                estimated_fare = 35 + (pred_dist * 6.5) + (pred_dur * 2.0)

                step = {
                    "pickup_zone": current_h3,
                    "dropoff_zone": dest_h3,
                    "fare": int(estimated_fare),
                    "distance_km": round(pred_dist, 1),
                    "duration_minutes": round(pred_dur, 1),
                }
                path.append(step)
                total_revenue += estimated_fare

                current_h3 = dest_h3
                current_time += timedelta(minutes=pred_dur + 10)

            if path:
                routes.append(
                    {
                        "path": path,
                        "total_revenue": total_revenue,
                        "total_trip_time_minutes": (current_time - start_time).seconds
                        / 60,
                        "total_trips": len(path),
                    }
                )

        routes.sort(key=lambda x: x["total_revenue"], reverse=True)
        return routes[:top_n]
