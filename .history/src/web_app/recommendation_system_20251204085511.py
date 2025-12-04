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

# ✅ FIX: Increase timeouts for MLflow operations to prevent 500 errors on large downloads
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "1000"  # Seconds
os.environ["MLFLOW_GUNICORN_OPTS"] = "--timeout 1000"


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
            print("   Loading Next Destination Model...")
            self.dest_model = mlflow.xgboost.load_model(
                "models:/next_destination_model/None"
            )
            print("   Loading Duration Model...")
            self.duration_model = mlflow.xgboost.load_model(
                "models:/trip_duration_model/None"
            )
            print("   Loading Distance Model...")
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
            latest_versions = client.get_latest_versions(
                "next_destination_model", stages=None
            )

            if not latest_versions:
                raise Exception(
                    "No registered version found for 'next_destination_model'"
                )

            # Get the absolute latest version (Converting version to int for correct sorting)
            latest_version = sorted(
                latest_versions, key=lambda x: int(x.version), reverse=True
            )[0]
            run_id = latest_version.run_id

            print(
                f"   Fetching artifacts from Run ID: {run_id} (Version {latest_version.version})"
            )

            # Download artifacts to current directory
            try:
                client.download_artifacts(run_id, "dest_feature_encoder.pkl", ".")
                client.download_artifacts(run_id, "dest_target_encoder.pkl", ".")
            except Exception as download_error:
                print(f"   ⚠️ Artifact download failed: {download_error}")
                print("   Attempting to use local fallback if available...")

            # Load encoders (will fail if download failed and file not present)
            if os.path.exists("dest_feature_encoder.pkl") and os.path.exists(
                "dest_target_encoder.pkl"
            ):
                self.le_start = joblib.load("dest_feature_encoder.pkl")
                self.le_end = joblib.load("dest_target_encoder.pkl")
                print("✅ Encoders loaded.")
            else:
                raise FileNotFoundError(
                    "Encoder files not found. MLflow download likely failed."
                )

        except Exception as e:
            print(f"❌ Error loading encoders: {e}")
            print("⚠️ Warning: Encoders failed to load. Predictions may fail.")

        # Thai taxi fare structure (2024 rates)
        self.fare_structure = {
            "base_fare": 35.0,
            "rate_1_10km": 6.50,
            "rate_10_20km": 7.00,
            "rate_20_40km": 8.00,
            "rate_40_60km": 8.50,
            "rate_60_80km": 9.00,
            "rate_80plus": 10.50,
        }

        # Bangkok center for distance calculations
        self.bkk_center = (13.7563, 100.5018)

    def calculate_fare(self, distance_km: float) -> float:
        """Calculate taxi fare based on distance using Thai taxi rates"""
        if distance_km <= 0:
            return 0

        fare = self.fare_structure["base_fare"]

        if distance_km > 1:
            fare += min(distance_km - 1, 9) * self.fare_structure["rate_1_10km"]
        if distance_km > 10:
            fare += min(distance_km - 10, 10) * self.fare_structure["rate_10_20km"]
        if distance_km > 20:
            fare += min(distance_km - 20, 20) * self.fare_structure["rate_20_40km"]
        if distance_km > 40:
            fare += min(distance_km - 40, 20) * self.fare_structure["rate_40_60km"]
        if distance_km > 60:
            fare += min(distance_km - 60, 20) * self.fare_structure["rate_60_80km"]
        if distance_km > 80:
            fare += (distance_km - 80) * self.fare_structure["rate_80plus"]

        return fare

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

    def predict_pickup_probability(self, h3_zone: str, timestamp: datetime) -> float:
        """Predict probability of getting a pickup in a zone at given time"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        # Simple heuristic based on time and location
        base_prob = 0.5

        # Rush hour bonus
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            base_prob += 0.2

        # Weekend adjustment
        if day_of_week >= 5:
            base_prob += 0.1

        # Late night penalty
        if hour >= 23 or hour <= 5:
            base_prob -= 0.2

        return max(0.1, min(0.95, base_prob))

    def predict_trip_metrics(self, start_h3, end_h3, current_time):
        slat, slon = self.h3_to_latlng(start_h3)
        elat, elon = self.h3_to_latlng(end_h3)

        hav_dist = self.haversine(slat, slon, elat, elon)
        center_lat, center_lon = 13.7563, 100.5018
        dist_center = self.haversine(slat, slon, center_lat, center_lon)

        day_of_week = current_time.weekday()

        try:
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
            pred_distance = hav_dist * 1.3

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
            top3_indices = np.argsort(probas)[-10:][::-1]  # Get top 10 for variety

            results = []
            for idx in top3_indices:
                dest_h3 = self.le_end.inverse_transform([idx])[0]
                prob = probas[idx]
                results.append({"h3": dest_h3, "probability": float(prob)})
            return results
        except:
            return []

    def simulate_route(
        self,
        start_h3: str,
        end_h3: str,
        start_time: datetime,
        end_time: datetime,
        max_trips: int = 20,
    ):
        """
        Simulate a single route using Monte Carlo approach
        Returns route details with revenue and timing
        """
        current_h3 = start_h3
        current_time = start_time
        time_limit = end_time

        trips = []
        total_revenue = 0
        total_distance = 0
        total_trip_time = 0
        total_idle_time = 0

        trip_count = 0

        while current_time < time_limit and trip_count < max_trips:
            # Get potential destinations
            next_options = self.predict_next_step(current_h3, current_time)
            if not next_options:
                break

            # Randomly select destination weighted by probability
            probs = np.array([x["probability"] for x in next_options])
            probs = probs / np.sum(probs)
            choice = np.random.choice(range(len(next_options)), p=probs)
            dest_h3 = next_options[choice]["h3"]

            # Predict pickup probability
            pickup_prob = self.predict_pickup_probability(current_h3, current_time)

            # Estimate idle time
            idle_minutes = np.random.exponential(
                scale=(1.0 / max(0.1, pickup_prob)) * 5
            )
            idle_minutes = min(idle_minutes, 30)

            current_time += timedelta(minutes=float(idle_minutes))

            if current_time >= time_limit:
                break

            # Predict trip details
            pred_dist, pred_dur = self.predict_trip_metrics(
                current_h3, dest_h3, current_time
            )

            # Calculate fare
            estimated_fare = self.calculate_fare(pred_dist)

            # Check if trip fits in remaining time
            time_after_trip = current_time + timedelta(minutes=pred_dur)

            if time_after_trip > time_limit:
                break

            # Accept trip
            trips.append(
                {
                    "trip_number": trip_count + 1,
                    "pickup_zone": current_h3,
                    "dropoff_zone": dest_h3,
                    "pickup_time": current_time.isoformat(),
                    "dropoff_time": time_after_trip.isoformat(),
                    "distance_km": float(pred_dist),
                    "duration_minutes": float(pred_dur),
                    "idle_minutes": float(idle_minutes),
                    "fare_thb": float(estimated_fare),
                    "pickup_probability": float(pickup_prob),
                }
            )

            total_revenue += estimated_fare
            total_distance += pred_dist
            total_trip_time += pred_dur
            total_idle_time += idle_minutes

            # Move to next zone
            current_h3 = dest_h3
            current_time = time_after_trip
            trip_count += 1

        # Calculate return journey time (empty time)
        total_empty_time = 0
        if current_h3 != end_h3:
            # Estimate return time based on distance
            slat, slon = self.h3_to_latlng(current_h3)
            elat, elon = self.h3_to_latlng(end_h3)
            return_dist = self.haversine(slat, slon, elat, elon)
            return_time = (return_dist / 30) * 60  # Assume 30 km/h average speed
            total_empty_time = return_time
            arrival_time = current_time + timedelta(minutes=float(return_time))
        else:
            arrival_time = current_time

        time_difference = (arrival_time - end_time).total_seconds() / 60
        time_accuracy = abs(time_difference)

        return {
            "trips": trips,
            "total_revenue": total_revenue,
            "total_trips": len(trips),
            "total_distance_km": total_distance,
            "total_trip_time_minutes": total_trip_time,
            "total_idle_time_minutes": total_idle_time,
            "total_empty_time_minutes": total_empty_time,
            "arrival_time": arrival_time,
            "time_difference_minutes": time_difference,
            "time_accuracy": time_accuracy,
        }

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
        Find top N routes using Monte Carlo simulation
        TWO-TIER RANKING: 1) Revenue (highest), 2) Time accuracy (closest to end_time)
        """
        start_h3 = self.latlng_to_h3(*start_location)
        end_h3 = self.latlng_to_h3(*end_location)

        routes = []

        print(f"🔄 Simulating {n_simulations} routes with AI predictions...")
        print(f"🎯 Ranking: PRIMARY=Revenue | SECONDARY=Time Accuracy\n")

        for i in range(n_simulations):
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{n_simulations} simulations completed...")

            route = self.simulate_route(start_h3, end_h3, start_time, end_time)

            if route["trips"]:  # Only include routes with at least one trip
                routes.append(route)

        # TWO-TIER SORTING
        # Primary: Revenue (descending - highest first)
        # Secondary: Time Accuracy (ascending - closest to target)
        routes.sort(key=lambda x: (-x["total_revenue"], x["time_accuracy"]))

        print(f"\n✅ Optimization complete!")
        print(f"📊 Generated {len(routes)} valid routes")
        print(f"🏆 Returning top {min(top_n, len(routes))} routes\n")

        return routes[:top_n]
