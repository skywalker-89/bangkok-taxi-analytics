from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, List, Tuple
import numpy as np
from flask import Flask, jsonify, render_template, request
import requests
from collections import defaultdict
import os
import sys

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import h3
except Exception as e:
    print("Warning: h3 not found")

from recommendation_system import BangkokTaxiOptimizer

APP_ROOT = Path(__file__).resolve().parent
BANGKOK_CENTER = [13.7563, 100.5018]
DEFAULT_ZOOM = 12
OSRM_API = "http://router.project-osrm.org/route/v1/driving/"


def get_osrm_route(start_lng, start_lat, end_lng, end_lat):
    try:
        url = f"{OSRM_API}{start_lng},{start_lat};{end_lng},{end_lat}?overview=full&geometries=geojson"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok" and data.get("routes"):
                return data["routes"][0]["geometry"]["coordinates"], 1
        return [[start_lng, start_lat], [end_lng, end_lat]], 0
    except Exception as e:
        print(f"OSRM routing error: {e}")
        return [[start_lng, start_lat], [end_lng, end_lat]], 0


def calculate_offset_position(lat, lng, offset_index, total_at_location):
    if total_at_location <= 1:
        return lat, lng
    radius = 0.0005
    angle = (2 * np.pi * offset_index) / total_at_location
    return lat + (radius * np.cos(angle)), lng + (radius * np.sin(angle))


app = Flask(__name__, template_folder="templates")

_optimizer: BangkokTaxiOptimizer | None = None


def _ensure_optimizer() -> BangkokTaxiOptimizer:
    global _optimizer
    if _optimizer is None:
        print(f"Loading optimizer (connecting to MLflow)...")
        _optimizer = BangkokTaxiOptimizer()
    return _optimizer


@app.route("/")
def index() -> Any:
    return render_template("index.html", center=BANGKOK_CENTER, zoom=DEFAULT_ZOOM)


@app.route("/api/optimize_route", methods=["POST"])
def api_optimize_route() -> Any:
    try:
        data = request.get_json()
        start_lat = float(data.get("start_lat"))
        start_lng = float(data.get("start_lng"))
        end_lat = float(data.get("end_lat"))
        end_lng = float(data.get("end_lng"))

        try:
            start_time = datetime.fromisoformat(
                data.get("start_time").replace("Z", "+00:00")
            )
            end_time = datetime.fromisoformat(
                data.get("end_time").replace("Z", "+00:00")
            )
        except ValueError:
            start_time = datetime.now()
            end_time = datetime.now() + timedelta(hours=8)

        n_simulations = int(data.get("n_simulations", 500))
        optimizer = _ensure_optimizer()

        print(
            f"NEW OPTIMIZATION REQUEST: ({start_lat}, {start_lng}) -> ({end_lat}, {end_lng})"
        )

        top_routes = optimizer.optimize_route(
            start_location=(start_lat, start_lng),
            end_location=(end_lat, end_lng),
            start_time=start_time,
            end_time=end_time,
            n_simulations=min(n_simulations, 1000),
            top_n=3,
        )

        def serialize_route(route):
            path_steps = route.get("path", [])
            enriched_trips = []
            current_trip_time = start_time

            for i, step in enumerate(path_steps):
                step_duration = float(step.get("duration_minutes", 20.0))
                enriched_trips.append(
                    {
                        "trip_number": i + 1,
                        "pickup_zone": step.get("pickup_zone"),
                        "dropoff_zone": step.get("dropoff_zone"),
                        "pickup_time": current_trip_time.isoformat(),
                        "dropoff_time": (
                            current_trip_time + timedelta(minutes=step_duration)
                        ).isoformat(),
                        "distance_km": float(step.get("distance_km", 5.0)),
                        "duration_minutes": step_duration,
                        "idle_minutes": 5.0,
                        "fare_thb": float(step.get("fare", 0)),
                        "pickup_probability": 0.85,
                    }
                )
                current_trip_time += timedelta(minutes=step_duration + 10)

            # ✅ FIX: Use correct key 'total_revenue'
            return {
                "total_revenue": float(route.get("total_revenue", 0)),
                "total_trips": int(route.get("total_trips", len(enriched_trips))),
                "total_distance_km": sum(t["distance_km"] for t in enriched_trips),
                "total_trip_time_minutes": float(
                    route.get("total_trip_time_minutes", 0)
                ),
                "total_idle_time_minutes": len(enriched_trips) * 5.0,
                "total_empty_time_minutes": 0.0,
                "arrival_time": current_trip_time.isoformat(),
                "time_difference_minutes": 0.0,
                "time_accuracy": 100.0,
                "trips": enriched_trips,
            }

        return jsonify(
            {
                "success": True,
                "routes": [serialize_route(r) for r in top_routes],
                "parameters": {
                    "start_location": {"lat": start_lat, "lng": start_lng},
                    "end_location": {"lat": end_lat, "lng": end_lng},
                },
            }
        )

    except Exception as e:
        print(f"Optimization error: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/route_geojson/<int:route_index>", methods=["POST"])
def api_route_geojson(route_index: int) -> Any:
    try:
        data = request.get_json()
        route = data.get("route", {})
        trips = route.get("trips", [])
        start_location = data.get("start_location")
        end_location = data.get("end_location")

        optimizer = _ensure_optimizer()
        features = []

        colors = ["#28a745", "#007bff", "#ff6600"]
        color = colors[route_index % len(colors)]

        # --- 1. Start Marker ---
        if start_location:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [start_location["lng"], start_location["lat"]],
                    },
                    "properties": {"type": "start", "label": "Start"},
                }
            )

        # --- 2. Process Trips (Lines + Markers) ---
        for i, trip in enumerate(trips):
            trip_num = trip["trip_number"]
            pickup_lat, pickup_lng = optimizer.h3_to_latlng(trip["pickup_zone"])
            dropoff_lat, dropoff_lng = optimizer.h3_to_latlng(trip["dropoff_zone"])

            # Route Line
            coords, _ = get_osrm_route(pickup_lng, pickup_lat, dropoff_lng, dropoff_lat)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "type": "trip_leg",
                        "color": color,
                        "trip_number": trip_num,
                    },
                }
            )

            # ✅ FIX: Pickup Marker (Trip 1, 2, 3...)
            # This generates the numbered pin on the map
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [pickup_lng, pickup_lat],
                    },
                    "properties": {
                        "type": "pickup",
                        "trip_number": trip_num,
                        "fare_thb": trip["fare_thb"],
                        "distance_km": trip["distance_km"],
                        "color": color,
                    },
                }
            )

            # Empty Leg Line (Previous Dropoff -> Current Pickup)
            if i > 0:
                prev_dropoff_lat, prev_dropoff_lng = optimizer.h3_to_latlng(
                    trips[i - 1]["dropoff_zone"]
                )
                empty_coords, _ = get_osrm_route(
                    prev_dropoff_lng, prev_dropoff_lat, pickup_lng, pickup_lat
                )
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": empty_coords},
                        "properties": {"type": "empty_leg", "color": "#FF13F0"},
                    }
                )

        # --- 3. End Marker ---
        if end_location:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [end_location["lng"], end_location["lat"]],
                    },
                    "properties": {"type": "end", "label": "End"},
                }
            )

        return jsonify({"type": "FeatureCollection", "features": features})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask Server on Port 8000...")
    app.run(host="0.0.0.0", port=8000, debug=False)
