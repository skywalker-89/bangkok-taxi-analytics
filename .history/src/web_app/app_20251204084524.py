from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, List, Tuple
import numpy as np
from flask import Flask, jsonify, render_template, request
import requests
from collections import defaultdict
import os

try:
    import h3
except Exception as e:
    raise ImportError(
        "The 'h3' package is required for generating hex boundaries. Install with: pip install h3"
    ) from e

# Import the route optimizer
from recommendation_system import BangkokTaxiOptimizer

APP_ROOT = Path(__file__).resolve().parent

# Map defaults
BANGKOK_CENTER = [13.7563, 100.5018]
DEFAULT_ZOOM = 12

# OSRM API endpoint
OSRM_API = "http://router.project-osrm.org/route/v1/driving/"


def get_osrm_route(
    start_lng: float, start_lat: float, end_lng: float, end_lat: float
) -> Tuple[List[List[float]], int]:
    """
    Get road route from OSRM API
    Returns tuple of (coordinates, trip_number) for progress tracking
    """
    try:
        url = f"{OSRM_API}{start_lng},{start_lat};{end_lng},{end_lat}?overview=full&geometries=geojson"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok" and data.get("routes"):
                coords = data["routes"][0]["geometry"]["coordinates"]
                return coords, 1

        return [[start_lng, start_lat], [end_lng, end_lat]], 0

    except Exception as e:
        print(f"OSRM routing error: {e}")
        return [[start_lng, start_lat], [end_lng, end_lat]], 0


def calculate_offset_position(
    lat: float, lng: float, offset_index: int, total_at_location: int
) -> Tuple[float, float]:
    """
    Calculate offset position for overlapping markers
    Distributes markers in a circle around the original point
    """
    if total_at_location <= 1:
        return lat, lng

    # Offset radius in degrees (roughly 50 meters)
    radius = 0.0005

    # Calculate angle for this marker
    angle = (2 * np.pi * offset_index) / total_at_location

    # Calculate offset
    lat_offset = radius * np.cos(angle)
    lng_offset = radius * np.sin(angle)

    return lat + lat_offset, lng + lng_offset


app = Flask(__name__, template_folder="templates")

# Lazy-load globals
_optimizer: BangkokTaxiOptimizer | None = None


def _ensure_optimizer() -> BangkokTaxiOptimizer:
    """Lazy load the route optimizer from MLflow"""
    global _optimizer
    if _optimizer is None:
        print(f"Loading optimizer (connecting to MLflow)...")
        # REFACTORED: No arguments needed, it self-configures from env vars
        _optimizer = BangkokTaxiOptimizer()
    return _optimizer


@app.route("/")
def index() -> Any:
    return render_template(
        "index.html",
        center=BANGKOK_CENTER,
        zoom=DEFAULT_ZOOM,
    )


# ==================== ROUTE OPTIMIZATION ENDPOINTS ====================


@app.route("/api/optimize_route", methods=["POST"])
def api_optimize_route() -> Any:
    """
    Start route optimization using Monte Carlo simulation
    """
    try:
        data = request.get_json()

        start_lat = float(data.get("start_lat"))
        start_lng = float(data.get("start_lng"))
        end_lat = float(data.get("end_lat"))
        end_lng = float(data.get("end_lng"))

        # Handle simple date strings or isoformat
        try:
            start_time = datetime.fromisoformat(
                data.get("start_time").replace("Z", "+00:00")
            )
            end_time = datetime.fromisoformat(
                data.get("end_time").replace("Z", "+00:00")
            )
        except ValueError:
            # Fallback for simple testing
            start_time = datetime.now()
            end_time = datetime.now() + timedelta(hours=8)

        n_simulations = int(data.get("n_simulations", 500))

        # Basic validation
        if not (13.0 <= start_lat <= 14.5 and 99.0 <= start_lng <= 102.0):
            # Expanded bounds slightly for greater Bangkok
            return jsonify({"error": "Start location outside Bangkok bounds"}), 400

        if end_time <= start_time:
            return jsonify({"error": "End time must be after start time"}), 400

        optimizer = _ensure_optimizer()

        print(f"\n{'='*80}")
        print(f"NEW OPTIMIZATION REQUEST")
        print(f"Start: ({start_lat}, {start_lng})")
        print(f"Simulations: {n_simulations}")

        # Run Optimization
        top_routes = optimizer.optimize_route(
            start_location=(start_lat, start_lng),
            end_location=(end_lat, end_lng),
            start_time=start_time,
            end_time=end_time,
            n_simulations=min(n_simulations, 1000),
            top_n=3,
        )

        def serialize_route(route):
            """
            Adapter: Converts the simple dictionary from the MLflow optimizer
            into the rich format expected by the frontend.
            """

            # The MLflow optimizer returns a simplified 'path' list.
            # We need to enrich it for the UI.
            path_steps = route.get("path", [])
            enriched_trips = []

            current_trip_time = start_time

            for i, step in enumerate(path_steps):
                # Mock or estimate duration if model didn't return it per-step
                step_duration = 20.0  # minutes default

                enriched_trips.append(
                    {
                        "trip_number": i + 1,
                        "pickup_zone": step.get("pickup_zone"),
                        "dropoff_zone": step.get("dropoff_zone"),
                        "pickup_time": current_trip_time.isoformat(),
                        "dropoff_time": (
                            current_trip_time + timedelta(minutes=step_duration)
                        ).isoformat(),
                        "distance_km": 5.0,  # Placeholder if distance model not queried yet
                        "duration_minutes": step_duration,
                        "idle_minutes": 5.0,
                        "fare_thb": float(step.get("fare", 0)),
                        "pickup_probability": 0.85,  # High confidence for top routes
                    }
                )
                current_trip_time += timedelta(
                    minutes=step_duration + 10
                )  # +10 min idle

            return {
                "total_revenue": float(route.get("total_estimated_revenue", 0)),
                "total_trips": len(enriched_trips),
                "total_distance_km": len(enriched_trips) * 5.0,  # Estimate
                "total_trip_time_minutes": float(route.get("duration_minutes", 0)),
                "total_idle_time_minutes": len(enriched_trips) * 5.0,
                "total_empty_time_minutes": 0.0,
                "arrival_time": current_trip_time.isoformat(),
                "time_difference_minutes": 0.0,
                "time_accuracy": 100.0,
                "trips": enriched_trips,
            }

        result = {
            "success": True,
            "routes": [serialize_route(r) for r in top_routes],
            "parameters": {
                "start_location": {"lat": start_lat, "lng": start_lng},
                "end_location": {"lat": end_lat, "lng": end_lng},
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "n_simulations": n_simulations,
            },
        }

        print(
            f"Top route revenue: {top_routes[0].get('total_estimated_revenue', 0):.0f} THB"
        )
        print(f"{'='*80}\n")

        return jsonify(result)

    except Exception as e:
        print(f"\nOptimization error: {str(e)}\n")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/route_geojson/<int:route_index>", methods=["POST"])
def api_route_geojson(route_index: int) -> Any:
    """
    Convert a route's trips into GeoJSON with OSRM routing.
    Connects start, end, and all intermediate trip points.
    """
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

        print(
            f"\nGenerating GeoJSON for route #{route_index + 1} with {len(trips)} trips"
        )

        # Track overlapping pickup locations only
        pickup_locations = defaultdict(list)
        for trip in trips:
            pickup_locations[trip["pickup_zone"]].append(trip["trip_number"])

        # Add route from Start Point to First Pickup
        if trips and start_location:
            start_lat, start_lng = start_location["lat"], start_location["lng"]
            first_pickup_lat, first_pickup_lng = optimizer.h3_to_latlng(
                trips[0]["pickup_zone"]
            )
            coords, _ = get_osrm_route(
                start_lng, start_lat, first_pickup_lng, first_pickup_lat
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"type": "empty_leg", "color": "#FF13F0"},
                }
            )

        # Process each trip
        for i, trip in enumerate(trips):
            trip_num = trip["trip_number"]

            # Get coordinates
            pickup_lat, pickup_lng = optimizer.h3_to_latlng(trip["pickup_zone"])
            dropoff_lat, dropoff_lng = optimizer.h3_to_latlng(trip["dropoff_zone"])

            # Add route from previous dropoff to current pickup
            if i > 0:
                prev_dropoff_lat, prev_dropoff_lng = optimizer.h3_to_latlng(
                    trips[i - 1]["dropoff_zone"]
                )
                coords, _ = get_osrm_route(
                    prev_dropoff_lng, prev_dropoff_lat, pickup_lng, pickup_lat
                )
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": {"type": "empty_leg", "color": "#FF13F0"},
                    }
                )

            # Calculate offsets for overlapping pickup markers
            pickup_trips_at_location = pickup_locations[trip["pickup_zone"]]
            pickup_offset_index = pickup_trips_at_location.index(trip_num)
            pickup_lat_offset, pickup_lng_offset = calculate_offset_position(
                pickup_lat,
                pickup_lng,
                pickup_offset_index,
                len(pickup_trips_at_location),
            )

            # Get OSRM road route for the actual trip
            route_coords, _ = get_osrm_route(
                pickup_lng, pickup_lat, dropoff_lng, dropoff_lat
            )

            # Create line feature for the trip
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": route_coords},
                    "properties": {
                        "type": "trip_leg",
                        "trip_number": trip_num,
                        "color": color,
                        "route_index": route_index,
                    },
                }
            )

            # Add pickup marker
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [pickup_lng_offset, pickup_lat_offset],
                    },
                    "properties": {
                        "type": "pickup",
                        "trip_number": trip_num,
                        "time": trip["pickup_time"],
                        "fare_thb": trip["fare_thb"],
                        "distance_km": trip["distance_km"],
                        "duration_minutes": trip["duration_minutes"],
                        "color": color,
                        "route_index": route_index,
                        "is_shared": len(pickup_trips_at_location) > 1,
                        "shared_trips": (
                            pickup_trips_at_location
                            if len(pickup_trips_at_location) > 1
                            else None
                        ),
                    },
                }
            )

        # Add route from Last Dropoff to End Point
        if trips and end_location:
            last_dropoff_lat, last_dropoff_lng = optimizer.h3_to_latlng(
                trips[-1]["dropoff_zone"]
            )
            end_lat, end_lng = end_location["lat"], end_location["lng"]
            coords, _ = get_osrm_route(
                last_dropoff_lng, last_dropoff_lat, end_lng, end_lat
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"type": "empty_leg", "color": "#FF13F0"},
                }
            )

        print(f"Generated {len(features)} features for a complete, connected route.")

        return jsonify({"type": "FeatureCollection", "features": features})

    except Exception as e:
        print(f"Error generating route GeoJSON: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"BANGKOK TAXI ROUTE OPTIMIZER WITH OSRM ROUTING")
    print(f"{'='*80}")
    print(f"App directory: {APP_ROOT}")
    print(f"MLflow connection: Active")
    print(f"OSRM API: {OSRM_API}")
    print(f"Starting server on http://0.0.0.0:8000")
    print(f"{'='*80}\n")

    app.run(host="0.0.0.0", port=8000, debug=True)
