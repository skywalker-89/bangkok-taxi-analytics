from flask import Flask, jsonify, render_template, request
from recommendation_system import BangkokTaxiOptimizer
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(
    __name__, template_folder="templates"
)  # Assuming index.html is in templates/

# Initialize Optimizer (Loads models once at startup)
try:
    optimizer = BangkokTaxiOptimizer()
    print("🚀 Optimizer Initialized!")
except Exception as e:
    print(f"⚠️ Optimizer failed to load: {e}")
    optimizer = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/optimize", methods=["POST"])
def optimize():
    if not optimizer:
        return jsonify({"error": "Models not loaded"}), 503

    data = request.json
    start_lat = data.get("start_lat")
    start_lng = data.get("start_lng")
    # ... parse other inputs ...

    # Mocking date for demo
    from datetime import datetime

    now = datetime.now()

    try:
        routes = optimizer.optimize_route(
            start_location=(start_lat, start_lng),
            end_location=(start_lat, start_lng),  # Demo: return to start
            start_time=now,
            end_time=now,
            n_simulations=5,
        )
        return jsonify({"routes": routes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
