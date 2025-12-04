import sys
import os
from datetime import datetime, timedelta

# Add src to path so we can import the app
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from web_app.recommendation_system import BangkokTaxiOptimizer
except ImportError:
    print("❌ Error: Could not import BangkokTaxiOptimizer. Check your paths.")
    sys.exit(1)


def test_optimizer_logic():
    print("🚀 Starting Integration Test...")

    # 1. Setup Environment (Point to your local MLflow)
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"

    # 2. Initialize Optimizer (This tests Model Loading)
    print("\n[Step 1] Loading Models from MLflow...")
    try:
        optimizer = BangkokTaxiOptimizer()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ FAILED to load models. Is MLflow running? Error: {e}")
        return

    # 3. Simulate a Route Request
    print("\n[Step 2] Simulating a route optimization...")
    start_loc = (13.7563, 100.5018)  # Bangkok City Pillar
    end_loc = (13.6520, 100.4930)  # King Mongkut's Univ (Thonburi)
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=4)

    try:
        routes = optimizer.optimize_route(
            start_location=start_loc,
            end_location=end_loc,
            start_time=start_time,
            end_time=end_time,
            n_simulations=5,  # Keep it small for testing
            top_n=1,
        )

        if not routes:
            print(
                "⚠️ Warning: No routes found (Logic might be too strict), but code ran."
            )
        else:
            best_route = routes[0]
            print(f"✅ Success! Found {len(routes)} routes.")
            print(f"   - Revenue: {best_route['total_revenue']:.2f} THB")
            print(f"   - Duration: {best_route['total_trip_time_minutes']:.2f} min")
            print(f"   - Trips Involved: {best_route['total_trips']}")

    except Exception as e:
        print(f"❌ Logic Error during optimization: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_optimizer_logic()
