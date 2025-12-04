import os
import requests
import zipfile
import io
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import sys

# Add project root to path so we can import flows
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from flows.main_flow import main_flow

# --- Configuration ---
DATA_URL = "https://traffic.longdo.com/opendata/probe-data/2023-12.zip"  # Example URL, usually you'd scrape the index for latest
# In a real scenario, you'd scrape https://traffic.longdo.com/opendata/ to find the newest link dynamically
DATA_DIR = "data/raw"
STATE_FILE = "data/latest_data_date.txt"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


def check_for_new_data():
    """Checks if the remote file is newer than what we last processed."""
    print(f"🔍 Checking for data updates at {DATA_URL}...")

    try:
        response = requests.head(DATA_URL)
        if response.status_code != 200:
            print(f"⚠️ Failed to reach data source. Status: {response.status_code}")
            return False

        remote_last_modified = response.headers.get("Last-Modified")
        print(f"   Remote Last-Modified: {remote_last_modified}")

        # Check local state
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                local_last_modified = f.read().strip()

            if remote_last_modified == local_last_modified:
                print("✅ Data is up to date. No retraining needed.")
                return False

        return remote_last_modified

    except Exception as e:
        print(f"❌ Error checking data source: {e}")
        return False


def download_and_extract(last_modified_header):
    """Downloads the ZIP and extracts to data/raw."""
    print("⬇️ New data found! Downloading...")
    r = requests.get(DATA_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(DATA_DIR)

    # Update state file
    with open(STATE_FILE, "w") as f:
        f.write(last_modified_header)
    print("✅ Download and extraction complete.")


def evaluate_and_promote(model_name, metric="test_mae", objective="minimize"):
    """
    Compares the latest run (Challenger) vs the current Production model (Champion).
    Promotes Challenger if it performs better.
    """
    print(f"🥊 Evaluating Challenger vs Champion for {model_name}...")
    client = MlflowClient()

    # 1. Get the latest run (Challenger)
    # Assuming the script just ran, the latest run is our Challenger
    experiment = client.get_experiment_by_name(
        f"bangkok_taxi_{model_name.replace('_model', '')}"
    )
    if not experiment:
        print(f"   Skipping: Experiment not found for {model_name}")
        return

    runs = client.search_runs(
        experiment.experiment_id, order_by=["start_time DESC"], max_results=1
    )
    challenger_run = runs[0]
    challenger_metric = challenger_run.data.metrics.get(metric)

    if challenger_metric is None:
        print(f"   ⚠️ Challenger has no metric '{metric}'. Skipping promotion.")
        return

    # 2. Get Production model (Champion)
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])

    if not latest_versions:
        print(
            f"   🚀 No Production model exists. Promoting Challenger (v{challenger_run.info.run_id}) immediately."
        )
        client.transition_model_version_stage(
            name=model_name,
            version=client.get_latest_versions(model_name, stages=["None"])[0].version,
            stage="Production",
        )
        return

    champion_version = latest_versions[0]
    champion_run = client.get_run(champion_version.run_id)
    champion_metric = champion_run.data.metrics.get(metric)

    print(f"   📊 Comparison ({metric}):")
    print(f"      Challenger: {challenger_metric:.4f}")
    print(f"      Champion:   {champion_metric:.4f}")

    # 3. Compare
    is_better = False
    if objective == "minimize":
        is_better = challenger_metric < champion_metric
    else:  # maximize (e.g., accuracy)
        is_better = challenger_metric > champion_metric

    if is_better:
        print("   🏆 Challenger wins! Promoting to Production...")
        # Get the version number of the challenger model created in this run
        # Note: This requires the training script to have registered the model
        model_versions = client.search_model_versions(
            f"run_id='{challenger_run.info.run_id}'"
        )
        if model_versions:
            new_version = model_versions[0].version
            client.transition_model_version_stage(
                name=model_name, version=new_version, stage="Production"
            )
            print(f"   ✅ Model {model_name} v{new_version} is now Production.")
        else:
            print(
                "   ⚠️ Challenger model artifact found, but it wasn't registered properly."
            )
    else:
        print(
            "   📉 Challenger failed to beat Champion. Keeping current Production model."
        )


def main():
    # 1. Check for updates
    new_data_header = check_for_new_data()

    # Force run for testing purposes if you want:
    # new_data_header = "Force Run"

    if not new_data_header:
        print("💤 No new data. Sleeping.")
        return

    # 2. Download new data
    download_and_extract(new_data_header)

    # 3. Run the full Pipeline (ETL -> Prep -> Train)
    print("⚙️ Triggering Main Pipeline...")
    main_flow()

    # 4. Evaluate & Promote Models
    # We check each model type specifically
    print("\n⚖️ Starting Model Evaluation Phase...")
    evaluate_and_promote("trip_distance_model", metric="test_mae", objective="minimize")
    evaluate_and_promote("trip_duration_model", metric="test_mae", objective="minimize")
    evaluate_and_promote(
        "next_destination_model", metric="test_accuracy", objective="maximize"
    )
    evaluate_and_promote(
        "pickup_rate_model", metric="test_accuracy", objective="maximize"
    )


if __name__ == "__main__":
    main()
