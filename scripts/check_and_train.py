import os
import requests
import tarfile
import zipfile
import io
import re
import mlflow
from mlflow.tracking import MlflowClient
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import sys

# Add project root to path so we can import flows
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from flows.main_flow import main_flow

# --- Configuration ---
DATA_INDEX_URL = "https://traffic.longdo.com/opendata/probe-data/"
DATA_DIR = "data/raw"
STATE_FILE = "data/latest_data_version.txt"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


def get_latest_file_url():
    """Scrapes the index page to find the latest PROBE-YYYYMM file."""
    print(f"🔍 Scraping index at {DATA_INDEX_URL}...")
    try:
        response = requests.get(DATA_INDEX_URL, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a")

        probe_files = []
        # Regex to match PROBE-YYYYMM.tar.bz2 or similar
        # Captures (YYYYMM) for sorting
        pattern = re.compile(r"PROBE-(\d{6})\.(tar\.bz2|zip)")

        for link in links:
            href = link.get("href")
            if not href:
                continue

            match = pattern.search(href)
            if match:
                date_str = match.group(1)
                full_url = urljoin(DATA_INDEX_URL, href)
                probe_files.append((int(date_str), full_url, href))

        if not probe_files:
            print("⚠️ No probe data files found matching pattern.")
            return None

        # Sort by date descending (latest first)
        probe_files.sort(key=lambda x: x[0], reverse=True)
        latest_date, latest_url, filename = probe_files[0]

        print(f"   Found latest dataset: {filename} (Date: {latest_date})")
        return latest_url, filename

    except Exception as e:
        print(f"❌ Error scraping data source: {e}")
        return None


def check_for_new_data():
    """Checks if the latest available file is new compared to local state."""

    # 1. Find the latest URL dynamically
    latest_info = get_latest_file_url()
    if not latest_info:
        return None

    latest_url, filename = latest_info

    # 2. Check local state (We store the filename as the version)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            last_processed_file = f.read().strip()

        if filename == last_processed_file:
            print(f"✅ Data ({filename}) is already processed. No action needed.")
            return None

    print(f"🆕 New data detected: {filename}")
    return latest_url, filename


def download_and_extract(url, filename):
    """Downloads the file and extracts it (handling both zip and tar.bz2)."""
    print(f"⬇️ Downloading {url}...")

    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()

        # Determine extraction method based on extension
        if filename.endswith(".tar.bz2"):
            print("   Extracting .tar.bz2 archive...")
            # TarFile requires a file-like object or path. stream=True helps memory.
            # We save to temp file first to be safe with tarfile
            temp_path = os.path.join(DATA_DIR, filename)
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            with tarfile.open(temp_path, "r:bz2") as tar:
                tar.extractall(path=DATA_DIR)

            # Clean up temp file
            os.remove(temp_path)

        elif filename.endswith(".zip"):
            print("   Extracting .zip archive...")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(DATA_DIR)

        else:
            print(f"❌ Unknown file format: {filename}")
            return False

        # Update state file ONLY after successful extraction
        with open(STATE_FILE, "w") as f:
            f.write(filename)

        print("✅ Extraction complete.")
        return True

    except Exception as e:
        print(f"❌ Failed to download/extract: {e}")
        return False


def evaluate_and_promote(model_name, metric="test_mae", objective="minimize"):
    """
    Compares the latest run (Challenger) vs the current Production model (Champion).
    """
    print(f"🥊 Evaluating Challenger vs Champion for {model_name}...")
    client = MlflowClient()

    # 1. Get the latest run (Challenger)
    experiment = client.get_experiment_by_name(
        f"bangkok_taxi_{model_name.replace('_model', '')}"
    )
    if not experiment:
        print(f"   Skipping: Experiment not found for {model_name}")
        return

    runs = client.search_runs(
        experiment.experiment_id, order_by=["start_time DESC"], max_results=1
    )
    if not runs:
        print("   No runs found.")
        return

    challenger_run = runs[0]
    challenger_metric = challenger_run.data.metrics.get(metric)

    if challenger_metric is None:
        print(f"   ⚠️ Challenger has no metric '{metric}'. Skipping.")
        return

    # 2. Get Production model (Champion)
    # Note: MLflow 2.x 'search_model_versions' filtering syntax
    latest_versions = client.search_model_versions(f"name='{model_name}'")
    production_version = next(
        (v for v in latest_versions if v.current_stage == "Production"), None
    )

    if not production_version:
        print(
            f"   🚀 No Production model exists. Promoting Challenger (Run {challenger_run.info.run_id}) immediately."
        )
        # Find the model version associated with this run
        versions = client.search_model_versions(
            f"run_id='{challenger_run.info.run_id}'"
        )
        if versions:
            client.transition_model_version_stage(
                name=model_name,
                version=versions[0].version,
                stage="Production",
                archive_existing_versions=True,
            )
        return

    champion_run = client.get_run(production_version.run_id)
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
        versions = client.search_model_versions(
            f"run_id='{challenger_run.info.run_id}'"
        )
        if versions:
            client.transition_model_version_stage(
                name=model_name,
                version=versions[0].version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"   ✅ Model {model_name} v{versions[0].version} is now Production.")
    else:
        print(
            "   📉 Challenger failed to beat Champion. Keeping current Production model."
        )


def main():
    # 1. Check for updates
    update_info = check_for_new_data()

    if not update_info:
        # Pass here if no data, or remove return to force run during testing
        print("💤 No new data found. Exiting.")
        return

    latest_url, filename = update_info

    # 2. Download new data
    if not download_and_extract(latest_url, filename):
        print("❌ Pipeline aborted due to download failure.")
        return

    # 3. Run the full Pipeline (ETL -> Prep -> Train)
    print("⚙️ Triggering Main Pipeline...")
    try:
        main_flow()
    except Exception as e:
        print(f"❌ Pipeline Failed: {e}")
        return

    # 4. Evaluate & Promote Models
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
