from prefect import flow, task
import subprocess
from pathlib import Path
import sys


# ---------------------
# Utility Functions
# ---------------------
def run_python(path: str):
    """Run a python script as a subprocess and fail if error occurs."""
    script_path = Path(path)
    if not script_path.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: Script not found at {path}")

    print(f"🚀 Running: {path}")
    subprocess.run(["python", path], check=True)


def check_files_exist(*files):
    """Check if all specified output files exist to potentially skip steps."""
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        return False
    print(f"✅ Files found: {list(files)}")
    return True


# ---------------------
# 1. ETL Task
# ---------------------
@task(name="Run ETL Pipeline")
def run_etl():
    # Runs the main cleaning/pipeline script at root
    run_python("run_pipeline.py")


# ---------------------
# 2. Feature Preparation Tasks
# ---------------------
@task(name="Prep Distance Features")
def prep_distance(force=False):
    outputs = ["X_features_distance.csv", "y_target_distance.csv"]
    if not force and check_files_exist(*outputs):
        print("⏭️  Skipping Distance Prep (files exist)")
        return
    run_python("model_data_prep/distance_prep.py")


@task(name="Prep Duration Features")
def prep_duration(force=False):
    outputs = ["X_features_duration.csv", "y_target_duration.csv"]
    if not force and check_files_exist(*outputs):
        print("⏭️  Skipping Duration Prep (files exist)")
        return
    run_python("model_data_prep/duration_prep.py")


@task(name="Prep Inter-Zone Features")
def prep_inter_zone(force=False):
    outputs = ["X_features_inter_zone.csv", "y_target_inter_zone.csv"]
    if not force and check_files_exist(*outputs):
        print("⏭️  Skipping Inter-Zone Prep (files exist)")
        return
    run_python("model_data_prep/inter_zone_prep.py")


@task(name="Prep Next Destination Features")
def prep_next_destination(force=False):
    # Adjusted to match model_data_prep folder structure
    if not force:
        print("ℹ️  Note: Next Destination usually writes to DB, checking script only.")
    run_python("model_data_prep/next_destination_prep.py")


@task(name="Prep Pickup Rate Features")
def prep_pickup_rate(force=False):
    outputs = ["X_features_5min.csv", "y_target_5min.csv", "meta_5min.csv"]
    if not force and check_files_exist(*outputs):
        print("⏭️  Skipping Pickup Rate Prep (files exist)")
        return
    run_python("model_data_prep/pickup_rate_prep.py")


# ---------------------
# 3. Model Training Tasks
# ---------------------
@task(name="Train Distance Model")
def train_distance():
    # Checks dependencies
    if not check_files_exist("X_features_distance.csv", "y_target_distance.csv"):
        print(
            "⚠️  Missing training data for Distance model. Attempting to run anyway..."
        )
    run_python("models/train_distance.py")


@task(name="Train Duration Model")
def train_duration():
    if not check_files_exist("X_features_duration.csv", "y_target_duration.csv"):
        print(
            "⚠️  Missing training data for Duration model. Attempting to run anyway..."
        )
    run_python("models/train_duration.py")


@task(name="Train Inter-Zone Model")
def train_inter_zone():
    if not check_files_exist("X_features_inter_zone.csv", "y_target_inter_zone.csv"):
        print(
            "⚠️  Missing training data for Inter-Zone model. Attempting to run anyway..."
        )
    run_python("models/train_inter_zone.py")


@task(name="Train Next Destination Model")
def train_next_destination():
    print("🧠 Training Next Destination Model...")
    run_python("models/train_next_destination.py")


@task(name="Train Pickup Rate Model")
def train_pickup_rate():
    if not check_files_exist("X_features_5min.csv", "y_target_5min.csv"):
        print(
            "⚠️  Missing training data for Pickup Rate model. Attempting to run anyway..."
        )
    run_python("models/train_pickup_rate.py")


# ---------------------
# Main Pipeline Flow
# ---------------------
@flow(name="End-to-End Taxi Analytics Pipeline")
def full_pipeline(skip_etl: bool = False, force_prep: bool = False):
    """
    Runs the complete pipeline from raw data to trained models.

    Args:
        skip_etl: If True, skips the initial run_pipeline.py
        force_prep: If True, re-runs data prep even if CSVs exist
    """
    print("🔥 Starting End-to-End Pipeline")

    # --- Step 1: ETL ---
    if not skip_etl:
        print("\n[Phase 1] Running ETL...")
        run_etl()
    else:
        print("\n[Phase 1] Skipping ETL (User Requested)")

    # --- Step 2: Feature Prep ---
    print("\n[Phase 2] Preparing Features...")
    prep_distance(force=force_prep)
    prep_duration(force=force_prep)
    prep_inter_zone(force=force_prep)
    prep_next_destination(force=force_prep)
    prep_pickup_rate(force=force_prep)

    # --- Step 3: Training ---
    # Note: Prefect will wait for Phase 2 tasks to complete before starting these
    print("\n[Phase 3] Training Models...")
    train_distance()
    train_duration()
    train_inter_zone()
    train_next_destination()
    train_pickup_rate()

    print("\n🎉 Pipeline Finished Successfully!")


# ---------------------
# Entry Point
# ---------------------
if __name__ == "__main__":
    # Default behavior: Run the full pipeline
    # You can add flags when running from CLI:
    # python prefect_flows.py --skip-etl
    # python prefect_flows.py --force-prep

    skip_etl_arg = "--skip-etl" in sys.argv
    force_prep_arg = "--force-prep" in sys.argv

    full_pipeline(skip_etl=skip_etl_arg, force_prep=force_prep_arg)
