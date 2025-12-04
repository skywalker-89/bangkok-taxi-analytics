from prefect import flow, task
import subprocess
import os
from pathlib import Path
import sys  # Added sys for the entry point


# ---------------------
# Utility functions
# ---------------------
def run_python(path: str):
    """Run a python script as a subprocess and fail if error occurs."""
    print(f"🚀 Running: {path}")
    subprocess.run(["python", path], check=True)


def check_files_exist(*files):
    """Check if all specified files exist."""
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    print(f"✅ All files exist: {list(files)}")
    return True


# ---------------------
# Data Prep Tasks
# ---------------------
@task(name="Run ETL Pipeline")
def run_etl():
    run_python("run_pipeline.py")


# --- CSV-based tasks (Unchanged) ---
@task(name="Prep Distance Features")
def prep_distance(force=False):
    files = ["X_features_distance.csv", "y_target_distance.csv"]
    if not force and check_files_exist(*files):
        print("⏭️  Skipping distance prep - files already exist")
        return
    run_python("model_data_prep/distance_prep.py")


@task(name="Prep Duration Features")
def prep_duration(force=False):
    files = ["X_features_duration.csv", "y_target_duration.csv"]
    if not force and check_files_exist(*files):
        print("⏭️  Skipping duration prep - files already exist")
        return
    run_python("model_data_prep/duration_prep.py")


@task(name="Prep Inter-Zone Features")
def prep_inter_zone(force=False):
    files = ["X_features_inter_zone.csv", "y_target_inter_zone.csv"]
    if not force and check_files_exist(*files):
        print("⏭️  Skipping inter-zone prep - files already exist")
        return
    run_python("model_data_prep/inter_zone_prep.py")


@task(name="Prep Pickup Rate Features")
def prep_pickup_rate(force=False):
    files = ["X_features_5min.csv", "y_target_5min.csv", "meta_5min.csv"]
    if not force and check_files_exist(*files):
        print("⏭️  Skipping pickup rate prep - files already exist")
        return
    run_python("model_data_prep/pickup_rate_prep.py")


# --- MODIFIED: Database task ---
@task(name="Prep Next Destination Features (DB)")
def prep_next_destination(force=False):
    """
    MODIFIED: Runs the database prep script to create 'model_destination_features'.
    'force' is ignored, but kept for flow compatibility.
    """
    print("  (Next Destination): Running DB script 'next_destination_prep.py'...")
    run_python("next_destination_prep.py")


# ---------------------
# Model Training Tasks
# ---------------------


# --- CSV-based tasks (Unchanged) ---
@task(name="Train Distance Model")
def train_distance():
    if not check_files_exist("X_features_distance.csv", "y_target_distance.csv"):
        raise FileNotFoundError(
            "Distance prep files not found. Run prep_distance() first."
        )
    script_path = "models/train_distance.py"
    if not Path(script_path).exists():
        print(f"⚠️  {script_path} not found - skipping")
        return
    run_python(script_path)


@task(name="Train Duration Model")
def train_duration():
    if not check_files_exist("X_features_duration.csv", "y_target_duration.csv"):
        raise FileNotFoundError(
            "Duration prep files not found. Run prep_duration() first."
        )
    script_path = "models/train_duration.py"
    if not Path(script_path).exists():
        print(f"⚠️  {script_path} not found - skipping")
        return
    run_python(script_path)


@task(name="Train Inter-Zone Model")
def train_inter_zone():
    if not check_files_exist("X_features_inter_zone.csv", "y_target_inter_zone.csv"):
        raise FileNotFoundError(
            "Inter-zone prep files not found. Run prep_inter_zone() first."
        )
    script_path = "models/train_inter_zone.py"
    if not Path(script_path).exists():
        print(f"⚠️  {script_path} not found - skipping")
        return
    run_python(script_path)


@task(name="Train Pickup Rate Model")
def train_pickup_rate():
    if not check_files_exist(
        "X_features_5min.csv", "y_target_5min.csv", "meta_5min.csv"
    ):
        raise FileNotFoundError(
            "Pickup rate prep files not found. Run prep_pickup_rate() first."
        )
    script_path = "models/train_pickup_rate.py"
    if not Path(script_path).exists():
        print(f"⚠️  {script_path} not found - skipping")
        return
    run_python(script_path)


# --- MODIFIED: Database task ---
@task(name="Train Next Destination Model (DB)")
def train_next_destination():
    """
    MODIFIED: Runs the database training script.
    Does not check for CSV files.
    """
    print("  (Next Destination): Running DB script 'train_next_destination.py'...")
    script_path = "train_next_destination.py"
    if not Path(script_path).exists():
        print(f"⚠️  {script_path} not found - skipping")
        return
    run_python(script_path)


# ---------------------
# Workflows (Unchanged)
# ---------------------
@flow(name="Full Pipeline (Smart)")
def full_pipeline(skip_etl=False, skip_prep=True, force_prep=False):
    """
    Smart pipeline that skips prep if files exist.
    """
    print("🔥 Starting Smart Pipeline")

    if not skip_etl:
        run_etl()
    else:
        print("⏭️  Skipping ETL")

    if not skip_prep or force_prep:
        print("\n📊 Feature Preparation Phase")
        prep_distance(force=force_prep)
        prep_duration(force=force_prep)
        prep_inter_zone(force=force_prep)
        prep_next_destination(force=force_prep)  # Calls the MODIFIED DB task
        prep_pickup_rate(force=force_prep)
    else:
        print("\n⏭️  Skipping all prep (files should exist)")

    print("\n🧠 Model Training Phase")
    train_distance()
    train_duration()
    train_inter_zone()
    train_next_destination()  # Calls the MODIFIED DB task
    train_pickup_rate()

    print("\n🎉 Pipeline Complete!")


@flow(name="Train All Models Only")
def train_only():
    """
    Only train models using existing prep files.
    """
    print("🧠 Training All Models (prep files must exist)")
    train_distance()
    train_duration()
    train_inter_zone()
    train_next_destination()  # Calls the MODIFIED DB task
    train_pickup_rate()
    print("🎉 Training Complete!")


@flow(name="Prep All Features Only")
def prep_only(force=False):
    """
    Only run feature preparation.
    """
    print("📊 Preparing All Features")
    prep_distance(force=force)
    prep_duration(force=force)
    prep_inter_zone(force=force)
    prep_next_destination(force=force)  # Calls the MODIFIED DB task
    prep_pickup_rate(force=force)
    print("✅ Prep Complete!")


@flow(name="Train Pickup Rate Only")
def train_pickup_rate_only():
    """Just train the pickup rate model."""
    print("🎯 Training Pickup Rate Model Only")
    train_pickup_rate()
    print("✅ Done!")


# ---------------------
# Entry Point (Unchanged)
# ---------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "train":
            train_only()
        elif mode == "prep":
            force = "--force" in sys.argv
            prep_only(force=force)
        elif mode == "pickup":
            train_pickup_rate_only()
        elif mode == "full":
            skip_etl = "--skip-etl" in sys.argv
            force_prep = "--force-prep" in sys.argv
            full_pipeline(skip_etl=skip_etl, skip_prep=True, force_prep=force_prep)
        else:
            print("Usage:")
            print("  python prefect_flows.py train          # Train models only")
            print("  python prefect_flows.py prep           # Prep features only")
            print("  python prefect_flows.py prep --force   # Force re-prep")
            print("  python prefect_flows.py pickup         # Train pickup rate only")
            print("  python prefect_flows.py full           # Full pipeline (smart)")
            print("  python prefect_flows.py full --skip-etl")
            sys.exit(1)
    else:
        print("ℹ️  No arguments - defaulting to 'train' mode")
        train_only()
