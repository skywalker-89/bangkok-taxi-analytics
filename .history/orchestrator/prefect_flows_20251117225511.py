from prefect import flow, task
import subprocess
import sys


# ---------------------
# Utility functions
# ---------------------
def run_python(path: str):
    """Run a python script as a subprocess and fail if error occurs."""
    print(f"🚀 Running: {path}")
    # Run the script, checking for errors
    subprocess.run(["python", path], check=True)


# ---------------------
# Tasks
# ---------------------
@task(name="Run DB Prep")
def run_db_prep():
    """
    Runs the database prep script to create 'model_destination_features' table.
    """
    run_python("next_destination_prep.py")


@task(name="Train Model from DB")
def train_model():
    """
    Runs the training script that reads from 'model_destination_features' table.
    """
    run_python("train_next_destination.py")


# ---------------------
# Workflows
# ---------------------
@flow(name="Full DB Pipeline")
def full_pipeline():
    """
    Runs the full pipeline:
    1. Prep data and create the database table.
    2. Train the model from that database table.
    """
    print("🔥 Starting Full Pipeline")

    # Step 1 — Prep DB Table
    print("\n📊 [1/2] Preparing database table...")
    run_db_prep()

    # Step 2 — Train Model
    print("\n🧠 [2/2] Training model from database...")
    train_model()

    print("\n🎉 Pipeline Complete!")


@flow(name="Train Model Only")
def train_only():
    """
    Only train the model.
    Assumes 'model_destination_features' table already exists.
    """
    print("🧠 Training Model Only (assumes DB table exists)")
    train_model()
    print("🎉 Training Complete!")


@flow(name="Prep DB Only")
def prep_only():
    """
    Only run the database preparation to create the
    'model_destination_features' table.
    """
    print("📊 Preparing Database Table Only")
    run_db_prep()
    print("✅ Prep Complete! Table 'model_destination_features' is ready.")


# ---------------------
# Entry Point
# ---------------------
if __name__ == "__main__":

    # Default to 'full' if no argument is given
    if len(sys.argv) < 2:
        print("ℹ️  No arguments provided, running 'full' pipeline...")
        full_pipeline()
        sys.exit(0)

    mode = sys.argv[1].lower()

    if mode == "prep":
        # Only prep features
        prep_only()

    elif mode == "train":
        # Only train models
        train_only()

    elif mode == "full":
        # Full pipeline
        full_pipeline()

    else:
        print(f"❌ Unknown mode: '{mode}'")
        print("Usage:")
        print(
            "  python prefect_flows.py prep    # Create 'model_destination_features' DB table"
        )
        print("  python prefect_flows.py train   # Train model (assumes table exists)")
        print("  python prefect_flows.py full    # Run both prep and train")
        sys.exit(1)
