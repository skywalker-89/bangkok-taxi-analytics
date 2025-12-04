from prefect import flow, task
import subprocess


# ---------------------
# Utility function
# ---------------------
def run_python(path: str):
    """Run a python script as a subprocess and fail if error occurs."""
    print(f"🚀 Running: {path}")
    subprocess.run(["python", path], check=True)


# ---------------------
# Tasks
# ---------------------
@task(name="Run ETL Pipeline")
def run_etl():
    run_python("run_pipeline.py")


@task(name="Prep Distance Features")
def prep_distance():
    run_python("model_data_prep/distance_prep.py")


@task(name="Prep Duration Features")
def prep_duration():
    run_python("model_data_prep/duration_prep.py")


@task(name="Prep Inter-Zone Features")
def prep_inter_zone():
    run_python("model_data_prep/inter_zone_prep.py")


@task(name="Prep Next Destination Features")
def prep_next_destination():
    run_python("model_data_prep/next_destination_prep.py")


@task(name="Prep Pickup Rate Features")
def prep_pickup_rate():
    run_python("model_data_prep/pickup_rate_prep.py")


@task(name="Train Distance Model")
def train_distance():
    run_python("models/train_distance.py")


@task(name="Train Duration Model")
def train_duration():
    run_python("models/train_duration.py")


@task(name="Train Inter-Zone Model")
def train_inter_zone():
    run_python("models/train_inter_zone.py")


@task(name="Train Next Destination Model")
def train_next_destination():
    run_python("models/train_next_destination.py")


@task(name="Train Pickup Rate Model")
def train_pickup_rate():
    run_python("models/train_pickup_rate.py")


# ---------------------
# Full Pipeline
# ---------------------
@flow(name="Full Taxi Analytics & ML Pipeline")
def full_pipeline():
    """
    Run everything end-to-end:
    1. ETL
    2. All 5 feature preparation steps
    3. All 5 model training steps
    """
    print("🔥 Starting Full Pipeline")

    # Step 1 — ETL
    run_etl()

    # Step 2 — Feature Preparations (parallelizable if you want)
    prep_distance()
    prep_duration()
    prep_inter_zone()
    prep_next_destination()
    prep_pickup_rate()

    # Step 3 — Model Training
    train_distance()
    train_duration()
    train_inter_zone()
    train_next_destination()
    train_pickup_rate()

    print("🎉 Full Pipeline Complete!")


# ---------------------
# Entry Point
# ---------------------
if __name__ == "__main__":
    full_pipeline()
