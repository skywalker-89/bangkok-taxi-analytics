from prefect import flow, task
from src.features.distance_prep import prepare_distance_data
from src.features.duration_prep import prepare_duration_data
from src.features.inter_zone_prep import prepare_inter_zone_data
from src.features.next_destination_prep import prepare_next_destination_data
from src.features.pickup_rate_prep import prepare_pickup_rate_data

from src.modeling.train_distance import train as train_distance
from src.modeling.train_duration import train as train_duration
from src.modeling.train_inter_zone import train as train_inter_zone
from src.modeling.train_next_destination import train as train_next_destination
from src.modeling.train_pickup_rate import train as train_pickup_rate


# --- 1. Feature Engineering Tasks ---
@task(name="Prep: Distance", tags=["feature-engineering"])
def task_prep_distance():
    prepare_distance_data()


@task(name="Prep: Duration", tags=["feature-engineering"])
def task_prep_duration():
    prepare_duration_data()


@task(name="Prep: Inter-Zone", tags=["feature-engineering"])
def task_prep_inter_zone():
    prepare_inter_zone_data()


@task(name="Prep: Next Destination", tags=["feature-engineering"])
def task_prep_next_dest():
    prepare_next_destination_data()


@task(name="Prep: Pickup Rate", tags=["feature-engineering"])
def task_prep_pickup_rate():
    prepare_pickup_rate_data()


# --- 2. Training Tasks ---
@task(name="Train: Distance", tags=["training"])
def task_train_distance():
    train_distance()


@task(name="Train: Duration", tags=["training"])
def task_train_duration():
    train_duration()


@task(name="Train: Inter-Zone", tags=["training"])
def task_train_inter_zone():
    train_inter_zone()


@task(name="Train: Next Destination", tags=["training"])
def task_train_next_dest():
    train_next_destination()


@task(name="Train: Pickup Rate", tags=["training"])
def task_train_pickup_rate():
    train_pickup_rate()


# --- 3. The Main Pipeline ---
@flow(name="Bangkok Taxi End-to-End Pipeline")
def main_flow():
    print("🚀 Starting Pipeline...")

    # Phase 1: Parallel Feature Engineering (Optional: Prefect can run these in parallel)
    # We run them sequentially here for safety on M2 Air
    task_prep_distance()
    task_prep_duration()
    task_prep_inter_zone()
    task_prep_next_dest()
    task_prep_pickup_rate()

    # Phase 2: Training
    task_train_distance()
    task_train_duration()
    task_train_inter_zone()
    task_train_next_dest()
    task_train_pickup_rate()

    print("✅ Pipeline Finished! Check MLflow for results.")


if __name__ == "__main__":
    main_flow()
