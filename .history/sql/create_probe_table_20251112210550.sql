CREATE TABLE IF NOT EXISTS taxi_probe_raw (
    id SERIAL PRIMARY KEY,
    vehicle_id VARCHAR(64),
    gps_valid SMALLINT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    timestamp TIMESTAMP,
    speed DOUBLE PRECISION,
    heading DOUBLE PRECISION,
    for_hire_light SMALLINT,
    engine_acc SMALLINT
);
