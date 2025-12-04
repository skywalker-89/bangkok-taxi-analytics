-- This creates a new table with ALL pickup events, 
-- binned by 5 mins and H3 cell.
-- THIS IS THE NEW "GROUND TRUTH" TABLE.
CREATE TABLE pickups_binned_5min AS (
  WITH 
  probes_with_new_trip AS (
    -- Find "is_new_trip" events
    SELECT
      *,
      (for_hire_light = 1 AND 
       LAG(for_hire_light, 1, 0) OVER (PARTITION BY vehicle_id ORDER BY timestamp) = 0) 
      AS is_new_trip
    FROM
      taxi_probe_raw
  ),
  pickup_events AS (
    -- Get just the pickup points
    SELECT
      *
    FROM
      probes_with_new_trip
    WHERE
      is_new_trip
  )
  -- Grid into 5-min bins and H3 cells, and count
  SELECT
    -- THIS IS THE FIXED LINE --
    h3_lat_lng_to_cell(POINT(longitude, latitude), 9) AS h3_cell,
    -- ---------------------- --
    date_trunc('hour', timestamp) + 
      (floor(extract(minute from timestamp) / 5) * interval '5 minute') 
    AS time_bin,
    COUNT(*) AS pickup_count
  FROM
    pickup_events
  GROUP BY
    h3_cell, time_bin
);

-- Add an index for lightning-fast queries
CREATE INDEX idx_pickups_binned_cell_time 
ON pickups_binned_5min (h3_cell, time_bin);