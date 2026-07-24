-- SQL Window Functions for Rolling Z-Score Anomaly Detection

-- 1. PostgreSQL Window Query for Rolling Latency Z-Score
-- Computes mean and standard deviation over a sliding window of preceding records
WITH telemetry_stats AS (
    SELECT 
        id,
        service_name,
        endpoint,
        status_code,
        response_time_ms,
        timestamp,
        AVG(response_time_ms) OVER (
            PARTITION BY service_name, endpoint 
            ORDER BY timestamp 
            ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
        ) AS rolling_mean_latency,
        STDDEV(response_time_ms) OVER (
            PARTITION BY service_name, endpoint 
            ORDER BY timestamp 
            ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
        ) AS rolling_std_latency
    FROM telemetry_logs
)
SELECT 
    id,
    service_name,
    endpoint,
    response_time_ms,
    rolling_mean_latency,
    rolling_std_latency,
    CASE 
        WHEN rolling_std_latency IS NULL OR rolling_std_latency = 0 THEN 0.0
        ELSE (response_time_ms - rolling_mean_latency) / rolling_std_latency
    END AS z_score,
    timestamp
FROM telemetry_stats
WHERE rolling_std_latency > 0
ORDER BY timestamp DESC;

-- 2. Error Rate Anomaly CTE (Calculating Error Percentage in sliding window)
WITH windowed_errors AS (
    SELECT
        service_name,
        endpoint,
        timestamp,
        CASE WHEN status_code >= 400 THEN 1 ELSE 0 END AS is_error,
        AVG(CASE WHEN status_code >= 400 THEN 1.0 ELSE 0.0 END) OVER (
            PARTITION BY service_name, endpoint
            ORDER BY timestamp
            ROWS BETWEEN 100 PRECEDING AND CURRENT ROW
        ) AS error_rate
    FROM telemetry_logs
)
SELECT 
    service_name,
    endpoint,
    error_rate,
    timestamp
FROM windowed_errors
WHERE error_rate > 0.15 -- Flag when error rate exceeds 15%
ORDER BY timestamp DESC;
