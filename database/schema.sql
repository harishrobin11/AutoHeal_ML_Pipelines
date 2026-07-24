-- AutoHeal-ML Database Schema (PostgreSQL & SQLite Compatible)

CREATE TABLE IF NOT EXISTS telemetry_logs (
    id VARCHAR(36) PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms DOUBLE PRECISION NOT NULL,
    payload_json TEXT,
    payload_schema_hash VARCHAR(64),
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS anomaly_events (
    id VARCHAR(36) PRIMARY KEY,
    telemetry_id VARCHAR(36) REFERENCES telemetry_logs(id),
    service_name VARCHAR(100) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- 'latency', 'error_rate', 'schema_drift'
    z_score DOUBLE PRECISION NOT NULL,
    root_cause_category VARCHAR(100), -- 'SchemaBreak', 'DataDrift', 'LatencySpike', 'TypeMismatch'
    top_feature VARCHAR(100),
    feature_importance_json TEXT,
    status VARCHAR(50) DEFAULT 'DETECTED', -- 'DETECTED', 'INVESTIGATING', 'PATCHED', 'RESOLVED'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_remediations (
    id VARCHAR(36) PRIMARY KEY,
    anomaly_id VARCHAR(36) REFERENCES anomaly_events(id),
    agent_name VARCHAR(100) NOT NULL,
    patch_code TEXT,
    patch_diff TEXT,
    unit_tests_passed BOOLEAN DEFAULT FALSE,
    guardrails_passed BOOLEAN DEFAULT FALSE,
    pr_url VARCHAR(255),
    status VARCHAR(50) DEFAULT 'IN_PROGRESS', -- 'IN_PROGRESS', 'VALIDATED', 'FAILED', 'PR_CREATED'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_telemetry_service_time ON telemetry_logs(service_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_anomaly_status ON anomaly_events(status);
