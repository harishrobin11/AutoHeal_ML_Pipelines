import sqlite3
import uuid
import json
import numpy as np
from typing import List, Dict, Any, Optional
from config.settings import settings

class ZScoreAnomalyEngine:
    """
    Computes rolling Z-score metrics over telemetry streams using SQL window logic & NumPy calculations.
    Flags statistical anomaly spikes (Z > 3.0) and persists anomaly events.
    """
    
    def __init__(self, db_path: Optional[str] = None, threshold: float = None):
        self.db_path = db_path or settings.SQLITE_DB_PATH
        self.threshold = threshold or settings.ZSCORE_THRESHOLD

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def compute_z_scores(self, window_size: int = 50) -> List[Dict[str, Any]]:
        """
        Calculates rolling Z-scores for response latencies per (service_name, endpoint) group.
        Returns all records with calculated rolling_mean, rolling_std, and z_score.
        """
        query = """
        SELECT id, service_name, endpoint, status_code, response_time_ms, payload_json, payload_schema_hash, error_message, timestamp
        FROM telemetry_logs
        ORDER BY service_name, endpoint, timestamp ASC
        """
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = [dict(r) for r in cursor.fetchall()]

        # Group records by (service_name, endpoint) to calculate windowed metrics
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in rows:
            key = f"{record['service_name']}|{record['endpoint']}"
            grouped.setdefault(key, []).append(record)

        processed_records = []
        for key, service_records in grouped.items():
            latencies = [r["response_time_ms"] for r in service_records]
            
            for idx, r in enumerate(service_records):
                # Window slicing
                start_idx = max(0, idx - window_size)
                window = latencies[start_idx:idx+1]
                
                if len(window) < 5:
                    mean_val = float(np.mean(window)) if window else 0.0
                    std_val = 0.0
                    z_score = 0.0
                else:
                    mean_val = float(np.mean(window))
                    std_val = float(np.std(window))
                    z_score = (r["response_time_ms"] - mean_val) / std_val if std_val > 1e-6 else 0.0

                r_copy = dict(r)
                r_copy["rolling_mean"] = round(mean_val, 2)
                r_copy["rolling_std"] = round(std_val, 2)
                r_copy["z_score"] = round(float(z_score), 3)
                processed_records.append(r_copy)

        return processed_records

    def detect_and_flag_anomalies(self) -> List[Dict[str, Any]]:
        """
        Executes rolling Z-score detection and identifies records breaching the Z-score threshold or error status.
        Inserts new anomaly events into `anomaly_events` table.
        """
        analyzed_logs = self.compute_z_scores()
        detected_anomalies = []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for log in analyzed_logs:
                is_z_anomaly = abs(log["z_score"]) >= self.threshold
                is_status_anomaly = log["status_code"] >= 500
                is_schema_anomaly = log.get("error_message") is not None and "KeyError" in log["error_message"]

                if is_z_anomaly or is_status_anomaly or is_schema_anomaly:
                    # Determine metric type and probable root cause category
                    metric_type = "latency" if is_z_anomaly else "error_rate"
                    if is_schema_anomaly:
                        root_cause = "SchemaBreak"
                    elif log["status_code"] == 400:
                        root_cause = "TypeMismatch"
                    elif is_z_anomaly:
                        root_cause = "LatencySpike"
                    else:
                        root_cause = "UnknownFailure"

                    anomaly_id = str(uuid.uuid4())
                    event = {
                        "id": anomaly_id,
                        "telemetry_id": log["id"],
                        "service_name": log["service_name"],
                        "endpoint": log["endpoint"],
                        "metric_type": metric_type,
                        "z_score": log["z_score"],
                        "root_cause_category": root_cause,
                        "top_feature": "response_time_ms" if is_z_anomaly else "payload_json",
                        "feature_importance_json": json.dumps({"z_score": log["z_score"], "status_code": log["status_code"]}),
                        "status": "DETECTED",
                        "timestamp": log["timestamp"]
                    }

                    # Check if already logged for this telemetry_id
                    cursor.execute("SELECT id FROM anomaly_events WHERE telemetry_id = ?", (log["id"],))
                    if not cursor.fetchone():
                        cursor.execute("""
                        INSERT INTO anomaly_events (
                            id, telemetry_id, service_name, endpoint, metric_type, z_score, 
                            root_cause_category, top_feature, feature_importance_json, status, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            event["id"], event["telemetry_id"], event["service_name"], event["endpoint"],
                            event["metric_type"], event["z_score"], event["root_cause_category"],
                            event["top_feature"], event["feature_importance_json"], event["status"], event["timestamp"]
                        ))
                        detected_anomalies.append(event)
            
            conn.commit()
            
        return detected_anomalies
