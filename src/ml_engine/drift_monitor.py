import numpy as np
import pandas as pd
from typing import Dict, Any, List

class TelemetryDriftMonitor:
    """
    Monitors statistical feature drift and payload schema drift between reference 
    and current production telemetry streams.
    """

    def __init__(self, reference_logs: List[Dict[str, Any]] = None):
        self.reference_logs = reference_logs or []

    def set_reference(self, logs: List[Dict[str, Any]]):
        self.reference_logs = logs

    def detect_drift(self, current_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes Kolmogorov-Smirnov (KS) test or statistical distribution drift 
        for numerical features (latency) and schema hash drift for categorical features.
        """
        if not self.reference_logs or not current_logs:
            return {"drift_detected": False, "drift_score": 0.0, "details": "Insufficient baseline telemetry"}

        ref_latencies = [r["response_time_ms"] for r in self.reference_logs]
        curr_latencies = [r["response_time_ms"] for r in current_logs]

        ref_mean = float(np.mean(ref_latencies))
        curr_mean = float(np.mean(curr_latencies))

        # Check mean latency shift
        latency_shift_pct = abs(curr_mean - ref_mean) / (ref_mean + 1e-5) * 100.0

        # Check schema hash stability
        ref_hashes = set(r.get("payload_schema_hash") for r in self.reference_logs if r.get("payload_schema_hash"))
        curr_hashes = set(r.get("payload_schema_hash") for r in current_logs if r.get("payload_schema_hash"))
        
        new_schemas = curr_hashes - ref_hashes
        schema_drift = len(new_schemas) > 0

        drift_detected = latency_shift_pct > 50.0 or schema_drift

        return {
            "drift_detected": drift_detected,
            "latency_shift_pct": round(latency_shift_pct, 2),
            "reference_latency_mean_ms": round(ref_mean, 2),
            "current_latency_mean_ms": round(curr_mean, 2),
            "schema_drift_detected": schema_drift,
            "new_schema_count": len(new_schemas),
            "status": "DRIFT_ALERT" if drift_detected else "STABLE"
        }
