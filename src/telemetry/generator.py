import uuid
import json
import random
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List

class TelemetryGenerator:
    """Generates synthetic API request telemetry logs, including normal traffic, latency spikes, and schema anomalies."""
    
    SERVICES = ["user-service", "payment-service", "recommendation-service", "inventory-service"]
    ENDPOINTS = {
        "user-service": ["/api/v1/user/profile", "/api/v1/user/auth"],
        "payment-service": ["/api/v1/payment/checkout", "/api/v1/payment/refund"],
        "recommendation-service": ["/api/v1/recommend/feed"],
        "inventory-service": ["/api/v1/inventory/items"]
    }
    
    NORMAL_SCHEMA_USER = {
        "user_id": "usr_9981",
        "user_tier": "premium",
        "device_type": "mobile_ios",
        "region": "us-east-1",
        "request_version": "v1.2.0"
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)

    @staticmethod
    def compute_schema_hash(payload: Dict[str, Any]) -> str:
        """Computes a SHA256 hash of sorted key names and their data types."""
        schema_representation = ",".join(sorted(f"{k}:{type(v).__name__}" for k, v in payload.items()))
        return hashlib.sha256(schema_representation.encode()).hexdigest()[:16]

    def generate_record(self, anomaly_type: str = None) -> Dict[str, Any]:
        """
        Generates a single telemetry log dictionary.
        anomaly_type options: None (normal), 'latency_spike', 'schema_break', 'type_mismatch', 'error_rate'
        """
        service = random.choice(self.SERVICES)
        endpoint = random.choice(self.ENDPOINTS[service])
        
        status_code = 200
        response_time_ms = float(random.normalvariate(45, 12))  # Base latency ~45ms
        response_time_ms = max(5.0, response_time_ms)
        error_message = None
        
        payload = dict(self.NORMAL_SCHEMA_USER)
        payload["user_id"] = f"usr_{random.randint(1000, 9999)}"

        if anomaly_type == "latency_spike":
            response_time_ms = float(random.uniform(450.0, 1200.0))  # Latency spike >450ms
        elif anomaly_type == "schema_break":
            # Simulate broken payload missing required field 'user_tier'
            payload.pop("user_tier", None)
            payload["legacy_tier_id"] = 101  # Unexpected breaking key change
            status_code = 500
            error_message = "KeyError: 'user_tier' missing in payload body"
        elif anomaly_type == "type_mismatch":
            # Simulate string user_id sent as integer
            payload["user_tier"] = 12345
            status_code = 400
            error_message = "TypeError: Expected user_tier to be String, got Int"
        elif anomaly_type == "error_rate":
            status_code = 503
            error_message = "ServiceUnavailable: Database Connection Timeout"
            response_time_ms = float(random.uniform(300.0, 800.0))

        schema_hash = self.compute_schema_hash(payload)

        return {
            "id": str(uuid.uuid4()),
            "service_name": service,
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time_ms": round(response_time_ms, 2),
            "payload_json": json.dumps(payload),
            "payload_schema_hash": schema_hash,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def generate_batch(self, count: int = 100, anomaly_ratio: float = 0.1) -> List[Dict[str, Any]]:
        """Generates a batch of telemetry logs with specified anomaly ratio."""
        records = []
        anomaly_types = ["latency_spike", "schema_break", "type_mismatch", "error_rate"]
        
        for i in range(count):
            if random.random() < anomaly_ratio:
                chosen_anomaly = random.choice(anomaly_types)
                records.append(self.generate_record(anomaly_type=chosen_anomaly))
            else:
                records.append(self.generate_record(anomaly_type=None))
                
        return records

if __name__ == "__main__":
    gen = TelemetryGenerator()
    sample = gen.generate_batch(5, anomaly_ratio=0.4)
    print(f"Generated {len(sample)} sample telemetry records:")
    print(json.dumps(sample[0], indent=2))
