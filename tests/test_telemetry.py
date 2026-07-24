import os
import pytest
from src.telemetry.generator import TelemetryGenerator
from src.telemetry.ingestor import TelemetryIngestor
from src.telemetry.zscore_engine import ZScoreAnomalyEngine

TEST_DB_PATH = "test_autoheal.db"

@pytest.fixture(autouse=True)
def cleanup():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    yield
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

def test_telemetry_generator():
    gen = TelemetryGenerator(seed=123)
    batch = gen.generate_batch(count=20, anomaly_ratio=0.3)
    assert len(batch) == 20
    assert any(b["status_code"] >= 400 or b["response_time_ms"] > 300 for b in batch)

def test_telemetry_ingestor():
    gen = TelemetryGenerator(seed=123)
    batch = gen.generate_batch(count=15, anomaly_ratio=0.1)
    
    ingestor = TelemetryIngestor(db_path=TEST_DB_PATH)
    inserted = ingestor.ingest_records(batch)
    assert inserted == 15
    
    fetched = ingestor.fetch_all(limit=50)
    assert len(fetched) == 15

def test_zscore_engine_detection():
    gen = TelemetryGenerator(seed=42)
    batch = gen.generate_batch(count=50, anomaly_ratio=0.2)
    
    ingestor = TelemetryIngestor(db_path=TEST_DB_PATH)
    ingestor.ingest_records(batch)
    
    z_engine = ZScoreAnomalyEngine(db_path=TEST_DB_PATH, threshold=2.5)
    anomalies = z_engine.detect_and_flag_anomalies()
    assert isinstance(anomalies, list)
