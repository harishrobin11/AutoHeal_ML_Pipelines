import pytest
from src.telemetry.generator import TelemetryGenerator
from src.ml_engine.train_classifier import RootCauseClassifier
from src.ml_engine.shap_explainer import SHAPExplainerEngine
from src.ml_engine.drift_monitor import TelemetryDriftMonitor

def test_ml_root_cause_classifier():
    gen = TelemetryGenerator(seed=100)
    train_batch = gen.generate_batch(count=40, anomaly_ratio=0.4)
    
    classifier = RootCauseClassifier()
    res = classifier.train(train_batch)
    assert res["status"] == "trained"
    assert classifier.is_trained
    
    sample_anomaly = gen.generate_record(anomaly_type="schema_break")
    pred = classifier.predict(sample_anomaly)
    assert "predicted_class" in pred
    assert "confidence" in pred

def test_shap_explainer():
    gen = TelemetryGenerator(seed=100)
    batch = gen.generate_batch(count=30, anomaly_ratio=0.3)
    
    classifier = RootCauseClassifier()
    classifier.train(batch)
    
    shap_eng = SHAPExplainerEngine(classifier)
    shap_eng.initialize_explainer(batch)
    
    sample_anomaly = gen.generate_record(anomaly_type="schema_break")
    exp = shap_eng.explain(sample_anomaly)
    assert "top_feature" in exp
    assert "feature_attributions" in exp

def test_drift_monitor():
    gen = TelemetryGenerator(seed=200)
    ref_batch = gen.generate_batch(count=30, anomaly_ratio=0.0)
    curr_batch = gen.generate_batch(count=30, anomaly_ratio=0.5)
    
    monitor = TelemetryDriftMonitor(reference_logs=ref_batch)
    drift_res = monitor.detect_drift(curr_batch)
    assert "drift_detected" in drift_res
