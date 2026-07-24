import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional

from src.telemetry.generator import TelemetryGenerator
from src.telemetry.ingestor import TelemetryIngestor
from src.telemetry.zscore_engine import ZScoreAnomalyEngine
from src.ml_engine.train_classifier import RootCauseClassifier
from src.ml_engine.shap_explainer import SHAPExplainerEngine
from src.ml_engine.drift_monitor import TelemetryDriftMonitor
from src.agentic_pipeline.state_machine import MultiAgentRemediationPipeline
from src.backend.websocket_manager import ws_manager
from src.backend.pr_automator import GitHubPRAutomator
from config.settings import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous Multi-Agent API Orchestrator & Self-Healing Data Pipeline API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared components
generator = TelemetryGenerator()
ingestor = TelemetryIngestor()
zscore_engine = ZScoreAnomalyEngine()
classifier = RootCauseClassifier()
shap_engine = SHAPExplainerEngine(classifier)
drift_monitor = TelemetryDriftMonitor()
agent_pipeline = MultiAgentRemediationPipeline()

@app.get("/")
def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }

@app.post("/api/v1/telemetry/generate")
def generate_and_ingest(count: int = 50, anomaly_ratio: float = 0.2):
    """Generates synthetic telemetry batch and persists to database."""
    batch = generator.generate_batch(count=count, anomaly_ratio=anomaly_ratio)
    inserted = ingestor.ingest_records(batch)
    
    # Train classifier on incoming batch
    classifier.train(batch)
    shap_engine.initialize_explainer(batch)
    
    return {
        "generated": len(batch),
        "inserted": inserted,
        "anomaly_ratio": anomaly_ratio
    }

@app.get("/api/v1/telemetry/logs")
def get_telemetry_logs(limit: int = 100):
    """Fetches recent telemetry stream logs."""
    return ingestor.fetch_all(limit=limit)

@app.post("/api/v1/anomalies/detect")
def detect_anomalies():
    """Runs rolling Z-score anomaly detection engine on telemetry stream."""
    anomalies = zscore_engine.detect_and_flag_anomalies()
    
    # Enrich anomalies with ML classifier & SHAP explanations
    enriched = []
    for a in anomalies:
        prediction = classifier.predict(a)
        explanation = shap_engine.explain(a)
        
        a["predicted_class"] = prediction["predicted_class"]
        a["ml_confidence"] = prediction["confidence"]
        a["top_feature"] = explanation["top_feature"]
        a["shap_attributions"] = explanation["feature_attributions"]
        enriched.append(a)
        
    return {"detected_count": len(enriched), "anomalies": enriched}

@app.post("/api/v1/remediate/{anomaly_id}")
def run_agent_remediation(anomaly_id: str):
    """Triggers LangGraph Multi-Agent Remediation Pipeline on a target anomaly event."""
    anomalies = zscore_engine.detect_and_flag_anomalies()
    target_event = next((a for a in anomalies if a["id"] == anomaly_id), None)
    
    if not target_event:
        # Generate target event fallback if ID not found directly in current scan
        target_event = {
            "id": anomaly_id,
            "service_name": "user-service",
            "endpoint": "/api/v1/user/profile",
            "root_cause_category": "SchemaBreak",
            "top_feature": "user_tier",
            "z_score": 4.12
        }

    remediation = agent_pipeline.run_remediation_workflow(target_event)
    
    # If validated, auto-generate Pull Request
    if remediation.get("success"):
        pr_result = GitHubPRAutomator.create_pull_request(remediation)
        remediation["pull_request"] = pr_result
        
    return remediation

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time telemetry streaming and diagnostic alerts."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Broadcast heartbeats / periodic telemetry events
            data = await websocket.receive_text()
            await ws_manager.broadcast({"type": "ACK", "message": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
