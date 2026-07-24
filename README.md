# ⚡ AutoHeal-ML: Autonomous Multi-Agent API Orchestrator & Self-Healing Data Pipeline

AutoHeal-ML is a production-grade self-healing MLOps and Data Engineering platform. It continuously ingests telemetry streams, detects latency/error anomalies using PostgreSQL window functions ($Z > 3.0$), diagnoses root causes with XGBoost and SHAP explainability, and deploys a LangGraph multi-agent architecture (Investigator, Developer, Validator) using Model Context Protocol (MCP) tools to auto-generate backward-compatible patches and pull requests.

---

## 🏗️ Architecture Pillars

1. **Data Analytics Pillar (PostgreSQL SQL Engine)**
   - Telemetry storage & window function Z-score calculation over sliding time windows.
2. **Data Science Pillar (XGBoost & SHAP Explainability)**
   - Classifies failure categories (`SchemaBreak`, `LatencySpike`, `TypeMismatch`) and outputs local feature attributions via SHAP.
3. **AI / ML Engineering Pillar (LangGraph & MCP)**
   - Multi-agent orchestration loop:
     - 🔍 **Investigator Agent**: Locates broken payload keys & codebase symbols.
     - 🛠️ **Developer Agent**: Writes backward-compatible code patches.
     - 🧪 **Validator Agent**: Runs AST syntax checks and unit tests.
4. **MLOps & UI Pillar (FastAPI, Streamlit & Docker)**
   - WebSocket stream backend, Streamlit diagnostic dashboard, and GitHub PR creation.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/
```

### 3. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### 4. Launch FastAPI Backend
```bash
uvicorn src.backend.app:app --reload
```

---

## 🐳 Docker Deployment
```bash
docker-compose up --build
```
