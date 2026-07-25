import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time

from src.telemetry.generator import TelemetryGenerator
from src.telemetry.ingestor import TelemetryIngestor
from src.telemetry.zscore_engine import ZScoreAnomalyEngine
from src.ml_engine.train_classifier import RootCauseClassifier
from src.ml_engine.shap_explainer import SHAPExplainerEngine
from src.ml_engine.drift_monitor import TelemetryDriftMonitor
from src.agentic_pipeline.state_machine import MultiAgentRemediationPipeline
from src.backend.pr_automator import GitHubPRAutomator

st.set_page_config(
    page_title="AutoHeal-ML | Autonomous Self-Healing Platform",
    page_icon="⚡",
    layout="wide"
)

# Custom High-End Cyberpunk Glassmorphism Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    /* Header Banner Styling */
    .hero-banner {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
    }
    
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; right: 0; width: 300px; height: 100%;
        background: radial-gradient(circle at 100% 0%, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 6px;
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 8px;
    }

    /* Glassmorphism Metric Cards */
    .metric-card-glass {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card-glass:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px -5px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Tab Header Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Styled Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Header Banner HTML
st.markdown("""
<div class="hero-banner">
    <div class="badge-pill">⚡ Autonomous MLOps & Self-Healing Pipeline</div>
    <div class="main-header">AutoHeal-ML Orchestration Platform</div>
    <div class="sub-header">Real-Time Telemetry Analytics • XGBoost & SHAP Diagnostics • LangGraph Multi-Agent Remediation</div>
</div>
""", unsafe_allow_html=True)

# Initialize engines
@st.cache_resource
def load_engines():
    gen = TelemetryGenerator()
    ing = TelemetryIngestor()
    z_eng = ZScoreAnomalyEngine()
    clf = RootCauseClassifier()
    shap_eng = SHAPExplainerEngine(clf)
    drift_mon = TelemetryDriftMonitor()
    pipeline = MultiAgentRemediationPipeline()
    return gen, ing, z_eng, clf, shap_eng, drift_mon, pipeline

generator, ingestor, zscore_engine, classifier, shap_engine, drift_monitor, agent_pipeline = load_engines()

# Sidebar Controls
st.sidebar.markdown("### 🎛️ Telemetry Stream Controls")
st.sidebar.markdown("---")
batch_size = st.sidebar.slider("Synthetic Batch Size", 20, 200, 50, help="Number of telemetry logs generated per stream ingestion")
anomaly_rate = st.sidebar.slider("Anomaly Injection Ratio", 0.0, 0.5, 0.2, 0.05, help="Percentage of failure modes injected (SchemaBreak, LatencySpike, TypeMismatch)")

if st.sidebar.button("🚀 Ingest Telemetry Stream", use_container_width=True):
    with st.spinner("Ingesting Stream & Retraining Classifier..."):
        batch = generator.generate_batch(count=batch_size, anomaly_ratio=anomaly_rate)
        ingestor.ingest_records(batch)
        classifier.train(batch)
        shap_engine.initialize_explainer(batch)
        st.sidebar.success(f"Ingested {len(batch)} records into telemetry store!")

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ Active System Engines")
st.sidebar.caption("• **Z-Score Engine**: Rolling Window Anomaly Spikes")
st.sidebar.caption("• **XGBoost Classifier**: Multi-Class Failure Diagnosis")
st.sidebar.caption("• **SHAP Engine**: Feature Importance Attributions")
st.sidebar.caption("• **LangGraph Agents**: Multi-Agent Code Repair")

# Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Telemetry Stream & Z-Scores",
    "🧠 Root-Cause ML & SHAP",
    "🤖 Multi-Agent Remediation",
    "🔀 Git Diffs & PR Automation"
])

# Fetch data
records = ingestor.fetch_all(limit=300)
df = pd.DataFrame(records) if records else pd.DataFrame()

# Tab 1: Telemetry Stream & Z-Scores
with tab1:
    st.subheader("Real-Time Telemetry Stream & Windowed Z-Score Spikes")
    
    if not df.empty:
        # Calculate Z-Scores
        analyzed_logs = zscore_engine.compute_z_scores()
        df_z = pd.DataFrame(analyzed_logs)
        
        error_count = len(df[df["status_code"] >= 400])
        avg_lat = round(df["response_time_ms"].mean(), 1)
        anomaly_count = len(df_z[abs(df_z["z_score"]) >= 2.0]) if not df_z.empty else 0
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card-glass"><div class="metric-label">Total Logs Processed</div><div class="metric-val">{len(df)}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card-glass"><div class="metric-label">HTTP Errors (4xx/5xx)</div><div class="metric-val" style="color: #f43f5e;">{error_count} ({round(error_count/len(df)*100, 1)}%)</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card-glass"><div class="metric-label">Mean Latency</div><div class="metric-val" style="color: #38bdf8;">{avg_lat} ms</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card-glass"><div class="metric-label">Z-Score Spikes (|Z| ≥ 2.0)</div><div class="metric-val" style="color: #fbbf24;">{anomaly_count}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if not df_z.empty:
            # Latency Z-Score Plot
            fig = px.scatter(
                df_z,
                x="timestamp",
                y="response_time_ms",
                color="z_score",
                size="response_time_ms",
                hover_data=["service_name", "endpoint", "status_code", "z_score"],
                title="Response Latency vs. Rolling Z-Score Spikes",
                color_continuous_scale="Purples",
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                font=dict(family="Plus Jakarta Sans, sans-serif"),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            fig.add_hline(y=300, line_dash="dash", line_color="#fbbf24", annotation_text="Latency Alert Threshold (300ms)", annotation_position="top left")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📋 Telemetry Log Registry")
            st.dataframe(df_z[["timestamp", "service_name", "endpoint", "status_code", "response_time_ms", "z_score", "error_message"]].head(25), use_container_width=True)
    else:
        st.info("No telemetry logs found. Click 'Ingest Telemetry Stream' in the sidebar to generate live traffic!")

# Tab 2: Root-Cause ML & SHAP
with tab2:
    st.subheader("Predictive Root-Cause Classification & SHAP Feature Attributions")
    
    anomalies = zscore_engine.detect_and_flag_anomalies()
    
    if anomalies:
        st.success(f"Detected {len(anomalies)} telemetry anomaly events requiring diagnosis.")
        selected_idx = st.selectbox("Select Anomaly Event to Diagnose", range(len(anomalies)), format_func=lambda i: f"[{anomalies[i]['root_cause_category']}] {anomalies[i]['service_name']} - {anomalies[i]['endpoint']} (Z={anomalies[i]['z_score']})", key="tab2_selectbox")
        
        target_anomaly = anomalies[selected_idx]
        
        pred = classifier.predict(target_anomaly)
        shap_res = shap_engine.explain(target_anomaly)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🎯 ML Failure Mode Prediction")
            st.write(f"**Predicted Failure Class**: `{pred['predicted_class']}`")
            st.write(f"**Confidence Level**: `{pred['confidence'] * 100:.1f}%`")
            
            # Probability Bar Chart
            prob_df = pd.DataFrame(list(pred["probabilities"].items()), columns=["Failure Mode", "Probability"])
            fig_prob = px.bar(
                prob_df, 
                x="Probability", 
                y="Failure Mode", 
                orientation="h", 
                color="Probability", 
                color_continuous_scale="Cividis",
                template="plotly_dark"
            )
            fig_prob.update_layout(
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                font=dict(family="Plus Jakarta Sans, sans-serif"),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with c2:
            st.markdown("#### 🔬 SHAP Feature Importance Attribution")
            st.write(f"**Top Isolated Feature**: `{shap_res['top_feature']}`")
            st.info(shap_res["summary"])
            
            attr_df = pd.DataFrame(list(shap_res["feature_attributions"].items()), columns=["Feature", "SHAP Value"])
            fig_attr = px.bar(
                attr_df, 
                x="SHAP Value", 
                y="Feature", 
                orientation="h", 
                color="SHAP Value", 
                color_continuous_scale="Tealgrn",
                template="plotly_dark"
            )
            fig_attr.update_layout(
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                font=dict(family="Plus Jakarta Sans, sans-serif"),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_attr, use_container_width=True)
    else:
        st.info("No anomalies detected in the current telemetry window.")

# Tab 3: Multi-Agent Remediation
with tab3:
    st.subheader("LangGraph Multi-Agent Remediation Orchestrator")
    st.markdown("Agents: **Investigator** (Diagnosis) ➔ **Developer** (Backward-Compatible Patch) ➔ **Validator** (AST & Test Sandbox)")
    
    anomalies = zscore_engine.detect_and_flag_anomalies()
    if anomalies:
        selected_anomaly_idx = st.selectbox("Target Anomaly Event for Auto-Healing", range(len(anomalies)), format_func=lambda i: f"ID: {anomalies[i]['id'][:8]} | {anomalies[i]['service_name']} | {anomalies[i]['root_cause_category']}", key="tab3_selectbox")
        anomaly_to_fix = anomalies[selected_anomaly_idx]
        
        if st.button("🤖 Trigger Multi-Agent Remediation"):
            with st.spinner("Multi-Agent System Orchestrating Remediation..."):
                remediation_res = agent_pipeline.run_remediation_workflow(anomaly_to_fix)
                st.session_state["last_remediation"] = remediation_res
                
            st.success("Remediation Workflow Complete!")
    else:
        st.info("No anomaly events detected for remediation yet. Click 'Ingest Telemetry Stream' in the sidebar!")
            
    if "last_remediation" in st.session_state:
        res = st.session_state["last_remediation"]
        
        st.markdown("### 📋 Multi-Agent Execution Trace")
        for trace_step in res.get("execution_trace", []):
            step_name = trace_step["step"]
            status = trace_step["status"]
            
            if status == "COMPLETED":
                with st.expander(f"✅ Step {step_name}: {status}", expanded=True):
                    st.json(trace_step.get("output", {}))

# Tab 4: Git Diffs & PR Automation
with tab4:
    st.subheader("Side-by-Side Patch Git Diff & GitHub Pull Request")
    
    if "last_remediation" in st.session_state:
        res = st.session_state["last_remediation"]
        patch = res.get("patch_solution", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📄 Generated Backward-Compatible Code")
            st.code(patch.get("patch_code", ""), language="python")
            
        with col2:
            st.markdown("#### 🔀 Unified Git Diff")
            st.code(patch.get("patch_diff", ""), language="diff")
            
        if st.button("🐙 Auto-Generate GitHub Pull Request"):
            pr_res = GitHubPRAutomator.create_pull_request(res)
            st.success(f"Pull Request Created! View PR at: [{pr_res['pr_url']}]({pr_res['pr_url']})")
            st.markdown(f"**PR Title**: `{pr_res['pr_title']}`")
            st.markdown(f"**Target Branch**: `{pr_res['branch_name']}`")
            with st.expander("View Pull Request Body", expanded=True):
                st.markdown(pr_res["pr_body"])
    else:
        st.info("Trigger a multi-agent remediation in Tab 3 to view generated patch diffs and pull requests!")
