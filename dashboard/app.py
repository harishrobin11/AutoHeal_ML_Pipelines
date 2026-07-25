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
    page_title="AutoHeal-ML | Self-Healing Data Pipeline",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ AutoHeal-ML Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous Multi-Agent API Orchestrator & Self-Healing Telemetry Pipeline</div>', unsafe_allow_html=True)

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
st.sidebar.title("🎛️ Telemetry Stream Controls")
batch_size = st.sidebar.slider("Synthetic Batch Size", 20, 200, 50)
anomaly_rate = st.sidebar.slider("Anomaly Injection Ratio", 0.0, 0.5, 0.2, 0.05)

if st.sidebar.button("🚀 Ingest Telemetry Stream"):
    batch = generator.generate_batch(count=batch_size, anomaly_ratio=anomaly_rate)
    ingestor.ingest_records(batch)
    classifier.train(batch)
    shap_engine.initialize_explainer(batch)
    st.sidebar.success(f"Ingested {len(batch)} records into telemetry store!")

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
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Logs Processed", len(df))
        error_count = len(df[df["status_code"] >= 400])
        col2.metric("Failed Requests", error_count, f"{round(error_count/len(df)*100, 1)}%")
        avg_lat = round(df["response_time_ms"].mean(), 2)
        col3.metric("Avg Latency (ms)", f"{avg_lat} ms")
        
        # Calculate Z-Scores
        analyzed_logs = zscore_engine.compute_z_scores()
        df_z = pd.DataFrame(analyzed_logs)
        
        anomaly_count = len(df_z[abs(df_z["z_score"]) >= 3.0]) if not df_z.empty else 0
        col4.metric("Z-Score Anomalies (Z > 3.0)", anomaly_count, delta_color="inverse")

        if not df_z.empty:
            # Latency Z-Score Plot
            fig = px.scatter(
                df_z,
                x="timestamp",
                y="response_time_ms",
                color="z_score",
                size="response_time_ms",
                hover_data=["service_name", "endpoint", "status_code", "z_score"],
                title="Response Latency vs. Rolling Z-Score Anomalies",
                color_continuous_scale="Reds"
            )
            fig.add_hline(y=300, line_dash="dash", line_color="orange", annotation_text="Latency Alert Threshold (300ms)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Telemetry Stream Table")
            st.dataframe(df_z[["timestamp", "service_name", "endpoint", "status_code", "response_time_ms", "z_score", "error_message"]].head(20), use_container_width=True)
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
            st.markdown("#### 🎯 ML Failure Classification")
            st.write(f"**Predicted Failure Mode**: `{pred['predicted_class']}`")
            st.write(f"**Confidence Score**: `{pred['confidence'] * 100:.1f}%`")
            
            # Probability Bar Chart
            prob_df = pd.DataFrame(list(pred["probabilities"].items()), columns=["Failure Mode", "Probability"])
            fig_prob = px.bar(prob_df, x="Probability", y="Failure Mode", orientation="h", color="Probability", color_continuous_scale="Viridis")
            st.plotly_chart(fig_prob, use_container_width=True)

        with c2:
            st.markdown("#### 🔬 SHAP Feature Importance Attribution")
            st.write(f"**Top Isolated Feature**: `{shap_res['top_feature']}`")
            st.info(shap_res["summary"])
            
            attr_df = pd.DataFrame(list(shap_res["feature_attributions"].items()), columns=["Feature", "SHAP Value"])
            fig_attr = px.bar(attr_df, x="SHAP Value", y="Feature", orientation="h", color="SHAP Value", color_continuous_scale="Purples")
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
