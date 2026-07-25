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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ─── GLOBAL RESET ─── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #030712 !important;
    }
    code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

    /* ─── PAGE LAYOUT ─── */
    .block-container { padding: 0 2rem 4rem 2rem !important; max-width: 100% !important; }
    header[data-testid="stHeader"] { background: rgba(3,7,18,0.95) !important; backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255,255,255,0.04); }

    /* ─── ANIMATED HERO BANNER ─── */
    .hero-wrap {
        position: relative;
        background: linear-gradient(135deg, #0f0c29, #1a1a3e, #0f0c29);
        border-radius: 20px;
        padding: 40px 44px;
        margin: 20px 0 28px 0;
        overflow: hidden;
        border: 1px solid rgba(139, 92, 246, 0.25);
        box-shadow: 0 0 80px -20px rgba(139, 92, 246, 0.3), 0 40px 60px -20px rgba(0,0,0,0.6);
    }
    .hero-wrap::before {
        content: '';
        position: absolute; inset: 0;
        background:
            radial-gradient(ellipse 60% 60% at 10% 50%, rgba(99,102,241,0.12) 0%, transparent 65%),
            radial-gradient(ellipse 50% 70% at 90% 20%, rgba(168,85,247,0.12) 0%, transparent 65%),
            radial-gradient(ellipse 40% 40% at 50% 100%, rgba(6,182,212,0.07) 0%, transparent 65%);
        pointer-events: none;
    }
    .hero-wrap::after {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.6), rgba(6,182,212,0.6), transparent);
    }
    .hero-pill {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(139,92,246,0.12);
        border: 1px solid rgba(139,92,246,0.35);
        border-radius: 100px;
        padding: 6px 16px 6px 10px;
        font-size: 0.78rem; font-weight: 600;
        color: #a78bfa;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 18px;
    }
    .hero-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #a78bfa;
        box-shadow: 0 0 8px #a78bfa;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.85)} }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem; font-weight: 700; line-height: 1.1;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #e0e7ff 0%, #c4b5fd 35%, #38bdf8 70%, #67e8f9 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 12px 0;
    }
    .hero-sub {
        font-size: 1.05rem; font-weight: 400; color: #64748b; line-height: 1.6;
        max-width: 680px;
    }
    .hero-sub span { color: #94a3b8; font-weight: 500; }
    .hero-tags { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
    .hero-tag {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 0.78rem; font-weight: 500; color: #94a3b8;
    }
    .hero-tag-icon { font-size: 0.85rem; }
    .hero-glow-orb {
        position: absolute; right: 60px; top: 50%; transform: translateY(-50%);
        width: 180px; height: 180px; border-radius: 50%;
        background: radial-gradient(circle, rgba(139,92,246,0.18) 0%, transparent 70%);
        border: 1px solid rgba(139,92,246,0.15);
        animation: orb-float 5s ease-in-out infinite;
    }
    .hero-glow-orb::after {
        content: '⚡';
        position: absolute; inset: 0;
        display: flex; align-items: center; justify-content: center;
        font-size: 4rem;
        animation: orb-float 5s ease-in-out infinite reverse;
    }
    @keyframes orb-float { 0%,100%{transform:translateY(-50%) scale(1)} 50%{transform:translateY(-56%) scale(1.05)} }

    /* ─── METRIC CARDS ─── */
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    .kpi-card {
        background: linear-gradient(145deg, rgba(15,20,40,0.9), rgba(10,15,30,0.95));
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        cursor: default;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(139,92,246,0.4);
        box-shadow: 0 20px 40px -10px rgba(139,92,246,0.2), 0 0 0 1px rgba(139,92,246,0.1);
    }
    .kpi-card::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        border-radius: 16px 16px 0 0;
    }
    .kpi-card.blue::before { background: linear-gradient(90deg, #3b82f6, #38bdf8); }
    .kpi-card.rose::before { background: linear-gradient(90deg, #f43f5e, #fb7185); }
    .kpi-card.violet::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
    .kpi-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .kpi-icon {
        width: 40px; height: 40px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; margin-bottom: 14px;
    }
    .kpi-icon.blue { background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.2); }
    .kpi-icon.rose { background: rgba(244,63,94,0.12); border: 1px solid rgba(244,63,94,0.2); }
    .kpi-icon.violet { background: rgba(139,92,246,0.12); border: 1px solid rgba(139,92,246,0.2); }
    .kpi-icon.amber { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.2); }
    .kpi-label { font-size: 0.78rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .kpi-value { font-family: 'Space Grotesk', monospace; font-size: 2.1rem; font-weight: 700; line-height: 1; color: #f1f5f9; }
    .kpi-sub { font-size: 0.78rem; font-weight: 500; margin-top: 6px; }
    .kpi-sub.up { color: #34d399; }
    .kpi-sub.down { color: #f43f5e; }
    .kpi-sub.neutral { color: #94a3b8; }

    /* ─── SECTION CARDS ─── */
    .section-card {
        background: rgba(10, 15, 30, 0.7);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 24px 26px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem; font-weight: 600; color: #e2e8f0;
        margin-bottom: 4px; display: flex; align-items: center; gap: 10px;
    }
    .section-desc { font-size: 0.85rem; color: #475569; margin-bottom: 20px; }

    /* ─── STATUS BADGES ─── */
    .badge { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .badge-success { background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.25); color: #34d399; }
    .badge-error   { background: rgba(244,63,94,0.1);  border: 1px solid rgba(244,63,94,0.25);  color: #f43f5e; }
    .badge-warning { background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.25); color: #fbbf24; }
    .badge-info    { background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.25); color: #38bdf8; }
    .badge-violet  { background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.25); color: #a78bfa; }

    /* ─── AGENT PIPELINE TRACE ─── */
    .agent-step {
        background: linear-gradient(135deg, rgba(15,23,42,0.8), rgba(10,15,30,0.9));
        border: 1px solid rgba(139,92,246,0.2);
        border-left: 3px solid #8b5cf6;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin-bottom: 12px;
        position: relative;
    }
    .agent-step-header { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
    .agent-step-name { font-weight: 700; color: #a78bfa; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .agent-step-status { font-size: 0.75rem; font-weight: 600; color: #34d399; }

    /* ─── TABS ─── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(10,15,30,0.8);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 6px;
        gap: 4px;
        backdrop-filter: blur(10px);
        margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 22px;
        color: #475569;
        font-weight: 600;
        font-size: 0.9rem;
        border: none !important;
        transition: all 0.2s ease;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #94a3b8 !important; background: rgba(255,255,255,0.04) !important; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15)) !important;
        color: #c4b5fd !important;
        box-shadow: inset 0 0 0 1px rgba(139,92,246,0.3), 0 4px 20px rgba(139,92,246,0.15) !important;
    }

    /* ─── SIDEBAR ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070c1a 0%, #03070f 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    section[data-testid="stSidebar"] .sidebar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem; font-weight: 700;
        color: #475569; text-transform: uppercase; letter-spacing: 0.08em;
        padding: 8px 0 4px;
    }
    .sidebar-engine-item {
        display: flex; align-items: center; gap: 10px;
        padding: 8px 10px; border-radius: 8px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 6px;
        font-size: 0.8rem; color: #64748b;
    }
    .sidebar-engine-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 6px #34d399;
        flex-shrink: 0;
    }

    /* ─── BUTTONS ─── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #fff !important;
        font-weight: 700 !important; font-size: 0.9rem !important;
        border-radius: 12px !important; border: none !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.15) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.01em !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(99,102,241,0.5), inset 0 1px 0 rgba(255,255,255,0.2) !important;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ─── SELECTBOX ─── */
    .stSelectbox > div > div {
        background: rgba(10,15,30,0.9) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important; color: #e2e8f0 !important;
    }
    .stSelectbox > div > div:focus-within {
        border-color: rgba(139,92,246,0.5) !important;
        box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
    }

    /* ─── SLIDERS ─── */
    .stSlider > div > div > div > div { background: #6366f1 !important; }
    .stSlider > div > div > div { background: rgba(255,255,255,0.08) !important; }

    /* ─── ALERTS & INFO ─── */
    .stAlert > div, .stInfo > div {
        background: rgba(56,189,248,0.07) !important;
        border: 1px solid rgba(56,189,248,0.2) !important;
        border-radius: 10px !important;
        color: #94a3b8 !important;
    }
    .stSuccess > div {
        background: rgba(52,211,153,0.08) !important;
        border: 1px solid rgba(52,211,153,0.25) !important;
        border-radius: 10px !important;
        color: #6ee7b7 !important;
    }

    /* ─── DATAFRAME ─── */
    .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.07); }

    /* ─── DIVIDER ─── */
    .fancy-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.4), rgba(56,189,248,0.4), transparent);
        margin: 20px 0;
        border: none;
    }

    /* ─── SPINNER ─── */
    .stSpinner > div { border-top-color: #8b5cf6 !important; }
</style>
""", unsafe_allow_html=True)

# ─── HERO BANNER ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-glow-orb"></div>
    <div class="hero-pill">
        <div class="hero-dot"></div>
        Autonomous MLOps Platform · Production Ready
    </div>
    <div class="hero-title">AutoHeal-ML Orchestration Platform</div>
    <div class="hero-sub">
        <span>Real-time telemetry analytics</span> with Z-Score anomaly detection,
        <span>XGBoost root-cause classification</span>, SHAP explainability, and
        <span>LangGraph multi-agent</span> autonomous code remediation pipelines.
    </div>
    <div class="hero-tags">
        <div class="hero-tag"><span class="hero-tag-icon">📡</span> Live Telemetry Ingestor</div>
        <div class="hero-tag"><span class="hero-tag-icon">🧬</span> XGBoost · SHAP · Drift Monitor</div>
        <div class="hero-tag"><span class="hero-tag-icon">🤖</span> LangGraph Agent Network</div>
        <div class="hero-tag"><span class="hero-tag-icon">🐙</span> GitHub PR Automation</div>
        <div class="hero-tag"><span class="hero-tag-icon">🐳</span> Docker · FastAPI · SQLite</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── ENGINE INIT ───────────────────────────────────────────────────────────
@st.cache_resource
def load_engines():
    gen      = TelemetryGenerator()
    ing      = TelemetryIngestor()
    z_eng    = ZScoreAnomalyEngine()
    clf      = RootCauseClassifier()
    shap_eng = SHAPExplainerEngine(clf)
    drift    = TelemetryDriftMonitor()
    pipeline = MultiAgentRemediationPipeline()
    return gen, ing, z_eng, clf, shap_eng, drift, pipeline

generator, ingestor, zscore_engine, classifier, shap_engine, drift_monitor, agent_pipeline = load_engines()

# ─── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Stream Controls</div>', unsafe_allow_html=True)
    batch_size   = st.slider("Batch Size", 20, 200, 50, help="Number of synthetic telemetry logs to ingest per run")
    anomaly_rate = st.slider("Anomaly Injection Ratio", 0.0, 0.5, 0.2, 0.05, help="Fraction of failure-mode logs to inject")

    st.markdown("<br>", unsafe_allow_html=True)
    ingest_btn = st.button("🚀 Ingest Telemetry Stream", width="stretch")
    if ingest_btn:
        with st.spinner("Ingesting stream & retraining classifier..."):
            batch = generator.generate_batch(count=batch_size, anomaly_ratio=anomaly_rate)
            ingestor.ingest_records(batch)
            classifier.train(batch)
            shap_engine.initialize_explainer(batch)
        st.success(f"✅ {len(batch)} records ingested!")

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🟢 Active Engines</div>', unsafe_allow_html=True)
    for engine, desc in [
        ("Z-Score Engine", "Rolling Window Anomaly"),
        ("XGBoost Classifier", "Multi-Class Root Cause"),
        ("SHAP Explainer", "Feature Attributions"),
        ("LangGraph Agents", "Code Remediation"),
        ("PR Automator", "GitHub Integration"),
    ]:
        st.markdown(f'<div class="sidebar-engine-item"><div class="sidebar-engine-dot"></div><div><strong>{engine}</strong><br><span style="font-size:0.72rem;color:#334155">{desc}</span></div></div>', unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Telemetry Stream & Z-Scores",
    "🧠  Root-Cause ML & SHAP",
    "🤖  Multi-Agent Remediation",
    "🔀  Git Diffs & PR Automation",
])

records = ingestor.fetch_all(limit=300)
df = pd.DataFrame(records) if records else pd.DataFrame()

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(5,10,25,0)",
    plot_bgcolor="rgba(5,10,25,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=46, b=10),
    title_font=dict(size=14, color="#e2e8f0", family="Space Grotesk"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.06)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
)

# ─────────────────────────────────────────────────────────────
# TAB 1 — TELEMETRY STREAM & Z-SCORES
# ─────────────────────────────────────────────────────────────
with tab1:
    if not df.empty:
        analyzed_logs  = zscore_engine.compute_z_scores()
        df_z           = pd.DataFrame(analyzed_logs)
        error_count    = len(df[df["status_code"] >= 400])
        avg_lat        = round(df["response_time_ms"].mean(), 1)
        anomaly_count  = len(df_z[abs(df_z["z_score"]) >= 2.0]) if not df_z.empty else 0
        error_pct      = round(error_count / len(df) * 100, 1)

        # KPI cards
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card blue">
                <div class="kpi-icon blue">📡</div>
                <div class="kpi-label">Total Logs Ingested</div>
                <div class="kpi-value">{len(df):,}</div>
                <div class="kpi-sub neutral">Rolling telemetry window</div>
            </div>
            <div class="kpi-card rose">
                <div class="kpi-icon rose">🔥</div>
                <div class="kpi-label">HTTP Errors (4xx/5xx)</div>
                <div class="kpi-value">{error_count:,}</div>
                <div class="kpi-sub down">↑ {error_pct}% failure rate</div>
            </div>
            <div class="kpi-card violet">
                <div class="kpi-icon violet">⚡</div>
                <div class="kpi-label">Z-Score Anomalies</div>
                <div class="kpi-value">{anomaly_count}</div>
                <div class="kpi-sub neutral">|Z| ≥ 2.0 threshold</div>
            </div>
            <div class="kpi-card amber">
                <div class="kpi-icon amber">⏱️</div>
                <div class="kpi-label">Mean Latency (ms)</div>
                <div class="kpi-value">{avg_lat}</div>
                <div class="kpi-sub up">↓ within SLA</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not df_z.empty:
            # Scatter Plot
            fig = px.scatter(
                df_z, x="timestamp", y="response_time_ms",
                color="z_score", size="response_time_ms",
                hover_data=["service_name", "endpoint", "status_code", "z_score"],
                title="📡 Response Latency vs. Rolling Z-Score Anomaly Detection",
                color_continuous_scale=[[0,"#1e1b4b"],[0.3,"#4f46e5"],[0.65,"#8b5cf6"],[1,"#f43f5e"]],
                template="plotly_dark"
            )
            fig.add_hline(y=300, line_dash="dash", line_color="#fbbf24", line_width=1.5,
                          annotation_text="SLA Threshold 300ms", annotation_font_color="#fbbf24", annotation_position="top left")
            fig.update_layout(**CHART_LAYOUT)
            fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5, color="rgba(255,255,255,0.15)")))
            st.plotly_chart(fig, width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                # Error rate by service
                if "service_name" in df_z.columns and "status_code" in df_z.columns:
                    svc_err = df_z.groupby("service_name").apply(lambda x: (x["status_code"] >= 400).mean() * 100).reset_index()
                    svc_err.columns = ["Service", "Error Rate (%)"]
                    svc_err = svc_err.sort_values("Error Rate (%)", ascending=True)
                    fig2 = px.bar(svc_err, x="Error Rate (%)", y="Service", orientation="h",
                                  title="🔥 Error Rate by Microservice",
                                  color="Error Rate (%)", color_continuous_scale=[[0,"#312e81"],[0.5,"#7c3aed"],[1,"#f43f5e"]],
                                  template="plotly_dark")
                    fig2.update_layout(**CHART_LAYOUT)
                    st.plotly_chart(fig2, width="stretch")

            with c2:
                # Latency distribution
                fig3 = px.histogram(df_z, x="response_time_ms", nbins=40,
                                    title="⏱️ Latency Distribution (ms)",
                                    color_discrete_sequence=["#6366f1"],
                                    template="plotly_dark")
                fig3.add_vline(x=300, line_dash="dash", line_color="#fbbf24", line_width=1.5)
                fig3.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig3, width="stretch")

            st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📋 Telemetry Log Registry</div>', unsafe_allow_html=True)
            st.dataframe(
                df_z[["timestamp", "service_name", "endpoint", "status_code", "response_time_ms", "z_score", "error_message"]].head(30),
                width="stretch"
            )
    else:
        st.markdown("""
        <div class="section-card" style="text-align:center;padding:60px 20px;">
            <div style="font-size:4rem;margin-bottom:16px">📡</div>
            <div style="color:#e2e8f0;font-size:1.2rem;font-weight:700;margin-bottom:8px">No Telemetry Data Yet</div>
            <div style="color:#475569">Click <strong style="color:#a78bfa">🚀 Ingest Telemetry Stream</strong> in the sidebar to generate live traffic!</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 2 — ROOT-CAUSE ML & SHAP
# ─────────────────────────────────────────────────────────────
with tab2:
    anomalies = zscore_engine.detect_and_flag_anomalies()

    if anomalies:
        col_head, col_badge = st.columns([5, 1])
        with col_head:
            st.markdown(f'<div class="section-title">🧠 Predictive Root-Cause Classification & SHAP Attributions</div>', unsafe_allow_html=True)
        with col_badge:
            st.markdown(f'<div class="badge badge-violet" style="margin-top:4px">{len(anomalies)} anomalies</div>', unsafe_allow_html=True)

        selected_idx = st.selectbox(
            "Select Anomaly Event to Diagnose",
            range(len(anomalies)),
            format_func=lambda i: f"[{anomalies[i]['root_cause_category']}]  {anomalies[i]['service_name']}  ·  {anomalies[i]['endpoint']}  (Z = {anomalies[i]['z_score']})",
            key="tab2_selectbox"
        )

        target_anomaly = anomalies[selected_idx]
        pred     = classifier.predict(target_anomaly)
        shap_res = shap_engine.explain(target_anomaly)

        # KPI row for selected anomaly
        conf_pct = pred['confidence'] * 100
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card violet">
                <div class="kpi-icon violet">🎯</div>
                <div class="kpi-label">Predicted Failure Mode</div>
                <div class="kpi-value" style="font-size:1.4rem">{pred['predicted_class']}</div>
                <div class="kpi-sub neutral">XGBoost Classification</div>
            </div>
            <div class="kpi-card blue">
                <div class="kpi-icon blue">📊</div>
                <div class="kpi-label">Model Confidence</div>
                <div class="kpi-value">{conf_pct:.1f}<span style="font-size:1.2rem">%</span></div>
                <div class="kpi-sub up">High certainty</div>
            </div>
            <div class="kpi-card amber">
                <div class="kpi-icon amber">🔬</div>
                <div class="kpi-label">Top SHAP Feature</div>
                <div class="kpi-value" style="font-size:1.1rem">{shap_res['top_feature']}</div>
                <div class="kpi-sub neutral">Primary signal</div>
            </div>
            <div class="kpi-card rose">
                <div class="kpi-icon rose">🚨</div>
                <div class="kpi-label">Root Cause Category</div>
                <div class="kpi-value" style="font-size:1.1rem">{target_anomaly['root_cause_category']}</div>
                <div class="kpi-sub down">Anomaly flagged</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            prob_df = pd.DataFrame(list(pred["probabilities"].items()), columns=["Failure Mode", "Probability"])
            prob_df = prob_df.sort_values("Probability", ascending=True)
            fig_p = px.bar(
                prob_df, x="Probability", y="Failure Mode", orientation="h",
                title="🎯 Failure Mode Probability Distribution",
                color="Probability",
                color_continuous_scale=[[0,"#1e1b4b"],[0.4,"#6366f1"],[0.75,"#a78bfa"],[1,"#38bdf8"]],
                template="plotly_dark"
            )
            fig_p.update_layout(**CHART_LAYOUT)
            fig_p.update_traces(marker_line_width=0)
            st.plotly_chart(fig_p, width="stretch")

        with c2:
            attr_df = pd.DataFrame(list(shap_res["feature_attributions"].items()), columns=["Feature", "SHAP Value"])
            attr_df = attr_df.sort_values("SHAP Value", ascending=True)
            colors  = ["#f43f5e" if v > 0 else "#34d399" for v in attr_df["SHAP Value"]]
            fig_s = go.Figure(go.Bar(
                x=attr_df["SHAP Value"], y=attr_df["Feature"],
                orientation="h", marker_color=colors,
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>"
            ))
            fig_s.update_layout(title="🔬 SHAP Feature Attribution Breakdown", **CHART_LAYOUT)
            st.plotly_chart(fig_s, width="stretch")

        st.markdown(f"""
        <div class="section-card" style="border-left: 3px solid #6366f1; padding: 16px 20px;">
            <div class="section-title">🧾 AI Diagnosis Summary</div>
            <div style="color:#94a3b8;font-size:0.9rem;line-height:1.7">{shap_res["summary"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="section-card" style="text-align:center;padding:60px 20px;">
            <div style="font-size:4rem;margin-bottom:16px">🧠</div>
            <div style="color:#e2e8f0;font-size:1.2rem;font-weight:700;margin-bottom:8px">No Anomalies Detected</div>
            <div style="color:#475569">Ingest a telemetry stream with anomalies to run root-cause ML analysis.</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 3 — MULTI-AGENT REMEDIATION
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="section-title">🤖 LangGraph Multi-Agent Remediation Orchestrator</div>
    <div class="section-desc">
        <span class="badge badge-violet">Investigator Agent</span>&nbsp;→&nbsp;
        <span class="badge badge-info">Developer Agent</span>&nbsp;→&nbsp;
        <span class="badge badge-success">Validator Agent</span>
        &nbsp;&nbsp;· Autonomous code patch synthesis & AST sandbox testing
    </div>
    """, unsafe_allow_html=True)

    anomalies = zscore_engine.detect_and_flag_anomalies()
    if anomalies:
        selected_anomaly_idx = st.selectbox(
            "Target Anomaly Event for Auto-Healing",
            range(len(anomalies)),
            format_func=lambda i: f"ID: {anomalies[i]['id'][:8]}  ·  {anomalies[i]['service_name']}  ·  [{anomalies[i]['root_cause_category']}]",
            key="tab3_selectbox"
        )
        anomaly_to_fix = anomalies[selected_anomaly_idx]

        a_col, b_col = st.columns([3, 1])
        with a_col:
            st.markdown(f"""
            <div class="agent-step">
                <div class="agent-step-header">
                    <span class="badge badge-warning">⚠️ Selected Anomaly</span>
                </div>
                <div style="color:#94a3b8;font-size:0.85rem;font-family:'JetBrains Mono',monospace;">
                    ID: {anomaly_to_fix['id'][:16]}…&emsp;|&emsp;
                    Service: <strong style="color:#a78bfa">{anomaly_to_fix['service_name']}</strong>&emsp;|&emsp;
                    Category: <strong style="color:#f43f5e">{anomaly_to_fix['root_cause_category']}</strong>&emsp;|&emsp;
                    Z-Score: <strong style="color:#fbbf24">{anomaly_to_fix['z_score']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🤖 Trigger Multi-Agent Remediation", width="content"):
            with st.spinner("Multi-Agent System orchestrating remediation pipeline..."):
                remediation_res = agent_pipeline.run_remediation_workflow(anomaly_to_fix)
                st.session_state["last_remediation"] = remediation_res
            st.success("✅ Remediation Workflow Complete! View results below.")
    else:
        st.markdown("""
        <div class="section-card" style="text-align:center;padding:60px 20px;">
            <div style="font-size:4rem;margin-bottom:16px">🤖</div>
            <div style="color:#e2e8f0;font-size:1.2rem;font-weight:700;margin-bottom:8px">Agents Standing By</div>
            <div style="color:#475569">Ingest a telemetry stream first to detect anomaly events for remediation.</div>
        </div>
        """, unsafe_allow_html=True)

    if "last_remediation" in st.session_state:
        res = st.session_state["last_remediation"]
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:16px">📋 Multi-Agent Execution Trace</div>', unsafe_allow_html=True)
        for trace_step in res.get("execution_trace", []):
            step_name = trace_step["step"]
            status    = trace_step["status"]
            if status == "COMPLETED":
                with st.expander(f"✅  {step_name}  ·  {status}", expanded=True):
                    st.json(trace_step.get("output", {}))

# ─────────────────────────────────────────────────────────────
# TAB 4 — GIT DIFFS & PR AUTOMATION
# ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="section-title">🔀 Automated Patch Review & GitHub Pull Request</div>
    <div class="section-desc">Inspect the AI-generated backward-compatible code fix, review the unified diff, and trigger automated PR creation.</div>
    """, unsafe_allow_html=True)

    if "last_remediation" in st.session_state:
        res   = st.session_state["last_remediation"]
        patch = res.get("patch_solution", {})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title" style="font-size:0.95rem;margin-bottom:10px">📄 Generated Backward-Compatible Code</div>', unsafe_allow_html=True)
            st.code(patch.get("patch_code", "# No patch generated"), language="python")

        with c2:
            st.markdown('<div class="section-title" style="font-size:0.95rem;margin-bottom:10px">🔀 Unified Git Diff</div>', unsafe_allow_html=True)
            st.code(patch.get("patch_diff", "# No diff available"), language="diff")

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

        # ── GitHub token config notice ──────────────────────────
        import os
        _token_set = bool(os.getenv("GITHUB_TOKEN", ""))
        _repo_set  = os.getenv("GITHUB_REPO", "harishrobin11/AutoHeal_ML_Pipelines")
        if not _token_set:
            st.markdown(f"""
            <div class="section-card" style="border-left:3px solid #fbbf24;padding:14px 18px;margin-bottom:18px;">
                <div style="color:#fbbf24;font-weight:700;font-size:0.9rem;margin-bottom:6px">⚠️ GitHub Token Not Configured</div>
                <div style="color:#94a3b8;font-size:0.85rem;line-height:1.7">
                    Set <code>GITHUB_TOKEN</code> and <code>GITHUB_REPO</code> environment variables to create real Pull Requests.<br>
                    <strong>Target repo:</strong> <code>{_repo_set}</code><br>
                    Without a token the PR will be <strong>simulated</strong> and no real GitHub URL will be generated.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🐙  Create GitHub Pull Request", width="content"):
            with st.spinner("Creating GitHub Pull Request..."):
                pr_res = GitHubPRAutomator.create_pull_request(res)
            st.session_state["last_pr"] = pr_res

        if "last_pr" in st.session_state:
            pr_res = st.session_state["last_pr"]
            mode   = pr_res.get("mode", "simulation")

            if mode == "live" and pr_res.get("success"):
                st.success("🎉 Real Pull Request Created on GitHub!")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f'<div class="badge badge-success">✅ PR #{pr_res["pr_number"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<a href="{pr_res["pr_url"]}" target="_blank" style="color:#38bdf8;font-size:0.85rem;font-weight:600;">{pr_res["pr_url"]}</a>', unsafe_allow_html=True)
                with col_b:
                    st.markdown(f'<div class="badge badge-info">🌿 Branch</div>&emsp;<code>{pr_res["branch_name"]}</code>', unsafe_allow_html=True)
                with col_c:
                    st.markdown(f'<div class="badge badge-violet">📋 Status</div>&emsp;<code>{pr_res["status"]}</code>', unsafe_allow_html=True)

            elif mode == "error":
                st.error(f"❌ PR creation failed: {pr_res.get('error', 'Unknown error')}")
                st.markdown(f"""
                <div class="section-card" style="border-left:3px solid #f43f5e;padding:14px 18px;">
                    <div style="color:#f43f5e;font-weight:700;font-size:0.9rem;margin-bottom:6px">🔧 Troubleshooting</div>
                    <div style="color:#94a3b8;font-size:0.85rem;line-height:1.7">
                        1. Make sure <code>GITHUB_TOKEN</code> has <strong>repo</strong> scope<br>
                        2. Confirm <code>GITHUB_REPO</code> = <code>{_repo_set}</code> is correct<br>
                        3. Ensure the token has write access to the repository
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:  # simulation
                st.info("🧪 **Simulation Mode** — No real PR created (token not set). Preview below:")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f'<div class="badge badge-warning">⚠️ Simulated</div>&emsp;<code>{pr_res["pr_title"]}</code>', unsafe_allow_html=True)
                with col_b:
                    st.markdown(f'<div class="badge badge-info">🌿 Branch</div>&emsp;<code>{pr_res["branch_name"]}</code>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📋 View Full Pull Request Description", expanded=True):
                st.markdown(pr_res["pr_body"])

    else:
        st.markdown("""
        <div class="section-card" style="text-align:center;padding:60px 20px;">
            <div style="font-size:4rem;margin-bottom:16px">🔀</div>
            <div style="color:#e2e8f0;font-size:1.2rem;font-weight:700;margin-bottom:8px">No Patch Available</div>
            <div style="color:#475569">Trigger a <strong style="color:#a78bfa">Multi-Agent Remediation</strong> in Tab 3 to view generated patch diffs and create PRs.</div>
        </div>
        """, unsafe_allow_html=True)
