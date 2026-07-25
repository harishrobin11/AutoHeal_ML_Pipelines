import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — Cyberpunk Deep Glassmorphism
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── TOKENS ── */
:root {
    --bg:          #080B10;
    --bg2:         #0D1117;
    --glass:       rgba(13, 20, 35, 0.7);
    --glass2:      rgba(255,255,255,0.03);
    --border:      rgba(255,255,255,0.07);
    --border-glow: rgba(0,240,255,0.2);
    --cyan:        #00F0FF;
    --indigo:      #6366F1;
    --violet:      #8B5CF6;
    --emerald:     #10B981;
    --amber:       #F59E0B;
    --rose:        #F43F5E;
    --slate:       #94A3B8;
    --slate2:      #475569;
    --text:        #E2E8F0;
    --text2:       #CBD5E1;
    --mono:        'JetBrains Mono', monospace;
    --sans:        'Plus Jakarta Sans', system-ui, sans-serif;
}

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
code, pre, .stCode { font-family: var(--mono) !important; }
.block-container { padding: 0 1.5rem 4rem !important; max-width: 100% !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 10px; }

/* ── HEADER ── */
header[data-testid="stHeader"] {
    background: rgba(8,11,16,0.9) !important;
    backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080B10 0%, #0A0E18 100%) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
.sidebar-logo {
    background: linear-gradient(135deg, rgba(0,240,255,0.07), rgba(99,102,241,0.07));
    border-bottom: 1px solid var(--border);
    padding: 20px 20px 16px;
    margin: 0 -1rem 20px;
    display: flex; align-items: center; gap: 12px;
}
.sidebar-logo-icon {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #00F0FF22, #6366F122);
    border: 1px solid rgba(0,240,255,0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 0 20px rgba(0,240,255,0.1);
}
.sidebar-logo-text { line-height: 1.2; }
.sidebar-logo-name {
    font-size: 0.92rem; font-weight: 800; letter-spacing: -0.01em;
    background: linear-gradient(90deg, #00F0FF, #6366F1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sidebar-logo-sub { font-size: 0.68rem; font-weight: 500; color: var(--slate2); text-transform: uppercase; letter-spacing: 0.07em; }
.sidebar-section { font-size: 0.7rem; font-weight: 700; color: var(--slate2); text-transform: uppercase; letter-spacing: 0.1em; margin: 16px 0 8px; }
.engine-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 8px;
    background: var(--glass2); border: 1px solid var(--border);
    margin-bottom: 5px;
}
.engine-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
    flex-shrink: 0;
    animation: glow-pulse 2.5s ease-in-out infinite;
}
@keyframes glow-pulse { 0%,100%{opacity:1;box-shadow:0 0 6px #10B981} 50%{opacity:0.5;box-shadow:0 0 12px #10B981} }
.engine-name { font-size: 0.78rem; font-weight: 600; color: var(--text2); }
.engine-desc { font-size: 0.68rem; color: var(--slate2); }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4E8 0%, #6366F1 100%) !important;
    color: #fff !important; font-weight: 700 !important;
    font-size: 0.85rem !important; letter-spacing: 0.02em !important;
    border: none !important; border-radius: 10px !important;
    padding: 10px 22px !important;
    box-shadow: 0 0 20px rgba(0,240,255,0.2), 0 4px 15px rgba(99,102,241,0.3) !important;
    transition: all 0.2s ease !important;
    font-family: var(--sans) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 35px rgba(0,240,255,0.35), 0 8px 25px rgba(99,102,241,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── SLIDERS ── */
.stSlider > div > div > div > div { background: var(--cyan) !important; }
.stSlider > div > div > div { background: rgba(255,255,255,0.08) !important; border-radius: 10px !important; }
.stSlider [data-testid="stThumbValue"] { font-family: var(--mono) !important; font-size: 0.75rem !important; color: var(--cyan) !important; }

/* ── SELECTBOX ── */
.stSelectbox > div > div {
    background: rgba(13,17,23,0.9) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-family: var(--mono) !important; font-size: 0.82rem !important;
}
.stSelectbox > div > div:focus-within { border-color: rgba(0,240,255,0.4) !important; box-shadow: 0 0 0 3px rgba(0,240,255,0.08) !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,17,23,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 5px !important; gap: 3px !important;
    backdrop-filter: blur(12px) !important;
    margin-bottom: 24px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important; padding: 9px 18px !important;
    color: var(--slate2) !important; font-weight: 600 !important;
    font-size: 0.85rem !important; border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
    font-family: var(--sans) !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--slate) !important; background: var(--glass2) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,240,255,0.08), rgba(99,102,241,0.12)) !important;
    color: var(--cyan) !important;
    box-shadow: inset 0 0 0 1px rgba(0,240,255,0.2), 0 0 20px rgba(0,240,255,0.08) !important;
}

/* ── ALERTS ── */
.stSuccess > div { background: rgba(16,185,129,0.08) !important; border: 1px solid rgba(16,185,129,0.25) !important; border-radius: 10px !important; color: #6EE7B7 !important; }
.stInfo > div, .stAlert > div { background: rgba(0,240,255,0.06) !important; border: 1px solid rgba(0,240,255,0.18) !important; border-radius: 10px !important; color: #94A3B8 !important; }
.stError > div { background: rgba(244,63,94,0.08) !important; border: 1px solid rgba(244,63,94,0.25) !important; border-radius: 10px !important; color: #FCA5A5 !important; }
.stWarning > div { background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.25) !important; border-radius: 10px !important; color: #FCD34D !important; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: var(--glass2) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text2) !important;
    font-family: var(--mono) !important; font-size: 0.82rem !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent { border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 10px 10px !important; background: rgba(8,11,16,0.6) !important; }

/* ── DATAFRAME ── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 12px !important; overflow: hidden !important; }

/* ── COMPONENTS ── */
.glass-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 20px 22px;
    backdrop-filter: blur(12px);
    position: relative; overflow: hidden;
    transition: border-color 0.25s, box-shadow 0.25s;
}
.glass-card:hover { border-color: var(--border-glow); box-shadow: 0 0 30px rgba(0,240,255,0.05); }
.glass-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,240,255,0.3), transparent);
}

/* KPI CARDS */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 22px; }
.kpi-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 20px;
    position: relative; overflow: hidden;
    transition: all 0.25s ease; cursor: default;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-card-top { height: 2px; position: absolute; top: 0; left: 0; right: 0; border-radius: 14px 14px 0 0; }
.kpi-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 9px;
    font-size: 1rem; margin-bottom: 12px;
}
.kpi-label { font-size: 0.72rem; font-weight: 700; color: var(--slate2); text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 5px; }
.kpi-value { font-family: var(--mono); font-size: 1.9rem; font-weight: 700; color: var(--text); line-height: 1; }
.kpi-sub { font-size: 0.75rem; font-weight: 600; margin-top: 6px; display: flex; align-items: center; gap: 4px; }

/* HERO */
.hero {
    background: linear-gradient(135deg, #080B10 0%, #0D1020 50%, #080B10 100%);
    border: 1px solid var(--border); border-radius: 18px;
    padding: 36px 40px; margin: 16px 0 24px;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 55% 70% at 5% 50%, rgba(0,240,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 95% 20%, rgba(99,102,241,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 30% 40% at 50% 110%, rgba(139,92,246,0.05) 0%, transparent 60%);
}
.hero::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(0,240,255,0.5) 40%, rgba(99,102,241,0.5) 60%, transparent 100%);
}
.hero-chip {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(0,240,255,0.07); border: 1px solid rgba(0,240,255,0.2);
    border-radius: 100px; padding: 5px 14px 5px 8px;
    font-size: 0.72rem; font-weight: 700; color: rgba(0,240,255,0.9);
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 16px;
}
.hero-chip-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 8px var(--cyan); animation: glow-pulse 2s infinite; }
.hero-title {
    font-size: 2.6rem; font-weight: 800; line-height: 1.08; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #E0F2FE 0%, #00F0FF 30%, #818CF8 65%, #C4B5FD 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 10px;
}
.hero-sub { font-size: 0.95rem; color: var(--slate); line-height: 1.65; max-width: 700px; }
.hero-sub strong { color: var(--text2); font-weight: 600; }
.hero-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--glass2); border: 1px solid var(--border);
    border-radius: 7px; padding: 4px 11px;
    font-size: 0.74rem; font-weight: 500; color: var(--slate);
}
.hero-orb {
    position: absolute; right: 48px; top: 50%; transform: translateY(-50%);
    width: 160px; height: 160px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,240,255,0.08) 0%, rgba(99,102,241,0.05) 50%, transparent 70%);
    border: 1px solid rgba(0,240,255,0.1);
    display: flex; align-items: center; justify-content: center;
    font-size: 3.5rem;
    animation: float 6s ease-in-out infinite;
}
@keyframes float { 0%,100%{transform:translateY(-50%) scale(1)} 50%{transform:translateY(-55%) scale(1.04)} }

/* BADGES */
.badge { display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:6px;font-size:0.72rem;font-weight:700;letter-spacing:0.02em; }
.b-cyan    { background:rgba(0,240,255,0.08);  border:1px solid rgba(0,240,255,0.2);  color:#67E8F9; }
.b-indigo  { background:rgba(99,102,241,0.1);  border:1px solid rgba(99,102,241,0.25);color:#A5B4FC; }
.b-emerald { background:rgba(16,185,129,0.1);  border:1px solid rgba(16,185,129,0.25);color:#6EE7B7; }
.b-rose    { background:rgba(244,63,94,0.1);   border:1px solid rgba(244,63,94,0.25); color:#FCA5A5; }
.b-amber   { background:rgba(245,158,11,0.1);  border:1px solid rgba(245,158,11,0.25);color:#FCD34D; }
.b-violet  { background:rgba(139,92,246,0.1);  border:1px solid rgba(139,92,246,0.25);color:#C4B5FD; }
.b-slate   { background:rgba(148,163,184,0.08);border:1px solid rgba(148,163,184,0.15);color:#94A3B8; }

/* SECTION HEADERS */
.sec-header { display:flex;align-items:center;gap:10px;margin-bottom:5px; }
.sec-title { font-size:1.05rem;font-weight:700;color:var(--text); }
.sec-desc  { font-size:0.82rem;color:var(--slate2);margin-bottom:18px; }
.divider   { height:1px;background:linear-gradient(90deg,transparent,rgba(0,240,255,0.2),rgba(99,102,241,0.2),transparent);margin:18px 0;border:none; }

/* WORKFLOW NODES (Tab 3) */
.workflow-bar {
    display:flex;align-items:center;gap:0;
    background:var(--glass);border:1px solid var(--border);
    border-radius:14px;padding:16px 24px;margin-bottom:22px;
}
.wf-node {
    display:flex;align-items:center;gap:12px;flex:1;
}
.wf-icon {
    width:44px;height:44px;border-radius:12px;
    display:flex;align-items:center;justify-content:center;font-size:1.3rem;
    flex-shrink:0;
}
.wf-icon.done { background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3); box-shadow:0 0 16px rgba(16,185,129,0.1); }
.wf-icon.active { background:rgba(0,240,255,0.1);border:1px solid rgba(0,240,255,0.3); box-shadow:0 0 16px rgba(0,240,255,0.15); animation:wf-glow 2s infinite; }
.wf-icon.idle { background:rgba(255,255,255,0.03);border:1px solid var(--border); }
@keyframes wf-glow { 0%,100%{box-shadow:0 0 10px rgba(0,240,255,0.1)} 50%{box-shadow:0 0 25px rgba(0,240,255,0.25)} }
.wf-text {}
.wf-label { font-size:0.75rem;font-weight:700;color:var(--slate2);text-transform:uppercase;letter-spacing:0.06em; }
.wf-name  { font-size:0.88rem;font-weight:700;color:var(--text2); }
.wf-arrow { font-size:1.2rem;color:var(--slate2);padding:0 10px;flex-shrink:0; }

/* TERMINAL CARD */
.terminal-card {
    background:#050810;border:1px solid rgba(0,240,255,0.12);
    border-radius:12px;padding:0;overflow:hidden;margin-bottom:10px;
}
.terminal-header {
    background:rgba(0,240,255,0.04);border-bottom:1px solid rgba(0,240,255,0.08);
    padding:10px 16px;display:flex;align-items:center;gap:8px;
}
.t-dot { width:9px;height:9px;border-radius:50%; }
.terminal-body { padding:14px 18px;font-family:var(--mono);font-size:0.78rem;color:#94A3B8;line-height:1.8; }

/* SHAP BAR */
.shap-row { display:flex;align-items:center;gap:12px;margin-bottom:10px; }
.shap-label { font-family:var(--mono);font-size:0.78rem;color:var(--slate);width:160px;flex-shrink:0;text-align:right; }
.shap-bar-wrap { flex:1;height:8px;background:rgba(255,255,255,0.05);border-radius:10px;overflow:hidden; }
.shap-bar { height:100%;border-radius:10px;transition:width 0.6s ease; }
.shap-val { font-family:var(--mono);font-size:0.75rem;width:70px;flex-shrink:0; }

/* CONFIDENCE GAUGE */
.conf-ring { display:flex;align-items:center;justify-content:center;margin:10px 0; }
.conf-center { position:absolute;text-align:center; }
.conf-pct { font-family:var(--mono);font-size:2rem;font-weight:700;color:var(--cyan); }
.conf-lbl { font-size:0.7rem;color:var(--slate2);text-transform:uppercase;letter-spacing:0.06em; }

/* DIFF VIEWER */
.diff-wrap { font-family:var(--mono);font-size:0.78rem;line-height:1.7;border-radius:10px;overflow:hidden;border:1px solid var(--border); }
.diff-header { background:rgba(255,255,255,0.03);border-bottom:1px solid var(--border);padding:8px 14px;font-size:0.72rem;color:var(--slate2);font-weight:600; }
.diff-body { background:#050810;padding:12px 14px;overflow-x:auto; }
.diff-add  { background:rgba(16,185,129,0.08);color:#6EE7B7;display:block; }
.diff-del  { background:rgba(244,63,94,0.08);color:#FCA5A5;display:block; }
.diff-ctx  { color:#475569;display:block; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  ENGINE INIT
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_engines():
    return (
        TelemetryGenerator(),
        TelemetryIngestor(),
        ZScoreAnomalyEngine(),
        RootCauseClassifier(),
        SHAPExplainerEngine(RootCauseClassifier()),
        TelemetryDriftMonitor(),
        MultiAgentRemediationPipeline(),
    )

generator, ingestor, zscore_engine, classifier, shap_engine, drift_monitor, agent_pipeline = load_engines()

# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">⚡</div>
        <div class="sidebar-logo-text">
            <div class="sidebar-logo-name">AutoHeal-ML</div>
            <div class="sidebar-logo-sub">MLOps Control Center</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">⚙️ Stream Controls</div>', unsafe_allow_html=True)
    batch_size   = st.slider("Batch Size", 20, 200, 50, help="Synthetic telemetry logs per ingest run")
    anomaly_rate = st.slider("Anomaly Injection Ratio", 0.0, 0.5, 0.2, 0.05, help="Fraction of failure-mode logs")

    st.markdown("<br>", unsafe_allow_html=True)
    ingest_btn = st.button("🚀  Ingest Telemetry Stream", use_container_width=True)
    if ingest_btn:
        with st.spinner("Ingesting & retraining…"):
            batch = generator.generate_batch(count=batch_size, anomaly_ratio=anomaly_rate)
            ingestor.ingest_records(batch)
            classifier.train(batch)
            shap_engine.initialize_explainer(batch)
        st.success(f"✅  {len(batch)} records ingested")

    st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:16px 0">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">🟢 Active Engines</div>', unsafe_allow_html=True)
    for nm, desc in [
        ("Z-Score Engine",     "Rolling Window Anomaly"),
        ("XGBoost Classifier", "Multi-Class Root Cause"),
        ("SHAP Explainer",     "Feature Attribution"),
        ("LangGraph Agents",   "Code Remediation"),
        ("PR Automator",       "GitHub Integration"),
    ]:
        st.markdown(f"""
        <div class="engine-item">
            <div class="engine-dot"></div>
            <div><div class="engine-name">{nm}</div><div class="engine-desc">{desc}</div></div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  HERO BANNER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-orb">⚡</div>
    <div class="hero-chip"><div class="hero-chip-dot"></div>Live · Autonomous MLOps & Self-Healing Pipeline</div>
    <div class="hero-title">AutoHeal-ML<br>Orchestration Platform</div>
    <div class="hero-sub">
        <strong>Real-time telemetry analytics</strong> with Z-Score anomaly detection ·
        <strong>XGBoost root-cause ML</strong> with SHAP explainability ·
        <strong>LangGraph multi-agent</strong> code remediation · Automated GitHub PRs
    </div>
    <div class="hero-chips">
        <div class="hero-tag">📡 Live Telemetry</div>
        <div class="hero-tag">🧬 XGBoost · SHAP</div>
        <div class="hero-tag">🤖 LangGraph Agents</div>
        <div class="hero-tag">🐙 GitHub PR Automation</div>
        <div class="hero-tag">🐳 Docker · FastAPI</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📡  Telemetry Stream & Z-Scores",
    "🧠  Root-Cause ML & SHAP",
    "🤖  Multi-Agent Remediation",
    "🔀  Git Diffs & PR Automation",
])

records = ingestor.fetch_all(limit=300)
df      = pd.DataFrame(records) if records else pd.DataFrame()

CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#64748B", size=11),
    margin=dict(l=8, r=8, t=44, b=8),
    title_font=dict(size=13, color="#CBD5E1", family="Plus Jakarta Sans"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.05)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.04)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.04)"),
)

EMPTY_STATE = lambda icon, title, msg: st.markdown(f"""
<div class="glass-card" style="text-align:center;padding:64px 20px;margin-top:10px">
    <div style="font-size:3.5rem;margin-bottom:14px">{icon}</div>
    <div style="font-size:1.15rem;font-weight:700;color:#E2E8F0;margin-bottom:8px">{title}</div>
    <div style="font-size:0.85rem;color:#475569">{msg}</div>
</div>""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────
# TAB 1 — TELEMETRY STREAM & Z-SCORES
# ───────────────────────────────────────────────────────────────────
with tab1:
    if not df.empty:
        analyzed    = zscore_engine.compute_z_scores()
        df_z        = pd.DataFrame(analyzed)
        total       = len(df)
        errors      = len(df[df["status_code"] >= 400])
        error_pct   = round(errors / total * 100, 1)
        avg_lat     = round(df["response_time_ms"].mean(), 1)
        spikes      = len(df_z[abs(df_z["z_score"]) >= 2.0]) if not df_z.empty else 0

        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-card" style="border-top:2px solid #00F0FF20">
            <div class="kpi-card-top" style="background:linear-gradient(90deg,#00F0FF,#38BDF8)"></div>
            <div class="kpi-badge" style="background:rgba(0,240,255,0.08);border:1px solid rgba(0,240,255,0.2)">📡</div>
            <div class="kpi-label">Total Logs Processed</div>
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-sub" style="color:#67E8F9">● Live window</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-card-top" style="background:linear-gradient(90deg,#F43F5E,#FB7185)"></div>
            <div class="kpi-badge" style="background:rgba(244,63,94,0.1);border:1px solid rgba(244,63,94,0.2)">🔥</div>
            <div class="kpi-label">HTTP Errors (4xx/5xx)</div>
            <div class="kpi-value">{errors:,}</div>
            <div class="kpi-sub" style="color:#FCA5A5">↑ {error_pct}% failure rate</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-card-top" style="background:linear-gradient(90deg,#F59E0B,#FCD34D)"></div>
            <div class="kpi-badge" style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.2)">⏱️</div>
            <div class="kpi-label">Mean Latency (ms)</div>
            <div class="kpi-value">{avg_lat}</div>
            <div class="kpi-sub" style="color:#FCD34D">SLA ≤ 300ms</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-card-top" style="background:linear-gradient(90deg,#8B5CF6,#C4B5FD)"></div>
            <div class="kpi-badge" style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.2)">⚡</div>
            <div class="kpi-label">Z-Score Spikes |Z| ≥ 2</div>
            <div class="kpi-value">{spikes}</div>
            <div class="kpi-sub" style="color:#C4B5FD">Anomaly events</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not df_z.empty:
            fig = px.scatter(
                df_z, x="timestamp", y="response_time_ms",
                color="z_score", size="response_time_ms",
                hover_data=["service_name","endpoint","status_code","z_score"],
                title="📡 Response Latency · Rolling Z-Score Anomaly Detection",
                color_continuous_scale=[[0,"#0D1117"],[0.35,"#6366F1"],[0.7,"#00F0FF"],[1,"#F43F5E"]],
                template="plotly_dark",
            )
            fig.add_hline(y=300, line_dash="dash", line_color="#F59E0B", line_width=1.2,
                          annotation_text="SLA 300ms", annotation_font_color="#F59E0B")
            fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.4, color="rgba(255,255,255,0.1)")))
            fig.update_layout(**CHART)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                if "service_name" in df_z.columns and "status_code" in df_z.columns:
                    svc = df_z.groupby("service_name").apply(
                        lambda x: (x["status_code"] >= 400).mean() * 100, include_groups=False
                    ).reset_index()
                    svc.columns = ["Service","Error %"]
                    svc = svc.sort_values("Error %", ascending=True)
                    f2 = px.bar(svc, x="Error %", y="Service", orientation="h",
                                title="🔥 Error Rate by Microservice",
                                color="Error %",
                                color_continuous_scale=[[0,"#1E1B4B"],[0.5,"#6366F1"],[1,"#F43F5E"]],
                                template="plotly_dark")
                    f2.update_layout(**CHART); f2.update_traces(marker_line_width=0)
                    st.plotly_chart(f2, use_container_width=True)
            with c2:
                f3 = px.histogram(df_z, x="response_time_ms", nbins=40,
                                  title="⏱️ Latency Distribution (ms)",
                                  color_discrete_sequence=["#00F0FF"],
                                  template="plotly_dark")
                f3.add_vline(x=300, line_dash="dash", line_color="#F59E0B", line_width=1.2)
                f3.update_layout(**CHART)
                st.plotly_chart(f3, use_container_width=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="sec-header"><div class="sec-title">📋 Telemetry Log Registry</div><div class="badge b-slate">last 30</div></div>', unsafe_allow_html=True)
            st.dataframe(
                df_z[["timestamp","service_name","endpoint","status_code","response_time_ms","z_score","error_message"]].head(30),
                use_container_width=True,
            )
    else:
        EMPTY_STATE("📡","No Telemetry Data","Click <strong>🚀 Ingest Telemetry Stream</strong> in the sidebar to generate live traffic.")

# ───────────────────────────────────────────────────────────────────
# TAB 2 — ROOT-CAUSE ML & SHAP
# ───────────────────────────────────────────────────────────────────
with tab2:
    anomalies = zscore_engine.detect_and_flag_anomalies()
    if anomalies:
        # Alert banner
        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(244,63,94,0.25);border-left:3px solid #F43F5E;padding:14px 20px;margin-bottom:18px;background:rgba(244,63,94,0.04)">
            <div style="display:flex;align-items:center;gap:12px">
                <span style="font-size:1.4rem">🚨</span>
                <div>
                    <div style="font-weight:700;color:#FCA5A5;font-size:0.9rem">{len(anomalies)} Anomaly Event{'s' if len(anomalies)>1 else ''} Detected</div>
                    <div style="color:#64748B;font-size:0.8rem">Select an event below to run XGBoost root-cause classification and SHAP attribution analysis.</div>
                </div>
                <div class="badge b-rose" style="margin-left:auto">{len(anomalies)} Events</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        idx = st.selectbox(
            "Select Anomaly Event to Diagnose",
            range(len(anomalies)),
            format_func=lambda i: f"[{anomalies[i]['root_cause_category']}]  {anomalies[i]['service_name']}  ·  {anomalies[i]['endpoint']}  (Z = {anomalies[i]['z_score']})",
        )
        anomaly  = anomalies[idx]
        pred     = classifier.predict(anomaly)
        shap_res = shap_engine.explain(anomaly)
        conf     = pred["confidence"] * 100

        c1, c2 = st.columns(2)

        # ── Left: Prediction + Confidence ──
        with c1:
            st.markdown(f"""
            <div class="glass-card" style="height:100%">
                <div class="sec-header">
                    <div class="sec-title">🎯 ML Failure Mode Prediction</div>
                    <div class="badge b-indigo">XGBoost</div>
                </div>
                <div style="margin-bottom:18px">
                    <div style="font-size:0.72rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px">Predicted Class</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;color:#00F0FF">{pred['predicted_class']}</div>
                </div>
                <div style="margin-bottom:18px">
                    <div style="font-size:0.72rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px">Model Confidence</div>
                    <div style="display:flex;align-items:center;gap:12px">
                        <div style="flex:1;height:10px;background:rgba(255,255,255,0.05);border-radius:10px;overflow:hidden">
                            <div style="height:100%;width:{conf:.1f}%;background:linear-gradient(90deg,#00F0FF,#6366F1);border-radius:10px;transition:width 0.8s ease"></div>
                        </div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:#00F0FF;width:52px;text-align:right">{conf:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            prob_df = pd.DataFrame(list(pred["probabilities"].items()), columns=["Mode","Prob"]).sort_values("Prob", ascending=True)
            fp = go.Figure(go.Bar(
                x=prob_df["Prob"], y=prob_df["Mode"], orientation="h",
                marker=dict(
                    color=prob_df["Prob"],
                    colorscale=[[0,"#0D1117"],[0.4,"#6366F1"],[0.75,"#00F0FF"],[1,"#38BDF8"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{y}</b><br>Prob: %{x:.4f}<extra></extra>",
            ))
            fp.update_layout(title="Failure Mode Probabilities", **CHART)
            st.plotly_chart(fp, use_container_width=True)

        # ── Right: SHAP ──
        with c2:
            attrs = shap_res.get("feature_attributions", {})
            sorted_attrs = sorted(attrs.items(), key=lambda x: abs(x[1]), reverse=True)
            max_val = max(abs(v) for _, v in sorted_attrs) if sorted_attrs else 1

            shap_bars = ""
            for feat, val in sorted_attrs[:8]:
                pct  = abs(val) / max_val * 100
                col  = "#00F0FF" if val > 0 else "#F43F5E"
                sign = "+" if val > 0 else ""
                shap_bars += f"""
                <div class="shap-row">
                    <div class="shap-label">{feat[:20]}</div>
                    <div class="shap-bar-wrap"><div class="shap-bar" style="width:{pct:.1f}%;background:{col};opacity:0.85"></div></div>
                    <div class="shap-val" style="color:{col};font-size:0.73rem">{sign}{val:.4f}</div>
                </div>"""

            st.markdown(f"""
            <div class="glass-card" style="height:auto">
                <div class="sec-header">
                    <div class="sec-title">🔬 SHAP Feature Attribution</div>
                    <div class="badge b-cyan">Top 8 Features</div>
                </div>
                <div style="margin-bottom:14px;font-size:0.78rem;color:#475569">
                    <span style="color:#00F0FF">■</span> Positive (pushes toward this class) &nbsp;
                    <span style="color:#F43F5E">■</span> Negative (pushes away)
                </div>
                {shap_bars}
                <div style="margin-top:16px;padding:12px 14px;background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.1);border-radius:10px;font-size:0.82rem;color:#94A3B8;line-height:1.65">
                    <span style="color:#00F0FF;font-weight:700">Top Feature:</span> <code style="color:#67E8F9">{shap_res.get('top_feature','—')}</code><br>
                    {shap_res.get('summary','')[:200]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        EMPTY_STATE("🧠","No Anomalies Detected","Ingest a telemetry stream with anomaly injection to run root-cause ML analysis.")

# ───────────────────────────────────────────────────────────────────
# TAB 3 — MULTI-AGENT REMEDIATION
# ───────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="workflow-bar">
        <div class="wf-node">
            <div class="wf-icon done">🔍</div>
            <div class="wf-text">
                <div class="wf-label">Step 01</div>
                <div class="wf-name">Investigator</div>
            </div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-node">
            <div class="wf-icon done">🛠️</div>
            <div class="wf-text">
                <div class="wf-label">Step 02</div>
                <div class="wf-name">Developer</div>
            </div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-node">
            <div class="wf-icon done">✅</div>
            <div class="wf-text">
                <div class="wf-label">Step 03</div>
                <div class="wf-name">Validator</div>
            </div>
        </div>
        <div style="margin-left:auto">
            <div class="badge b-emerald" style="font-size:0.75rem;padding:5px 12px">● Agents Online</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    anomalies = zscore_engine.detect_and_flag_anomalies()
    if anomalies:
        idx3 = st.selectbox(
            "Target Anomaly for Auto-Healing",
            range(len(anomalies)),
            format_func=lambda i: f"ID: {anomalies[i]['id'][:8]}  ·  {anomalies[i]['service_name']}  ·  [{anomalies[i]['root_cause_category']}]",
            key="tab3_sel",
        )
        target = anomalies[idx3]

        st.markdown(f"""
        <div class="terminal-card" style="margin-bottom:16px">
            <div class="terminal-header">
                <div class="t-dot" style="background:#F43F5E"></div>
                <div class="t-dot" style="background:#F59E0B"></div>
                <div class="t-dot" style="background:#10B981"></div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.73rem;color:#475569;margin-left:8px">anomaly-event.json</span>
            </div>
            <div class="terminal-body">
                <span style="color:#64748B">id:</span>       <span style="color:#67E8F9">{target['id'][:32]}…</span><br>
                <span style="color:#64748B">service:</span>  <span style="color:#C4B5FD">{target['service_name']}</span><br>
                <span style="color:#64748B">endpoint:</span> <span style="color:#94A3B8">{target['endpoint']}</span><br>
                <span style="color:#64748B">category:</span> <span style="color:#FCA5A5">{target['root_cause_category']}</span><br>
                <span style="color:#64748B">z_score:</span>  <span style="color:#FCD34D">{target['z_score']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🤖  Trigger Multi-Agent Remediation Pipeline", use_container_width=False):
            with st.spinner("🤖  Agents investigating, patching, and validating…"):
                result = agent_pipeline.run_remediation_workflow(target)
                st.session_state["last_remediation"] = result
            st.success("✅  Remediation workflow completed — view execution trace below!")
    else:
        EMPTY_STATE("🤖","Agents Standing By","Ingest a telemetry stream to detect anomaly events for autonomous remediation.")

    if "last_remediation" in st.session_state:
        res = st.session_state["last_remediation"]
        # Only keep the final COMPLETED entry for each unique step name
        seen, completed_steps = set(), []
        for step in reversed(res.get("execution_trace", [])):
            name = step["step"].split("·")[0].strip().split(" ")[0]
            if step["status"] == "COMPLETED" and name not in seen:
                seen.add(name)
                completed_steps.insert(0, step)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Header row
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px">
            <div class="sec-title">📋 Agent Execution Trace</div>
            <div class="badge b-emerald">● {len(completed_steps)} Steps Completed</div>
            <div class="badge b-cyan">Pipeline OK</div>
        </div>
        """, unsafe_allow_html=True)

        STEP_COLORS = {
            "INVESTIGATION": ("🔍", "#6366F1", "rgba(99,102,241,0.1)", "rgba(99,102,241,0.25)"),
            "DEVELOPMENT":   ("🛠️", "#00F0FF", "rgba(0,240,255,0.08)", "rgba(0,240,255,0.2)"),
            "VALIDATION":    ("✅", "#10B981", "rgba(16,185,129,0.08)", "rgba(16,185,129,0.2)"),
        }

        for i, step in enumerate(completed_steps):
            step_key = next((k for k in STEP_COLORS if k in step["step"].upper()), "INVESTIGATION")
            icon, accent, bg, border = STEP_COLORS[step_key]
            step_name = step["step"]
            output    = step.get("output", {})
            is_last   = i == len(completed_steps) - 1

            # Build a one-line summary from the output dict
            summary_parts = []
            for k, v in list(output.items())[:3]:
                summary_parts.append(f'<span style="color:#64748B">{k}:</span> <span style="color:#94A3B8">{str(v)[:40]}</span>')
            summary_html = " &nbsp;·&nbsp; ".join(summary_parts) if summary_parts else "No output data"

            st.markdown(f"""
            <div style="display:flex;gap:0;margin-bottom:{'4' if not is_last else '0'}px">
                <!-- Step line -->
                <div style="display:flex;flex-direction:column;align-items:center;margin-right:16px;flex-shrink:0">
                    <div style="width:38px;height:38px;border-radius:10px;background:{bg};border:1px solid {border};
                                display:flex;align-items:center;justify-content:center;font-size:1.1rem;
                                box-shadow:0 0 16px {bg}">
                        {icon}
                    </div>
                    {'<div style="width:2px;flex:1;background:rgba(255,255,255,0.06);margin:4px 0;min-height:24px"></div>' if not is_last else ''}
                </div>
                <!-- Card -->
                <div style="flex:1;background:rgba(13,20,35,0.6);border:1px solid {border};border-radius:12px;
                            padding:14px 18px;margin-bottom:8px;position:relative;overflow:hidden">
                    <div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,{accent}44,{accent}22,transparent)"></div>
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:700;
                                    color:{accent};text-transform:uppercase;letter-spacing:0.08em">
                            Step {i+1:02d} · {step_key}
                        </div>
                        <div class="badge" style="background:{bg};border:1px solid {border};color:{accent};font-size:0.65rem">
                            ● COMPLETED
                        </div>
                    </div>
                    <div style="font-size:0.82rem;font-weight:600;color:#CBD5E1;margin-bottom:6px">{step_name}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#475569;line-height:1.6">
                        {summary_html}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"  View full output — {step_key}", expanded=False):
                st.markdown(f"""
                <div class="terminal-card">
                    <div class="terminal-header">
                        <div class="t-dot" style="background:#F43F5E"></div>
                        <div class="t-dot" style="background:#F59E0B"></div>
                        <div class="t-dot" style="background:#10B981"></div>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#475569;margin-left:8px">{step_key.lower()}_output.json</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.json(output)

# ───────────────────────────────────────────────────────────────────
# TAB 4 — GIT DIFFS & PR AUTOMATION
# ───────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="sec-header"><div class="sec-title">🔀 Patch Review & GitHub Pull Request Automation</div></div>
    <div class="sec-desc">AI-generated backward-compatible code fix · Unified diff viewer · One-click GitHub PR creation</div>
    """, unsafe_allow_html=True)

    if "last_remediation" in st.session_state:
        res   = st.session_state["last_remediation"]
        patch = res.get("patch_solution", {})
        code  = patch.get("patch_code",  "# No patch generated")
        diff  = patch.get("patch_diff",  "# No diff available")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sec-header" style="margin-bottom:8px"><div class="sec-title" style="font-size:0.9rem">📄 Generated Code</div><div class="badge b-cyan">Backward-Compatible</div></div>', unsafe_allow_html=True)
            st.code(code, language="python")

        with c2:
            # Render styled diff
            lines = diff.split("\n")
            rendered = ""
            for ln in lines:
                if ln.startswith("+") and not ln.startswith("+++"):
                    rendered += f'<span class="diff-add">+ {ln[1:]}</span>'
                elif ln.startswith("-") and not ln.startswith("---"):
                    rendered += f'<span class="diff-del">- {ln[1:]}</span>'
                else:
                    rendered += f'<span class="diff-ctx">  {ln}</span>'

            st.markdown(f"""
            <div style="margin-bottom:8px;display:flex;align-items:center;gap:10px">
                <div class="sec-title" style="font-size:0.9rem">🔀 Unified Git Diff</div>
                <div class="badge b-emerald">+additions</div>
                <div class="badge b-rose">-deletions</div>
            </div>
            <div class="diff-wrap">
                <div class="diff-header">diff --git a/service.py b/service.py</div>
                <div class="diff-body"><pre style="margin:0;white-space:pre-wrap">{rendered}</pre></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Token status notice
        _token = bool(os.getenv("GITHUB_TOKEN", ""))
        _repo  = os.getenv("GITHUB_REPO", "harishrobin11/AutoHeal_ML_Pipelines")
        if not _token:
            st.warning(f"⚠️  **GITHUB_TOKEN** not set — PRs will run in simulation mode. Set it in `.env` with `GITHUB_REPO={_repo}`.")

        if st.button("🐙  Create GitHub Pull Request", use_container_width=False):
            with st.spinner("Creating GitHub Pull Request via API…"):
                pr = GitHubPRAutomator.create_pull_request(res)
            st.session_state["last_pr"] = pr

        if "last_pr" in st.session_state:
            pr   = st.session_state["last_pr"]
            mode = pr.get("mode", "simulation")

            if mode == "live" and pr.get("success"):
                st.success(f"🎉 Pull Request **#{pr['pr_number']}** created successfully on GitHub!")
                st.markdown(f"""
                <div class="glass-card" style="border-color:rgba(16,185,129,0.3);margin-top:12px">
                    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
                        <div><div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px">PR Link</div>
                        <a href="{pr['pr_url']}" target="_blank" style="color:#00F0FF;font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:600;text-decoration:none">{pr['pr_url']}</a></div>
                        <div><div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px">Branch</div>
                        <code style="color:#C4B5FD;font-size:0.82rem">{pr['branch_name']}</code></div>
                        <div><div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px">Status</div>
                        <div class="badge b-emerald">{pr['status']}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif mode == "error":
                st.error(f"❌  PR creation failed: {pr.get('error','Unknown error')}")
            else:
                st.info("🧪  **Simulation Mode** — Set `GITHUB_TOKEN` in `.env` to create real PRs.")
                st.markdown(f'<div class="badge b-amber">⚠️ Simulated</div>&emsp;<code style="color:#C4B5FD">{pr["branch_name"]}</code>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📋  View Full Pull Request Description", expanded=False):
                st.markdown(pr["pr_body"])
    else:
        EMPTY_STATE("🔀","No Patch Available","Trigger a <strong>Multi-Agent Remediation</strong> in Tab 3 first.")
