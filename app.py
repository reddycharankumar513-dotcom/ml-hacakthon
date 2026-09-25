import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src/ directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from business_entity_resolution.config import load_config
from business_entity_resolution.io import load_all_datasets
from business_entity_resolution.normalization import process_normalization
from business_entity_resolution.blocking import MultiStrategyBlocker
from business_entity_resolution.features import extract_pair_features
from business_entity_resolution.model import EntityResolutionMatcher
from business_entity_resolution.submission import run_submission_validator

# ----------------------------------------------------
# 1. FastAPI Web Server & WSGI/ASGI Export (Vercel / Render / Uvicorn)
# ----------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="Amazon ML Challenge — Business Entity Resolution API",
    description="Production ML Pipeline API for Business Entity Resolution",
    version="1.0.0"
)

# Export aliases for WSGI/ASGI server discovery
application = app
handler = app

class EntityQuery(BaseModel):
    business_name: str
    business_address: str
    country: str = "US"

@app.get("/", response_class=HTMLResponse)
def root_index():
    return """
    <!Inherit>
    <html>
    <head>
        <title>Business Entity Resolution System</title>
        <style>
            body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 40px; background: #F8F9FA; color: #1A202C; }
            .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); max-width: 800px; margin: 0 auto; }
            h1 { color: #1E88E5; font-size: 2.2rem; }
            .btn { display: inline-block; background: #7B1FA2; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 600; margin-top: 15px; }
            code { background: #EDF2F7; padding: 3px 6px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🚀 Business Entity Resolution AI API</h1>
            <p>Welcome to the competition-grade Business Entity Resolution machine learning pipeline service.</p>
            <hr/>
            <h3>Available Endpoints:</h3>
            <ul>
                <li><code>GET /api/health</code> — Health Check</li>
                <li><code>GET /api/metrics</code> — Validation Metrics</li>
                <li><code>POST /api/resolve</code> — Resolve Single Entity Match</li>
                <li><code>GET /docs</code> — Interactive Swagger API Documentation</li>
            </ul>
            <a class="btn" href="/docs">View Interactive API Documentation</a>
        </div>
    </body>
    </html>
    """

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "business_entity_resolution", "version": "1.0.0"}

@app.get("/api/metrics")
def get_metrics():
    cfg = load_config()
    val_path = os.path.join(cfg["artifacts"]["metrics_dir"], "validation_summary.json")
    if os.path.exists(val_path):
        with open(val_path, "r") as f:
            return json.load(f)
    return {"status": "no_metrics_found"}

@app.post("/api/resolve")
def resolve_entity(query: EntityQuery):
    try:
        cfg = load_config()
        df_s1 = pd.DataFrame([{
            "entity_id": "S1-API-QUERY",
            "business_name": query.business_name,
            "business_address": query.business_address,
            "country": query.country
        }])
        
        try:
            datasets, _ = load_all_datasets(cfg, sample_size=500)
            df_target = pd.concat([datasets["s2"], datasets["s3"]], ignore_index=True)
        except Exception:
            df_target = pd.DataFrame([{
                "entity_id": "S2-00047", "business_name": "Google Incorporated",
                "business_address": "1600 Amphitheatre Parkway, Mountain View, CA 94043", "country": "US"
            }])

        df_s1_norm = process_normalization(df_s1)
        df_target_norm = process_normalization(df_target)

        blocker = MultiStrategyBlocker(max_candidates_per_entity=20)
        df_cands = blocker.generate_candidates(df_s1_norm, df_target_norm)

        if df_cands.empty:
            return {"entity_id": "S1-API-QUERY", "matched": False, "candidates": []}

        df_feats = extract_pair_features(df_cands, df_s1_norm, df_target_norm)
        
        model_path = os.path.join(cfg["artifacts"]["models_dir"], "matcher_model.joblib")
        if os.path.exists(model_path):
            matcher = EntityResolutionMatcher()
            matcher.load(model_path)
            probas = matcher.predict_proba(df_feats)
        else:
            probas = df_feats["name_fuzz_wratio"].values

        df_feats["match_probability"] = probas
        df_res = df_feats.merge(
            df_target[["entity_id", "business_name", "business_address", "country"]],
            left_on="candidate_entity_id", right_on="entity_id"
        ).sort_values("match_probability", ascending=False)

        results = []
        for row in df_res.head(5).itertuples():
            results.append({
                "candidate_id": row.candidate_entity_id,
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country,
                "probability": float(row.match_probability)
            })

        return {"entity_id": "S1-API-QUERY", "matched": True, "candidates": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# 2. Streamlit Dashboard Mode
# ----------------------------------------------------
def run_streamlit_dashboard():
    import streamlit as st
    import plotly.express as px

    st.set_page_config(
        page_title="Amazon ML Challenge — Business Entity Resolution AI Studio",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .hero-title { font-size: 2.6rem; font-weight: 800; background: linear-gradient(135deg, #1E88E5 0%, #7B1FA2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .hero-sub { font-size: 1.15rem; color: #4A5568; font-weight: 400; margin-bottom: 1.8rem; }
        .metric-card-glass { background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(10px); border: 1px solid rgba(226, 232, 240, 0.8); border-radius: 12px; padding: 1.2rem; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05); }
        .card-val { font-size: 2.2rem; font-weight: 800; color: #1A202C; }
        .card-lbl { font-size: 0.88rem; font-weight: 600; color: #718096; text-transform: uppercase; }
        .badge-tag { display: inline-block; padding: 0.25rem 0.6rem; font-size: 0.75rem; font-weight: 700; border-radius: 20px; background: #EBF8FF; color: #2B6CB0; }
    </style>
    """, unsafe_allow_html=True)

    cfg = load_config()

    st.sidebar.markdown("<h2 style='text-align: center; color: #1E88E5;'>⚡ EntityRes AI Studio</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigation",
        [
            "🏆 Executive Overview",
            "⚡ Live Entity Resolution Query",
            "📈 Threshold & Model Analytics",
            "📁 Submission & Outputs Explorer",
            "⚙️ Control Center"
        ]
    )

    if page == "🏆 Executive Overview":
        st.markdown('<div class="hero-title">Business Entity Resolution System</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Production-grade ML pipeline resolving noisy commercial entities across independent sources</div>', unsafe_allow_html=True)
        
        val_summary_path = os.path.join(cfg["artifacts"]["metrics_dir"], "validation_summary.json")
        val_f05, cand_rec, val_prec, val_rec, sing_acc = 0.0, 0.0, 0.0, 0.0, 1.0
        
        if os.path.exists(val_summary_path):
            with open(val_summary_path, "r") as f:
                m = json.load(f)
                val_f05 = m.get('validation_macro_f0_5', 0.0)
                cand_rec = m.get('candidate_recall', 0.0)
                val_prec = m.get('validation_precision', 0.0)
                val_rec = m.get('validation_recall', 0.0)
                sing_acc = m.get('singleton_accuracy', 1.0)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.markdown(f'<div class="metric-card-glass"><div class="card-val">{val_f05:.4f}</div><div class="card-lbl">Validation F0.5</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card-glass"><div class="card-val">{cand_rec:.4f}</div><div class="card-lbl">Blocking Recall</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card-glass"><div class="card-val">{val_prec:.4f}</div><div class="card-lbl">Precision</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card-glass"><div class="card-val">{val_rec:.4f}</div><div class="card-lbl">Recall</div></div>', unsafe_allow_html=True)
        m5.markdown(f'<div class="metric-card-glass"><div class="card-val">{sing_acc:.4f}</div><div class="card-lbl">Singleton Acc</div></div>', unsafe_allow_html=True)

    elif page == "⚡ Live Entity Resolution Query":
        st.markdown('<div class="hero-title">Live Entity Resolution Query Engine</div>', unsafe_allow_html=True)
        s1_name = st.text_input("Business Name", "Google LLC")
        s1_addr = st.text_input("Business Address", "1600 Amphitheatre Pkwy, Mountain View, CA 94043")
        s1_country = st.selectbox("Country", ["US", "India", "France", "Other"])
        
        if st.button("🚀 Resolve Match Candidates"):
            with st.spinner("Resolving..."):
                res = resolve_entity(EntityQuery(business_name=s1_name, business_address=s1_addr, country=s1_country))
                st.write(res)

    elif page == "📈 Threshold & Model Analytics":
        st.markdown('<div class="hero-title">Model Analytics</div>', unsafe_allow_html=True)
        imp_path = os.path.join(cfg["artifacts"]["reports_dir"], "feature_importance.csv")
        if os.path.exists(imp_path):
            df_imp = pd.read_csv(imp_path)
            st.dataframe(df_imp.head(20), use_container_width=True)

    elif page == "📁 Submission & Outputs Explorer":
        st.markdown('<div class="hero-title">Submission Files</div>', unsafe_allow_html=True)
        m_p = os.path.join(cfg["output"]["dir"], cfg["output"]["matching_results"])
        if os.path.exists(m_p):
            df_m = pd.read_csv(m_p, sep="\t", keep_default_na=False)
            st.dataframe(df_m.head(100), use_container_width=True)

    elif page == "⚙️ Control Center":
        st.markdown('<div class="hero-title">Pipeline Control Center</div>', unsafe_allow_html=True)
        if st.button("▶️ Run Submission Format Validator"):
            raw_dir = cfg["data"]["raw_dir"]
            test_dir = os.path.join(raw_dir, "test") if os.path.exists(os.path.join(raw_dir, "test")) else raw_dir
            m_p = os.path.join(cfg["output"]["dir"], cfg["output"]["matching_results"])
            c_p = os.path.join(cfg["output"]["dir"], cfg["output"]["candidate_pairs"])
            passed = run_submission_validator(m_p, c_p, test_dir)
            st.success("Validator executed successfully.")

# Detect execution context
is_streamlit = any("streamlit" in arg for arg in sys.argv) or "STREAMLIT_RUN" in os.environ or os.environ.get("SERVER_PORT")

if is_streamlit:
    run_streamlit_dashboard()
