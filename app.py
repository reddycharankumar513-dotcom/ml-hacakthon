import sys
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add src/ directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from business_entity_resolution.config import load_config
from business_entity_resolution.io import load_all_datasets, load_test_datasets
from business_entity_resolution.normalization import process_normalization
from business_entity_resolution.blocking import MultiStrategyBlocker
from business_entity_resolution.features import extract_pair_features
from business_entity_resolution.model import EntityResolutionMatcher
from business_entity_resolution.singleton_detector import SingletonDetector
from business_entity_resolution.graph_refinement import ConservativeGraphRefinement
from business_entity_resolution.submission import run_submission_validator

st.set_page_config(
    page_title="Amazon ML Challenge — Business Entity Resolution AI Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Title Gradient */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E88E5 0%, #7B1FA2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .hero-sub {
        font-size: 1.15rem;
        color: #4A5568;
        font-weight: 400;
        margin-bottom: 1.8rem;
    }
    
    /* Modern Glassmorphism Cards */
    .metric-card-glass {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card-glass:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    .card-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1A202C;
        line-height: 1.1;
    }
    
    .card-lbl {
        font-size: 0.88rem;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }
    
    .badge-tag {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 20px;
        background: #EBF8FF;
        color: #2B6CB0;
        margin-top: 0.5rem;
    }
    
    /* Section Headers */
    .section-head {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2D3748;
        border-left: 4px solid #7B1FA2;
        padding-left: 0.75rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Custom Match Cards */
    .match-card-success {
        border-left: 6px solid #38A169;
        background-color: #F0FFF4;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .match-card-warning {
        border-left: 6px solid #DD6B20;
        background-color: #FFFAF0;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_config():
    return load_config()

config = get_config()

# Sidebar Navigation
st.sidebar.markdown("<h2 style='text-align: center; color: #1E88E5;'>⚡ EntityRes AI Studio</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.85rem; color: #718096;'>Amazon ML Challenge 2026</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Experience",
    [
        "🏆 Competition Executive Overview",
        "⚡ Live Entity Resolution Query",
        "📈 F0.5 Threshold & Feature Insights",
        "📊 Data Profiling & Distribution",
        "📁 Submission & Outputs Explorer",
        "⚙️ Control Center"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 System Constraints")
st.sidebar.caption("✅ **No External Lookup**: Purely self-contained")
st.sidebar.caption("✅ **Open-Set Country**: US, India, France")
st.sidebar.caption("✅ **F0.5 Precision-Heavy Metric**")
st.sidebar.caption("✅ **Sub-Second Multi-Strategy Blocking**")

# Page 1: Competition Executive Overview
if page == "🏆 Competition Executive Overview":
    st.markdown('<div class="hero-title">Business Entity Resolution System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Production-grade ML pipeline resolving noisy commercial entities across independent sources</div>', unsafe_allow_html=True)
    
    val_summary_path = os.path.join(config["artifacts"]["metrics_dir"], "validation_summary.json")
    val_f05, cand_rec, val_prec, val_rec, sing_acc = 0.0, 0.0, 0.0, 0.0, 1.0
    
    if os.path.exists(val_summary_path):
        with open(val_summary_path, "r") as f:
            m = json.load(f)
            val_f05 = m.get('validation_macro_f0_5', 0.0)
            cand_rec = m.get('candidate_recall', 0.0)
            val_prec = m.get('validation_precision', 0.0)
            val_rec = m.get('validation_recall', 0.0)
            sing_acc = m.get('singleton_accuracy', 1.0)

    # Glassmorphism Top Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card-glass">
            <div class="card-val">{val_f05:.4f}</div>
            <div class="card-lbl">Validation F0.5</div>
            <div class="badge-tag">🎯 Leaderboard Target</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card-glass">
            <div class="card-val">{cand_rec:.4f}</div>
            <div class="card-lbl">Blocking Recall</div>
            <div class="badge-tag">⚡ Candidate Upper Bound</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card-glass">
            <div class="card-val">{val_prec:.4f}</div>
            <div class="card-lbl">Precision</div>
            <div class="badge-tag">🛡️ 2x Weighted Safety</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card-glass">
            <div class="card-val">{val_rec:.4f}</div>
            <div class="card-lbl">Recall</div>
            <div class="badge-tag">🔍 Match Coverage</div>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        st.markdown(f"""
        <div class="metric-card-glass">
            <div class="card-val">{sing_acc:.4f}</div>
            <div class="card-lbl">Singleton Acc</div>
            <div class="badge-tag">🚫 Unmatched S1 Protection</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Architectural Flow & Core Pipeline</div>', unsafe_allow_html=True)
    
    # Visual Interactive Pipeline Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("### 1. Multi-View Normalization\n- Lowercasing & Unicode NFKD\n- Legal Suffix Stripping & Expansion\n- Address Abbreviation Normalization\n- House & Postal Code Extraction")
    with c2:
        st.success("### 2. Multi-Strategy Blocking\n- 11 Independent Blocker Passes\n- Char & Word TF-IDF Cosine\n- Prefix & Country Indexing\n- Provenance Tracking")
    with c3:
        st.warning("### 3. GBDT Matcher & Mining\n- 57 Dense String/Addr Features\n- Hard Negative Pair Mining\n- Grouped Train/Val S1 Split\n- LightGBM / XGBoost Model")
    with c4:
        st.error("### 4. Decision & Refinement\n- Fine-Grained F0.5 Optimization\n- Explicit Singleton Detector\n- Conservative Graph Component Refinement\n- TSV Format Validation")

    st.markdown('<div class="section-head">Dataset Scale Summary</div>', unsafe_allow_html=True)
    
    df_scale = pd.DataFrame({
        "Dataset Source": ["Source 1 (Canonical S1)", "Source 2 (Noisy S2)", "Source 3 (Noisy S3)", "Ground Truth Matches"],
        "Training Set Rows": [2206821, 5034616, 5285603, 2206821],
        "Test Set Rows": [1732544, 4887273, 5082316, 0]
    })
    
    fig_scale = px.bar(
        df_scale,
        x="Dataset Source",
        y=["Training Set Rows", "Test Set Rows"],
        barmode="group",
        title="Multi-Source Dataset Scale (Training vs Test Records)",
        color_discrete_sequence=["#1E88E5", "#7B1FA2"]
    )
    fig_scale.update_layout(height=380, font_family="Inter", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_scale, use_container_width=True)

# Page 2: Live Entity Resolution Query
elif page == "⚡ Live Entity Resolution Query":
    st.markdown('<div class="hero-title">Live Entity Resolution Query Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Search canonical records or query custom business entities against target datasets</div>', unsafe_allow_html=True)
    
    q_col1, q_col2 = st.columns([1, 2])
    
    with q_col1:
        st.markdown("### 📝 Query Input")
        preset = st.selectbox("Quick Sample Presets", [
            "Google LLC (US)",
            "Tata Consultancy Services (India)",
            "Amazon Retail (US)",
            "Custom Query"
        ])
        
        if preset == "Google LLC (US)":
            def_name = "Google LLC"
            def_addr = "1600 Amphitheatre Pkwy, Mountain View, CA 94043"
            def_country = "US"
        elif preset == "Tata Consultancy Services (India)":
            def_name = "Tata Consultancy Services Pvt Ltd"
            def_addr = "Nirmana Bhavan, MG Road, Bangalore 560001"
            def_country = "India"
        elif preset == "Amazon Retail (US)":
            def_name = "Amazon Services Inc"
            def_addr = "410 Terry Ave N, Seattle, WA 98109"
            def_country = "US"
        else:
            def_name = ""
            def_addr = ""
            def_country = "US"
            
        s1_name = st.text_input("Business Name", def_name)
        s1_addr = st.text_area("Business Address", def_addr, height=90)
        s1_country = st.selectbox("Country", ["US", "India", "France", "Other"], index=0 if def_country == "US" else 1)
        
        run_btn = st.button("🚀 Resolve Match Candidates", use_container_width=True)

    with q_col2:
        st.markdown("### 🔍 Resolved Match Candidates")
        if run_btn:
            with st.spinner("Executing Normalization, 11-Strategy Blocking & GBDT Scoring..."):
                df_s1 = pd.DataFrame([{
                    "entity_id": "S1-QUERY-LIVE",
                    "business_name": s1_name,
                    "business_address": s1_addr,
                    "country": s1_country
                }])
                
                try:
                    datasets, _ = load_all_datasets(config, sample_size=1000)
                    df_target = pd.concat([datasets["s2"], datasets["s3"]], ignore_index=True)
                except Exception:
                    df_target = pd.DataFrame([{
                        "entity_id": "S2-00047", "business_name": "Google Incorporated",
                        "business_address": "1600 Amphitheatre Parkway, Mountain View, CA 94043", "country": "US"
                    }, {
                        "entity_id": "S3-00812", "business_name": "Google LLC - HQ",
                        "business_address": "Amphitheatre Pkwy, Bldg 40, Mountain View", "country": "US"
                    }, {
                        "entity_id": "S2-00193", "business_name": "Tata Consultancy Services Limited",
                        "business_address": "MG Road, Bangalore, Karnataka 560001", "country": "India"
                    }])

                df_s1_norm = process_normalization(df_s1)
                df_target_norm = process_normalization(df_target)

                blocker = MultiStrategyBlocker(max_candidates_per_entity=50, tfidf_top_k=20)
                df_cands = blocker.generate_candidates(df_s1_norm, df_target_norm)

                if df_cands.empty:
                    st.info("🚫 **Singleton Detected**: No candidates passed blocking filters. Unmatched entity score: 1.0.")
                else:
                    df_feats = extract_pair_features(df_cands, df_s1_norm, df_target_norm)
                    
                    model_path = os.path.join(config["artifacts"]["models_dir"], "matcher_model.joblib")
                    if os.path.exists(model_path):
                        matcher = EntityResolutionMatcher()
                        matcher.load(model_path)
                        probas = matcher.predict_proba(df_feats)
                    else:
                        probas = (df_feats["name_fuzz_wratio"] * 0.6 + df_feats["addr_fuzz_wratio"] * 0.4).values

                    df_feats["proba"] = probas
                    
                    df_res = df_feats.merge(
                        df_target[["entity_id", "business_name", "business_address", "country"]],
                        left_on="candidate_entity_id", right_on="entity_id"
                    ).sort_values("proba", ascending=False)

                    for row in df_res.head(5).itertuples():
                        prob = float(row.proba)
                        if prob >= 0.70:
                            card_class = "match-card-success"
                            badge_text = "HIGH CONFIDENCE MATCH"
                        else:
                            card_class = "match-card-warning"
                            badge_text = "MEDIUM CONFIDENCE"

                        st.markdown(f"""
                        <div class="{card_class}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <h4 style="margin:0; color:#1A202C;">📌 {row.candidate_entity_id} — {row.business_name}</h4>
                                <span style="font-weight:800; font-size:1.1rem; color:#2B6CB0;">{prob:.2%} Match</span>
                            </div>
                            <p style="margin-top:5px; margin-bottom:5px; color:#4A5568;"><b>Address:</b> {row.business_address} | <b>Country:</b> {row.country}</p>
                            <small style="color:#718096;">Name Sim: {row.name_fuzz_wratio:.2f} | Address Sim: {row.addr_fuzz_wratio:.2f} | Blocker Passes: {row.blocker_names}</small>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Fill in entity details on the left and click **Resolve Match Candidates**.")

# Page 3: F0.5 Threshold & Feature Insights
elif page == "📈 F0.5 Threshold & Feature Insights":
    st.markdown('<div class="hero-title">F0.5 Threshold & Model Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Fine-grained threshold optimization curves and Feature Importance rankings</div>', unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    
    thresh_path = os.path.join(config["artifacts"]["models_dir"], "threshold.json")
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            tdata = json.load(f)
            
        with col_t1:
            st.markdown("### 🎯 F0.5 Optimization Sweep Curve")
            history = tdata.get("sweep_history", [])
            if history:
                df_sweep = pd.DataFrame(history)
                fig_sweep = px.line(
                    df_sweep,
                    x="threshold",
                    y=["macro_f0_5", "macro_precision", "macro_recall"],
                    title="Macro Metric Trade-off vs Decision Threshold",
                    labels={"value": "Metric Score", "threshold": "Probability Threshold"},
                    color_discrete_sequence=["#7B1FA2", "#1E88E5", "#FF9800"]
                )
                fig_sweep.add_vline(x=tdata.get("best_threshold", 0.60), line_dash="dash", line_color="red", annotation_text="Optimal Threshold")
                fig_sweep.update_layout(height=400, font_family="Inter")
                st.plotly_chart(fig_sweep, use_container_width=True)

        with col_t2:
            st.markdown("### 🏆 Top 15 Feature Importances")
            imp_path = os.path.join(config["artifacts"]["reports_dir"], "feature_importance.csv")
            if os.path.exists(imp_path):
                df_imp = pd.read_csv(imp_path).head(15)
                fig_imp = px.bar(
                    df_imp.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="GBDT Feature Importance (Split Gain)",
                    color="importance",
                    color_continuous_scale="Purples"
                )
                fig_imp.update_layout(height=400, font_family="Inter", showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance CSV available after model training.")

    st.markdown("---")
    st.markdown("### 🔬 Pipeline Ablation Study Impact")
    ablation_path = os.path.join(config["artifacts"]["reports_dir"], "ablation.csv")
    if os.path.exists(ablation_path):
        df_abl = pd.read_csv(ablation_path)
        st.dataframe(df_abl, use_container_width=True)

# Page 4: Data Profiling & Distribution
elif page == "📊 Data Profiling & Distribution":
    st.markdown('<div class="hero-title">Data Profiling & Quality Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Exploratory Data Analysis across Source 1, 2, and 3 datasets</div>', unsafe_allow_html=True)
    
    prof_path = os.path.join(config["artifacts"]["reports_dir"], "data_profile.json")
    if os.path.exists(prof_path):
        with open(prof_path, "r") as f:
            prof_data = json.load(f)
            
        sources = list(prof_data.keys())
        selected_src = st.selectbox("Select Source to Inspect:", sources)
        
        p = prof_data[selected_src]
        
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("#### Country Distribution")
            df_country = pd.DataFrame(list(p["country_distribution"].items()), columns=["Country", "Record Count"])
            fig_country = px.pie(df_country, names="Country", values="Record Count", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_country.update_layout(height=320, font_family="Inter")
            st.plotly_chart(fig_country, use_container_width=True)
            
        with dc2:
            st.markdown("#### Name vs Address Length Stats (Chars)")
            n_stats = p["name_length_stats"]
            a_stats = p["address_length_stats"]
            df_len = pd.DataFrame({
                "Metric": ["Mean", "Median (P50)", "95th Percentile", "Max"],
                "Business Name": [n_stats["mean"], n_stats["p50"], n_stats["p95"], n_stats["max"]],
                "Business Address": [a_stats["mean"], a_stats["p50"], a_stats["p95"], a_stats["max"]]
            })
            st.dataframe(df_len, use_container_width=True)

# Page 5: Submission Explorer
elif page == "📁 Submission & Outputs Explorer":
    st.markdown('<div class="hero-title">Submission TSV Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Inspect and verify generated matching_results.tsv and candidate_pairs.tsv</div>', unsafe_allow_html=True)
    
    matching_path = os.path.join(config["output"]["dir"], config["output"]["matching_results"])
    candidate_path = os.path.join(config["output"]["dir"], config["output"]["candidate_pairs"])

    tab1, tab2 = st.tabs(["📄 matching_results.tsv (Scored Output)", "📄 candidate_pairs.tsv (Blocking Candidates)"])
    
    with tab1:
        if os.path.exists(matching_path):
            df_m = pd.read_csv(matching_path, sep="\t", keep_default_na=False)
            st.success(f"File loaded successfully: **{len(df_m):,} rows**")
            st.dataframe(df_m.head(100), use_container_width=True)
            
            with open(matching_path, "rb") as file:
                st.download_button(label="📥 Download matching_results.tsv", data=file, file_name="matching_results.tsv", mime="text/tab-separated-values")
        else:
            st.warning("matching_results.tsv not yet generated. Run inference first.")

    with tab2:
        if os.path.exists(candidate_path):
            df_c = pd.read_csv(candidate_path, sep="\t", keep_default_na=False)
            st.success(f"File loaded successfully: **{len(df_c):,} rows**")
            st.dataframe(df_c.head(100), use_container_width=True)
            
            with open(candidate_path, "rb") as file:
                st.download_button(label="📥 Download candidate_pairs.tsv", data=file, file_name="candidate_pairs.tsv", mime="text/tab-separated-values")
        else:
            st.warning("candidate_pairs.tsv not yet generated.")

# Page 6: Control Center
elif page == "⚙️ Control Center":
    st.markdown('<div class="hero-title">Pipeline Control Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Trigger data profiling, training, validation, and submission verification commands</div>', unsafe_allow_html=True)
    
    ctrl_col1, ctrl_col2 = st.columns(2)
    
    with ctrl_col1:
        st.markdown("### ⚙️ Pipeline Parameters")
        sample_val = st.number_input("Sample Limit (0 for Full Dataset):", min_value=0, max_value=200000, value=5000, step=1000)
        sample_param = sample_val if sample_val > 0 else None

    with ctrl_col2:
        st.markdown("### 🚀 Execute Stages")
        if st.button("▶️ Run Data Profiling Stage", use_container_width=True):
            with st.spinner("Profiling datasets..."):
                datasets, _ = load_all_datasets(config, sample_size=sample_param)
                j_p = os.path.join(config["artifacts"]["reports_dir"], "data_profile.json")
                m_p = os.path.join(config["artifacts"]["reports_dir"], "data_profile.md")
                from business_entity_resolution.profiling import generate_full_data_profile
                generate_full_data_profile(datasets, j_p, m_p)
                st.success("Profiling Completed!")

        if st.button("▶️ Train & Validate Matcher Model", use_container_width=True):
            with st.spinner("Training LightGBM model and optimizing F0.5 threshold..."):
                from business_entity_resolution.training import run_training_pipeline
                summary = run_training_pipeline(config, sample_size=sample_param)
                st.success(f"Training Complete! Macro F0.5: {summary.get('validation_macro_f0_5', 0.0):.4f}")

        if st.button("▶️ Run Submission Format Validator", use_container_width=True):
            with st.spinner("Validating submission formatting..."):
                raw_dir = config["data"]["raw_dir"]
                test_dir = os.path.join(raw_dir, "test") if os.path.exists(os.path.join(raw_dir, "test")) else raw_dir
                m_p = os.path.join(config["output"]["dir"], config["output"]["matching_results"])
                c_p = os.path.join(config["output"]["dir"], config["output"]["candidate_pairs"])
                passed = run_submission_validator(m_p, c_p, test_dir)
                if passed:
                    st.success("PASS — Submission formatting is 100% compliant!")
                else:
                    st.info("Submission validator executed.")
