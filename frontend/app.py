"""
Streamlit Frontend — Fake Review Intelligence Platform 🔍

Two-tab premium dashboard with Dark / Light mode toggle:
  Tab 1 — Admin Review Analyzer
  Tab 2 — Fraud Analytics Dashboard
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Review Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = "http://127.0.0.1:5000"

# ─── Theme State ───────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

is_dark = st.session_state.dark_mode

# ─── CSS ───────────────────────────────────────────────────────────────────────

DARK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
.stApp {
    background: linear-gradient(140deg, #060614 0%, #0d0b26 30%, #111830 60%, #0a1520 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04); border-radius: 14px;
    padding: 5px 6px; gap: 4px; border: 1px solid rgba(255,255,255,0.09);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #7a8fa6 !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.92rem !important; padding: 8px 28px !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,rgba(0,210,255,0.15),rgba(58,123,213,0.15)) !important;
    color: #00d2ff !important; border: 1px solid rgba(0,210,255,0.25) !important;
}

/* Header */
.main-header { text-align: center; padding: 0.5rem 0 0.4rem; }
.main-header h1 {
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(90deg, #00d2ff 0%, #5b9cff 50%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.2rem; line-height: 1.1;
}
.main-header p { color: #6a7f96; font-size: 0.95rem; margin: 0; }

/* Section title */
.section-title {
    color: #c8d8ec; font-size: 1.0rem; font-weight: 700;
    margin: 1.2rem 0 0.6rem; display: flex; align-items: center; gap: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.4rem;
}

/* KPI Card */
.kpi-card {
    background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px; padding: 1.1rem 0.7rem 0.9rem; text-align: center;
    backdrop-filter: blur(16px); transition: transform 0.25s, border-color 0.25s;
    height: 125px; display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(0,210,255,0.25); }
.kpi-icon  { font-size: 1.3rem; line-height: 1; }
.kpi-value { font-size: 1.9rem; font-weight: 800; line-height: 1.15; }
.kpi-label { font-size: 0.68rem; color: #607080; font-weight: 600; text-transform: uppercase; letter-spacing: .09em; }

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px; padding: 1.3rem 1.4rem; margin: 0.7rem 0; backdrop-filter: blur(12px);
}
.admin-banner {
    background: rgba(255,255,255,0.04); border-radius: 12px; padding: 0.85rem 1.2rem;
    margin-bottom: 0.8rem; border-left: 3px solid #3a7bd5;
    border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #3a7bd5;
}

/* Prediction badges */
.pred-badge { text-align:center; padding:1rem; border-radius:14px; font-size:1.7rem; font-weight:800; margin:0.4rem 0; }
.pred-genuine      { background:linear-gradient(135deg,#0a4820,#166534); color:#4ade80; border:2px solid #22c55e; }
.pred-suspicious   { background:linear-gradient(135deg,#6b2d0a,#92400e); color:#fbbf24; border:2px solid #f59e0b; }
.pred-manipulative { background:linear-gradient(135deg,#6b1a1a,#991b1b); color:#f87171; border:2px solid #ef4444; }

/* Confidence bar */
.conf-bar-wrap { height:20px; border-radius:10px; background:rgba(255,255,255,0.08); overflow:hidden; margin-top:0.5rem; }
.conf-bar-fill  { height:100%; border-radius:10px; }

/* Reason item */
.reason-item {
    background: rgba(255,255,255,0.035); padding: 0.6rem 0.9rem;
    border-radius: 9px; margin: 0.3rem 0; border-left: 3px solid #3a7bd5;
    color: #c8d8ec; font-size: 0.88rem; line-height: 1.5;
}

/* Activity table & Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
.act-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.act-table th { position:sticky; top:0; z-index:10; background:rgba(18,25,48,0.95); color:#00d2ff; padding:0.5rem 0.8rem;
    text-align:left; font-weight:700; font-size:0.7rem; text-transform:uppercase; letter-spacing:.07em; box-shadow:0 1px 0 rgba(255,255,255,0.05); }
.act-table td { padding:0.45rem 0.8rem; color:#b0c4d8; border-bottom:1px solid rgba(255,255,255,0.045); }
.act-table tr:hover td { background:rgba(255,255,255,0.025); }
.badge-genuine { color:#4ade80; font-weight:700; }
.badge-suspicious { color:#fbbf24; font-weight:700; }
.badge-manipulative { color:#f87171; font-weight:700; }

/* Widgets */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; color: #dce8f4 !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(0,210,255,0.45) !important;
    box-shadow: 0 0 0 2px rgba(0,210,255,0.12) !important;
}
.stTextArea label, .stTextInput label, .stSlider label, .stSelectbox label {
    color: #8aabcc !important; font-weight: 600 !important;
    font-size: 0.82rem !important; text-transform: uppercase !important; letter-spacing: .06em !important;
}
.stButton > button {
    background: linear-gradient(90deg, #00d2ff, #3a7bd5) !important; color: #fff !important;
    border: none !important; padding: 0.68rem 1.8rem !important; font-size: 1.0rem !important;
    font-weight: 700 !important; border-radius: 12px !important; width: 100% !important;
    letter-spacing: .04em !important; transition: all 0.25s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(0,210,255,0.28) !important; }

/* Theme toggle specific */
.theme-btn > button {
    background: rgba(255,255,255,0.07) !important; color: #c8d8ec !important;
    border: 1px solid rgba(255,255,255,0.15) !important; border-radius: 20px !important;
    padding: 0.4rem 1.1rem !important; font-size: 0.82rem !important;
    font-weight: 600 !important; width: auto !important;
    transition: all 0.25s !important; letter-spacing: .03em !important;
}
.theme-btn > button:hover { background: rgba(255,255,255,0.12) !important; transform:none !important; box-shadow:none !important; }

.empty-state { text-align:center; padding:3rem 2rem; color:#3a4f60; font-size:0.92rem; }
.empty-state .icon { font-size:2.8rem; margin-bottom:0.8rem; }
hr { border-color: rgba(255,255,255,0.07) !important; }
"""

LIGHT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
.stApp {
    background: #ffffff;
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,0,0,0.04); border-radius: 14px;
    padding: 5px 6px; gap: 4px; border: 1px solid rgba(0,0,0,0.08);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #5a7090 !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.92rem !important; padding: 8px 28px !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,rgba(0,150,200,0.12),rgba(58,100,213,0.12)) !important;
    color: #0077aa !important; border: 1px solid rgba(0,150,200,0.3) !important;
}

/* Header */
.main-header { text-align: center; padding: 0.5rem 0 0.4rem; }
.main-header h1 {
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(90deg, #0077aa 0%, #2255cc 50%, #6633cc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.2rem; line-height: 1.1;
}
.main-header p { color: #5a7090; font-size: 0.95rem; margin: 0; }

/* Section title */
.section-title {
    color: #1a2840; font-size: 1.0rem; font-weight: 700;
    margin: 1.2rem 0 0.6rem; display: flex; align-items: center; gap: 0.4rem;
    border-bottom: 1px solid rgba(0,0,0,0.08); padding-bottom: 0.4rem;
}

/* KPI Card */
.kpi-card {
    background: rgba(255,255,255,0.85); border: 1px solid rgba(0,0,0,0.08);
    border-radius: 18px; padding: 1.1rem 0.7rem 0.9rem; text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); transition: transform 0.25s, box-shadow 0.25s;
    height: 125px; display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,100,200,0.12); }
.kpi-icon  { font-size: 1.3rem; line-height: 1; }
.kpi-value { font-size: 1.9rem; font-weight: 800; line-height: 1.15; }
.kpi-label { font-size: 0.68rem; color: #7090a8; font-weight: 600; text-transform: uppercase; letter-spacing: .09em; }

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.85); border: 1px solid rgba(0,0,0,0.08);
    border-radius: 18px; padding: 1.3rem 1.4rem; margin: 0.7rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.admin-banner {
    background: rgba(255,255,255,0.85); border-radius: 12px; padding: 0.85rem 1.2rem;
    margin-bottom: 0.8rem; border: 1px solid rgba(0,0,0,0.08); border-left: 3px solid #2255cc;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}

/* Prediction badges */
.pred-badge { text-align:center; padding:1rem; border-radius:14px; font-size:1.7rem; font-weight:800; margin:0.4rem 0; }
.pred-genuine      { background:linear-gradient(135deg,#d4f7e0,#a5f0c0); color:#146624; border:2px solid #22c55e; }
.pred-suspicious   { background:linear-gradient(135deg,#fff3cd,#ffe08a); color:#8a5a00; border:2px solid #f59e0b; }
.pred-manipulative { background:linear-gradient(135deg,#ffe0e0,#ffb8b8); color:#8a0000; border:2px solid #ef4444; }

/* Confidence bar */
.conf-bar-wrap { height:20px; border-radius:10px; background:rgba(0,0,0,0.08); overflow:hidden; margin-top:0.5rem; }
.conf-bar-fill  { height:100%; border-radius:10px; }

/* Reason item */
.reason-item {
    background: rgba(0,0,0,0.03); padding: 0.6rem 0.9rem;
    border-radius: 9px; margin: 0.3rem 0; border-left: 3px solid #2255cc;
    color: #1a2840; font-size: 0.88rem; line-height: 1.5;
}

/* Activity table & Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.02); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.25); }
.act-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.act-table th { position:sticky; top:0; z-index:10; background:rgba(240,245,250,0.95); color:#0055aa; padding:0.5rem 0.8rem;
    text-align:left; font-weight:700; font-size:0.7rem; text-transform:uppercase; letter-spacing:.07em; box-shadow:0 1px 0 rgba(0,0,0,0.06); }
.act-table td { padding:0.45rem 0.8rem; color:#2a3a50; border-bottom:1px solid rgba(0,0,0,0.06); }
.act-table tr:hover td { background:rgba(0,0,0,0.02); }
.badge-genuine { color:#146624; font-weight:700; }
.badge-suspicious { color:#8a5a00; font-weight:700; }
.badge-manipulative { color:#8a0000; font-weight:700; }

/* Widgets */
.stTextArea textarea, .stTextInput input {
    background: #fff !important; border: 1px solid rgba(0,0,0,0.15) !important;
    border-radius: 10px !important; color: #1a2840 !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(0,100,200,0.4) !important;
    box-shadow: 0 0 0 2px rgba(0,100,200,0.10) !important;
}
.stTextArea label, .stTextInput label, .stSlider label, .stSelectbox label {
    color: #2a4a6a !important; font-weight: 600 !important;
    font-size: 0.82rem !important; text-transform: uppercase !important; letter-spacing: .06em !important;
}
.stButton > button {
    background: linear-gradient(90deg, #0077cc, #2255cc) !important; color: #fff !important;
    border: none !important; padding: 0.68rem 1.8rem !important; font-size: 1.0rem !important;
    font-weight: 700 !important; border-radius: 12px !important; width: 100% !important;
    letter-spacing: .04em !important; transition: all 0.25s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(0,100,200,0.25) !important; }

/* Theme toggle specific */
.theme-btn > button {
    background: rgba(0,0,0,0.06) !important; color: #2a4a6a !important;
    border: 1px solid rgba(0,0,0,0.14) !important; border-radius: 20px !important;
    padding: 0.4rem 1.1rem !important; font-size: 0.82rem !important;
    font-weight: 600 !important; width: auto !important;
    transition: all 0.25s !important; letter-spacing: .03em !important;
}
.theme-btn > button:hover { background: rgba(0,0,0,0.10) !important; transform:none !important; box-shadow:none !important; }

.empty-state { text-align:center; padding:3rem 2rem; color:#8090a8; font-size:0.92rem; }
.empty-state .icon { font-size:2.8rem; margin-bottom:0.8rem; }
hr { border-color: rgba(0,0,0,0.08) !important; }
"""

# Inject theme CSS
st.markdown(f"<style>{DARK_CSS if is_dark else LIGHT_CSS}</style>", unsafe_allow_html=True)

# ─── Plotly base layout (theme-aware) ─────────────────────────────────────────
_text_color = "#b0c4d8" if is_dark else "#2a3a50"
_grid_color = "rgba(255,255,255,0.05)" if is_dark else "rgba(0,0,0,0.07)"

PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=_text_color, family='Inter, sans-serif', size=12),
    margin=dict(l=12, r=12, t=36, b=12),
)


# ─── Helper utilities ─────────────────────────────────────────────────────────
def kpi_card_html(icon, value, label, color):
    return f"""<div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""


def fetch_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def fmt_ts(ts_str):
    try:
        return datetime.fromisoformat(ts_str).strftime("%b %d  %H:%M:%S")
    except Exception:
        return ts_str or "—"


# ─── Page Header + Theme Toggle ────────────────────────────────────────────────
hdr_col, toggle_col = st.columns([9, 1])

with hdr_col:
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Fake Review Intelligence</h1>
        <p>Advanced fraud detection · DistilBERT · GPT-2 · Semantic Mismatch · Temporal Burst Analysis</p>
    </div>""", unsafe_allow_html=True)

with toggle_col:
    st.markdown("<div style='padding-top:1.4rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="theme-btn">', unsafe_allow_html=True)
    toggle_label = "☀️ Light" if is_dark else "🌙 Dark"
    if st.button(toggle_label, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_analyze, tab_dashboard = st.tabs(["🔎  Review Analyzer", "📊  Fraud Analytics Dashboard"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — REVIEW ANALYZER (Admin Investigation Panel)
# ══════════════════════════════════════════════════════════════════════════
with tab_analyze:
    _sub_color   = "#8aabcc" if is_dark else "#2a4a6a"
    _body_color  = "#c8d8ec" if is_dark else "#1a2840"
    _hint_color  = "#607080" if is_dark else "#7090a8"
    _chip_bg     = "rgba(0,210,255,0.06)" if is_dark else "rgba(0,100,200,0.06)"
    _chip_border = "rgba(0,210,255,0.15)" if is_dark else "rgba(0,100,200,0.15)"
    _chip_text   = "#00d2ff" if is_dark else "#0055aa"

    st.markdown(f'<div class="section-title">📝 Review Content</div>', unsafe_allow_html=True)

    review_text = st.text_area(
        "Review Body",
        height=145,
        placeholder="Paste the full review text here (headline + body)…",
        key="body_input",
    )

    st.markdown('<div class="section-title">📋 Review Metadata</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        rating   = st.slider("⭐ Star Rating", min_value=1, max_value=5, value=3)
        verified = st.selectbox("✅ Verified Purchase", ["Yes", "No"], index=0)
    with col_right:
        ip_address = st.text_input("🌐 IP Address",   placeholder="e.g., 203.0.113.42", key="ip_input")
        user_id    = st.text_input("👤 User ID",       placeholder="e.g., user_abc123",  key="uid_input")

    st.markdown('<div class="section-title">🕐 Review Timestamp</div>', unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:{_hint_color};font-size:0.81rem;'>"
        "Enter the date and time when this review was originally posted (from your dataset metadata)."
        "</span>", unsafe_allow_html=True,
    )

    col_date, col_time = st.columns(2)
    with col_date:
        review_date = st.date_input("📅 Review Date",
                                    value=datetime.now().date(), key="review_date_input")
    with col_time:
        review_time = st.time_input("⏰ Review Time",
                                    value=datetime.now().replace(second=0, microsecond=0).time(),
                                    key="review_time_input", step=60)

    review_timestamp = datetime.combine(review_date, review_time).isoformat()
    st.markdown(f"""
    <div style="margin:.35rem 0 .85rem;padding:.48rem .9rem;
                background:{_chip_bg};border-radius:8px;border:1px solid {_chip_border};">
        <span style="color:{_hint_color};font-size:.76rem;font-weight:600;
               text-transform:uppercase;letter-spacing:.06em;">Captured Timestamp</span><br>
        <span style="color:{_chip_text};font-size:.93rem;font-weight:600;font-family:monospace;">
            {review_timestamp}
        </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    if st.button("🔎 Analyze Review", key="analyze_btn"):
        if not review_text or len(review_text.strip()) < 10:
            st.warning("⚠️ Please enter a review body with at least 10 characters.")
        else:
            with st.spinner("🧠 Running DistilBERT + GPT-2 + Semantic Engine…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={
                            "review_text":     review_text,
                            "rating":          rating,
                            "verified":        1 if verified == "Yes" else 0,
                            "ip":              ip_address,
                            "user_id":         user_id,
                            "timestamp":       review_timestamp,
                        },
                        timeout=90,
                    )

                    if resp.status_code == 200:
                        result     = resp.json()
                        prediction = result["prediction"]
                        confidence = result["confidence"]
                        reasons    = result.get("reasons", [])

                        st.markdown("---")
                        st.markdown('<div class="section-title">📊 Analysis Results</div>',
                                    unsafe_allow_html=True)

                        badge_map = {
                            "Genuine":      ("pred-genuine",      "✅", "Genuine Review"),
                            "Suspicious":   ("pred-suspicious",   "⚠️", "Suspicious Review"),
                            "Manipulative": ("pred-manipulative", "🚨", "Manipulative Review"),
                        }
                        badge_cls, emoji, lbl = badge_map.get(
                            prediction, ("pred-suspicious", "⚠️", prediction)
                        )

                        st.markdown(f"""<div class="glass-card">
                            <div class="pred-badge {badge_cls}">{emoji}&nbsp; {lbl}</div>
                        </div>""", unsafe_allow_html=True)

                        if prediction == "Genuine":
                            st.success(f"✅ Genuine — Confidence: {confidence*100:.1f}%")
                            bar_col = "#22c55e"
                        elif prediction == "Suspicious":
                            st.warning(f"⚠️ Suspicious — Confidence: {confidence*100:.1f}%")
                            bar_col = "#f59e0b"
                        else:
                            st.error(f"🚨 Manipulative Detected — Confidence: {confidence*100:.1f}%")
                            bar_col = "#ef4444"

                        pct = confidence * 100
                        st.markdown(f"""<div class="glass-card">
                            <span style="color:{_sub_color};font-weight:700;font-size:.80rem;
                                         text-transform:uppercase;letter-spacing:.07em;">🎯 Confidence Score</span>
                            <div style="font-size:1.8rem;font-weight:800;color:{bar_col};
                                        margin:.22rem 0 .45rem;">{pct:.1f}%</div>
                            <div class="conf-bar-wrap">
                                <div class="conf-bar-fill" style="width:{pct}%;background:{bar_col};"></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                        if reasons:
                            signals = "".join(
                                f'<div class="reason-item">• {r}</div>' for r in reasons
                            )
                            st.markdown(f"""<div class="glass-card">
                                <div style="color:{_sub_color};font-weight:700;font-size:.80rem;
                                            text-transform:uppercase;letter-spacing:.07em;margin-bottom:.55rem;">
                                    📋 Detection Signals
                                </div>
                                {signals}
                            </div>""", unsafe_allow_html=True)

                    elif resp.status_code == 400:
                        st.error(f"❌ Bad request: {resp.json().get('error','Unknown error')}")
                    else:
                        st.error(f"❌ API error (HTTP {resp.status_code})")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Make sure Flask is running: `python backend/app.py`")
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out — model may still be loading.")
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — FRAUD ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    _pie_colors = ["#22c55e", "#f59e0b", "#ef4444"]
    _bar_scale  = [[0, "#7f1d1d"], [0.5, "#c0392b"], [1, "#ff5252"]] if is_dark else \
                  [[0, "#fca5a5"], [0.5, "#ef4444"], [1, "#991b1b"]]

    btn_col, _ = st.columns([1, 7])
    with btn_col:
        if st.button("🔄 Refresh Data", key="refresh_btn"):
            st.rerun()

    stats = fetch_stats()

    if stats is None:
        st.markdown("""<div class="empty-state"><div class="icon">📡</div>
            <p>Cannot reach the API at <b>localhost:5000</b>.<br>
               Make sure Flask is running: <code>python backend/app.py</code></p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    if stats.get("total", 0) == 0:
        st.markdown("""<div class="empty-state"><div class="icon">📊</div>
            <p>No data yet.<br>
               Run <code>python backend/precompute_dataset_stats.py</code> to load the
               dataset baseline, then analyze reviews to see live updates here.</p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    total        = stats.get("total",        0)
    genuine      = stats.get("genuine",      0)
    suspicious   = stats.get("suspicious",   0)
    manipulative = stats.get("manipulative", 0)
    mal_users    = stats.get("malicious_users", 0)
    mal_ips      = stats.get("malicious_ips",   0)
    detect_rate  = round((suspicious + manipulative) / total * 100, 1) if total else 0

    # ── KPI Cards ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Live Intelligence Overview</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_data = [
        (c1, "📋", total,             "Total Reviewed",  "#00d2ff" if is_dark else "#0066bb"),
        (c2, "⚠️", suspicious,       "Suspicious",       "#f59e0b"),
        (c3, "🚨", manipulative,      "Manipulative",     "#ef4444"),
        (c4, "👤", mal_users,         "Malicious Users",  "#a78bfa"),
        (c5, "🌐", mal_ips,           "Flagged IPs",      "#f472b6"),
        (c6, "🎯", f"{detect_rate}%", "Detection Rate",   "#34d399" if is_dark else "#059669"),
    ]
    for col, icon, val, lbl, clr in kpi_data:
        with col:
            st.markdown(kpi_card_html(icon, val, lbl, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pie + Bar ─────────────────────────────────────────────────────────
    col_pie, col_bar = st.columns([1, 1.6])

    with col_pie:
        st.markdown('<div class="section-title">🥧 Review Distribution</div>',
                    unsafe_allow_html=True)

        fig_pie = go.Figure(go.Pie(
            labels=["Genuine", "Suspicious", "Manipulative"],
            values=[genuine, suspicious, manipulative],
            hole=0.52,
            marker=dict(
                colors=_pie_colors,
                line=dict(color='rgba(0,0,0,0.3)', width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=12, color="#fff" if is_dark else "#1a2840"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
            pull=[0.02, 0.03, 0.04],
        ))
        fig_pie.add_annotation(
            text=f"<b>{total:,}</b><br><span style='font-size:10px'>Reviews</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=_text_color),
        )
        fig_pie.update_layout(**PLOTLY_BASE, showlegend=False, height=290)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with col_bar:
        st.markdown('<div class="section-title">🌐 Top Flagged IP Addresses</div>',
                    unsafe_allow_html=True)

        top_ips = stats.get("top_malicious_ips", [])
        if top_ips:
            ips_df = pd.DataFrame(top_ips, columns=["IP Address", "Fake Reviews"])
            fig_bar = go.Figure(go.Bar(
                x=ips_df["Fake Reviews"],
                y=ips_df["IP Address"],
                orientation='h',
                marker=dict(
                    color=ips_df["Fake Reviews"],
                    colorscale=_bar_scale,
                    showscale=False,
                    line=dict(width=0),
                ),
                text=ips_df["Fake Reviews"],
                textposition="outside",
                textfont=dict(color="#f87171" if is_dark else "#991b1b", size=12),
                hovertemplate="<b>%{y}</b><br>Fake Reviews: %{x:,}<extra></extra>",
            ))
            fig_bar.update_layout(
                **PLOTLY_BASE, height=290,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color=_text_color)),
                bargap=0.25,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("""<div class="empty-state" style="padding:2.5rem;">
                <div class="icon">✅</div>
                <p>No flagged IP addresses yet.</p></div>""", unsafe_allow_html=True)

    # ── Recent Activity Table ─────────────────────────────────────────────
    st.markdown('<div class="section-title">🕐 Recent Activity Log</div>',
                unsafe_allow_html=True)

    recent = stats.get("recent_entries", [])
    if recent:
        rows_html = ""
        for e in recent:
            pred   = e.get("prediction", "")
            cls    = f"badge-{pred.lower()}"
            icon   = "✅" if pred == "Genuine" else ("⚠️" if pred == "Suspicious" else "🚨")
            ts     = fmt_ts(e.get("timestamp", ""))
            uid    = e.get("user_id", "") or "<i style='opacity:.4'>anonymous</i>"
            ip_val = e.get("ip", "")      or "<i style='opacity:.4'>—</i>"
            rows_html += f"""<tr>
                <td>{ts}</td><td>{uid}</td><td>{ip_val}</td>
                <td><span class="{cls}">{icon} {pred}</span></td>
            </tr>"""

        st.markdown(f"""<div class="glass-card" style="padding:0;overflow:hidden;">
            <div style="max-height: 420px; overflow-y: auto;">
            <table class="act-table">
                <thead><tr>
                    <th>Timestamp</th><th>User ID</th>
                    <th>IP Address</th><th>Verdict</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="empty-state" style="padding:2rem;">
            <div class="icon">📋</div><p>No activity logged yet.</p>
        </div>""", unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
_footer_color = "#2a3a4a" if is_dark else "#8090a8"
st.markdown(f"""<div style="text-align:center;color:{_footer_color};font-size:0.78rem;padding:.5rem 0;">
    🧠 DistilBERT · GPT-2 Perplexity · SentenceTransformer Semantic Similarity ·
    Temporal Burst Detection · Flask · Streamlit
</div>""", unsafe_allow_html=True)
