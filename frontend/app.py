"""
Streamlit Frontend — Fake Review Detector 🔍

Premium UI for the Fake Review Detection System.
Calls the Flask backend API for predictions.
"""

import streamlit as st
import requests
import time

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Review Detector 🔍",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS for premium look ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #aab;
        font-size: 1.1rem;
    }

    /* Card styling */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Prediction badge */
    .pred-badge {
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .pred-genuine {
        background: linear-gradient(135deg, #0d5524, #1a8a42);
        color: #4ade80;
        border: 2px solid #22c55e;
    }
    .pred-suspicious {
        background: linear-gradient(135deg, #78350f, #a16207);
        color: #fbbf24;
        border: 2px solid #f59e0b;
    }
    .pred-manipulative {
        background: linear-gradient(135deg, #7f1d1d, #b91c1c);
        color: #f87171;
        border: 2px solid #ef4444;
    }

    /* Confidence bar */
    .confidence-container {
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 24px;
        border-radius: 12px;
        background: rgba(255,255,255,0.1);
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 1s ease;
    }

    /* Reason items */
    .reason-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin: 0.4rem 0;
        border-left: 3px solid #3a7bd5;
        color: #ddd;
        font-size: 0.95rem;
    }

    /* Input labels */
    .stTextArea label, .stSlider label, .stSelectbox label, .stTextInput label {
        color: #ccd !important;
        font-weight: 600 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5) !important;
        color: white !important;
        border: none !important;
        padding: 0.7rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(0, 210, 255, 0.4) !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── API Config ────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:5000"

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Fake Review Detector</h1>
    <p>AI-powered review analysis using DistilBERT, GPT-2 & Metadata Analytics</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Input Section ─────────────────────────────────────────────────────────────
st.markdown("### 📝 Enter Review Details")

review_text = st.text_area(
    "Review Text",
    height=150,
    placeholder="Paste or type the review text here..."
)

col1, col2 = st.columns(2)

with col1:
    rating = st.slider("⭐ Rating", min_value=1, max_value=5, value=3)
    verified = st.selectbox("✅ Verified Purchase", ["Yes", "No"], index=0)

with col2:
    ip_address = st.text_input("🌐 IP Address", placeholder="e.g., 192.168.1.1")
    user_id = st.text_input("👤 User ID", placeholder="e.g., user123")

st.markdown("")

# ─── Analyze Button ───────────────────────────────────────────────────────────
if st.button("🔎 Analyze Review"):
    if not review_text or len(review_text.strip()) < 10:
        st.warning("⚠️ Please enter a review with at least 10 characters.")
    else:
        with st.spinner("🧠 Analyzing review with SBERT + GPT-2..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "review_text": review_text,
                        "rating": rating,
                        "verified": 1 if verified == "Yes" else 0,
                        "ip": ip_address,
                        "user_id": user_id
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    reasons = result.get("reasons", [])

                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")

                    # ─── Prediction Badge ──────────────────────────────────
                    if prediction == "Genuine":
                        badge_class = "pred-genuine"
                        emoji = "✅"
                        label = "Genuine Review"
                    elif prediction == "Suspicious":
                        badge_class = "pred-suspicious"
                        emoji = "⚠️"
                        label = "Suspicious Review"
                    else:  # Manipulative
                        badge_class = "pred-manipulative"
                        emoji = "🚨"
                        label = "Manipulative Review"

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="pred-badge {badge_class}">
                            {emoji} {label}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ─── Also show with Streamlit native for color ─────────
                    if prediction == "Genuine":
                        st.success(f"✅ Genuine Review — Confidence: {confidence * 100:.1f}%")
                    elif prediction == "Suspicious":
                        st.warning(f"⚠️ Suspicious Review — Confidence: {confidence * 100:.1f}%")
                    else:
                        st.error(f"🚨 Manipulative Review Detected — Confidence: {confidence * 100:.1f}%")

                    # ─── Confidence Bar ────────────────────────────────────
                    conf_pct = confidence * 100
                    if prediction == "Genuine":
                        bar_color = "#22c55e"
                    elif prediction == "Suspicious":
                        bar_color = "#f59e0b"
                    else:
                        bar_color = "#ef4444"

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="confidence-container">
                            <span style="color: #ccc; font-weight: 600;">
                                🎯 Confidence Score: <span style="color: {bar_color}; font-size: 1.3rem;">{conf_pct:.1f}%</span>
                            </span>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {conf_pct}%; background: {bar_color};"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ─── Reasons ───────────────────────────────────────────
                    if reasons:
                        st.markdown("""
                        <div class="result-card">
                            <span style="color: #ccc; font-weight: 600; font-size: 1.1rem;">📋 Reasons:</span>
                        """, unsafe_allow_html=True)

                        for reason in reasons:
                            st.markdown(f"""
                            <div class="reason-item">• {reason}</div>
                            """, unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

                elif response.status_code == 400:
                    st.error(f"❌ Bad request: {response.json().get('error', 'Unknown error')}")
                else:
                    st.error(f"❌ API error (HTTP {response.status_code})")

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to the backend API. "
                    "Make sure the Flask server is running:\n\n"
                    "`python backend/app.py`"
                )
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The model may still be loading.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem; padding: 1rem 0;">
    <p>🧠 Powered by DistilBERT Sequence Classification + GPT-2 Perplexity + Metadata Analytics</p>
    <p>Full Stack AI Project — NLP × Deep Learning × ML × Flask × Streamlit</p>
</div>
""", unsafe_allow_html=True)
