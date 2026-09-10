"""
=============================================================================
app.py — Main Streamlit Application Entry Point
=============================================================================
CNC Predictive Maintenance System
Final Year Engineering Project
=============================================================================
"""

import sys
import os
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Auto-bootstrap Streamlit if invoked directly via bare python (e.g. Railway default `python app.py`)
if not st.runtime.exists():
    from streamlit.web import cli as stcli
    raw_port = os.environ.get("PORT", "8501")
    try:
        port = str(int(raw_port))
    except (ValueError, TypeError):
        port = "8501"
    sys.argv = [
        "streamlit",
        "run",
        os.path.abspath(__file__),
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "CNC Predictive Maintenance | ML System",
    page_icon  = "⚙️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

from app.pages import (
    page_home,
    page_dataset,
    page_eda,
    page_train,
    page_predict,
    page_visual_predict,
    page_metrics,
    page_maintenance,
    page_docs,
    page_about,
    page_live,
)
from app.style import inject_css

# ─────────────────────────────────────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'dark'

# ─────────────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────────────
def login_page():
    inject_css()
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align:center;padding:2rem;background:var(--card-bg);border:1px solid var(--border);border-radius:15px;box-shadow:0 10px 40px rgba(0,0,0,0.2);">
            <div style="font-size:4rem;margin-bottom:1rem;">🔐</div>
            <h2 style="font-family:'Outfit', sans-serif;font-weight:700;color:var(--text);">System Login</h2>
            <p style="color:var(--muted);margin-bottom:2rem;">Please authenticate to access the Predictive Maintenance System.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="admin123")
            submitted = st.form_submit_button("🔑 Login", use_container_width=True)
            
            if submitted:
                if username == "admin" and password == "admin123":
                    st.session_state['logged_in'] = True
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try admin / admin123")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

PAGES = {
    "🏠  Home"                 : page_home,
    "📊  Live Dashboard"       : page_live,
    "🔮  Prediction"           : page_predict,
    "🖼️  Visual Prediction"    : page_visual_predict,
    "📈  Visualizations (EDA)" : page_eda,
    "📂  Dataset Analysis"     : page_dataset,
    "🤖  Model Training"       : page_train,
    "📉  Performance Metrics"  : page_metrics,
    "🔧  Maintenance"          : page_maintenance,
    "📚  Documentation"        : page_docs,
    "📌  About Project"        : page_about,
}

def sidebar() -> str:
    inject_css()

    with st.sidebar:
        # Logo / Title
        st.markdown("""
        <div style="text-align:center;padding:1rem 0;">
          <div style="font-size:3rem;">⚙️</div>
          <div style="font-family:'Outfit',sans-serif;font-weight:800;font-size:1.05rem;
                      background:linear-gradient(135deg,#6366F1,#EC4899);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            CNC Predictive<br>Maintenance
          </div>
        </div>
        <hr style="border-color:var(--border);margin:0 0 1rem;">
        """, unsafe_allow_html=True)

        selected = st.selectbox("Navigate to", list(PAGES.keys()), label_visibility="collapsed")

        # Theme Switch
        st.markdown("<br>", unsafe_allow_html=True)
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            if st.button("🌙 Dark", use_container_width=True):
                st.session_state['theme'] = 'dark'
                st.rerun()
        with t_col2:
            if st.button("☀️ Light", use_container_width=True):
                st.session_state['theme'] = 'light'
                st.rerun()

        # Status indicator
        model_ready = os.path.exists(os.path.join(ROOT, 'models', 'best_classifier.joblib'))
        st.markdown("<br>", unsafe_allow_html=True)
        if model_ready:
            st.markdown("""
            <div style="background:rgba(34,197,94,0.12);border:1px solid #22C55E;
                        border-radius:8px;padding:0.6rem 1rem;font-size:0.82rem;color:var(--text);">
              ✅ <b>Models Ready</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(245,158,11,0.12);border:1px solid #F59E0B;
                        border-radius:8px;padding:0.6rem 1rem;font-size:0.82rem;color:var(--text);">
              ⚠️ <b>Models not trained</b>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    return selected

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if not st.session_state['logged_in']:
        login_page()
    else:
        selected = sidebar()
        page_module = PAGES[selected]
        page_module.render()

if __name__ == '__main__':
    main()
