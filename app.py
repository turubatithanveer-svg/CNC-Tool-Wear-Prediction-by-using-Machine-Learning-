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
    import subprocess
    import socket
    import threading
    import time

    raw_port = os.environ.get("PORT", "8501")
    try:
        port_int = int(raw_port)
        port = str(port_int)
    except (ValueError, TypeError):
        port_int = 8501
        port = "8501"

    def _monitor(p: int):
        time.sleep(2.0)
        print(f"[PORT CHECK] Starting connection verification on port {p}...")
        sys.stdout.flush()
        for _ in range(30):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0)
                    if s.connect_ex(("127.0.0.1", p)) == 0:
                        print("============================================================")
                        print(f"[PORT STATUS] ✅ SUCCESS: Port {p} is OPEN and CONNECTED!")
                        print(f"[PORT STATUS] 🚀 Streamlit is LIVE at http://0.0.0.0:{p}")
                        print(f"[PORT STATUS] 🌐 Traffic is being accepted from Railway router.")
                        print("============================================================")
                        sys.stdout.flush()
                        return
            except Exception:
                pass
            time.sleep(1.0)
        print(f"[PORT STATUS] ⚠️ WARNING: Port {p} did not accept connection.")
        sys.stdout.flush()

    threading.Thread(target=_monitor, args=(port_int,), daemon=True).start()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        os.path.abspath(__file__),
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    print(f"[STARTUP] Re-launching through Streamlit runner on port {port} ...")
    sys.stdout.flush()
    sys.exit(subprocess.call(cmd))

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "CNC Predictive Maintenance | ML System",
    page_icon  = "⚙️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Port Connectivity Console Logger
# ─────────────────────────────────────────────────────────────────────────────
_current_port = os.environ.get("PORT", "8501")
print(f"[CONSOLE STATUS] 🚀 App loaded successfully | Target Port: {_current_port} | Host: 0.0.0.0", flush=True)

if 'port_verified' not in st.session_state:
    st.session_state['port_verified'] = True
    import socket
    try:
        p_val = int(_current_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_sock:
            probe_sock.settimeout(1.5)
            status = probe_sock.connect_ex(("127.0.0.1", p_val))
            if status == 0:
                print("============================================================", flush=True)
                print(f"[PORT CHECK] ✅ SUCCESS: Port {p_val} is CONNECTED & LISTENING!", flush=True)
                print(f"[PORT CHECK] 🌐 Ready to accept traffic from Railway router.", flush=True)
                print("============================================================", flush=True)
            else:
                print(f"[PORT CHECK] ℹ️ Probe status code on port {p_val}: {status}", flush=True)
    except Exception as err:
        print(f"[PORT CHECK] Probe exception: {err}", flush=True)

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

        # Port status badge
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);
                    border-radius:8px;padding:0.5rem 0.8rem;font-size:0.78rem;color:var(--text);margin-top:0.5rem;">
          🌐 <b>Port:</b> <code>{_current_port}</code> (Active)
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
