"""
=============================================================================
page_home.py — Home / Landing Page
=============================================================================
Shows hero banner, project overview, benefits, and how-it-works section.
=============================================================================
"""

import streamlit as st
from app.style import inject_css, hero_banner, section_header, footer, metric_card


def render() -> None:
    inject_css()

    # ── Hero ─────────────────────────────────────────────────────────────────
    hero_banner(
        title    = "CNC Tool Wear Prediction System",
        subtitle = ("An AI-powered Predictive Maintenance platform for CNC machines. "
                    "Detect tool failures before they happen — save cost, time, and quality."),
        emoji    = "⚙️",
    )

    # ── Quick stats ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("10,000+", "Training Samples", "AI4I Dataset"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("7", "ML Algorithms", "Compared & Benchmarked"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("~97%", "Model Accuracy", "Best Classifier"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Real-Time", "Predictions", "Instant Inference"), unsafe_allow_html=True)

    # ── About ──────────────────────────────────────────────────────────────
    section_header("📌 About the Project")
    st.markdown("""
    <div class="info-card">
      <p style="margin:0;line-height:1.8;">
        This system leverages cutting-edge <strong>Machine Learning</strong> to predict CNC machine
        tool wear and failure probability in real-time. The platform analyses sensor data including
        temperature, rotational speed, torque, and accumulated tool wear to generate instant
        maintenance recommendations — helping factories avoid costly unplanned downtime.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Benefits ────────────────────────────────────────────────────────────
    section_header("🚀 Key Benefits")
    b1, b2, b3 = st.columns(3)
    benefits = [
        ("💰", "Cost Reduction", "Prevent unplanned shutdowns that cost 5–10× more than scheduled maintenance."),
        ("⏱️", "Increased Uptime", "Maximise machine utilisation by replacing tools exactly when needed."),
        ("🎯", "Quality Assurance", "Consistent machining quality by catching tool degradation early."),
        ("🌱", "Sustainability", "Reduce tool waste by extending life safely with data-driven insights."),
        ("📊", "Data-Driven", "Move from experience-based guessing to scientific, ML-driven decisions."),
        ("🔔", "Smart Alerts", "Automated recommendations with colour-coded severity indicators."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(benefits):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;margin-bottom:1rem;">
              <div style="font-size:2rem;margin-bottom:0.5rem;">{icon}</div>
              <div style="font-weight:700;font-size:1rem;margin-bottom:0.3rem;color:#C7D2FE;">{title}</div>
              <div style="font-size:0.88rem;color:#94A3B8;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── How it works ────────────────────────────────────────────────────────
    section_header("🔄 How It Works")
    steps = [
        ("1", "Collect Data",       "Sensor readings — temperature, speed, torque, wear time — are collected from CNC machine."),
        ("2", "Preprocess",         "Data is cleaned, engineered, and scaled using industry-standard pipelines."),
        ("3", "ML Inference",       "Trained ensemble models classify failure risk and estimate remaining tool life."),
        ("4", "Recommendation",     "System generates colour-coded alerts and maintenance action suggestions instantly."),
        ("5", "Report & Export",    "Download PDF reports, export batch predictions, and maintain history logs."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;margin-bottom:1rem;gap:1rem;">
          <div style="min-width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#6366F1,#EC4899);
                      display:flex;align-items:center;justify-content:center;font-weight:800;font-size:0.9rem;color:white;">
            {num}
          </div>
          <div>
            <div style="font-weight:600;color:#C7D2FE;margin-bottom:0.2rem;">{title}</div>
            <div style="font-size:0.9rem;color:#94A3B8;line-height:1.5;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── CTA ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_c, _ = st.columns([1, 3])
    with col_c:
        st.info("👈  Use the sidebar menu to navigate to Prediction, EDA, Model Training, and more.")

    footer()
