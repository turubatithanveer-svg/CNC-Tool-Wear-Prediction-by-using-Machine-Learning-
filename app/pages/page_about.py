"""
=============================================================================
page_about.py — About Developers Page
=============================================================================
Shows:
  • Project title and institution details
  • Guide / mentor information
  • Student team cards
  • Technology stack badges
  • Version info
=============================================================================
"""

import streamlit as st
from app.style import inject_css, hero_banner, section_header, footer


# ─────────────────────────────────────────────────────────────────────────────
# ✏️  EDIT THIS SECTION WITH YOUR REAL DETAILS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_INFO = {
    "title"     : "Predictive Maintenance of Tool Wear in CNC Machines Using Machine Learning",
    "college"   : "Narasaraopeta Engineering College",
    "dept"      : "Mechanical Engineering",
    "guide"     : "Mr. K. John Babu",
    "year"      : "2026–2027",
    "version"   : "v1.0.0",
}

STUDENTS = [
    {"name": "S. Sai Kiran",       "roll": "Team Member", "emoji": "👨‍💻"},
    {"name": "T. Thanveer",        "roll": "Team Member", "emoji": "👨‍💻"},
    {"name": "D. Janardhan Naidu", "roll": "Team Member", "emoji": "👨‍💻"},
]

TECH_STACK = [
    ("🐍", "Python 3.10+",       "Primary programming language"),
    ("📊", "Streamlit",           "Interactive web dashboard"),
    ("🤖", "Scikit-learn",        "Machine learning framework"),
    ("⚡", "XGBoost",             "Gradient boosting library"),
    ("🐼", "Pandas / NumPy",      "Data manipulation"),
    ("📈", "Plotly",              "Interactive visualisations"),
    ("💾", "Joblib",              "Model serialisation"),
    ("🎨", "HTML / CSS",          "Custom UI styling"),
]


# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    inject_css()
    hero_banner(
        title    = "About the Project",
        subtitle = "Final Year Engineering Project — Development Team & Technology Stack",
        emoji    = "👥",
    )

    # ── Project Info ─────────────────────────────────────────────────────────
    section_header("🏫 Project Information")
    st.markdown(f"""
    <div class="info-card">
      <table style="width:100%;border-collapse:collapse;font-size:0.95rem;">
        <tr><td style="padding:0.5rem 0;color:#94A3B8;width:35%;">Project Title</td>
            <td style="color:#E2E8F0;font-weight:600;">{PROJECT_INFO['title']}</td></tr>
        <tr><td style="padding:0.5rem 0;color:#94A3B8;">Institution</td>
            <td style="color:#E2E8F0;">{PROJECT_INFO['college']}</td></tr>
        <tr><td style="padding:0.5rem 0;color:#94A3B8;">Department</td>
            <td style="color:#E2E8F0;">{PROJECT_INFO['dept']}</td></tr>
        <tr><td style="padding:0.5rem 0;color:#94A3B8;">Project Guide</td>
            <td style="color:#E2E8F0;font-weight:600;">{PROJECT_INFO['guide']}</td></tr>
        <tr><td style="padding:0.5rem 0;color:#94A3B8;">Academic Year</td>
            <td style="color:#E2E8F0;">{PROJECT_INFO['year']}</td></tr>
        <tr><td style="padding:0.5rem 0;color:#94A3B8;">Version</td>
            <td><span class="badge badge-primary">{PROJECT_INFO['version']}</span></td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Student Cards ─────────────────────────────────────────────────────────
    section_header("👨‍🎓 Development Team")
    cols = st.columns(len(STUDENTS))
    for i, student in enumerate(STUDENTS):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
              <div style="font-size:3rem;margin-bottom:0.5rem;">{student['emoji']}</div>
              <div style="font-weight:700;font-size:1rem;color:#C7D2FE;">{student['name']}</div>
              <div style="font-size:0.82rem;color:#94A3B8;margin-top:0.3rem;">{student['roll']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────────────────────
    section_header("🛠️ Technology Stack")
    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(TECH_STACK):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;margin-bottom:1rem;">
              <div style="font-size:2rem;">{icon}</div>
              <div style="font-weight:700;font-size:0.9rem;color:#C7D2FE;margin-top:0.3rem;">{name}</div>
              <div style="font-size:0.78rem;color:#94A3B8;margin-top:0.2rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Acknowledgement ───────────────────────────────────────────────────────
    section_header("🙏 Acknowledgements")
    st.markdown("""
    <div class="info-card">
      We sincerely thank our project guide for the invaluable mentorship and guidance throughout
      this project. We also acknowledge the <b>UCI Machine Learning Repository</b> for providing
      the AI4I 2020 Predictive Maintenance Dataset used in this work. This project was developed
      as part of the Final Year Engineering curriculum.
    </div>
    """, unsafe_allow_html=True)

    footer()
