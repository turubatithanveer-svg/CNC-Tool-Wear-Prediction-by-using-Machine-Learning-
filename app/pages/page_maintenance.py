"""
=============================================================================
page_maintenance.py — Maintenance Recommendation Page
=============================================================================
Shows intelligent maintenance recommendations based on tool wear thresholds
and failure probability categories.
Includes:
  • Severity level guide
  • Recommendation cards
  • Industry best practices
  • Downloadable maintenance schedule
=============================================================================
"""

import streamlit as st
from app.style import inject_css, hero_banner, section_header, footer


RECOMMENDATIONS = [
    {
        "level"   : 1,
        "icon"    : "✅",
        "title"   : "Healthy Tool — Continue Production",
        "css"     : "status-green",
        "triggers": "Failure Probability < 25% | Tool Life > 50%",
        "actions" : [
            "Continue normal production without interruption",
            "Perform routine visual inspection weekly",
            "Log current sensor readings for trend analysis",
            "Check coolant levels and tool mounting torque",
        ],
        "interval": "Weekly inspection cycle",
    },
    {
        "level"   : 2,
        "icon"    : "⚠️",
        "title"   : "Moderate Wear — Schedule Maintenance",
        "css"     : "status-orange",
        "triggers": "Failure Probability 25–50% | Tool Life 25–50%",
        "actions" : [
            "Plan tool replacement within next 2 production shifts",
            "Increase monitoring frequency to every 4 hours",
            "Reduce feed rate by 10–15% to extend tool life",
            "Inspect tool edge under microscope for chipping",
            "Pre-order replacement tool to minimise downtime",
        ],
        "interval": "Daily monitoring recommended",
    },
    {
        "level"   : 3,
        "icon"    : "🔴",
        "title"   : "High Wear — Replace Tool Soon",
        "css"     : "status-red",
        "triggers": "Failure Probability 50–75% | Tool Life < 25%",
        "actions" : [
            "Replace cutting tool at the next scheduled break",
            "Do not start new batch runs with current tool",
            "Reduce cutting depth by 20% immediately",
            "Check for vibration anomalies and machine alignment",
            "Inspect spindle bearings and coolant nozzle",
        ],
        "interval": "Replace within 1 shift",
    },
    {
        "level"   : 4,
        "icon"    : "🚨",
        "title"   : "Critical — Immediate Maintenance Required",
        "css"     : "status-dark",
        "triggers": "Failure Probability > 75% | Tool Life < 10%",
        "actions" : [
            "STOP MACHINE IMMEDIATELY",
            "Replace tool before resuming any operation",
            "Inspect workpiece for dimensional errors caused by worn tool",
            "Conduct full spindle and fixture inspection",
            "Document failure event in maintenance log",
            "Root-cause analysis required before restarting",
        ],
        "interval": "Immediate — Do not operate",
    },
]

BEST_PRACTICES = [
    ("🌡️", "Temperature Control",    "Maintain process temperature within ±2K of target. Thermal drift accelerates tool wear."),
    ("💧", "Coolant Management",     "Check coolant concentration daily (8–10%). Degraded coolant increases thermal load by up to 40%."),
    ("⚙️", "Spindle Calibration",    "Calibrate spindle runout monthly. Runout > 5μm significantly reduces tool life."),
    ("📊", "Data Logging",           "Record all sensor readings per shift for trend-based predictive scheduling."),
    ("🔩", "Torque Monitoring",      "Alert when torque exceeds 20% above baseline — indicator of edge chipping."),
    ("🗓️", "Preventive Schedule",    "Replace tools at 80% of rated life, not at failure — reduces scrap and rework costs."),
]


def render() -> None:
    inject_css()
    hero_banner(
        title    = "Maintenance Recommendations",
        subtitle = "Intelligent maintenance guidance based on ML predictions and industry best practices.",
        emoji    = "🔧",
    )

    section_header("🚦 Severity Levels & Actions")
    for rec in RECOMMENDATIONS:
        with st.expander(f"{rec['icon']}  Level {rec['level']}: {rec['title']}", expanded=(rec['level'] == 1)):
            st.markdown(f"""
            <div class="{rec['css']}">
              <div style="margin-bottom:0.6rem;">
                <span class="badge badge-primary">Triggers</span>
                <span style="margin-left:0.5rem;font-size:0.9rem;color:#E2E8F0;">{rec['triggers']}</span>
              </div>
              <div style="margin-bottom:0.6rem;">
                <span class="badge badge-warning">Inspection</span>
                <span style="margin-left:0.5rem;font-size:0.9rem;color:#E2E8F0;">{rec['interval']}</span>
              </div>
              <hr style="border-color:rgba(255,255,255,0.15);">
              <b>Recommended Actions:</b>
              <ul style="margin-top:0.5rem;line-height:2;">
                {''.join(f"<li>{a}</li>" for a in rec['actions'])}
              </ul>
            </div>
            """, unsafe_allow_html=True)

    section_header("📋 Industry Best Practices")
    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(BEST_PRACTICES):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="info-card" style="margin-bottom:1rem;">
              <div style="font-size:1.5rem;margin-bottom:0.3rem;">{icon}</div>
              <div style="font-weight:700;color:#C7D2FE;margin-bottom:0.2rem;">{title}</div>
              <div style="font-size:0.88rem;color:#94A3B8;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Maintenance Schedule Download ────────────────────────────────────────
    section_header("📥 Download Maintenance Schedule Template")
    import pandas as pd, datetime
    today = datetime.date.today()
    schedule = pd.DataFrame({
        "Week"         : [f"Week {i+1}" for i in range(8)],
        "Date"         : [(today + datetime.timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(8)],
        "Task"         : [
            "Visual inspection + coolant check",
            "Tool wear measurement + spindle check",
            "Replace if wear > 120 min",
            "Full machine inspection",
            "Visual inspection + coolant check",
            "Tool wear measurement",
            "Replace if wear > 120 min",
            "Monthly performance report",
        ],
        "Responsible"  : ["Operator"] * 8,
        "Status"       : ["Pending"] * 8,
    })
    st.dataframe(schedule, use_container_width=True)
    st.download_button(
        label     = "📥 Download Schedule (CSV)",
        data      = schedule.to_csv(index=False).encode('utf-8'),
        file_name = "maintenance_schedule.csv",
        mime      = 'text/csv',
    )

    footer()
