"""
=============================================================================
page_docs.py — Project Documentation Page
=============================================================================
Shows:
  • Project objective
  • Dataset description
  • ML workflow
  • Algorithms used
  • Advantages & applications
  • Future scope
  • Conclusion
=============================================================================
"""

import streamlit as st
from app.style import inject_css, hero_banner, section_header, footer


def render() -> None:
    inject_css()
    hero_banner(
        title    = "Project Documentation",
        subtitle = "Technical documentation — ML Workflow · Algorithms · Dataset · Applications",
        emoji    = "📚",
    )

    # Tabs
    tabs = st.tabs([
        "🎯 Objective",
        "📂 Dataset",
        "⚙️ ML Workflow",
        "🤖 Algorithms",
        "✅ Advantages",
        "🌐 Applications",
        "🔮 Future Scope",
        "🏁 Conclusion",
    ])

    with tabs[0]:
        section_header("Project Objective")
        st.markdown("""
        <div class="info-card">
          <h4 style="color:#C7D2FE;">Goal</h4>
          <p>Develop an <b>intelligent predictive maintenance system</b> that predicts tool wear
          and failure probability of CNC machine cutting tools using Machine Learning, enabling
          industries to:</p>
          <ul style="line-height:2.0;">
            <li>Avoid unexpected tool failures and machine downtime</li>
            <li>Reduce maintenance costs by switching from reactive to predictive maintenance</li>
            <li>Improve machining quality and dimensional accuracy</li>
            <li>Extend tool life through data-driven usage optimisation</li>
            <li>Provide non-expert operators with actionable, colour-coded alerts</li>
          </ul>
          <h4 style="color:#C7D2FE;margin-top:1rem;">Problem Statement</h4>
          <p>Traditional CNC maintenance relies on fixed-interval schedules or reactive repairs after
          failure occurs. Both approaches are sub-optimal: scheduled maintenance wastes good tools,
          while reactive repair causes costly unplanned downtime. This project bridges the gap with
          ML-powered real-time prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    with tabs[1]:
        section_header("Dataset Description")
        st.markdown("""
        <div class="info-card">
          <h4 style="color:#C7D2FE;">AI4I 2020 Predictive Maintenance Dataset</h4>
          <ul style="line-height:2.0;">
            <li><b>Rows:</b> 10,000 samples</li>
            <li><b>Features:</b> 12 columns (11 input + 1 target)</li>
            <li><b>Source:</b> UCI Machine Learning Repository (AI4I 2020)</li>
            <li><b>Task:</b> Binary Classification (failure / no failure) + Tool Wear Regression</li>
          </ul>
          <h4 style="color:#C7D2FE;margin-top:1rem;">Feature Descriptions</h4>
        </div>
        """, unsafe_allow_html=True)

        import pandas as pd
        features = [
            ("Type",                 "Categorical", "Machine quality type: L (Low), M (Medium), H (High)"),
            ("Air temperature",      "Float (K)",   "Ambient air temperature around the machine"),
            ("Process temperature",  "Float (K)",   "Temperature during the machining process"),
            ("Rotational speed",     "Int (RPM)",   "Spindle rotational speed"),
            ("Torque",               "Float (Nm)",  "Torque applied during cutting"),
            ("Tool wear",            "Int (min)",   "Accumulated tool wear time in minutes"),
            ("Machine failure",      "Binary",      "Target: 1=Failure, 0=Normal (3.4% failure rate)"),
            ("TWF",                  "Binary",      "Tool Wear Failure sub-category"),
            ("HDF",                  "Binary",      "Heat Dissipation Failure sub-category"),
            ("PWF",                  "Binary",      "Power Failure sub-category"),
            ("OSF",                  "Binary",      "Overstrain Failure sub-category"),
            ("RNF",                  "Binary",      "Random Noise Failure sub-category"),
        ]
        df = pd.DataFrame(features, columns=['Feature', 'Type', 'Description'])
        st.dataframe(df, use_container_width=True)

    with tabs[2]:
        section_header("Machine Learning Workflow")
        steps_html = """
        <div style="position:relative;padding-left:2rem;">
        """
        steps = [
            ("Data Collection",     "Raw CSV dataset with 10,000 CNC sensor records"),
            ("Data Inspection",     "Shape, dtypes, null check, duplicate detection"),
            ("Data Cleaning",       "Remove duplicates, forward-fill nulls, drop identifiers"),
            ("Feature Engineering", "Derive Power, Temp diff, Torque/Speed ratio, Wear rate"),
            ("Label Encoding",      "Encode Type (L/M/H) → (0/1/2) using LabelEncoder"),
            ("Scaling",             "StandardScaler applied to all numeric features"),
            ("Train-Test Split",    "80/20 stratified split, random_state=42"),
            ("Model Training",      "Train 7 classifiers + 1 regressor with cross-validation"),
            ("Evaluation",          "F1 Score, Accuracy, AUC, Confusion Matrix, CV Score"),
            ("Model Selection",     "Best model by F1 score auto-selected and saved"),
            ("Deployment",          "Streamlit app with real-time inference and PDF export"),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            steps_html += f"""
            <div style="display:flex;gap:1rem;margin-bottom:1rem;align-items:flex-start;">
              <div style="min-width:32px;height:32px;border-radius:50%;
                          background:linear-gradient(135deg,#6366F1,#EC4899);
                          display:flex;align-items:center;justify-content:center;
                          font-weight:800;font-size:0.85rem;color:white;flex-shrink:0;">{i}</div>
              <div class="info-card" style="flex:1;margin:0;padding:0.75rem 1rem;">
                <b style="color:#C7D2FE;">{title}</b>
                <span style="color:#94A3B8;margin-left:0.5rem;font-size:0.9rem;">{desc}</span>
              </div>
            </div>"""
        steps_html += "</div>"
        st.markdown(steps_html, unsafe_allow_html=True)

    with tabs[3]:
        section_header("ML Algorithms Used")
        algos = [
            ("Decision Tree",        "Interpretable tree-based classifier; baseline model"),
            ("Random Forest",        "Ensemble of 200 decision trees; robust to noise"),
            ("Gradient Boosting",    "Sequential ensemble; minimises classification error"),
            ("XGBoost",              "Optimised gradient boosting with regularisation"),
            ("Extra Trees",          "Extremely randomised forest; fast and accurate"),
            ("Support Vector Machine","Maximum-margin classifier; effective in high dimensions"),
            ("Logistic Regression",  "Linear probabilistic classifier; strong baseline"),
            ("RF Regressor",         "Random Forest for continuous tool wear estimation"),
        ]
        import pandas as pd
        df = pd.DataFrame(algos, columns=['Algorithm', 'Description'])
        df.index = df.index + 1
        st.dataframe(df, use_container_width=True, height=320)

    with tabs[4]:
        section_header("Advantages of this System")
        advs = [
            "Real-time prediction — instant results within milliseconds",
            "Multi-model comparison — automatically picks the best performer",
            "Explainable AI — feature importance shows which sensors matter most",
            "Beginner-friendly UI — no ML expertise required to use the system",
            "Batch processing — predict for thousands of records via CSV upload",
            "Downloadable reports — export predictions as CSV for documentation",
            "Physics-based features — domain knowledge improves model accuracy",
            "Scalable — easily extended to new machine types or sensor streams",
        ]
        for a in advs:
            st.markdown(f"✅ {a}")

    with tabs[5]:
        section_header("Industrial Applications")
        apps = [
            ("🏭", "CNC Machining",          "Predict tool failure in milling, turning, drilling operations"),
            ("✈️", "Aerospace Manufacturing","Monitor precision tool wear for aircraft component machining"),
            ("🚗", "Automotive",             "High-volume production lines — reduce scrap from worn tools"),
            ("🏗️", "Heavy Industry",         "Monitor cutting tools in large structural component machining"),
            ("💊", "Medical Devices",        "Ultra-precision tool monitoring for implant manufacturing"),
            ("⚡", "Electronics",            "PCB drilling and precision micro-machining quality control"),
        ]
        cols = st.columns(2)
        for i, (icon, title, desc) in enumerate(apps):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="metric-card" style="text-align:left;margin-bottom:1rem;">
                  <div style="font-size:2rem;">{icon}</div>
                  <div style="font-weight:700;color:#C7D2FE;margin:0.3rem 0;">{title}</div>
                  <div style="font-size:0.88rem;color:#94A3B8;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with tabs[6]:
        section_header("Future Scope")
        futures = [
            "🔗 IoT Integration — Connect directly to CNC machine PLCs for live sensor streaming",
            "📱 Mobile App — React Native companion app for shop-floor alerts",
            "🧠 Deep Learning — LSTM / Transformer models for time-series wear prediction",
            "☁️ Cloud Deployment — AWS/Azure hosted system for multi-plant monitoring",
            "🗃️ Database Backend — PostgreSQL for prediction history and audit trails",
            "🔔 Push Notifications — Email/SMS alerts when failure probability crosses thresholds",
            "🔧 Digital Twin — Simulate tool wear under different cutting conditions",
            "📊 Advanced Analytics — Pareto analysis of failure modes for continuous improvement",
        ]
        for f in futures:
            st.markdown(f"→ {f}")

    with tabs[7]:
        section_header("Conclusion")
        st.markdown("""
        <div class="info-card">
          <p style="line-height:1.9;">
            This project successfully demonstrates the application of Machine Learning to the
            industrial problem of CNC tool wear prediction. By training and comparing <b>7 ML
            classifiers</b> on the AI4I 2020 Predictive Maintenance Dataset, the system achieves
            high accuracy (&gt;96%) and strong F1-score (&gt;95%) in predicting machine failures.
          </p>
          <p style="line-height:1.9;">
            The addition of <b>physics-based engineered features</b> (mechanical power, temperature
            differential, torque-to-speed ratio) significantly improves model interpretability and
            prediction quality. The Streamlit dashboard makes this technology accessible to
            non-expert operators through a clean, colour-coded interface.
          </p>
          <p style="line-height:1.9;">
            This work validates that <b>predictive maintenance driven by machine learning</b>
            can replace both reactive and fixed-interval maintenance strategies, delivering
            measurable cost savings and quality improvements to CNC manufacturing environments.
          </p>
        </div>
        """, unsafe_allow_html=True)

    footer()
