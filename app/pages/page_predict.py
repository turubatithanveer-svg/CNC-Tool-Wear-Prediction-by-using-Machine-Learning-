"""
=============================================================================
page_predict.py — Prediction Page
=============================================================================
User enters CNC sensor values → system returns:
  • Failure Prediction (binary)
  • Failure Probability %
  • Predicted Tool Wear (regression)
  • Health Status (Green / Yellow / Red)
  • Remaining Tool Life
  • Maintenance Recommendation
  • Confidence Score
  • Downloadable Report (CSV)
=============================================================================
"""

import os
import io
import json
import datetime
import streamlit as st
import pandas as pd
from fpdf import FPDF
from app.style import inject_css, hero_banner, section_header, footer


MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


def _models_ready() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, 'best_classifier.joblib'))


def _colour_box(result: dict) -> str:
    colour = result.get('color', 'gray')
    css_map = {
        'green'  : 'status-green',
        'orange' : 'status-orange',
        'red'    : 'status-red',
        'darkred': 'status-dark',
    }
    cls = css_map.get(colour, 'info-card')
    return cls


def _render_result_card(result: dict) -> None:
    css_cls = _colour_box(result)
    prob    = result['failure_probability']
    status  = result['status']
    action  = result['action']
    alert   = result['alert']
    conf    = result['confidence']
    rem     = result['remaining_life']
    pct     = result['pct_remaining']
    wear    = result['predicted_wear_min']

    st.markdown(f"""
    <div class="{css_cls}" style="margin-bottom:1.5rem;">
      <div style="font-size:2rem;font-weight:900;margin-bottom:0.4rem;">{status}</div>
      <div style="font-size:1.1rem;font-weight:600;margin-bottom:1rem;color:#E2E8F0;">{alert}</div>
      <hr style="border-color:rgba(255,255,255,0.15);margin:0.75rem 0;">
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;text-align:center;">
        <div>
          <div style="font-size:1.8rem;font-weight:800;color:#6366F1;">{prob:.1f}%</div>
          <div style="font-size:0.8rem;color:#94A3B8;text-transform:uppercase;">Failure Probability</div>
        </div>
        <div>
          <div style="font-size:1.8rem;font-weight:800;color:#14B8A6;">{rem} min</div>
          <div style="font-size:0.8rem;color:#94A3B8;text-transform:uppercase;">Remaining Life</div>
        </div>
        <div>
          <div style="font-size:1.8rem;font-weight:800;color:#EC4899;">{conf:.1f}%</div>
          <div style="font-size:0.8rem;color:#94A3B8;text-transform:uppercase;">Confidence</div>
        </div>
      </div>
      <hr style="border-color:rgba(255,255,255,0.15);margin:0.75rem 0;">
      <div style="font-size:0.95rem;">
        🔧 <b>Recommended Action:</b> {action}<br>
        🛠️ <b>Predicted Wear:</b> {wear} min &nbsp;|&nbsp;
        🔋 <b>Life Remaining:</b> {pct:.1f}%
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar for remaining life
    st.progress(int(pct), text=f"Tool Life Remaining: {pct:.1f}%")


def render() -> None:
    inject_css()
    hero_banner(
        title    = "Tool Wear Prediction",
        subtitle = "Enter CNC sensor readings to get instant failure prediction and maintenance advice.",
        emoji    = "🔮",
    )

    if not _models_ready():
        st.error("⚠️ Models not trained yet. Please go to **Model Training** page and train the models first.")
        footer()
        return

    # ── Input Form ────────────────────────────────────────────────────────────
    section_header("🎛️ Enter Sensor Parameters")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            type_val  = st.selectbox("Machine Type", ['L', 'M', 'H'],
                                      index=1, help="L=Low, M=Medium, H=High grade")
            air_temp  = st.number_input("Air Temperature (K)",
                                         value=300.0, step=0.1,
                                         help="Ambient air temperature in Kelvin")
            process_temp = st.number_input("Process Temperature (K)",
                                            value=310.0, step=0.1,
                                            help="Machining process temperature in Kelvin")

        with c2:
            rot_speed = st.number_input("Rotational Speed (RPM)",
                                         value=1500, step=50,
                                         help="Spindle rotational speed")
            torque    = st.number_input("Torque (Nm)",
                                         value=40.0, step=0.5,
                                         help="Cutting torque applied to the tool")
            tool_wear = st.number_input("Tool Wear (min)",
                                         value=50, step=1,
                                         help="Accumulated tool wear time in minutes")

        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card" style="font-size:0.85rem;line-height:1.7;">
              <b>Parameter Ranges:</b><br>
              🌡️ Air Temp: 295–305 K<br>
              🔥 Process Temp: 305–315 K<br>
              ⚙️ Speed: 1168–2886 RPM<br>
              🔩 Torque: 3.8–76.6 Nm<br>
              🕰️ Wear: 0–253 min
            </div>
            """, unsafe_allow_html=True)

        submitted = st.form_submit_button("🔮 Predict Failure", use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if submitted:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from src.predict import predict_single

        input_data = {
            'type_val'    : type_val,
            'air_temp'    : air_temp,
            'process_temp': process_temp,
            'rot_speed'   : rot_speed,
            'torque'      : torque,
            'tool_wear'   : tool_wear,
        }

        with st.spinner("🤖 Running ML inference…"):
            try:
                result = predict_single(input_data, model_dir=MODEL_DIR)
            except RuntimeError as e:
                st.error(str(e))
                footer()
                return

        section_header("📊 Prediction Result")
        _render_result_card(result)

        # Save to session history
        if 'prediction_history' not in st.session_state:
            st.session_state['prediction_history'] = []

        entry = {
            'Timestamp'          : datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Machine Type'       : type_val,
            'Air Temp (K)'       : air_temp,
            'Process Temp (K)'   : process_temp,
            'Speed (RPM)'        : rot_speed,
            'Torque (Nm)'        : torque,
            'Tool Wear (min)'    : tool_wear,
            'Failure Probability': result['failure_probability'],
            'Confidence (%)'     : result['confidence'],
            'Predicted Wear'     : result['predicted_wear_min'],
            'Remaining Life'     : result['remaining_life'],
            'Status'             : result['status'],
            'Action'             : result['action'],
        }
        st.session_state['prediction_history'].append(entry)

        # Download CSV and PDF reports
        report_df = pd.DataFrame([entry])
        csv_data = report_df.to_csv(index=False).encode('utf-8')
        
        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=15, style='B')
        pdf.cell(200, 10, txt="CNC Predictive Maintenance Report", ln=1, align='C')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Date: {entry['Timestamp']}", ln=1, align='L')
        pdf.ln(10)
        
        for k, v in entry.items():
            if k != 'Timestamp':
                clean_v = str(v).encode('latin-1', 'replace').decode('latin-1')
                pdf.cell(200, 10, txt=f"{k}: {clean_v}", ln=1, align='L')
        
        pdf_bytes = bytes(pdf.output())

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label     = "📥 Download CSV Report",
                data      = csv_data,
                file_name = f"prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime      = 'text/csv',
                use_container_width=True
            )
        with col2:
            st.download_button(
                label     = "📥 Download PDF Report",
                data      = pdf_bytes,
                file_name = f"prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime      = 'application/pdf',
                use_container_width=True
            )

    # ── Prediction History ────────────────────────────────────────────────────
    if st.session_state.get('prediction_history'):
        section_header("📜 Prediction History (this session)")
        hist_df = pd.DataFrame(st.session_state['prediction_history'])
        st.dataframe(hist_df, use_container_width=True, height=260)

        # Add graph for history
        if len(hist_df) > 1:
            st.markdown("### Failure Probability Trend")
            st.line_chart(hist_df.set_index('Timestamp')['Failure Probability'])

        csv_hist = hist_df.to_csv(index=False).encode('utf-8')
        
        # Generate Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            hist_df.to_excel(writer, index=False, sheet_name='History')
        excel_data = output.getvalue()

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label     = "📥 Export History (CSV)",
                data      = csv_hist,
                file_name = "prediction_history.csv",
                mime      = 'text/csv',
                use_container_width=True
            )
        with col2:
            st.download_button(
                label     = "📥 Export History (Excel)",
                data      = excel_data,
                file_name = "prediction_history.xlsx",
                mime      = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )

    # ── Batch Prediction ──────────────────────────────────────────────────────
    section_header("📂 Batch Prediction — Upload CSV")
    uploaded = st.file_uploader(
        "Upload a CSV file with CNC sensor columns",
        type=['csv'],
        key='batch_upload',
    )
    if uploaded:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from src.predict import predict_batch

        batch_df = pd.read_csv(uploaded)
        st.write(f"Uploaded: **{len(batch_df)} rows**")

        with st.spinner("Running batch predictions…"):
            try:
                result_df = predict_batch(batch_df, model_dir=MODEL_DIR)
                st.success(f"✅ Batch prediction complete for {len(result_df)} rows.")
                st.dataframe(result_df, use_container_width=True, height=320)

                # Add graphs for batch prediction
                st.markdown("### Batch Prediction Analytics")
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Failure Probability Distribution**")
                    if 'failure_probability' in result_df.columns:
                        import plotly.express as px
                        fig = px.histogram(result_df, x='failure_probability', nbins=20, 
                                           title="Probability Distribution", color_discrete_sequence=['#6366F1'])
                        st.plotly_chart(fig, use_container_width=True)
                with colB:
                    st.markdown("**Status Breakdown**")
                    if 'status' in result_df.columns:
                        import plotly.express as px
                        status_counts = result_df['status'].value_counts().reset_index()
                        status_counts.columns = ['Status', 'Count']
                        fig2 = px.pie(status_counts, names='Status', values='Count', 
                                      title="Predicted Status Breakdown", hole=0.4)
                        st.plotly_chart(fig2, use_container_width=True)

                st.download_button(
                    label     = "📥 Download Batch Results (CSV)",
                    data      = result_df.to_csv(index=False).encode('utf-8'),
                    file_name = "batch_predictions.csv",
                    mime      = 'text/csv',
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

    footer()
