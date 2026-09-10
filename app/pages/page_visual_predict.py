"""
=============================================================================
page_visual_predict.py — Visual Prediction Page
=============================================================================
Allows users to upload a CNC tool / machine image or video clip and receive:
  • Extracted visual wear metrics (edge sharpness, texture, discoloration …)
  • Estimated sensor parameters derived from visual analysis
  • Full ML failure prediction (same result card as manual prediction)
  • For video: per-frame predictions + failure probability trend chart
=============================================================================
"""

import io
import os
import datetime
import numpy as np
import streamlit as st
import pandas as pd
from fpdf import FPDF
from app.style import inject_css, hero_banner, section_header, footer

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _models_ready() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, 'best_classifier.joblib'))


def _colour_box_cls(result: dict) -> str:
    css_map = {
        'green'  : 'status-green',
        'orange' : 'status-orange',
        'red'    : 'status-red',
        'darkred': 'status-dark',
    }
    return css_map.get(result.get('color', 'gray'), 'info-card')


def _render_result_card(result: dict) -> None:
    css_cls = _colour_box_cls(result)
    prob   = result['failure_probability']
    status = result['status']
    action = result['action']
    alert  = result['alert']
    conf   = result['confidence']
    rem    = result['remaining_life']
    pct    = result['pct_remaining']
    wear   = result['predicted_wear_min']

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
    st.progress(int(pct), text=f"Tool Life Remaining: {pct:.1f}%")


def _render_visual_metrics(vf: dict) -> None:
    """Render a styled card showing extracted visual metrics."""
    def _bar(val: float, color: str = '#6366F1') -> str:
        pct = int(val * 100)
        return (
            f'<div style="background:rgba(255,255,255,0.08);border-radius:6px;height:8px;margin-top:4px;">'
            f'<div style="width:{pct}%;background:{color};height:8px;border-radius:6px;'
            f'transition:width 0.5s ease;"></div></div>'
        )

    metrics = [
        ('🔪 Edge Sharpness',    vf.get('edge_sharpness', 0),    '#14B8A6',
         'High = sharp tool, Low = worn/blunt'),
        ('🏔️ Surface Roughness', vf.get('texture_roughness', 0),  '#F59E0B',
         'High = rough surface (high torque zone)'),
        ('🌡️ Discoloration',     vf.get('discoloration', 0),      '#EC4899',
         'High = heat tints / oxidation detected'),
        ('⚫ Wear Area %',       vf.get('wear_area_pct', 0),      '#EF4444',
         'Dark cavities / deep wear regions'),
        ('💡 Brightness',        vf.get('brightness', 0),         '#6366F1',
         'Overall surface brightness level'),
        ('🎨 Saturation',        vf.get('hsv_saturation', 0),     '#A855F7',
         'Colour saturation — rust / oxidation indicator'),
    ]

    cols = st.columns(3)
    for i, (label, val, color, tip) in enumerate(metrics):
        with cols[i % 3]:
            st.markdown(
                f'<div class="info-card" style="padding:0.9rem 1rem;">'
                f'<div style="font-size:0.82rem;font-weight:600;margin-bottom:2px;">{label}</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{color};">{val*100:.1f}%</div>'
                f'{_bar(val, color)}'
                f'<div style="font-size:0.72rem;color:#94A3B8;margin-top:4px;">{tip}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


def _render_sensor_estimates(sp: dict) -> None:
    """Show estimated sensor values derived from visual analysis."""
    rows = [
        ('Machine Type',         sp.get('type_val', 'M'),   ''),
        ('Air Temperature',      sp.get('air_temp', 0),     'K'),
        ('Process Temperature',  sp.get('process_temp', 0), 'K'),
        ('Rotational Speed',     sp.get('rot_speed', 0),    'RPM'),
        ('Torque',               sp.get('torque', 0),       'Nm'),
        ('Tool Wear',            sp.get('tool_wear', 0),    'min'),
    ]
    st.markdown("""
    <div class="info-card" style="padding:1rem 1.4rem;">
      <div style="font-weight:700;margin-bottom:0.7rem;font-size:0.95rem;">
        🎛️ Estimated Sensor Parameters (from visual analysis)
      </div>
    """, unsafe_allow_html=True)
    html = '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
    for name, val, unit in rows:
        html += (
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.07);">'
            f'<td style="padding:0.35rem 0;color:#94A3B8;">{name}</td>'
            f'<td style="padding:0.35rem 0;font-weight:700;color:#E2E8F0;text-align:right;">'
            f'{val} {unit}</td></tr>'
        )
    html += '</table></div>'
    st.markdown(html, unsafe_allow_html=True)


def _generate_pdf_report(entry: dict, mode: str = 'image') -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15, style='B')
    pdf.cell(200, 10, txt="CNC Visual Prediction Report", ln=1, align='C')
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 8, txt=f"Mode: {mode.upper()} | Date: {entry.get('Timestamp', '')}", ln=1, align='L')
    pdf.ln(6)
    for k, v in entry.items():
        if k != 'Timestamp':
            clean_v = str(v).encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(200, 8, txt=f"{k}: {clean_v}", ln=1, align='L')
    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# Image Prediction Tab
# ─────────────────────────────────────────────────────────────────────────────

def _render_image_tab() -> None:
    import cv2
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.visual_predict import predict_from_image, get_wear_label

    st.markdown("""
    <div class="info-card">
      <b>📸 How it works:</b> Upload a photo of your CNC tool or machined surface.
      The system uses <b>computer vision</b> to extract wear indicators (edge sharpness,
      surface texture, discoloration, wear area) and maps them to sensor estimates —
      then runs the ML model to predict failure probability.
    </div>
    """, unsafe_allow_html=True)

    col_up, col_cfg = st.columns([3, 1])
    with col_up:
        uploaded_img = st.file_uploader(
            "Upload tool / machine image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
            key='visual_img_upload',
            help="Supports JPG, PNG, BMP, TIFF, WebP"
        )
    with col_cfg:
        machine_type = st.selectbox(
            "Machine Type",
            ['L', 'M', 'H'],
            index=1,
            key='visual_img_type',
            help="L=Low, M=Medium, H=High grade"
        )

    if not uploaded_img:
        st.markdown("""
        <div style="border:2px dashed #2D2D55;border-radius:14px;padding:3rem;text-align:center;color:#94A3B8;margin-top:1rem;">
          <div style="font-size:3rem;margin-bottom:0.5rem;">🖼️</div>
          <div style="font-size:1rem;font-weight:600;">Drop an image here or click Browse</div>
          <div style="font-size:0.82rem;margin-top:0.4rem;">Supported: JPG · PNG · BMP · TIFF · WebP</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Decode image
    file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("❌ Could not decode the uploaded image. Please try a different file.")
        return

    # Display image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(img_rgb, caption=f"📷 {uploaded_img.name}", use_container_width=True)

    with st.spinner("🔍 Analysing image…"):
        try:
            result_bundle = predict_from_image(img_bgr, machine_type, MODEL_DIR)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

    vf  = result_bundle['visual_features']
    sp  = result_bundle['sensor_params']
    res = result_bundle['prediction']

    with col_info:
        wear_label = get_wear_label(vf)
        st.markdown(f"""
        <div class="info-card" style="margin-bottom:0.8rem;">
          <div style="font-size:1rem;font-weight:700;">🔬 Visual Wear Assessment</div>
          <div style="font-size:1.3rem;font-weight:800;margin-top:0.4rem;">{wear_label}</div>
        </div>
        """, unsafe_allow_html=True)
        _render_sensor_estimates(sp)

    section_header("📊 Visual Metrics")
    _render_visual_metrics(vf)

    section_header("📊 Prediction Result")
    _render_result_card(res)

    # Save to history
    if 'visual_prediction_history' not in st.session_state:
        st.session_state['visual_prediction_history'] = []

    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = {
        'Timestamp'          : ts,
        'Source'             : f'Image: {uploaded_img.name}',
        'Machine Type'       : machine_type,
        'Edge Sharpness'     : vf['edge_sharpness'],
        'Texture Roughness'  : vf['texture_roughness'],
        'Discoloration'      : vf['discoloration'],
        'Wear Area %'        : vf['wear_area_pct'],
        'Est. Air Temp (K)'  : sp['air_temp'],
        'Est. Process Temp (K)': sp['process_temp'],
        'Est. Speed (RPM)'   : sp['rot_speed'],
        'Est. Torque (Nm)'   : sp['torque'],
        'Est. Tool Wear (min)': sp['tool_wear'],
        'Failure Probability': res['failure_probability'],
        'Confidence (%)'     : res['confidence'],
        'Predicted Wear (min)': res['predicted_wear_min'],
        'Remaining Life (min)': res['remaining_life'],
        'Status'             : res['status'],
        'Action'             : res['action'],
    }
    st.session_state['visual_prediction_history'].append(entry)

    # Downloads
    csv_data  = pd.DataFrame([entry]).to_csv(index=False).encode('utf-8')
    pdf_bytes = _generate_pdf_report(entry, mode='image')

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_data,
            file_name=f"visual_pred_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            use_container_width=True,
            key='img_csv_dl'
        )
    with col2:
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"visual_pred_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime='application/pdf',
            use_container_width=True,
            key='img_pdf_dl'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Video Prediction Tab
# ─────────────────────────────────────────────────────────────────────────────

def _render_video_tab() -> None:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.visual_predict import predict_from_video, get_wear_label
    import cv2

    st.markdown("""
    <div class="info-card">
      <b>🎬 How it works:</b> Upload a short video clip of your CNC machine in operation
      or a tool inspection video. The system samples <b>evenly-spaced frames</b>, runs visual
      analysis on each, and plots how failure probability evolves across the clip —
      helping detect wear progression over time.
    </div>
    """, unsafe_allow_html=True)

    col_up, col_cfg = st.columns([3, 1])
    with col_up:
        uploaded_vid = st.file_uploader(
            "Upload video clip",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
            key='visual_vid_upload',
            help="Supports MP4, AVI, MOV, MKV, WMV"
        )
    with col_cfg:
        machine_type = st.selectbox(
            "Machine Type",
            ['L', 'M', 'H'],
            index=1,
            key='visual_vid_type',
        )
        n_frames = st.slider(
            "Frames to Sample",
            min_value=4,
            max_value=20,
            value=10,
            key='visual_vid_frames',
            help="Number of frames to extract and analyse"
        )

    if not uploaded_vid:
        st.markdown("""
        <div style="border:2px dashed #2D2D55;border-radius:14px;padding:3rem;text-align:center;color:#94A3B8;margin-top:1rem;">
          <div style="font-size:3rem;margin-bottom:0.5rem;">🎬</div>
          <div style="font-size:1rem;font-weight:600;">Drop a video file here or click Browse</div>
          <div style="font-size:0.82rem;margin-top:0.4rem;">Supported: MP4 · AVI · MOV · MKV · WMV</div>
        </div>
        """, unsafe_allow_html=True)
        return

    video_bytes = uploaded_vid.read()

    with st.spinner(f"🎥 Processing video — sampling {n_frames} frames…"):
        try:
            frame_results = predict_from_video(
                video_bytes,
                machine_type=machine_type,
                model_dir=MODEL_DIR,
                n_frames=n_frames
            )
        except Exception as e:
            st.error(f"Video processing failed: {e}")
            return

    if not frame_results:
        st.error("No frames could be extracted from the video.")
        return

    st.success(f"✅ Analysed **{len(frame_results)} frames** from the video clip.")

    # ── Trend Chart ───────────────────────────────────────────────────────────
    section_header("📈 Failure Probability Trend")

    import plotly.graph_objects as go

    timestamps = [r['timestamp_sec'] for r in frame_results]
    fail_probs = [r['prediction']['failure_probability'] for r in frame_results]
    wear_pcts  = [r['prediction']['pct_remaining']       for r in frame_results]
    sharpness  = [r['visual_features']['edge_sharpness'] * 100 for r in frame_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=fail_probs,
        mode='lines+markers',
        name='Failure Probability (%)',
        line=dict(color='#EF4444', width=3),
        marker=dict(size=8, color='#EF4444'),
        fill='tozeroy',
        fillcolor='rgba(239,68,68,0.15)',
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=sharpness,
        mode='lines+markers',
        name='Edge Sharpness (%)',
        line=dict(color='#14B8A6', width=2, dash='dot'),
        marker=dict(size=6, color='#14B8A6'),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=wear_pcts,
        mode='lines+markers',
        name='Remaining Life (%)',
        line=dict(color='#6366F1', width=2),
        marker=dict(size=6, color='#6366F1'),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0', family='Inter'),
        xaxis=dict(title='Time (seconds)', gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(title='Value (%)', range=[0, 105], gridcolor='rgba(255,255,255,0.06)'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=30, b=0),
    )
    # Add danger zone
    fig.add_hrect(y0=75, y1=105, fillcolor='rgba(239,68,68,0.08)',
                  line_width=0, annotation_text='⚠️ Danger Zone',
                  annotation_position='top right',
                  annotation=dict(font_color='#EF4444', font_size=12))
    st.plotly_chart(fig, use_container_width=True)

    # ── Frame Thumbnails ──────────────────────────────────────────────────────
    section_header("🖼️ Frame-by-Frame Analysis")

    thumb_cols = st.columns(min(4, len(frame_results)))
    for i, fr in enumerate(frame_results):
        prob   = fr['prediction']['failure_probability']
        status = fr['prediction']['status']
        ts_sec = fr['timestamp_sec']

        color = '#22C55E' if prob < 25 else ('#F59E0B' if prob < 50 else ('#EF4444' if prob < 75 else '#B91C1C'))

        with thumb_cols[i % 4]:
            st.markdown(
                f'<div class="metric-card" style="padding:0.7rem;text-align:center;margin-bottom:0.7rem;">'
                f'<div style="font-size:0.72rem;color:#94A3B8;">⏱️ {ts_sec}s</div>'
                f'<div style="font-size:1.3rem;font-weight:800;color:{color};">{prob:.1f}%</div>'
                f'<div style="font-size:0.7rem;color:#94A3B8;">Failure Prob.</div>'
                f'<div style="font-size:0.72rem;margin-top:4px;">{status.split(" ", 1)[0]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Aggregate Result ──────────────────────────────────────────────────────
    section_header("📊 Aggregate Prediction (Worst Frame)")

    probs_arr = np.array(fail_probs)
    worst_idx = int(np.argmax(probs_arr))
    worst_res = frame_results[worst_idx]['prediction']
    avg_prob  = float(probs_arr.mean())

    st.markdown(f"""
    <div class="info-card" style="margin-bottom:1rem;">
      <b>📊 Summary across {len(frame_results)} frames:</b><br>
      Avg Failure Probability: <b>{avg_prob:.1f}%</b> &nbsp;|&nbsp;
      Peak Failure Probability: <b>{probs_arr.max():.1f}%</b> at {frame_results[worst_idx]['timestamp_sec']}s &nbsp;|&nbsp;
      Min Failure Probability: <b>{probs_arr.min():.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

    _render_result_card(worst_res)

    # ── Per-Frame Table ───────────────────────────────────────────────────────
    with st.expander("📋 View Per-Frame Data Table"):
        rows = []
        for fr in frame_results:
            rows.append({
                'Frame'              : fr['frame_index'],
                'Time (s)'           : fr['timestamp_sec'],
                'Edge Sharpness'     : fr['visual_features']['edge_sharpness'],
                'Discoloration'      : fr['visual_features']['discoloration'],
                'Est. Tool Wear'     : fr['sensor_params']['tool_wear'],
                'Est. Torque'        : fr['sensor_params']['torque'],
                'Failure Prob (%)'   : fr['prediction']['failure_probability'],
                'Remaining Life (min)': fr['prediction']['remaining_life'],
                'Status'             : fr['prediction']['status'],
            })
        frame_df = pd.DataFrame(rows)
        st.dataframe(frame_df, use_container_width=True)

        csv_data = frame_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Frame Analysis (CSV)",
            data=csv_data,
            file_name=f"video_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            use_container_width=True,
            key='vid_csv_dl'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Render
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    inject_css()
    hero_banner(
        title    = "Visual Prediction",
        subtitle = "Upload a CNC tool image or video clip — AI analyses visual wear patterns and predicts machine health instantly.",
        emoji    = "🖼️",
    )

    if not _models_ready():
        st.error("⚠️ Models not trained yet. Please go to **Model Training** page and train the models first.")
        footer()
        return

    # Check OpenCV
    try:
        import cv2  # noqa: F401
    except ImportError:
        st.error("❌ OpenCV not installed. Run: `pip install opencv-python-headless` and restart the app.")
        footer()
        return

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_img, tab_vid = st.tabs(["📸  Image Prediction", "🎬  Video Prediction"])

    with tab_img:
        section_header("📸 Image-Based Prediction")
        _render_image_tab()

    with tab_vid:
        section_header("🎬 Video-Based Prediction")
        _render_video_tab()

    # ── Session History ───────────────────────────────────────────────────────
    history = st.session_state.get('visual_prediction_history', [])
    if history:
        section_header("📜 Visual Prediction History (this session)")
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True, height=220)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            hist_df.to_excel(writer, index=False, sheet_name='Visual History')
        excel_data = output.getvalue()

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Export History (CSV)",
                data=hist_df.to_csv(index=False).encode('utf-8'),
                file_name="visual_prediction_history.csv",
                mime='text/csv',
                use_container_width=True,
                key='hist_csv'
            )
        with col2:
            st.download_button(
                label="📥 Export History (Excel)",
                data=excel_data,
                file_name="visual_prediction_history.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True,
                key='hist_excel'
            )

    footer()
