"""
=============================================================================
page_live.py — Live Dashboard
=============================================================================
Simulates a live dashboard for CNC Predictive Maintenance.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from collections import deque
from app.style import inject_css, hero_banner, section_header, footer

def render() -> None:
    inject_css()
    hero_banner(
        title    = "Live Monitoring Dashboard",
        subtitle = "Real-time CNC sensor data simulation and active prediction.",
        emoji    = "📡",
    )

    section_header("Live Sensor Feed")
    
    # Session state for simulated live data
    if 'live_data' not in st.session_state:
        st.session_state['live_data'] = {
            'time': deque(maxlen=50),
            'air_temp': deque(maxlen=50),
            'process_temp': deque(maxlen=50),
            'torque': deque(maxlen=50),
            'speed': deque(maxlen=50),
            'wear': deque(maxlen=50)
        }
        
    start_btn, stop_btn, clear_btn = st.columns(3)
    
    with start_btn:
        start_sim = st.button("▶️ Start Live Feed", use_container_width=True)
    with stop_btn:
        stop_sim = st.button("⏸️ Stop Feed", use_container_width=True)
    with clear_btn:
        if st.button("🔄 Clear Data", use_container_width=True):
            for k in st.session_state['live_data']:
                st.session_state['live_data'][k].clear()
            st.rerun()

    # Placeholders for live charts
    chart_col1, chart_col2 = st.columns(2)
    chart1_placeholder = chart_col1.empty()
    chart2_placeholder = chart_col2.empty()
    
    # Placeholder for live metrics
    metrics_placeholder = st.empty()
    
    if start_sim:
        st.session_state['sim_running'] = True
    if stop_sim:
        st.session_state['sim_running'] = False
        
    is_running = st.session_state.get('sim_running', False)
    
    if is_running:
        st.toast("Live simulation started!", icon="📡")
        
    # Simulation loop
    while st.session_state.get('sim_running', False):
        # Generate random new data points based on normal ranges
        t = pd.Timestamp.now().strftime('%H:%M:%S')
        air_t = np.random.normal(300.0, 1.0)
        proc_t = np.random.normal(310.0, 1.0)
        trq = np.random.normal(40.0, 5.0)
        spd = np.random.normal(1500, 50)
        
        # Increment wear slowly
        prev_wear = st.session_state['live_data']['wear'][-1] if len(st.session_state['live_data']['wear']) > 0 else 50
        wear = prev_wear + np.random.uniform(0, 0.5)
        
        st.session_state['live_data']['time'].append(t)
        st.session_state['live_data']['air_temp'].append(air_t)
        st.session_state['live_data']['process_temp'].append(proc_t)
        st.session_state['live_data']['torque'].append(trq)
        st.session_state['live_data']['speed'].append(spd)
        st.session_state['live_data']['wear'].append(wear)
        
        # Update charts
        with chart1_placeholder.container():
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=list(st.session_state['live_data']['time']), y=list(st.session_state['live_data']['torque']), mode='lines+markers', name='Torque', line=dict(color='#EC4899')))
            fig1.update_layout(title="Live Torque (Nm)", height=300, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='gray'))
            st.plotly_chart(fig1, use_container_width=True)
            
        with chart2_placeholder.container():
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(st.session_state['live_data']['time']), y=list(st.session_state['live_data']['process_temp']), mode='lines+markers', name='Process Temp', line=dict(color='#14B8A6')))
            fig2.update_layout(title="Live Process Temp (K)", height=300, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='gray'))
            st.plotly_chart(fig2, use_container_width=True)
            
        # Update metrics
        with metrics_placeholder.container():
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Current Torque", f"{trq:.2f} Nm", f"{trq - 40:.2f}")
            mc2.metric("Process Temp", f"{proc_t:.2f} K", f"{proc_t - 310:.2f}")
            mc3.metric("Spindle Speed", f"{spd:.0f} RPM")
            mc4.metric("Tool Wear", f"{wear:.1f} min", f"+{(wear - prev_wear):.2f}", delta_color="inverse")
            
            # Simple threshold check for notifications
            if trq > 55:
                st.warning(f"⚠️ High Torque Detected: {trq:.2f} Nm")
            if proc_t > 313:
                st.error(f"🚨 Critical Process Temperature: {proc_t:.2f} K")
                
        time.sleep(2)
        st.rerun()
        
    footer()
