"""
=============================================================================
page_eda.py — Exploratory Data Analysis Page
=============================================================================
Generates beautiful interactive charts:
  • Tool Wear Distribution (histogram + KDE)
  • Machine Type Pie Chart
  • Machine Failure Distribution
  • Correlation Heatmap
  • Scatter Plots (Torque vs Speed, Wear vs Torque)
  • Box Plots
  • Missing Value Chart
  • Feature vs Failure Bar Chart
  • Tool Wear Trend Line
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.style import inject_css, hero_banner, section_header, footer


@st.cache_data
def _load_df():
    import os
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'ai4i2020.csv')
    df = pd.read_csv(path)
    rename_map = {
        'Air temperature [K]': 'Air temperature',
        'Process temperature [K]': 'Process temperature',
        'Rotational speed [rpm]': 'Rotational speed',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Tool wear'
    }
    return df.rename(columns=rename_map)


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _layout(fig, title='', h=420):
    fig.update_layout(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor ='rgba(15,15,26,0.6)',
        font=dict(family='Inter', color='#E2E8F0', size=13),
        height=h,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.07)', showgrid=True)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.07)', showgrid=True)
    return fig


def chart_tool_wear_dist(df):
    fig = px.histogram(df, x='Tool wear', nbins=60,
                        color_discrete_sequence=['#6366F1'],
                        marginal='violin')
    return _layout(fig, '📊 Tool Wear Distribution (minutes)')


def chart_machine_type_pie(df):
    vc = df['Type'].value_counts().reset_index()
    vc.columns = ['Type', 'Count']
    fig = px.pie(vc, names='Type', values='Count',
                  color_discrete_sequence=['#6366F1', '#EC4899', '#14B8A6'],
                  hole=0.45)
    fig.update_traces(textposition='outside', textinfo='percent+label')
    return _layout(fig, '🔩 Machine Type Distribution')


def chart_failure_distribution(df):
    vc = df['Machine failure'].value_counts().reset_index()
    vc.columns = ['Failure', 'Count']
    vc['Failure'] = vc['Failure'].map({0: 'Normal (0)', 1: 'Failure (1)'})
    fig = px.bar(vc, x='Failure', y='Count',
                  color='Failure',
                  color_discrete_map={'Normal (0)': '#22C55E', 'Failure (1)': '#EF4444'},
                  text='Count')
    fig.update_traces(textposition='outside')
    return _layout(fig, '⚠️ Machine Failure Distribution')


def chart_corr_heatmap(df):
    num_df = df.select_dtypes(include=np.number)
    corr   = num_df.corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis',
        text=corr.values,
        texttemplate='%{text}',
        textfont=dict(size=9),
        showscale=True,
    ))
    return _layout(fig, '🌡️ Correlation Heatmap', h=550)


def chart_scatter_torque_speed(df):
    fig = px.scatter(df, x='Rotational speed', y='Torque',
                      color='Machine failure',
                      color_discrete_map={0: '#6366F1', 1: '#EF4444'},
                      opacity=0.6, size_max=6,
                      labels={'Machine failure': 'Failure'})
    return _layout(fig, '🔄 Rotational Speed vs Torque (coloured by Failure)')


def chart_scatter_wear_torque(df):
    fig = px.scatter(df, x='Tool wear', y='Torque',
                      color='Type',
                      color_discrete_sequence=['#6366F1', '#EC4899', '#14B8A6'],
                      opacity=0.6, trendline='ols')
    return _layout(fig, '📉 Tool Wear vs Torque by Machine Type')


def chart_boxplot(df):
    numeric_cols = ['Air temperature', 'Process temperature',
                    'Rotational speed', 'Torque', 'Tool wear']
    fig = make_subplots(rows=1, cols=len(numeric_cols),
                         subplot_titles=numeric_cols)
    colours = ['#6366F1', '#EC4899', '#14B8A6', '#F59E0B', '#EF4444']
    for i, col in enumerate(numeric_cols, 1):
        fig.add_trace(go.Box(
            y=df[col], name=col,
            marker_color=colours[i-1],
            boxmean='sd',
        ), row=1, col=i)
    fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(15,15,26,0.6)',
                       font=dict(family='Inter', color='#E2E8F0'),
                       height=430, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_annotations(font_color='#C7D2FE')
    return fig


def chart_missing_values(df):
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        return None
    fig = px.bar(x=miss.index, y=miss.values,
                  color=miss.values, color_continuous_scale='Reds',
                  labels={'x': 'Column', 'y': 'Missing Count'})
    return _layout(fig, '❓ Missing Values per Column')


def chart_feature_vs_failure(df):
    """Mean of each numeric feature grouped by failure status."""
    num = df.select_dtypes(include=np.number).columns.tolist()
    exclude = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    cols = [c for c in num if c not in exclude]

    grouped = df.groupby('Machine failure')[cols].mean().T.reset_index()
    grouped.columns = ['Feature', 'Normal', 'Failure']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grouped['Feature'], y=grouped['Normal'],
                          name='Normal', marker_color='#22C55E'))
    fig.add_trace(go.Bar(x=grouped['Feature'], y=grouped['Failure'],
                          name='Failure', marker_color='#EF4444'))
    fig.update_layout(barmode='group')
    return _layout(fig, '📊 Mean Feature Values — Normal vs Failure', h=460)


def chart_wear_trend(df):
    """Tool wear trend over first 500 samples (index order)."""
    sample = df.head(500).copy().reset_index(drop=True)
    fig = px.line(sample, y='Tool wear', x=sample.index,
                   color_discrete_sequence=['#6366F1'])
    fig.update_traces(line_width=1.5)
    return _layout(fig, '📈 Tool Wear Trend (first 500 samples)')


def chart_failure_mode_breakdown(df):
    modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    counts = [df[m].sum() for m in modes if m in df.columns]
    labels = [m for m in modes if m in df.columns]
    fig = px.bar(x=labels, y=counts,
                  color=labels,
                  color_discrete_sequence=['#6366F1','#EC4899','#14B8A6','#F59E0B','#EF4444'],
                  text=counts)
    fig.update_traces(textposition='outside')
    return _layout(fig, '💥 Failure Mode Breakdown (TWF / HDF / PWF / OSF / RNF)')


# ─────────────────────────────────────────────────────────────────────────────
# Page renderer
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    inject_css()
    hero_banner(
        title    = "Exploratory Data Analysis",
        subtitle = "Interactive visualisations of the CNC Predictive Maintenance dataset",
        emoji    = "📈",
    )

    df = _load_df()

    # ── Chart tabs ───────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Distributions",
        "🌡️ Correlations",
        "🔄 Scatter Plots",
        "📦 Box Plots",
        "💥 Failures",
        "📈 Trends",
        "📊 Graphs",
    ])

    with tabs[0]:
        section_header("Tool Wear & Machine Type Distribution")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_tool_wear_dist(df), use_container_width=True)
        with c2:
            st.plotly_chart(chart_machine_type_pie(df), use_container_width=True)
        st.plotly_chart(chart_failure_distribution(df), use_container_width=True)

    with tabs[1]:
        section_header("Correlation Heatmap")
        st.plotly_chart(chart_corr_heatmap(df), use_container_width=True)

    with tabs[2]:
        section_header("Scatter Plots")
        st.plotly_chart(chart_scatter_torque_speed(df), use_container_width=True)
        st.plotly_chart(chart_scatter_wear_torque(df),  use_container_width=True)

    with tabs[3]:
        section_header("Box Plots — Numeric Features")
        st.plotly_chart(chart_boxplot(df), use_container_width=True)

    with tabs[4]:
        section_header("Machine Failure Analysis")
        st.plotly_chart(chart_failure_mode_breakdown(df), use_container_width=True)
        st.plotly_chart(chart_feature_vs_failure(df),     use_container_width=True)
        miss_fig = chart_missing_values(df)
        if miss_fig:
            st.plotly_chart(miss_fig, use_container_width=True)
        else:
            st.success("✅ Dataset has zero missing values!")

    with tabs[5]:
        section_header("Tool Wear Trend Over Time")
        st.plotly_chart(chart_wear_trend(df), use_container_width=True)

    with tabs[6]:
        section_header("Prediction Output Graphs")
        import os
        MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        if not os.path.exists(os.path.join(MODEL_DIR, 'best_classifier.joblib')):
            st.warning("⚠️ Models not trained yet. Please train models first to see prediction graphs on the dataset.")
        else:
            with st.spinner("Generating predictions on the dataset..."):
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
                from src.predict import predict_batch
                
                try:
                    df_pred = predict_batch(df, model_dir=MODEL_DIR)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        # Histogram of failure probabilities
                        fig_prob = px.histogram(df_pred, x='Failure_Probability', nbins=30,
                                               title="Distribution of Failure Probabilities",
                                               color_discrete_sequence=['#6366F1'])
                        fig_prob = _layout(fig_prob, "Distribution of Failure Probabilities")
                        st.plotly_chart(fig_prob, use_container_width=True)
                        
                    with c2:
                        # Pie chart of health status
                        status_counts = df_pred['Health_Status'].value_counts().reset_index()
                        status_counts.columns = ['Status', 'Count']
                        fig_status = px.pie(status_counts, names='Status', values='Count',
                                            title="Predicted Health Status", hole=0.45)
                        fig_status = _layout(fig_status, "Predicted Health Status Breakdown")
                        st.plotly_chart(fig_status, use_container_width=True)
                        
                    # Scatter: predicted wear vs actual wear
                    if 'Predicted_Wear_min' in df_pred.columns and 'Tool wear' in df_pred.columns:
                        fig_wear = px.scatter(df_pred, x='Tool wear', y='Predicted_Wear_min',
                                              color='Failure_Prediction',
                                              title="Actual Tool Wear vs Predicted Tool Wear",
                                              opacity=0.6,
                                              labels={'Failure_Prediction': 'Predicted Failure'})
                        fig_wear = _layout(fig_wear, "Actual vs Predicted Tool Wear")
                        st.plotly_chart(fig_wear, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Failed to generate prediction graphs: {e}")

    footer()

