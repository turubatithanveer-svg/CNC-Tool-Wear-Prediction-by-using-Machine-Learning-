"""
=============================================================================
evaluation.py — CNC Predictive Maintenance Project
=============================================================================
Provides rich evaluation utilities:
  • Model comparison table
  • Confusion matrix plots
  • ROC curves
  • Feature importance charts
  • Cross-validation summary
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              classification_report)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_df(clf_results: list) -> pd.DataFrame:
    """
    Build a clean comparison DataFrame from list of classifier result dicts.

    Parameters
    ----------
    clf_results : list of dicts (output of train_all_classifiers)

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for r in clf_results:
        rows.append({
            'Model'        : r['name'],
            'Accuracy'     : round(r['accuracy'],  4),
            'Precision'    : round(r['precision'], 4),
            'Recall'       : round(r['recall'],    4),
            'F1 Score'     : round(r['f1'],        4),
            'ROC AUC'      : round(r['roc_auc'],   4) if r['roc_auc'] else None,
            'CV Mean (F1)' : round(r['cv_mean'],   4),
            'CV Std'       : round(r['cv_std'],    4),
            'Train Time(s)': round(r['train_time'],3),
            'Pred Time(s)' : round(r['pred_time'], 4),
        })
    df = pd.DataFrame(rows).sort_values('F1 Score', ascending=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plotly Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list, model_name: str = 'Model') -> go.Figure:
    """
    Return an interactive Plotly heatmap for a confusion matrix.

    Parameters
    ----------
    cm         : list or 2-D array — confusion matrix values
    model_name : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    cm = np.array(cm)
    labels = ['Normal (0)', 'Failure (1)']

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 18},
    ))
    fig.update_layout(
        title=f'Confusion Matrix — {model_name}',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        font=dict(family='Inter', size=13),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROC Curves (all models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(clf_results: list, y_test) -> go.Figure:
    """
    Plot overlaid ROC curves for all classifiers.

    Parameters
    ----------
    clf_results : list of dicts
    y_test      : array-like — true labels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Random baseline
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             line=dict(dash='dash', color='gray', width=1),
                             name='Random Classifier'))

    colours = px.colors.qualitative.Plotly
    for i, r in enumerate(clf_results):
        if r.get('y_prob') is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{r['name']} (AUC={roc_auc:.3f})",
            line=dict(width=2, color=colours[i % len(colours)]),
        ))

    fig.update_layout(
        title='ROC Curves — All Models',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.6, y=0.1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=13),
        height=500,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_cols: list,
                             model_name: str = 'Model',
                             top_n: int = 15) -> go.Figure:
    """
    Plot horizontal bar chart of feature importances.

    Parameters
    ----------
    model        : fitted sklearn model with feature_importances_
    feature_cols : list of column names
    model_name   : str
    top_n        : int — how many features to show

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    if not hasattr(model, 'feature_importances_'):
        return None

    imp = model.feature_importances_
    df  = pd.DataFrame({'Feature': feature_cols, 'Importance': imp})
    df  = df.sort_values('Importance', ascending=True).tail(top_n)

    fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                 title=f'Feature Importance — {model_name}',
                 color='Importance', color_continuous_scale='Blues')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=13),
        height=450,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(clf_results: list) -> go.Figure:
    """
    Grouped bar chart comparing all models across key metrics.

    Parameters
    ----------
    clf_results : list of dicts

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df   = build_comparison_df(clf_results)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

    fig = go.Figure()
    colours = ['#6366F1', '#EC4899', '#14B8A6', '#F59E0B', '#EF4444']
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            fig.add_trace(go.Bar(
                x=df['Model'],
                y=df[metric],
                name=metric,
                marker_color=colours[i],
            ))

    fig.update_layout(
        barmode='group',
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.05]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=13),
        legend=dict(orientation='h', y=-0.2),
        height=500,
    )
    return fig
