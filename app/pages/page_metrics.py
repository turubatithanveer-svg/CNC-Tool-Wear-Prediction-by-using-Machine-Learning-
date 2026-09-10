"""
=============================================================================
page_metrics.py — Model Performance Metrics Page
=============================================================================
Shows:
  • Saved model metrics (accuracy, F1, AUC, etc.)
  • Confusion Matrix (Plotly heatmap)
  • ROC Curve
  • Feature Importance
  • Model Comparison Bar Chart
  • Cross Validation Summary
=============================================================================
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from app.style import inject_css, hero_banner, section_header, footer, metric_card


MODEL_DIR    = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'ai4i2020.csv')


def _metrics_exist() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, 'metrics.json'))


def render() -> None:
    inject_css()
    hero_banner(
        title    = "Model Performance Metrics",
        subtitle = "Detailed evaluation — Confusion Matrix · ROC Curve · Feature Importance",
        emoji    = "📉",
    )

    if not _metrics_exist():
        st.warning("⚠️ No trained models found. Please visit **Model Training** and train the models first.")
        footer()
        return

    # ── Load saved metrics ────────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
        m = json.load(f)

    # ── Summary cards ──────────────────────────────────────────────────────────
    section_header("🏆 Best Model Summary")
    st.markdown(f"""
    <div class="info-card" style="margin-bottom:1.5rem;">
      <b>Best Model:</b> {m.get('best_model','—')} &nbsp;|&nbsp;
      <b>CV Mean F1:</b> {m.get('cv_mean',0)*100:.2f}% ± {m.get('cv_std',0)*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [f"{m.get('accuracy',0)*100:.1f}%",
         f"{m.get('precision',0)*100:.1f}%",
         f"{m.get('recall',0)*100:.1f}%",
         f"{m.get('f1',0)*100:.1f}%",
         f"{m.get('roc_auc',0)*100:.1f}%"],
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    ):
        with col:
            st.markdown(metric_card(val, lbl), unsafe_allow_html=True)

    # ── Regressor metrics ──────────────────────────────────────────────────────
    section_header("📐 Regression Model (Tool Wear Estimator)")
    r1, r2, r3, _ = st.columns(4)
    with r1:
        st.markdown(metric_card(f"{m.get('regressor_r2',0):.4f}", "R² Score"), unsafe_allow_html=True)
    with r2:
        st.markdown(metric_card(f"{m.get('regressor_mae',0):.2f}", "MAE (min)"), unsafe_allow_html=True)
    with r3:
        st.markdown(metric_card(f"{m.get('regressor_rmse',0):.2f}", "RMSE (min)"), unsafe_allow_html=True)

    # ── Live evaluation (re-run on dataset) ───────────────────────────────────
    section_header("🔄 Live Evaluation on Test Set")
    if st.button("🔄 Run Live Evaluation", key='eval_btn'):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from src.preprocessing import load_dataset, preprocess
        from src.evaluation import (plot_confusion_matrix, plot_roc_curves,
                                     plot_feature_importance, plot_model_comparison,
                                     build_comparison_df)
        import joblib

        with st.spinner("Preprocessing…"):
            df       = load_dataset(DATASET_PATH)
            data_dict = preprocess(df)

        clf = joblib.load(os.path.join(MODEL_DIR, 'best_classifier.joblib'))
        X_test  = data_dict['X_test']
        y_test  = data_dict['y_test_cls']

        y_pred = clf.predict(X_test)
        try:
            y_prob = clf.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = y_pred.astype(float)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Confusion matrix
        section_header("📊 Confusion Matrix")
        fig_cm = plot_confusion_matrix(cm, m.get('best_model', 'Best Model'))
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report table
        from sklearn.metrics import classification_report
        report_dict = classification_report(y_test, y_pred,
                                             output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).T.round(4)
        st.write("**Classification Report**")
        st.dataframe(report_df, use_container_width=True)

        # ROC curve (single model)
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve, auc as sk_auc
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_val     = sk_auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                      line=dict(dash='dash', color='gray')))
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                      name=f"AUC={roc_val:.3f}",
                                      line=dict(color='#6366F1', width=2.5)))
        fig_roc.update_layout(title='ROC Curve',
                               xaxis_title='False Positive Rate',
                               yaxis_title='True Positive Rate',
                               paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               font=dict(family='Inter', color='#E2E8F0'),
                               height=430)
        section_header("📈 ROC Curve")
        st.plotly_chart(fig_roc, use_container_width=True)

        # Feature importance
        section_header("🔍 Feature Importance")
        feat_cols = data_dict['feature_cols']
        imp_fig   = plot_feature_importance(clf, feat_cols,
                                             m.get('best_model', 'Best Model'))
        if imp_fig:
            st.plotly_chart(imp_fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    footer()
