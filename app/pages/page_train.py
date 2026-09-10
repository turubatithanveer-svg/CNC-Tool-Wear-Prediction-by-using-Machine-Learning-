"""
=============================================================================
page_train.py — Model Training Page
=============================================================================
Allows users to trigger the full ML training pipeline from the UI.
Shows:
  • Training progress
  • Live model metrics
  • Best model summary
  • Comparison table
  • Feature importance chart
=============================================================================
"""

import os
import json
import streamlit as st
import pandas as pd
from app.style import inject_css, hero_banner, section_header, footer, metric_card


MODEL_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'ai4i2020.csv')


def _metrics_exist() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, 'metrics.json'))


def _load_metrics() -> dict:
    with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
        return json.load(f)


def render() -> None:
    inject_css()
    hero_banner(
        title    = "Model Training",
        subtitle = "Train 7 ML models, compare them, and auto-select the best performer.",
        emoji    = "🤖",
    )

    # ── Training trigger ────────────────────────────────────────────────────
    section_header("🚀 Start Training Pipeline")
    st.markdown("""
    <div class="info-card">
      <b>Training includes:</b> Decision Tree · Random Forest · Gradient Boosting ·
      XGBoost · Extra Trees · SVM · Logistic Regression<br>
      <b>Plus:</b> Random Forest Regressor for continuous tool-wear prediction.<br>
      All models are evaluated and the best F1-score model is saved automatically.
    </div>
    """, unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        do_train = st.button("⚙️ Train All Models", key='train_btn', use_container_width=True)

    if do_train:
        with st.spinner("🔄 Loading and preprocessing dataset…"):
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from src.preprocessing import load_dataset, preprocess
            from src.train import run_training_pipeline
            from src.predict import clear_cache

            df        = load_dataset(DATASET_PATH)
            data_dict = preprocess(df)

        st.success("✅ Data preprocessed!")

        # Progress bar animation
        bar = st.progress(0, text="Training models…")
        import time

        with st.spinner("🤖 Training classifiers — this may take 1–3 minutes…"):
            results = run_training_pipeline(data_dict, model_dir=MODEL_DIR)
            clear_cache()   # invalidate prediction cache

        for i in range(100):
            bar.progress(i + 1, text=f"Training complete! ({i+1}%)")
            time.sleep(0.01)

        st.balloons()
        st.success(f"🏆 Best Model: **{results['best']['name']}** — F1 = {results['best']['f1']:.4f}")

        # Store in session for display below
        st.session_state['train_results'] = results

    # ── Show results if trained ─────────────────────────────────────────────
    if _metrics_exist():
        metrics = _load_metrics()
        section_header("📊 Best Model Performance")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card(f"{metrics.get('accuracy',0)*100:.1f}%",  "Accuracy"),  unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card(f"{metrics.get('f1',0)*100:.1f}%",       "F1 Score"),  unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card(f"{metrics.get('roc_auc',0)*100:.1f}%",  "ROC AUC"),   unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card(f"{metrics.get('cv_mean',0)*100:.1f}%",  "CV Mean F1"), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-card" style="margin-top:1rem;">
          <b>🏆 Best Classifier:</b> {metrics.get('best_model','—')}<br>
          <b>📐 Regressor R²:</b> {metrics.get('regressor_r2',0):.4f} &nbsp;|&nbsp;
          <b>MAE:</b> {metrics.get('regressor_mae',0):.2f} min &nbsp;|&nbsp;
          <b>RMSE:</b> {metrics.get('regressor_rmse',0):.2f} min
        </div>
        """, unsafe_allow_html=True)

    elif not do_train:
        st.info("ℹ️ Models not yet trained. Click the **Train All Models** button above.")

    # ── Show comparison table from session ──────────────────────────────────
    if 'train_results' in st.session_state:
        section_header("📋 Model Comparison Table")
        from src.evaluation import build_comparison_df, plot_model_comparison, plot_feature_importance

        results = st.session_state['train_results']
        comp_df = build_comparison_df(results['classifiers'])
        st.dataframe(comp_df, use_container_width=True)

        section_header("📊 Comparison Bar Chart")
        st.plotly_chart(plot_model_comparison(results['classifiers']),
                         use_container_width=True)

        section_header("🔍 Feature Importance — Best Model")
        best_model = results['best']['model']
        feat_cols  = st.session_state.get('feature_cols',
                     list(results['best'].get('report', {}).keys()))

        # Retrieve feature_cols from data_dict saved in session
        if 'data_dict' in st.session_state:
            feat_cols = st.session_state['data_dict']['feature_cols']
        else:
            fc_path = os.path.join(MODEL_DIR, 'feature_cols.json')
            if os.path.exists(fc_path):
                with open(fc_path) as f:
                    feat_cols = json.load(f)

        imp_fig = plot_feature_importance(best_model, feat_cols,
                                           results['best']['name'])
        if imp_fig:
            st.plotly_chart(imp_fig, use_container_width=True)

    footer()
