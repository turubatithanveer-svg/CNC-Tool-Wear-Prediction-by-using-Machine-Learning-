"""
=============================================================================
page_dataset.py — Dataset Analysis Page
=============================================================================
Shows:
  • Dataset overview table
  • Shape, dtypes, missing values, duplicates
  • Statistical summary
  • Outlier count
  • Download raw data
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.style import inject_css, hero_banner, section_header, footer, metric_card


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


def render() -> None:
    inject_css()
    hero_banner(
        title    = "Dataset Analysis",
        subtitle = "AI4I 2020 Predictive Maintenance Dataset — Deep Inspection & Statistics",
        emoji    = "📂",
    )

    df = _load_df()

    # ── Quick stats ─────────────────────────────────────────────────────────
    total_rows   = len(df)
    total_cols   = len(df.columns)
    missing      = int(df.isnull().sum().sum())
    duplicates   = int(df.duplicated().sum())
    failure_rate = f"{df['Machine failure'].mean()*100:.1f}%"

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [f"{total_rows:,}", total_cols, missing, duplicates, failure_rate],
        ["Total Rows", "Features", "Missing Values", "Duplicates", "Failure Rate"],
    ):
        with col:
            st.markdown(metric_card(val, lbl), unsafe_allow_html=True)

    # ── Dataset preview ─────────────────────────────────────────────────────
    section_header("📋 Dataset Preview")
    n = st.slider("Rows to show", 5, 100, 20, key='ds_rows')
    st.dataframe(df.head(n), use_container_width=True, height=350)

    # ── Column Info ─────────────────────────────────────────────────────────
    section_header("🔍 Column Information")
    col_info = pd.DataFrame({
        'Column'     : df.columns,
        'Data Type'  : df.dtypes.values,
        'Non-Null'   : df.notnull().sum().values,
        'Null Count' : df.isnull().sum().values,
        'Unique'     : df.nunique().values,
        'Sample'     : [df[c].iloc[0] for c in df.columns],
    })
    st.dataframe(col_info, use_container_width=True)

    # ── Statistical Summary ─────────────────────────────────────────────────
    section_header("📊 Statistical Summary")
    st.dataframe(df.describe().T.round(3), use_container_width=True)

    # ── Value Counts for categorical ─────────────────────────────────────────
    section_header("🔡 Categorical Distribution")
    cat_cols = ['Type'] if 'Type' in df.columns else []
    if cat_cols:
        for col_name in cat_cols:
            vc = df[col_name].value_counts().reset_index()
            vc.columns = [col_name, 'Count']
            vc['Percentage'] = (vc['Count'] / len(df) * 100).round(2)
            st.write(f"**{col_name}**")
            st.dataframe(vc, use_container_width=True)

    # ── Outlier report ───────────────────────────────────────────────────────
    section_header("🔴 Outlier Detection (IQR × 3.0)")
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_rows = []
    for col in numeric_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3.0 * IQR
        upper = Q3 + 3.0 * IQR
        cnt   = int(((df[col] < lower) | (df[col] > upper)).sum())
        if cnt > 0:
            outlier_rows.append({'Feature': col, 'Outlier Count': cnt,
                                  'Lower Bound': round(lower, 3),
                                  'Upper Bound': round(upper, 3)})
    if outlier_rows:
        st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)
    else:
        st.success("✅ No significant outliers detected at IQR × 3.0 threshold.")

    # ── Download button ──────────────────────────────────────────────────────
    section_header("⬇️ Download Dataset")
    st.download_button(
        label    = "📥 Download ai4i2020.csv",
        data     = df.to_csv(index=False).encode('utf-8'),
        file_name= "cnc_ai4i2020.csv",
        mime     = "text/csv",
    )

    footer()
