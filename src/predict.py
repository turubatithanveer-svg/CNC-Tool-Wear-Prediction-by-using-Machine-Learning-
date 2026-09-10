"""
=============================================================================
predict.py — CNC Predictive Maintenance Project
=============================================================================
Prediction engine:
  • Single-row inference (user inputs from Streamlit form)
  • Batch inference (CSV upload)
  • Returns structured result with health status, confidence, recommendation
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


# ─────────────────────────────────────────────────────────────────────────────
# Model Loader (cached on first call)
# ─────────────────────────────────────────────────────────────────────────────

_cache: dict = {}

def load_models(model_dir: str = MODEL_DIR) -> dict:
    """
    Load trained models from disk (cached in module-level dict).

    Returns
    -------
    dict with keys: classifier, regressor, scaler, encoder, feature_cols
    """
    global _cache
    if _cache:
        return _cache

    try:
        clf  = joblib.load(os.path.join(model_dir, 'best_classifier.joblib'))
        reg  = joblib.load(os.path.join(model_dir, 'tool_wear_regressor.joblib'))
        scaler  = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))

        with open(os.path.join(model_dir, 'feature_cols.json')) as f:
            feature_cols = json.load(f)

        with open(os.path.join(model_dir, 'metrics.json')) as f:
            metrics = json.load(f)

        _cache = {
            'classifier' : clf,
            'regressor'  : reg,
            'scaler'     : scaler,
            'encoder'    : encoder,
            'feature_cols': feature_cols,
            'metrics'    : metrics,
        }
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model files not found in {model_dir}.\n"
            "Please run 'python -m src.train' first to train the models.\n"
            f"Details: {e}"
        )

    return _cache


def clear_cache() -> None:
    """Clear the module-level model cache (used after retraining)."""
    global _cache
    _cache = {}


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_row(input_data: dict, models: dict) -> pd.DataFrame:
    """
    Convert raw user input dict to a scaled feature DataFrame.

    Expected keys in input_data
    ---------------------------
    type_val      : str   — 'L', 'M', or 'H'
    air_temp      : float — Kelvin
    process_temp  : float — Kelvin
    rot_speed     : float — RPM
    torque        : float — Nm
    tool_wear     : float — minutes

    Returns
    -------
    pd.DataFrame (1 row, scaled)
    """
    from src.feature_engineering import engineer_features

    # Build raw single row
    row = {
        'Type'               : input_data.get('type_val', 'M'),
        'Air temperature'    : float(input_data.get('air_temp', 300.0)),
        'Process temperature': float(input_data.get('process_temp', 310.0)),
        'Rotational speed'   : float(input_data.get('rot_speed', 1500.0)),
        'Torque'             : float(input_data.get('torque', 40.0)),
        'Tool wear'          : float(input_data.get('tool_wear', 0.0)),
    }
    df = pd.DataFrame([row])

    # Encode 'Type'
    encoder = models['encoder']
    df['Type_encoded'] = encoder.transform(df['Type'].astype(str))

    # Feature engineering
    df = engineer_features(df)

    # Select and order features
    feature_cols = models['feature_cols']
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()

    # Scale
    X_scaled = models['scaler'].transform(X)
    return pd.DataFrame(X_scaled, columns=available)


# ─────────────────────────────────────────────────────────────────────────────
# Health & Recommendation Engine
# ─────────────────────────────────────────────────────────────────────────────

def _build_recommendation(failure_prob: float, tool_wear: float,
                            predicted_wear: float) -> dict:
    """
    Derive health status, colour, and action recommendation.

    Parameters
    ----------
    failure_prob   : float [0–1] — P(failure)
    tool_wear      : float       — current tool wear (min)
    predicted_wear : float       — regressor prediction (min)

    Returns
    -------
    dict with status, color, alert, recommendation, remaining_life
    """
    # Max useful life of tool ~200 min (domain knowledge)
    MAX_LIFE = 200.0
    remaining = max(0.0, MAX_LIFE - max(tool_wear, predicted_wear))
    pct_remaining = remaining / MAX_LIFE * 100

    if failure_prob < 0.25 and pct_remaining > 50:
        status     = '✅ Healthy'
        color      = 'green'
        alert      = 'No immediate action needed'
        action     = 'Continue Production — Monitor Weekly'
        level      = 1

    elif failure_prob < 0.50 or pct_remaining > 25:
        status     = '⚠️ Moderate Wear'
        color      = 'orange'
        alert      = 'Schedule preventive maintenance soon'
        action     = 'Plan Tool Replacement — Monitor Daily'
        level      = 2

    elif failure_prob < 0.75 or pct_remaining > 10:
        status     = '🔴 High Wear'
        color      = 'red'
        alert      = 'Tool nearing end of life!'
        action     = 'Replace Tool Soon — Reduce Feed Rate'
        level      = 3

    else:
        status     = '🚨 Critical / Failure'
        color      = 'darkred'
        alert      = 'IMMEDIATE MAINTENANCE REQUIRED!'
        action     = 'Stop Machine — Replace Tool Immediately'
        level      = 4

    return {
        'status'         : status,
        'color'          : color,
        'alert'          : alert,
        'action'         : action,
        'level'          : level,
        'remaining_life' : round(remaining, 1),
        'pct_remaining'  : round(pct_remaining, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(input_data: dict, model_dir: str = MODEL_DIR) -> dict:
    """
    Run prediction for a single set of sensor inputs.

    Parameters
    ----------
    input_data : dict  — raw sensor inputs from user form
    model_dir  : str   — path to saved models

    Returns
    -------
    dict with full prediction result
    """
    models = load_models(model_dir)
    X      = _build_feature_row(input_data, models)

    clf = models['classifier']
    reg = models['regressor']

    # Classification
    failure_pred = int(clf.predict(X)[0])
    try:
        failure_prob = float(clf.predict_proba(X)[0][1])
    except AttributeError:
        failure_prob = float(failure_pred)

    # Regression (tool wear estimate)
    predicted_wear = float(reg.predict(X)[0])
    tool_wear      = float(input_data.get('tool_wear', 0))

    recommendation = _build_recommendation(failure_prob, tool_wear,
                                            predicted_wear)

    return {
        'failure_prediction' : failure_pred,
        'failure_probability': round(failure_prob * 100, 2),
        'confidence'         : round(max(failure_prob, 1 - failure_prob) * 100, 2),
        'predicted_wear_min' : round(predicted_wear, 1),
        **recommendation,
        'input_data'         : input_data,
    }


def predict_batch(df: pd.DataFrame, model_dir: str = MODEL_DIR) -> pd.DataFrame:
    """
    Run batch prediction on an uploaded DataFrame.

    Parameters
    ----------
    df        : pd.DataFrame — must have the same raw columns as ai4i2020.csv
    model_dir : str

    Returns
    -------
    pd.DataFrame — original df with prediction columns appended
    """
    from src.preprocessing import clean_data, encode_labels
    from src.feature_engineering import engineer_features

    models = load_models(model_dir)

    df_clean = clean_data(df.copy())
    df_enc, _ = encode_labels(df_clean, fit=False, encoder=models['encoder'])
    df_feat   = engineer_features(df_enc)

    feature_cols = models['feature_cols']
    available    = [c for c in feature_cols if c in df_feat.columns]
    X_scaled     = models['scaler'].transform(df_feat[available])

    clf = models['classifier']
    reg = models['regressor']

    preds      = clf.predict(X_scaled)
    try:
        probs = clf.predict_proba(X_scaled)[:, 1]
    except AttributeError:
        probs = preds.astype(float)

    wear_preds = reg.predict(X_scaled)

    df_out = df_clean.copy()
    df_out['Failure_Prediction']  = preds
    df_out['Failure_Probability'] = (probs * 100).round(2)
    df_out['Predicted_Wear_min']  = wear_preds.round(1)

    # Derive health status
    statuses = []
    for prob, wear, pred_wear in zip(probs, df_out.get('Tool wear', [0]*len(df_out)),
                                      wear_preds):
        rec = _build_recommendation(prob, float(wear), float(pred_wear))
        statuses.append(rec['status'])
    df_out['Health_Status'] = statuses

    return df_out
