"""
=============================================================================
train.py — CNC Predictive Maintenance Project
=============================================================================
Trains multiple ML classifiers on CNC failure data and selects the best.

Models trained
--------------
  • Decision Tree
  • Random Forest
  • Gradient Boosting
  • XGBoost
  • Support Vector Machine
  • Extra Trees
  • Linear Regression (tool wear regression)

Best model is selected by F1-score and saved via Joblib.
=============================================================================
"""

import os
import time
import json
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (RandomForestClassifier,
                                    GradientBoostingClassifier,
                                    ExtraTreesClassifier)
from sklearn.svm           import SVC
from sklearn.linear_model  import LogisticRegression, LinearRegression
from sklearn.metrics       import (accuracy_score, precision_score,
                                    recall_score, f1_score,
                                    roc_auc_score, confusion_matrix,
                                    classification_report, mean_absolute_error,
                                    mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("XGBoost not installed. Skipping XGBClassifier.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_classifier(name: str, model,
                          X_train, y_train,
                          X_test,  y_test) -> dict:
    """Fit, predict, and return metrics dict for a classifier."""
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0

    # Probability for AUC
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    # Cross-validation (F1)
    try:
        cv = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='f1', n_jobs=-1)
        cv_mean, cv_std = float(cv.mean()), float(cv.std())
    except Exception:
        cv_mean, cv_std = 0.0, 0.0

    return {
        'name'        : name,
        'model'       : model,
        'accuracy'    : accuracy_score(y_test, y_pred),
        'precision'   : precision_score(y_test, y_pred, zero_division=0),
        'recall'      : recall_score(y_test, y_pred, zero_division=0),
        'f1'          : f1_score(y_test, y_pred, zero_division=0),
        'roc_auc'     : auc,
        'cv_mean'     : cv_mean,
        'cv_std'      : cv_std,
        'train_time'  : train_time,
        'pred_time'   : pred_time,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'report'      : classification_report(y_test, y_pred, output_dict=True,
                                              zero_division=0),
        'y_pred'      : y_pred,
        'y_prob'      : y_prob if auc is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Classifier Suite
# ─────────────────────────────────────────────────────────────────────────────

def train_all_classifiers(X_train, y_train,
                          X_test,  y_test) -> list:
    """
    Train all classification models and return their metrics.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : array-like

    Returns
    -------
    list of dicts (one per model, sorted by F1 score descending)
    """
    models = [
        ('Decision Tree',
         DecisionTreeClassifier(max_depth=10, random_state=42,
                                class_weight='balanced')),

        ('Random Forest',
         RandomForestClassifier(n_estimators=200, max_depth=12,
                                class_weight='balanced',
                                random_state=42, n_jobs=-1)),

        ('Gradient Boosting',
         GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                    max_depth=5, random_state=42)),

        ('Extra Trees',
         ExtraTreesClassifier(n_estimators=200, max_depth=12,
                              class_weight='balanced',
                              random_state=42, n_jobs=-1)),

        ('SVM',
         SVC(kernel='rbf', C=10, gamma='scale',
             probability=True, class_weight='balanced',
             random_state=42)),

        ('Logistic Regression',
         LogisticRegression(C=1.0, max_iter=1000,
                            class_weight='balanced',
                            random_state=42, n_jobs=-1)),
    ]

    if HAS_XGB:
        # Compute scale_pos_weight for imbalanced data
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = neg / max(pos, 1)
        models.append(('XGBoost',
                        XGBClassifier(n_estimators=200, max_depth=6,
                                      learning_rate=0.1,
                                      scale_pos_weight=spw,
                                      use_label_encoder=False,
                                      eval_metric='logloss',
                                      random_state=42,
                                      n_jobs=-1)))

    if HAS_LGB:
        models.append(('LightGBM',
                        LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                       class_weight='balanced',
                                       random_state=42, n_jobs=-1,
                                       verbose=-1)))

    results = []
    for name, clf in models:
        print(f"   🔄 Training {name}…", end=' ', flush=True)
        try:
            res = _evaluate_classifier(name, clf, X_train, y_train,
                                        X_test, y_test)
            auc_str = f"{res['roc_auc']:.4f}" if res['roc_auc'] else "N/A"
            print(f"F1={res['f1']:.4f} | AUC={auc_str}")
            results.append(res)
        except Exception as exc:
            print(f"⚠ FAILED: {exc}")

    # Sort by F1
    results.sort(key=lambda r: r['f1'], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Regression Model (Tool Wear Prediction)
# ─────────────────────────────────────────────────────────────────────────────

def train_regression_model(X_train, y_train,
                            X_test,  y_test) -> dict:
    """
    Train a Random Forest Regressor to predict continuous tool wear value.

    Returns
    -------
    dict with model + metrics
    """
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=200, max_depth=15,
                                   random_state=42, n_jobs=-1)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"   📐 Regressor → MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")

    return {
        'model'      : model,
        'mae'        : mae,
        'rmse'       : rmse,
        'r2'         : r2,
        'train_time' : train_time,
        'y_pred'     : y_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save everything
# ─────────────────────────────────────────────────────────────────────────────

def save_models(best_clf_result: dict,
                reg_result: dict,
                feature_cols: list,
                model_dir: str = MODEL_DIR) -> None:
    """
    Save best classifier, regressor, and metadata to disk.

    Parameters
    ----------
    best_clf_result : dict   — top classifier result dict
    reg_result      : dict   — regression result dict
    feature_cols    : list   — feature column names
    model_dir       : str    — output directory
    """
    os.makedirs(model_dir, exist_ok=True)

    # Best classifier
    joblib.dump(best_clf_result['model'],
                os.path.join(model_dir, 'best_classifier.joblib'))

    # Regressor
    joblib.dump(reg_result['model'],
                os.path.join(model_dir, 'tool_wear_regressor.joblib'))

    # Feature list
    with open(os.path.join(model_dir, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)

    # Serialisable metrics snapshot (strip non-JSON-safe values)
    safe_metrics = {
        'best_model'  : best_clf_result['name'],
        'accuracy'    : best_clf_result['accuracy'],
        'precision'   : best_clf_result['precision'],
        'recall'      : best_clf_result['recall'],
        'f1'          : best_clf_result['f1'],
        'roc_auc'     : best_clf_result['roc_auc'],
        'cv_mean'     : best_clf_result['cv_mean'],
        'cv_std'      : best_clf_result['cv_std'],
        'regressor_r2': reg_result['r2'],
        'regressor_mae': reg_result['mae'],
        'regressor_rmse': reg_result['rmse'],
    }
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(safe_metrics, f, indent=2)

    print(f"✅ Models saved to {model_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Full Training Pipeline (entry point)
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(data_dict: dict, model_dir: str = MODEL_DIR) -> dict:
    """
    Run the full model training pipeline.

    Parameters
    ----------
    data_dict  : dict — output from preprocessing.preprocess()
    model_dir  : str  — where to save models

    Returns
    -------
    dict with 'classifiers', 'best', 'regressor'
    """
    X_train = data_dict['X_train']
    X_test  = data_dict['X_test']
    y_train_cls = data_dict['y_train_cls']
    y_test_cls  = data_dict['y_test_cls']
    y_train_reg = data_dict['y_train_reg']
    y_test_reg  = data_dict['y_test_reg']
    feature_cols = data_dict['feature_cols']

    print("\n📊 Training Classification Models…")
    clf_results = train_all_classifiers(X_train, y_train_cls,
                                         X_test,  y_test_cls)

    print("\n📐 Training Regression Model…")
    reg_result = train_regression_model(X_train, y_train_reg,
                                         X_test,  y_test_reg)

    best = clf_results[0]
    print(f"\n🏆 Best Model: {best['name']}  F1={best['f1']:.4f}")

    save_models(best, reg_result, feature_cols, model_dir)

    return {
        'classifiers': clf_results,
        'best'       : best,
        'regressor'  : reg_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from src.preprocessing import load_dataset, preprocess

    dataset_path = os.path.join(os.path.dirname(__file__),
                                '..', 'dataset', 'ai4i2020.csv')
    df       = load_dataset(dataset_path)
    data     = preprocess(df)
    results  = run_training_pipeline(data)
    print("\n✅ Training complete!")
