"""
=============================================================================
preprocessing.py — CNC Predictive Maintenance Project
=============================================================================
Handles:
  • Dataset loading and inspection
  • Missing value handling
  • Duplicate removal
  • Label encoding
  • Feature scaling
  • Outlier detection (IQR method)
  • Train-test split
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'ai4i2020.csv')
MODEL_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')

FEATURE_COLS = [
    'Type_encoded',
    'Air temperature',
    'Process temperature',
    'Rotational speed',
    'Torque',
    'Tool wear',
    'Power',
    'Temp_diff',
    'Torque_speed_ratio',
    'Wear_rate',
]

TARGET_COL   = 'Machine failure'    # classification target
WEAR_COL     = 'Tool wear'          # regression target


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the CNC dataset from CSV.

    Parameters
    ----------
    path : str
        Path to ai4i2020.csv

    Returns
    -------
    pd.DataFrame
        Raw dataframe
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Please place ai4i2020.csv inside the dataset/ folder."
        )
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded — Shape: {df.shape}")
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Return a summary dictionary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Inspection report with shape, dtypes, nulls, duplicates, statistics
    """
    report = {
        'shape'       : df.shape,
        'columns'     : list(df.columns),
        'dtypes'      : df.dtypes.to_dict(),
        'null_counts' : df.isnull().sum().to_dict(),
        'duplicates'  : int(df.duplicated().sum()),
        'describe'    : df.describe(include='all'),
        'value_counts': {col: df[col].value_counts().to_dict()
                         for col in df.select_dtypes(include='object').columns},
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataset:
      1. Remove duplicate rows
      2. Drop rows where all values are NaN
      3. Forward-fill remaining NaN values (sensor continuity)
      4. Drop the 'UDI' and 'Product ID' columns if present (not useful for ML)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    original_len = len(df)

    # 1. Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"   🗑  Removed {original_len - len(df)} duplicate rows.")

    # 2. Drop completely empty rows
    df = df.dropna(how='all').reset_index(drop=True)

    # 3. Forward-fill remaining NaN
    df = df.ffill().bfill()
    
    # Rename columns to remove bracketed units for standardisation
    rename_map = {
        'Air temperature [K]': 'Air temperature',
        'Process temperature [K]': 'Process temperature',
        'Rotational speed [rpm]': 'Rotational speed',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Tool wear'
    }
    df = df.rename(columns=rename_map)

    # 4. Drop identifier columns if present
    drop_cols = [c for c in ['UDI', 'Product ID'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"   🗑  Dropped non-feature columns: {drop_cols}")

    print(f"✅ Data cleaned — Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Outlier Detection (IQR)
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, factor: float = 3.0) -> pd.Series:
    """
    Detect outlier rows using the IQR method across all numeric columns.

    Parameters
    ----------
    df     : pd.DataFrame
    factor : float
        Multiplier for IQR range (default 3.0 to keep manufacturing variability)

    Returns
    -------
    pd.Series (bool)
        True where row is an outlier in at least one column
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_mask = pd.Series(False, index=df.index)

    for col in numeric_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        outlier_mask |= (df[col] < lower) | (df[col] > upper)

    return outlier_mask


# ─────────────────────────────────────────────────────────────────────────────
# Label Encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(df: pd.DataFrame, fit: bool = True,
                  encoder: LabelEncoder = None) -> tuple:
    """
    Encode the 'Type' column (L/M/H) to integers.

    Parameters
    ----------
    df      : pd.DataFrame
    fit     : bool
        True to fit a new encoder, False to reuse existing
    encoder : LabelEncoder
        Pre-fitted encoder (required when fit=False)

    Returns
    -------
    (pd.DataFrame, LabelEncoder)
    """
    df = df.copy()
    if 'Type' in df.columns:
        if fit:
            encoder = LabelEncoder()
            df['Type_encoded'] = encoder.fit_transform(df['Type'].astype(str))
        else:
            df['Type_encoded'] = encoder.transform(df['Type'].astype(str))
        print(f"   🔡 Type classes: {list(encoder.classes_)}")
    return df, encoder


# ─────────────────────────────────────────────────────────────────────────────
# Full Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame,
               scaler: StandardScaler  = None,
               encoder: LabelEncoder   = None,
               fit: bool               = True,
               test_size: float        = 0.2,
               random_state: int       = 42) -> dict:
    """
    Full preprocessing pipeline.

    Steps
    -----
    1. Clean data
    2. Label encode 'Type'
    3. Feature engineering (imported from feature_engineering.py)
    4. Outlier removal (optional, kept conservative)
    5. Scale features
    6. Train-test split

    Parameters
    ----------
    df           : pd.DataFrame  — Raw dataframe
    scaler       : StandardScaler — Pre-fitted scaler (when fit=False)
    encoder      : LabelEncoder   — Pre-fitted encoder (when fit=False)
    fit          : bool           — Fit new scaler/encoder if True
    test_size    : float          — Fraction for test set
    random_state : int

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train_cls, y_test_cls,
        y_train_reg, y_test_reg, scaler, encoder,
        feature_cols, df_processed
    """
    from src.feature_engineering import engineer_features

    # ── 1. Clean ──────────────────────────────────────────────────────────────
    df = clean_data(df)

    # ── 2. Encode Type ────────────────────────────────────────────────────────
    df, encoder = encode_labels(df, fit=fit, encoder=encoder)

    # ── 3. Feature engineering ────────────────────────────────────────────────
    df = engineer_features(df)

    # ── 4. Select features ────────────────────────────────────────────────────
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].copy()
    y_cls = df[TARGET_COL].values  # classification (0 / 1)
    y_reg = df[WEAR_COL].values    # regression (tool wear in minutes)

    # ── 5. Scale ──────────────────────────────────────────────────────────────
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=available_features)

    # ── 6. Train-test split ────────────────────────────────────────────────────
    (X_train, X_test,
     y_train_cls, y_test_cls,
     y_train_reg, y_test_reg) = train_test_split(
        X_scaled, y_cls, y_reg,
        test_size=test_size,
        random_state=random_state,
        stratify=y_cls
    )

    print(f"✅ Preprocessing done — Train: {len(X_train)} | Test: {len(X_test)}")

    return {
        'X_train'      : X_train,
        'X_test'       : X_test,
        'y_train_cls'  : y_train_cls,
        'y_test_cls'   : y_test_cls,
        'y_train_reg'  : y_train_reg,
        'y_test_reg'   : y_test_reg,
        'scaler'       : scaler,
        'encoder'      : encoder,
        'feature_cols' : available_features,
        'df_processed' : df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load artefacts
# ─────────────────────────────────────────────────────────────────────────────

def save_preprocessors(scaler: StandardScaler,
                        encoder: LabelEncoder,
                        model_dir: str = MODEL_DIR) -> None:
    """Persist scaler and encoder to disk."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler,  os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(encoder, os.path.join(model_dir, 'label_encoder.joblib'))
    print(f"✅ Preprocessors saved to {model_dir}/")


def load_preprocessors(model_dir: str = MODEL_DIR) -> tuple:
    """Load saved scaler and encoder."""
    scaler  = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
    return scaler, encoder
