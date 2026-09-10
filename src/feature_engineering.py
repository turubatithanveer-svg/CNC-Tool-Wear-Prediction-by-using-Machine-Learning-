"""
=============================================================================
feature_engineering.py — CNC Predictive Maintenance Project
=============================================================================
Creates domain-specific features from raw CNC sensor readings:
  • Power (Torque × Rotational speed)
  • Temperature differential
  • Torque-to-speed ratio
  • Wear rate estimation
  • Failure mode flags
=============================================================================
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create physically meaningful derived features for CNC tool wear prediction.

    Derived features
    ----------------
    Power              — Mechanical power = Torque × (2π × RPM / 60)  [W]
    Temp_diff          — Process temp − Air temp  [K]
    Torque_speed_ratio — Torque / Rotational speed
    Wear_rate          — Tool wear / (Rotational speed × Torque / 1000)
    TWF_flag           — Already in dataset: Tool Wear Failure
    HDF_flag           — Heat Dissipation Failure
    PWF_flag           — Power Failure
    OSF_flag           — Overstrain Failure

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (must contain columns from ai4i2020.csv)

    Returns
    -------
    pd.DataFrame
        Dataframe with new columns appended
    """
    df = df.copy()

    # ── 1. Mechanical Power (W) ───────────────────────────────────────────────
    # P = T × ω  where ω (rad/s) = 2π × RPM / 60
    df['Power'] = df['Torque'] * (2 * np.pi * df['Rotational speed'] / 60)

    # ── 2. Temperature Differential (K) ──────────────────────────────────────
    df['Temp_diff'] = df['Process temperature'] - df['Air temperature']

    # ── 3. Torque-to-Speed Ratio ──────────────────────────────────────────────
    df['Torque_speed_ratio'] = df['Torque'] / (df['Rotational speed'] + 1e-6)

    # ── 4. Wear Rate (tool wear per unit energy) ──────────────────────────────
    df['Wear_rate'] = df['Tool wear'] / (df['Power'] / 1000 + 1e-6)

    return df


def get_feature_descriptions() -> dict:
    """
    Return human-readable descriptions for each feature, including engineered ones.

    Returns
    -------
    dict : {column_name: description_string}
    """
    return {
        'Type_encoded'       : 'Machine type encoded (L=0, M=1, H=2)',
        'Air temperature'    : 'Ambient air temperature (K)',
        'Process temperature': 'Machining process temperature (K)',
        'Rotational speed'   : 'Spindle rotational speed (RPM)',
        'Torque'             : 'Cutting torque applied (Nm)',
        'Tool wear'          : 'Accumulated tool wear time (min)',
        'Power'              : 'Mechanical power = Torque × ω (W)',
        'Temp_diff'          : 'Process − Air temperature differential (K)',
        'Torque_speed_ratio' : 'Torque / Rotational speed',
        'Wear_rate'          : 'Tool wear per unit power (min/kW)',
        'Machine failure'    : 'Target: 1 = Failure, 0 = Normal',
        'TWF'                : 'Tool Wear Failure indicator',
        'HDF'                : 'Heat Dissipation Failure indicator',
        'PWF'                : 'Power Failure indicator',
        'OSF'                : 'Overstrain Failure indicator',
        'RNF'                : 'Random Noise Failure indicator',
    }
