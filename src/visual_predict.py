"""
=============================================================================
visual_predict.py — CNC Predictive Maintenance Project
=============================================================================
Visual Prediction Engine:
  • Accepts a CNC tool / machine image (numpy array) or a video file path
  • Extracts visual wear indicators using OpenCV:
      – Edge sharpness   → tool wear estimate
      – Surface texture  → torque estimate
      – Discoloration    → process temperature estimate
      – Wear area %      → rotational speed estimate
  • Maps visual features → sensor parameters → runs existing predict_single()
  • For video: samples N frames evenly, predicts each frame, returns trend list
=============================================================================
"""

import os
import tempfile
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Sensor Range Constants (domain knowledge from AI4I 2020 dataset)
# ─────────────────────────────────────────────────────────────────────────────
_AIR_TEMP_MIN,   _AIR_TEMP_MAX    = 295.0, 308.0   # Kelvin
_PROC_TEMP_MIN,  _PROC_TEMP_MAX   = 305.0, 315.0   # Kelvin
_ROT_SPEED_MIN,  _ROT_SPEED_MAX   = 1168.0, 2886.0  # RPM
_TORQUE_MIN,     _TORQUE_MAX      = 3.8, 76.6       # Nm
_TOOL_WEAR_MIN,  _TOOL_WEAR_MAX   = 0.0, 253.0      # minutes

# Number of frames to sample from a video
_VIDEO_SAMPLE_FRAMES = 12


# ─────────────────────────────────────────────────────────────────────────────
# Visual Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_image_features(image: np.ndarray) -> dict:
    """
    Extract visual wear-related features from a BGR/RGB numpy image array.

    Parameters
    ----------
    image : np.ndarray
        Image in BGR (OpenCV) or RGB format, shape (H, W, 3) or (H, W).

    Returns
    -------
    dict with keys:
        edge_sharpness    : float [0–1] — 1 = sharp (new tool), 0 = blurry (worn)
        texture_roughness : float [0–1] — 1 = rough surface, 0 = smooth
        discoloration     : float [0–1] — 1 = heavy discoloration (heat), 0 = normal
        wear_area_pct     : float [0–1] — fraction of image showing wear/dark spots
        brightness        : float [0–1] — overall image brightness
        hsv_saturation    : float [0–1] — colour saturation (rust/oxidation indicator)
    """
    import cv2

    # Ensure we have a usable image
    if image is None or image.size == 0:
        return _default_features()

    # Convert to BGR if needed (OpenCV expects BGR)
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        bgr = image.copy()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    h, w = gray.shape
    total_pixels = float(h * w)

    # ── 1. Edge Sharpness (Laplacian variance) ────────────────────────────────
    # High variance → sharp edges → new/sharp tool
    # Low variance  → blurry     → worn/dull tool
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: typically 0–3000 for CNC tool images; clamp at 3000
    edge_sharpness = float(np.clip(lap_var / 3000.0, 0.0, 1.0))

    # ── 2. Surface Texture Roughness (local std via Gaussian subtraction) ─────
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    residual = cv2.absdiff(gray, blurred).astype(np.float32)
    texture_roughness = float(np.clip(residual.mean() / 40.0, 0.0, 1.0))

    # ── 3. Discoloration Score (hue deviation from metallic grey) ────────────
    # Metallic surfaces have low saturation. Oxidation/heat → higher saturation
    # Also check for brownish/bluish hues (heat tint colours for metals)
    sat_channel   = hsv[:, :, 1].astype(np.float32) / 255.0
    hue_channel   = hsv[:, :, 0].astype(np.float32)          # 0–180 in OpenCV

    # Heat tints: blue (120°), gold (30°), brown (15°) — far from 0/180 (grey/red)
    hue_dev = np.minimum(hue_channel, 180.0 - hue_channel)   # 0 = neutral grey
    hue_dev_norm = np.clip(hue_dev.mean() / 60.0, 0.0, 1.0)
    sat_mean   = float(sat_channel.mean())
    discoloration = float(np.clip((sat_mean + hue_dev_norm) / 2.0, 0.0, 1.0))

    # ── 4. Wear Area % (very dark regions = cavities / deep wear) ────────────
    _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    wear_area_pct = float(cv2.countNonZero(dark_mask)) / total_pixels

    # ── 5. Brightness ─────────────────────────────────────────────────────────
    val_channel = hsv[:, :, 2].astype(np.float32) / 255.0
    brightness  = float(val_channel.mean())

    # ── 6. HSV Saturation ─────────────────────────────────────────────────────
    hsv_saturation = sat_mean

    return {
        'edge_sharpness'    : round(edge_sharpness,    4),
        'texture_roughness' : round(texture_roughness,  4),
        'discoloration'     : round(discoloration,      4),
        'wear_area_pct'     : round(wear_area_pct,      4),
        'brightness'        : round(brightness,         4),
        'hsv_saturation'    : round(hsv_saturation,     4),
    }


def _default_features() -> dict:
    """Return neutral/default features when extraction fails."""
    return {
        'edge_sharpness'    : 0.5,
        'texture_roughness' : 0.3,
        'discoloration'     : 0.2,
        'wear_area_pct'     : 0.1,
        'brightness'        : 0.5,
        'hsv_saturation'    : 0.15,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature → Sensor Parameter Mapping
# ─────────────────────────────────────────────────────────────────────────────

def _lerp(val: float, out_min: float, out_max: float) -> float:
    """Linear interpolation: val in [0,1] → [out_min, out_max]."""
    return out_min + float(np.clip(val, 0.0, 1.0)) * (out_max - out_min)


def visual_features_to_sensor_params(features: dict,
                                      machine_type: str = 'M') -> dict:
    """
    Map extracted visual features to estimated CNC sensor parameters.

    Mapping rationale
    -----------------
    tool_wear      ← inverse of edge_sharpness (dull tool = blurry edges)
    torque         ← texture_roughness (rough surface → higher cutting force)
    process_temp   ← discoloration (heat tints → higher temperature)
    rot_speed      ← brightness (brighter/cleaner surface → tool still cutting well)
    air_temp       ← weak proxy from overall brightness (ambient estimate)
    type_val       ← user-supplied machine type

    Parameters
    ----------
    features     : dict from extract_image_features()
    machine_type : str — 'L', 'M', or 'H'

    Returns
    -------
    dict compatible with predict_single() input_data
    """
    sharpness     = features.get('edge_sharpness',    0.5)
    roughness     = features.get('texture_roughness',  0.3)
    discolor      = features.get('discoloration',      0.2)
    wear_area     = features.get('wear_area_pct',      0.1)
    brightness    = features.get('brightness',         0.5)

    # Tool wear: 0=new (sharp), 1=worn (blunt) → inverse of sharpness
    # Also weighted by wear_area
    wear_score   = (1.0 - sharpness) * 0.7 + wear_area * 0.3
    tool_wear    = _lerp(wear_score, _TOOL_WEAR_MIN, _TOOL_WEAR_MAX)

    # Torque: rough surface requires more force
    torque       = _lerp(roughness, _TORQUE_MIN, _TORQUE_MAX)

    # Process temperature: discoloration indicates heat buildup
    process_temp = _lerp(discolor, _PROC_TEMP_MIN, _PROC_TEMP_MAX)

    # Rotational speed: high brightness → cleaner surface → speed OK
    rot_speed    = _lerp(brightness, _ROT_SPEED_MIN, _ROT_SPEED_MAX)

    # Air temperature: small ambient variation, proxy from brightness midpoint
    air_temp     = _lerp(brightness * 0.3 + 0.35, _AIR_TEMP_MIN, _AIR_TEMP_MAX)

    return {
        'type_val'    : machine_type,
        'air_temp'    : round(air_temp,    2),
        'process_temp': round(process_temp, 2),
        'rot_speed'   : round(rot_speed,   1),
        'torque'      : round(torque,       2),
        'tool_wear'   : round(tool_wear,    1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_from_image(image: np.ndarray,
                        machine_type: str = 'M',
                        model_dir: str | None = None) -> dict:
    """
    Run the full visual-prediction pipeline on a single image.

    Parameters
    ----------
    image        : np.ndarray  — BGR/RGB image
    machine_type : str         — 'L', 'M', or 'H'
    model_dir    : str | None  — path to model directory (uses default if None)

    Returns
    -------
    dict with keys:
        visual_features  : dict  — raw visual metrics
        sensor_params    : dict  — estimated sensor values
        prediction       : dict  — full result from predict_single()
    """
    from src.predict import predict_single, MODEL_DIR as _DEFAULT_MODEL_DIR

    md = model_dir or _DEFAULT_MODEL_DIR

    visual_features = extract_image_features(image)
    sensor_params   = visual_features_to_sensor_params(visual_features, machine_type)
    prediction      = predict_single(sensor_params, model_dir=md)

    return {
        'visual_features' : visual_features,
        'sensor_params'   : sensor_params,
        'prediction'      : prediction,
    }


def predict_from_video(video_bytes: bytes,
                        machine_type: str = 'M',
                        model_dir: str | None = None,
                        n_frames: int = _VIDEO_SAMPLE_FRAMES) -> list:
    """
    Run visual prediction on evenly-sampled frames from a video.

    Parameters
    ----------
    video_bytes  : bytes  — raw video file bytes
    machine_type : str    — 'L', 'M', or 'H'
    model_dir    : str    — path to model directory
    n_frames     : int    — number of frames to sample

    Returns
    -------
    list of dicts, each with keys: frame_index, timestamp_sec, visual_features,
                                    sensor_params, prediction
    """
    import cv2

    # Write bytes to a temp file so OpenCV can open it
    suffix = '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    results = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file. Try a different format (MP4, AVI).")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total_frames <= 0:
            raise RuntimeError("Video has no readable frames.")

        sample_count = min(n_frames, total_frames)
        indices      = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            timestamp = round(int(idx) / fps, 2)
            result    = predict_from_image(frame, machine_type, model_dir)
            result['frame_index']    = int(idx)
            result['timestamp_sec']  = timestamp
            results.append(result)

        cap.release()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return results


def get_wear_label(features: dict) -> str:
    """Return a human-readable wear label from visual features."""
    sharpness = features.get('edge_sharpness', 0.5)
    wear_area = features.get('wear_area_pct', 0.1)
    score     = (1.0 - sharpness) * 0.7 + wear_area * 0.3

    if score < 0.25:
        return '✅ Visually New / Minimal Wear'
    elif score < 0.50:
        return '⚠️ Moderate Visual Wear'
    elif score < 0.75:
        return '🔴 Heavy Visual Wear Detected'
    else:
        return '🚨 Severe Wear — Critical Condition'
