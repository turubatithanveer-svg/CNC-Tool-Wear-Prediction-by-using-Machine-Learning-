@echo off
title CNC Predictive Maintenance System

echo.
echo ============================================================
echo   CNC PREDICTIVE MAINTENANCE — STARTUP SCRIPT
echo ============================================================
echo.

REM Step 1: Train models (skip if models exist)
if not exist "models\best_classifier.joblib" (
    echo [1/2] Training ML models...
    python train_model.py
    echo.
) else (
    echo [1/2] Models already trained. Skipping training.
)

REM Step 2: Launch Streamlit
echo [2/2] Launching Streamlit dashboard...
echo.
echo Open your browser at: http://localhost:8501
echo.
python -m streamlit run app.py --server.port 8501 --server.headless false

pause
