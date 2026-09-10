"""
start.py — Production startup launcher for Railway / Cloud deployment
Resolves port dynamically, binds to 0.0.0.0, and runs Streamlit safely.
"""

import os
import sys
import subprocess

def main():
    raw_port = os.environ.get("PORT", "8501")
    try:
        port = str(int(raw_port))
    except (ValueError, TypeError):
        print(f"[STARTUP WARNING] PORT environment variable '{raw_port}' is not an integer. Defaulting to 8501.")
        port = "8501"

    print(f"[STARTUP] Launching CNC Predictive Maintenance Streamlit app on 0.0.0.0:{port} ...")
    sys.stdout.flush()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        port,
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]

    # Run and exit with the process return code
    exit_code = subprocess.call(cmd)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
