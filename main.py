"""
main.py — Production Entrypoint for Railway, Railpack, and Cloud Deployments
Starts Streamlit bound to 0.0.0.0 and dynamically resolves $PORT.
"""

import os
import sys
import subprocess

def main():
    raw_port = os.environ.get("PORT", "8501")
    try:
        port = str(int(raw_port))
    except (ValueError, TypeError):
        port = "8501"

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]

    print(f"[STARTUP] Starting CNC Predictive Maintenance Streamlit on 0.0.0.0:{port} ...")
    sys.stdout.flush()

    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
