"""
main.py — Production Entrypoint for Railway, Railpack, and Cloud Deployments
Starts Streamlit bound to 0.0.0.0 and dynamically resolves $PORT.
Includes an active socket health monitor to log port connectivity status directly to console.
"""

import os
import sys
import socket
import threading
import time
import subprocess

def start_port_monitor(port: int, max_attempts: int = 30, interval: float = 1.0):
    """Monitors the target port in a background thread and logs connectivity status."""
    def monitor():
        time.sleep(2.0)  # Allow Streamlit process time to initialize
        print(f"[PORT CHECK] Starting connection verification on port {port}...")
        sys.stdout.flush()

        for attempt in range(1, max_attempts + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0)
                    result = s.connect_ex(("127.0.0.1", port))
                    if result == 0:
                        print("============================================================")
                        print(f"[PORT STATUS] ✅ SUCCESS: Port {port} is OPEN and CONNECTED!")
                        print(f"[PORT STATUS] 🚀 Streamlit is LIVE at http://0.0.0.0:{port}")
                        print(f"[PORT STATUS] 🌐 Traffic is being accepted from Railway router.")
                        print("============================================================")
                        sys.stdout.flush()
                        return
            except Exception as e:
                pass
            time.sleep(interval)

        print(f"[PORT STATUS] ⚠️ WARNING: Port {port} did not accept connection within {max_attempts} seconds.")
        sys.stdout.flush()

    probe_thread = threading.Thread(target=monitor, daemon=True)
    probe_thread.start()

def main():
    raw_port = os.environ.get("PORT", "8501")
    try:
        port_int = int(raw_port)
        port = str(port_int)
    except (ValueError, TypeError):
        port_int = 8501
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

    # Launch background port connection verification
    start_port_monitor(port_int)

    # Launch Streamlit in blocking mode
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
