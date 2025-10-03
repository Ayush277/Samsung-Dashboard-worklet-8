import os
import sys
import time
import socket
import subprocess
from urllib.request import urlopen
from urllib.error import URLError
from flask import Flask, render_template, redirect, url_for, jsonify

app = Flask(__name__)

# Workspace root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Apps to manage (run via `flask run` so we can set ports without touching app code)
APPS = {
    "case2": {
        "name": "No sales prediction",
        "cwd": os.path.join(ROOT_DIR, "Case2-Nosalesuplift(pipeline) 2"),
        "port": 7001,
    },
    "loan": {
        "name": "Loan predictor",
        "cwd": os.path.join(ROOT_DIR, "loan_app"),
        "port": 7002,
    },
    "rossmann": {
        "name": "Sales prediction",
        "cwd": os.path.join(ROOT_DIR, "Sales Uplift 2", "pipeline"),
        "port": 7003,
    },
}

# Track subprocesses
PROCS: dict[str, subprocess.Popen] = {}

# Logs directory
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _log_path(app_id: str) -> str:
    return os.path.join(LOG_DIR, f"{app_id}.log")


def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except Exception:
            return False


def _wait_until_up(port: int, timeout: float = 40.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if _is_port_open(port):
            time.sleep(0.5)
            try:
                resp = urlopen(f"http://127.0.0.1:{port}/", timeout=2)
                if getattr(resp, "status", 200) < 500:
                    return True
            except Exception:
                pass
        time.sleep(0.5)
    return False


def _start_app(app_id: str) -> tuple[bool, str]:
    cfg = APPS[app_id]
    port = cfg["port"]

    # If already running, don't start again
    if _is_port_open(port):
        return True, "already running"

    # Build environment and command for Flask CLI
    env = os.environ.copy()
    env["FLASK_APP"] = "app.py"
    # Do NOT enable debug in child apps from here; avoid reloader/FD shenanigans
    env["FLASK_DEBUG"] = "0"
    # Remove inherited Werkzeug/Flask runner env from the dashboard server
    for k in env.copy().keys():
        if k.startswith("WERKZEUG_") or k in ("FLASK_RUN_FROM_CLI",):
            env.pop(k, None)
    # Ensure module imports resolve within the app folder and workspace
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [cfg["cwd"], ROOT_DIR, env.get("PYTHONPATH", "")]))

    cmd = [sys.executable, "-m", "flask", "run", f"--port={port}", "--host=127.0.0.1", "--no-reload"]

    log_file_path = _log_path(app_id)
    # Open log file in append mode and truncate if large
    try:
        if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 2_000_000:
            os.remove(log_file_path)
        log_fh = open(log_file_path, "a", buffering=1)
        log_fh.write("\n" + "=" * 80 + f"\nStarting {app_id} at {time.ctime()} on port {port}\n" + "=" * 80 + "\n")
    except Exception as e:
        log_fh = subprocess.DEVNULL

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cfg["cwd"],
            env=env,
            stdout=log_fh,
            stderr=log_fh,
        )
        PROCS[app_id] = proc
    except Exception as e:
        return False, f"failed to spawn: {e}"

    # Wait for server
    ok = _wait_until_up(port, timeout=40)
    if not ok:
        # Check if process is still alive
        if proc.poll() is not None:
            return False, f"process died (exit code: {proc.returncode}). See logs: {log_file_path}"
        return False, f"failed to start in time. See logs: {log_file_path}"
    return True, "started"


def _stop_app(app_id: str) -> tuple[bool, str]:
    proc = PROCS.get(app_id)
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return True, "stopped"
        except Exception as e:
            return False, str(e)
    return True, "not running"


@app.route("/")
def index():
    status = {}
    for key, cfg in APPS.items():
        status[key] = {
            "name": cfg["name"],
            "port": cfg["port"],
            "running": _is_port_open(cfg["port"]),
        }
    return render_template("index.html", status=status)


@app.route("/open/<app_id>")
def open_app(app_id: str):
    if app_id not in APPS:
        return redirect(url_for("index"))

    ok, msg = _start_app(app_id)
    if not ok:
        # Try to read last 60 lines of the log for display
        tail = ""
        try:
            with open(_log_path(app_id), "r", errors="ignore") as fh:
                lines = fh.readlines()
                tail = "".join(lines[-60:])
        except Exception:
            pass
        return render_template(
            "view.html",
            app_name=APPS[app_id]["name"],
            app_url=None,
            error=f"Unable to start app: {msg}\n\n{tail}"
        )

    port = APPS[app_id]["port"]
    return render_template(
        "view.html",
        app_name=APPS[app_id]["name"],
        app_url=f"http://127.0.0.1:{port}/",
        error=None,
    )


@app.route("/api/start/<app_id>", methods=["POST"])
def api_start(app_id: str):
    if app_id not in APPS:
        return jsonify({"ok": False, "error": "unknown app"}), 404
    ok, msg = _start_app(app_id)
    return jsonify({"ok": ok, "message": msg, "port": APPS[app_id]["port"]})


@app.route("/api/stop/<app_id>", methods=["POST"])
def api_stop(app_id: str):
    if app_id not in APPS:
        return jsonify({"ok": False, "error": "unknown app"}), 404
    ok, msg = _stop_app(app_id)
    return jsonify({"ok": ok, "message": msg})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
