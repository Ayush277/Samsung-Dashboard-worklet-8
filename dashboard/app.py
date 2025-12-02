import os
from flask import Flask, render_template, redirect, url_for, jsonify

app = Flask(__name__)

# Apps configuration for Vercel deployment
APPS = {
    "loan": {
        "name": "Loan delinquency risk",
        "url": "/loan/",
        "description": "Advanced ML-powered risk assessment for loan delinquency prediction",
        "icon": "calculator",
        "color": "primary"
    },
    "campaign": {
        "name": "Campaign performance (marketing)",
        "url": "/campaign/",
        "description": "Store performance analysis and marketing campaign optimization",
        "icon": "chart-line",
        "color": "success"
    },
    "sales": {
        "name": "Sell-out performance forecasting (sales uplift)",
        "url": "/sales/",
        "description": "AI-driven sales forecasting with uplift prediction capabilities",
        "icon": "trending-up",
        "color": "warning"
    },
}

@app.route("/")
def index():
    # On Vercel, all apps are "running"
    status = {}
    for key, cfg in APPS.items():
        status[key] = {
            "name": cfg["name"],
            "running": True,
            "port": "N/A", # Not applicable on Vercel
            "url": cfg["url"]
        }
    return render_template("index.html", status=status)


@app.route("/open/<app_id>")
def open_app(app_id: str):
    if app_id not in APPS:
        return redirect(url_for("index"))
    
    # Direct redirect to the mounted app
    return redirect(APPS[app_id]["url"])


# API routes for compatibility (mocked)
@app.route("/api/start/<app_id>", methods=["POST"])
def api_start(app_id: str):
    return jsonify({"ok": True, "message": "App is running", "port": 0})


@app.route("/api/stop/<app_id>", methods=["POST"])
def api_stop(app_id: str):
    return jsonify({"ok": True, "message": "Cannot stop apps on Vercel"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
