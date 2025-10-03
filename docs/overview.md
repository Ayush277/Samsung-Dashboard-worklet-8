# Project Overview

Samsung Worklet 8 unifies three Flask apps under a single control plane (the dashboard). Each app remains independent and can be launched, stopped, and accessed through the dashboard UI.

## Components

- `dashboard/` — Flask dashboard that manages child apps
  - `app.py` — Starts target apps with `flask run`, monitors ports, and shows status
  - `templates/` — Dashboard views (`index.html`, `view.html`)
  - `logs/` — Child app stdout/stderr logs per app

- `Case2-Nosalesuplift(pipeline) 2/` — Single and batch sales prediction app
  - `app.py` — Loads models/scaler, exposes `/` form, `/predict`, `/batch_predict`, and `/download/<file>`
  - `templates/index.html` — UI for form and batch upload
  - `train2.csv` — For dropdowns (store, item)
  - `*.pkl` — Model artifacts (`lgbm_model.pkl`, `catboost_model.pkl`, `ridge_model.pkl`, `scaler.pkl`)

- `loan_app/` — Loan default risk classifier
  - `app.py` — Loads a classifier, scaler, and dummy columns; exposes `/` and `/predict`
  - `templates/index.html` — UI for applicant form submission
  - `models/` — Expected to contain `tabpfn.pkl`, `scaler.pkl`, and `dummy_columns.pkl` (not shown in tree if missing)

- `Sales Uplift 2/pipeline/` — Rossmann sales prediction (batch CSV)
  - `app.py` — Loads encoder/scaler/model, handles CSV upload `/predict` and `/download/<file>`
  - `config.py` — Configuration (upload folder, expected features)
  - `utils/` — Data processing, model loading, file handling
  - `templates/index.html` — CSV upload UI

## Dashboard Launch Flow

1. The dashboard exposes `GET /` listing the three apps with status and Open buttons.
2. On `GET /open/<app_id>`, it:
   - Spawns `python -m flask run --host 127.0.0.1 --port <PORT> --no-reload` in the app folder
   - Waits until the port responds to HTTP
   - Renders `view.html` with an iframe to the child app
3. On failure, the dashboard shows the log tail to help diagnose startup issues.

## Ports

- Dashboard: 5050
- Case2 Sales Prediction: 7001
- Loan Default Predictor: 7002
- Rossmann Sales Uplift: 7003

These are configurable in `dashboard/app.py` under `APPS`.

## Data and Artifacts

- Case2 requires `train2.csv` and the scaler/model pickles in its root folder.
- Loan app expects artifacts in `loan_app/models/`.
- Rossmann app expects artifacts in `Sales Uplift 2/pipeline/` alongside its code (encoder/scaler/model).

## Development Notes

- All apps run with `--no-reload` from the launcher to avoid FD inheritance issues.
- The dashboard strips `WERKZEUG_*` env and `FLASK_RUN_FROM_CLI` from children.
- The dashboard sets `PYTHONUNBUFFERED=1` for real-time logs.

## Security

- Intended for local development/demo.
- Do not expose directly to the Internet without a production WSGI server and proper hardening.
