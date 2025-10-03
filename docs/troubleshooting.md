# Troubleshooting

## Apps fail to start from the dashboard
- Open the relevant log file in `dashboard/logs/<app>.log`.
- Common issues:
  - Missing artifacts (e.g., scaler/model `.pkl` files). Place them in the app folder.
  - Version mismatch warnings when unpickling scikit-learn models. Align `scikit-learn` versions or re-export models.
  - Port already in use. Edit the port mapping in `dashboard/app.py` or free the port.

## Batch CSV upload errors
- Ensure the CSV schema matches the app requirements:
  - Case2 requires columns: `date`, `store`, `item`.
  - Rossmann requires the expected set of input features; do not include `Sales`.

## Dashboard shows “failed to start in time”
- Increase the timeout in `dashboard/app.py` (`_wait_until_up`).
- Check that the app’s root route `/` returns 200 locally when run manually.

## Flask debug/reloader issues
- The dashboard starts apps with `--no-reload` and strips some env vars to avoid FD inheritance.
- If starting an app manually, prefer:

```bash
FLASK_APP=app.py python -m flask run --host 127.0.0.1 --port 7001 --no-reload
```

## macOS specific
- If `lsof`/`kill` are needed to free ports:

```bash
lsof -ti:5050,7001,7002,7003 | xargs kill -9
```

## Virtual environment
- Always activate the same venv used for development:

```bash
source .venv/bin/activate
```
