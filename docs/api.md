# API Reference

This reference lists the HTTP endpoints across all apps.

## Dashboard (`dashboard/`)

- `GET /` — List apps, status, and Open buttons.
- `GET /open/<app_id>` — Launch app if needed, then embed it.
- `POST /api/start/<app_id>` — Programmatic start.
- `POST /api/stop/<app_id>` — Programmatic stop.

App IDs
- `case2` — Case2 Sales Prediction
- `loan` — Loan Default Predictor
- `rossmann` — Rossmann Sales Uplift


## Case2 Sales Prediction

- `GET /` — HTML page with tabs for single/batch prediction.
- `POST /predict`
  - Form fields: `store`, `item`, `month`, `day`, `model_choice`
  - Returns HTML page with prediction text.
- `POST /batch_predict`
  - Multipart with file: `file`
  - Returns JSON `{ success, download_url, records_processed }`
- `GET /download/<filename>` — CSV download.


## Loan Default Predictor

- `GET /` — HTML form.
- `POST /predict`
  - Body: application/x-www-form-urlencoded
  - Returns JSON `{ prediction, class }`.


## Rossmann Sales Uplift

- `GET /` — HTML uploader.
- `POST /predict`
  - Multipart with file: `file`
  - Returns JSON `{ success, row_count, download_url }`.
- `GET /download/<filename>` — CSV download.
