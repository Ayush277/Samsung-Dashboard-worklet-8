# Applications Guide

## Case2 Sales Prediction (`Case2-Nosalesuplift(pipeline) 2/`)

Endpoints
- `GET /` — Form for single prediction and tab for batch CSV
- `POST /predict` — Single prediction based on selected fields
- `POST /batch_predict` — Upload CSV, receive JSON with download URL
- `GET /download/<filename>` — Download processed CSV with predictions

Inputs (single prediction)
- `store` (int)
- `item` (int)
- `month` (1-12)
- `day` (1-31)
- `model_choice` — Uses available models loaded from artifacts

Inputs (batch CSV)
- CSV with columns: `date`, `store`, `item`

Outputs
- Single: Renders result message (`Predicted Sales with <Model>: <value>`)
- Batch: JSON `{ success, download_url, records_processed }`

Artifacts
- `scaler.pkl` (required)
- One or more models: `lgbm_model.pkl`, `catboost_model.pkl`, `ridge_model.pkl` (optional; only loaded if present)


## Loan Default Predictor (`loan_app/`)

Endpoints
- `GET /` — Applicant details form
- `POST /predict` — Returns classification JSON

Inputs
- Numeric and categorical fields such as `interest_rate`, `EducationLevel`, `source`, `loan_purpose`, `MaritalStatus`, `Gender`, `EmploymentStatus`

Output
- JSON: `{ "prediction": 0|1, "class": "Default"|"Non-Default" }`

Artifacts (expected in `loan_app/models/`)
- `tabpfn.pkl` — Trained classifier
- `scaler.pkl` — Feature scaler
- `dummy_columns.pkl` — List of one-hot encoded training columns


## Rossmann Sales Uplift (`Sales Uplift 2/pipeline/`)

Endpoints
- `GET /` — CSV upload UI
- `POST /predict` — Accepts CSV upload, returns JSON with summary + download link
- `GET /download/<filename>` — Download predictions

Inputs (CSV)
- Required test features: Date, Store, DayOfWeek, Open, Promo, StateHoliday, SchoolHoliday, StoreType, Assortment, CompetitionDistance, etc.
- Do not include the `Sales` column; the app will predict it.

Outputs
- JSON `{ success, row_count, download_url }`
- CSV download includes original columns and `Predicted_Sales`

Artifacts
- `encoder.pkl`, `scaler.pkl`, `xgb_model.pkl` inside `Sales Uplift 2/pipeline/`
