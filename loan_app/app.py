from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import logging
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load artifacts lazily to avoid import-time crashes
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
scaler = None
dummy_columns = None
TRAINING_DONE = False
MEDIANS = {}
STDS = {}
FEATURE_GROUP_IMPORTANCE = {}

ARTIFACT_ERR = None

MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Expanded numeric feature list (subset of available columns)
NUMERIC_COLS = [
    'interest_rate','unpaid_principal_bal','Loan_term','loan_to_value','number_of_borrowers',
    'debt_to_income_ratio','borrower_credit_score','insurance_percent','co-borrower_credit_score',
    'Age','NumberOfDependents','Annual_Income','total_on_time_payments','total_late_payments',
    'avg_payment_delay','current_dpd'
]

FEATURE_NUMERIC = ['interest_rate']  # kept for backward compatibility, superseded by NUMERIC_COLS

CATEGORICAL_COLS = ['source', 'loan_purpose', 'EducationLevel', 
                   'MaritalStatus', 'Gender', 'EmploymentStatus']

MEDIANS_PATH = os.path.join(MODEL_DIR, 'medians.json')  # now stores medians + stds

# Risk level thresholds (probability of delinquency)
RISK_THRESHOLDS = [0.25, 0.50, 0.75]  # low, moderate, high, critical
RISK_LABELS = ["Low", "Moderate", "High", "Critical"]

# Utility to standardize column names (train csv has spaces / casing inconsistencies)
_DEF_RENAMES = {
    'Annual Income': 'Annual_Income',
    'Occupation ': 'Occupation',
}

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip())
    for k, v in _DEF_RENAMES.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    return df

def _train_artifacts_if_missing():
    global model, scaler, dummy_columns, ARTIFACT_ERR, TRAINING_DONE, MEDIANS, STDS
    if TRAINING_DONE:
        return
    missing_any = any(
        not os.path.exists(os.path.join(MODEL_DIR, fname))
        for fname in ['tabpfn.pkl', 'scaler.pkl', 'dummy_columns.pkl', 'medians.json']
    )
    if missing_any:
        try:
            logging.warning("Loan model artifacts missing – training enhanced fallback model.")
            df_path = os.path.join(BASE_DIR, 'approach_train.csv')
            df = pd.read_csv(df_path, low_memory=False)
            df = _standardize_columns(df)

            # Target
            if 'mx' not in df.columns:
                raise RuntimeError("Training data missing target column 'mx'")
            y = df['mx'].astype(int)

            # Keep only needed columns
            cols_needed = set(NUMERIC_COLS + CATEGORICAL_COLS)
            present_cols = [c for c in df.columns if c in cols_needed]
            missing_numeric = [c for c in NUMERIC_COLS if c not in df.columns]
            if missing_numeric:
                logging.warning(f"Numeric columns missing in training data and will be skipped: {missing_numeric}")
            present_numeric = [c for c in NUMERIC_COLS if c in df.columns]

            X = df[present_cols].copy()

            # Convert numeric columns
            for col in present_numeric:
                X[col] = pd.to_numeric(X[col], errors='coerce')

            # Compute medians & stds for numeric
            MEDIANS = {col: float(X[col].median()) for col in present_numeric}
            STDS = {col: float(X[col].std(ddof=0) or 1.0) for col in present_numeric}
            # Fill missing numeric
            for col, med in MEDIANS.items():
                X[col] = X[col].fillna(med)

            # Ensure categorical present (fill NA with 'Unknown')
            for c in CATEGORICAL_COLS:
                if c not in X.columns:
                    X[c] = 'Unknown'
                X[c] = X[c].fillna('Unknown')

            # One-hot encode
            X_enc = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
            dummy_columns = list(X_enc.columns)  # store order

            # Scale numeric subset only (but simpler: scale all encoded columns)
            scaler_local = StandardScaler()
            X_scaled = scaler_local.fit_transform(X_enc)

            # Model
            clf = RandomForestClassifier(
                n_estimators=160,
                max_depth=12,
                min_samples_leaf=25,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_scaled, y)

            # Persist
            joblib.dump(clf, os.path.join(MODEL_DIR, 'tabpfn.pkl'))
            joblib.dump(scaler_local, os.path.join(MODEL_DIR, 'scaler.pkl'))
            joblib.dump(dummy_columns, os.path.join(MODEL_DIR, 'dummy_columns.pkl'))
            with open(MEDIANS_PATH, 'w') as f:
                json.dump({'medians': MEDIANS, 'stds': STDS}, f)
            logging.info("Enhanced fallback loan model trained and saved.")
        except Exception as e:
            ARTIFACT_ERR = f"Auto-train failed: {e}"
            logging.exception(ARTIFACT_ERR)
        finally:
            TRAINING_DONE = True

def load_artifacts():
    global model, scaler, dummy_columns, ARTIFACT_ERR, MEDIANS, STDS, FEATURE_GROUP_IMPORTANCE
    if model is not None and scaler is not None and dummy_columns is not None and MEDIANS:
        return True
    _train_artifacts_if_missing()
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'tabpfn.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        dummy_columns = joblib.load(os.path.join(MODEL_DIR, 'dummy_columns.pkl'))
        if os.path.exists(MEDIANS_PATH):
            with open(MEDIANS_PATH, 'r') as f:
                data_json = json.load(f)
                if isinstance(data_json, dict) and 'medians' in data_json:
                    MEDIANS = data_json.get('medians', {})
                    STDS.update(data_json.get('stds', {}))
                else:
                    MEDIANS.update(data_json)
        # Build feature group importances (only once)
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            group = {}
            for col, imp in zip(dummy_columns, fi):
                base = col
                # collapse one-hots for categoricals
                for cat in CATEGORICAL_COLS:
                    if col.startswith(cat + '_'):
                        base = cat
                        break
                group[base] = group.get(base, 0.0) + float(imp)
            FEATURE_GROUP_IMPORTANCE.update(group)
        ARTIFACT_ERR = None
        return True
    except Exception as e:
        ARTIFACT_ERR = str(e)
        return False

def preprocess(input_data):
    """Replicate training preprocessing with median imputation for unseen numeric features."""
    if dummy_columns is None or scaler is None:
        raise RuntimeError('Artifacts not loaded')

    # Build full feature dict using provided form values overriding medians
    row = {}
    # Numeric features: use medians if not provided (only interest_rate is on form)
    for n in NUMERIC_COLS:
        val = input_data.get(n)
        if val is None or val == '':
            # fallback to median if available
            if n in MEDIANS:
                row[n] = MEDIANS[n]
                continue
            else:
                # skip if not in dummy columns (was missing in training)
                continue
        try:
            row[n] = float(val)
        except ValueError:
            row[n] = MEDIANS.get(n, 0.0)

    # Categorical features: pull from input (form enforces selection)
    for c in CATEGORICAL_COLS:
        row[c] = input_data.get(c, 'Unknown') or 'Unknown'

    X = pd.DataFrame([row])

    # One-hot encode
    x_encoded = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)

    # Add any missing training columns
    for col in dummy_columns:
        if col not in x_encoded.columns:
            x_encoded[col] = 0
    # Remove any extra unseen columns (should not happen with handle_unknown strategy)
    x_encoded = x_encoded[dummy_columns]

    return scaler.transform(x_encoded)

def _risk_bucket(p: float | None) -> str:
    if p is None:
        return "Unknown"
    for idx, thr in enumerate(RISK_THRESHOLDS):
        if p < thr:
            return RISK_LABELS[idx]
    return RISK_LABELS[-1]

@app.route('/')
def home():
    load_artifacts()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not load_artifacts():
            return jsonify({'error': f'Artifacts not loaded: {ARTIFACT_ERR}'}), 500
        data = request.form.to_dict()
        x_processed = preprocess(data)
        prediction = int(model.predict(x_processed)[0])
        proba = None
        try:
            proba = float(model.predict_proba(x_processed)[0][1])  # probability of delinquent class (1)
        except Exception:
            pass
        risk_level = _risk_bucket(proba)

        # Driver analysis (simple standardized deviation * group importance for numeric features only)
        drivers = []
        if proba is not None and FEATURE_GROUP_IMPORTANCE:
            for n in NUMERIC_COLS:
                if n not in MEDIANS or n not in STDS:
                    continue
                user_val = request.form.get(n)
                if user_val is None or user_val == '':
                    # user did not supply; skip explaining median-imputed values
                    continue
                try:
                    v = float(user_val)
                except ValueError:
                    continue
                std = STDS.get(n, 1.0) or 1.0
                z = (v - MEDIANS[n]) / std
                importance = FEATURE_GROUP_IMPORTANCE.get(n, 0.0)
                driver_score = abs(z) * importance
                direction = 'Higher Risk' if z > 0 else 'Lower Risk'
                drivers.append({
                    'feature': n,
                    'value': v,
                    'median': MEDIANS[n],
                    'zscore': round(z, 3),
                    'importance': round(importance, 4),
                    'driver_score': round(driver_score, 4),
                    'direction': direction
                })
            drivers.sort(key=lambda d: d['driver_score'], reverse=True)
            drivers = drivers[:5]

        risk_summary = (
            f"Predicted delinquency probability {proba:.1%} => {risk_level} risk band. "
            f"Top drivers reflect deviation from portfolio medians weighted by model importance." if proba is not None else
            "Risk probability unavailable for this model variant."
        )

        return jsonify({
            'binary_prediction': prediction,              # 1 = predicted delinquent, 0 = predicted on-time
            'delinquency_probability': proba,             # may be None if model lacks predict_proba
            'risk_level': risk_level,                     # Low / Moderate / High / Critical
            'delinquency_flag': bool(prediction),         # boolean convenience
            'risk_summary': risk_summary,
            'top_drivers': drivers,
            'explanation': 'Probability-based delinquency risk classification with simple driver analysis.'
        })
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=True, host='0.0.0.0', port=port)