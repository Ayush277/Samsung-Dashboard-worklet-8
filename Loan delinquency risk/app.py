# CRITICAL NOTE: In the training data, the target variable 'mx' is inverted:
# mx=1 = GOOD LOAN (non-default), mx=0 = BAD LOAN (default)
# This is contrary to typical ML convention, so predictions are inverted in the code.

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

# Complete numeric feature list based on CSV analysis (EXACT COLUMN NAMES)
NUMERIC_COLS = [
    'interest_rate', 'unpaid_principal_bal', 'Loan_term', 'loan_to_value', 'number_of_borrowers',
    'debt_to_income_ratio', 'borrower_credit_score', 'insurance_percent', 'co-borrower_credit_score',
    'Age', 'NumberOfDependents', 'Annual Income', 'total_on_time_payments', 'total_late_payments',
    'avg_payment_delay', 'current_dpd'
]

# Categorical columns with proper mapping from CSV (EXACT COLUMN NAMES)
CATEGORICAL_COLS = ['source', 'loan_purpose', 'EducationLevel', 'MaritalStatus', 'Gender', 'EmploymentStatus']

# Expected value mappings from training data (EXACT VALUES FROM CSV)
EXPECTED_VALUES = {
    'source': ['X', 'Y', 'Z'],
    'loan_purpose': ['A23', 'B12', 'C86'], 
    'EducationLevel': ['Bachelor\'s', 'Doctorate', 'High School', 'Master\'s', 'PhD'],
    'MaritalStatus': ['Divorced', 'Married', 'Single'],
    'Gender': ['Female', 'Male', 'Other'],
    'EmploymentStatus': ['Employed', 'Self-Employed', 'Unemployed']
}

MEDIANS_PATH = os.path.join(MODEL_DIR, 'medians.json')  # now stores medians + stds

# Risk level thresholds (probability of delinquency)
RISK_THRESHOLDS = [0.25, 0.50, 0.75]  # low, moderate, high, critical
RISK_LABELS = ["Low", "Moderate", "High", "Critical"]

# Utility to standardize column names (train csv has spaces / casing inconsistencies)
_DEF_RENAMES = {
    # Keep exact column names as in CSV - no renaming needed for core features
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
    """Replicate training preprocessing with comprehensive feature handling."""
    if dummy_columns is None or scaler is None:
        raise RuntimeError('Artifacts not loaded')

    # Build comprehensive feature dict
    row = {}
    
    # Process numeric features with validation and median fallback
    for n in NUMERIC_COLS:
        val = input_data.get(n)
        if val is None or val == '' or val == 'None':
            # Use median if available, otherwise skip
            if n in MEDIANS:
                row[n] = MEDIANS[n]
            continue
        try:
            numeric_val = float(val)
            # Basic validation for reasonable ranges
            if n == 'borrower_credit_score' or n == 'co-borrower_credit_score':
                numeric_val = max(300, min(850, numeric_val))  # FICO score range
            elif n == 'interest_rate':
                numeric_val = max(0, min(50, numeric_val))  # Reasonable interest rate
            elif n == 'Age':
                numeric_val = max(18, min(100, numeric_val))  # Valid age range
            elif n in ['loan_to_value', 'insurance_percent']:
                numeric_val = max(0, min(100, numeric_val))  # Percentage fields
            elif n in ['total_on_time_payments', 'total_late_payments', 'current_dpd']:
                numeric_val = max(0, numeric_val)  # Non-negative counts
            
            row[n] = numeric_val
        except (ValueError, TypeError):
            # Fall back to median if conversion fails
            row[n] = MEDIANS.get(n, 0.0)

    # Process categorical features with validation
    for c in CATEGORICAL_COLS:
        val = input_data.get(c, 'Unknown')
        if val is None or val == '':
            val = 'Unknown'
        
        # Validate against expected values
        if c in EXPECTED_VALUES and val not in EXPECTED_VALUES[c]:
            # Map common variations or default to most common value
            if c == 'source' and val not in ['X', 'Y', 'Z']:
                val = 'X'  # Default to most common
            elif c == 'loan_purpose' and val not in ['A23', 'B12', 'C86']:
                val = 'A23'  # Default to most common
            else:
                val = EXPECTED_VALUES[c][0] if EXPECTED_VALUES[c] else 'Unknown'
        
        row[c] = val

    # Create DataFrame
    X = pd.DataFrame([row])

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)

    # Ensure all training columns are present
    for col in dummy_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    
    # Keep only training columns in correct order
    X_encoded = X_encoded[dummy_columns]

    return scaler.transform(X_encoded), row

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

@app.route('/comprehensive')
def comprehensive():
    load_artifacts()
    return render_template('index_new.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not load_artifacts():
            return jsonify({'error': f'Artifacts not loaded: {ARTIFACT_ERR}'}), 500
        
        data = request.form.to_dict()
        x_processed, processed_row = preprocess(data)
          # Model predictions
        # NOTE: In training data, mx=1 means GOOD loan, mx=0 means BAD loan
        # So we need to invert the prediction for default probability
        raw_prediction = int(model.predict(x_processed)[0])
        prediction = 1 - raw_prediction  # Invert: 0=good, 1=default
        
        proba = None
        try:
            proba_array = model.predict_proba(x_processed)[0]
            # proba_array[1] is P(mx=1) = P(good loan)
            # We want P(default) = P(mx=0) = proba_array[0]
            proba = float(proba_array[0])  # probability of default (mx=0)
        except Exception as e:
            logging.warning(f"Probability calculation failed: {e}")
        
        risk_level = _risk_bucket(proba)

        # Enhanced driver analysis
        drivers = []
        if proba is not None and FEATURE_GROUP_IMPORTANCE:
            for n in NUMERIC_COLS:
                if n not in MEDIANS or n not in STDS:
                    continue
                
                # Get the actual processed value
                processed_val = processed_row.get(n)
                if processed_val is None:
                    continue
                
                std = STDS.get(n, 1.0) or 1.0
                median = MEDIANS[n]
                z = (processed_val - median) / std
                importance = FEATURE_GROUP_IMPORTANCE.get(n, 0.0)
                driver_score = abs(z) * importance
                
                # Determine risk direction based on feature characteristics
                if n in ['borrower_credit_score', 'co-borrower_credit_score', 'Annual_Income', 'total_on_time_payments']:
                    # Higher values typically mean lower risk
                    direction = 'Lower Risk' if z > 0 else 'Higher Risk'
                else:
                    # Higher values typically mean higher risk for most other features
                    direction = 'Higher Risk' if z > 0 else 'Lower Risk'
                
                drivers.append({
                    'feature': n.replace('_', ' ').title(),
                    'value': round(processed_val, 2),
                    'median': round(median, 2),
                    'zscore': round(z, 3),
                    'importance': round(importance, 4),
                    'driver_score': round(driver_score, 4),
                    'direction': direction
                })
            
            # Sort by driver score and take top 5
            drivers.sort(key=lambda d: d['driver_score'], reverse=True)
            drivers = drivers[:5]        # Enhanced risk summary
        if proba is not None:
            risk_summary = (
                f"Loan default probability: {proba:.1%} ({risk_level} Risk). "
                f"Model analyzed {len(NUMERIC_COLS)} numeric features and {len(CATEGORICAL_COLS)} categorical features. "
                f"Risk assessment based on borrower profile, loan characteristics, and payment history patterns."
            )
        else:
            risk_summary = "Risk assessment completed using enhanced fallback model. Comprehensive analysis of borrower and loan characteristics performed."

        # Additional metrics for comprehensive output
        confidence_score = abs(proba - 0.5) * 2 if proba is not None else 0.5
        
        # Loan characteristics summary
        loan_summary = {
            'loan_amount': processed_row.get('unpaid_principal_bal', 0),
            'interest_rate': processed_row.get('interest_rate', 0),
            'loan_term': processed_row.get('Loan_term', 0),
            'ltv_ratio': processed_row.get('loan_to_value', 0),
            'borrower_age': processed_row.get('Age', 0),
            'credit_score': processed_row.get('borrower_credit_score', 0),
            'annual_income': processed_row.get('Annual_Income', 0),
            'debt_to_income': processed_row.get('debt_to_income_ratio', 0)
        }

        return jsonify({
            'binary_prediction': prediction,
            'delinquency_probability': proba,
            'risk_level': risk_level,
            'delinquency_flag': bool(prediction),
            'confidence_score': round(confidence_score, 3),
            'risk_summary': risk_summary,
            'top_drivers': drivers,
            'loan_summary': loan_summary,
            'model_info': {
                'features_analyzed': len(NUMERIC_COLS) + len(CATEGORICAL_COLS),
                'numeric_features': len(NUMERIC_COLS),
                'categorical_features': len(CATEGORICAL_COLS),
                'model_type': 'RandomForest Enhanced'
            },
            'explanation': 'Comprehensive loan default risk analysis using machine learning model trained on historical loan performance data.',
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({'error': str(e), 'details': 'Please check input values and try again.'}), 400

if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', '5001'))
    app.run(debug=True, host='0.0.0.0', port=port)