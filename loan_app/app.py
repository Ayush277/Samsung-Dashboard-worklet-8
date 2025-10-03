from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load artifacts lazily to avoid import-time crashes
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
scaler = None
dummy_columns = None

ARTIFACT_ERR = None

def load_artifacts():
    global model, scaler, dummy_columns, ARTIFACT_ERR
    if model is not None and scaler is not None and dummy_columns is not None:
        return True
    try:
        model = joblib.load(os.path.join(BASE_DIR, 'models', 'tabpfn.pkl'))
        scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
        dummy_columns = joblib.load(os.path.join(BASE_DIR, 'models', 'dummy_columns.pkl'))
        ARTIFACT_ERR = None
        return True
    except Exception as e:
        ARTIFACT_ERR = str(e)
        return False

# Original categorical columns (must match training)
CATEGORICAL_COLS = ['source', 'loan_purpose', 'EducationLevel', 
                   'MaritalStatus', 'Gender', 'EmploymentStatus']

def preprocess(input_data):
    """Replicate training preprocessing"""
    if dummy_columns is None or scaler is None:
        raise RuntimeError('Artifacts not loaded')
    # 1. Convert to DataFrame
    X = pd.DataFrame([input_data])
    
    # 2. One-hot encode (mimic pd.get_dummies())
    x_encoded = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    
    # 3. Add missing columns with 0s
    for col in dummy_columns:
        if col not in x_encoded.columns:
            x_encoded[col] = 0
    
    # 4. Ensure column order matches training
    x_encoded = x_encoded[dummy_columns]
    
    # 5. Apply scaling
    return scaler.transform(x_encoded)

@app.route('/')
def home():
    load_artifacts()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not load_artifacts():
            return jsonify({'error': f'Artifacts not loaded: {ARTIFACT_ERR}'}), 500
        # Get raw input
        data = request.form.to_dict()
        
        # Preprocess and predict
        x_processed = preprocess(data)
        prediction = model.predict(x_processed)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'class': 'Default' if prediction == 1 else 'Non-Default'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=True, host='0.0.0.0', port=port)