import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, jsonify
import pickle
import os

# Initialize the Flask app
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
TEMPLATE_MAIN = 'index.html'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# --- Load Models and Scaler (dynamic) ---
MODEL_FILES = {
    'random_forest': 'rf_model.pkl',
    'lightgbm': 'lgbm_model.pkl',
    'catboost': 'catboost_model.pkl',
    'ridge': 'ridge_model.pkl'
}

models = {}
scaler = None

try:
    # Load models that actually exist to avoid hard failures
    for key, fname in MODEL_FILES.items():
        if os.path.exists(fname):
            try:
                with open(fname, 'rb') as f:
                    models[key] = pickle.load(f)
            except Exception as e:
                print(f"WARNING: Failed to load {fname}: {e}")
    # Load scaler if present
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
except Exception as e:
    print(f"ERROR while loading artifacts: {e}")
    models = {}
    scaler = None

# --- Load Data for Dropdowns ---
try:
    df = pd.read_csv('train2.csv')
    dropdown_options = {
        'store_list': sorted(df['store'].unique()),
        'item_list': sorted(df['item'].unique())
    }
except FileNotFoundError:
    print("ERROR: train2.csv not found. Dropdowns will be disabled.")
    dropdown_options = {'store_list': [], 'item_list': []}


# --- Routes ---
@app.route('/')
def home():
    """Renders the main page."""
    warning = None
    missing = []
    if not models:
        missing.append('models')
    if scaler is None:
        missing.append('scaler')
    if missing:
        warning = f"Missing artifacts: {', '.join(missing)}. Place pickle files in the project root."
    return render_template(
        TEMPLATE_MAIN,
        options=dropdown_options,
        form_data={},
        prediction_text=None,
        warning=warning,
        available_models=list(models.keys())
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handles a single prediction from the form."""
    if not models or scaler is None:
        return render_template(TEMPLATE_MAIN, options=dropdown_options, form_data={}, prediction_text=None, warning='Artifacts not loaded.', available_models=list(models.keys()))

    # Store the user's selections
    form_data = request.form.to_dict()

    try:
        model_choice = form_data['model_choice']
        model = models.get(model_choice)
        if model is None:
            return render_template(TEMPLATE_MAIN, options=dropdown_options, prediction_text='Selected model not available.', form_data=form_data, warning=None, available_models=list(models.keys()))
        
        day = int(form_data['day'])
        month = int(form_data['month'])
        
        try:
            date = pd.to_datetime(f'2024-{month}-{day}')
        except ValueError:
            return render_template(TEMPLATE_MAIN, options=dropdown_options, prediction_text="Error: Invalid date.", form_data=form_data, available_models=list(models.keys()))

        features = [
            int(form_data['store']),
            int(form_data['item']),
            month,
            day,
            date.dayofweek,
            date.dayofyear,
            int(date.isocalendar().week)
        ]
        
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)
        output = int(np.round(np.ravel(prediction)[0]))
        model_name = model_choice.replace('_', ' ').title()
        
        prediction_text = f'Predicted Sales with {model_name}: {output}'
        # Pass form_data back to the template
        return render_template(TEMPLATE_MAIN, options=dropdown_options, prediction_text=prediction_text, form_data=form_data, available_models=list(models.keys()))

    except Exception as e:
        return render_template(TEMPLATE_MAIN, options=dropdown_options, prediction_text=f'An error occurred: {e}', form_data=form_data, available_models=list(models.keys()))


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handles batch prediction from an uploaded CSV file."""
    if scaler is None or not models:
        return jsonify({'success': False, 'error': 'Artifacts not loaded (models/scaler).'}), 400

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file and file.filename.endswith('.csv'):
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            test_df = pd.read_csv(filepath, parse_dates=['date'])
            
            test_df['month'] = test_df['date'].dt.month
            test_df['day'] = test_df['date'].dt.day
            test_df['dayofweek'] = test_df['date'].dt.dayofweek
            test_df['dayofyear'] = test_df['date'].dt.dayofyear
            test_df['weekofyear'] = test_df['date'].dt.isocalendar().week.astype(int)

            feature_cols = ['store', 'item', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear']
            X_test = test_df[feature_cols]

            x_test_scaled = scaler.transform(X_test)
            # Prefer random_forest if available; else pick the first available model
            model = models.get('random_forest') or next(iter(models.values()))
            predictions = model.predict(x_test_scaled)

            test_df['predicted_sales'] = np.round(predictions).astype(int)
            result_filename = 'predictions_' + file.filename
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            test_df.to_csv(result_filepath, index=False)

            return jsonify({
                'success': True, 
                'download_url': f'/download/{result_filename}',
                'records_processed': int(len(test_df))
            })

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    return jsonify({'success': False, 'error': 'Invalid file type, please upload a .csv file'})


@app.route('/download/<filename>')
def download(filename):
    """Provides the results file for download."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)