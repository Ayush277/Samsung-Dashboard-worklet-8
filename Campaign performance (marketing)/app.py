import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, jsonify
import pickle
import os
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load the sales dataset for store performance analysis
try:
    SALES_DATA = pd.read_csv('train2.csv')
    SALES_DATA['date'] = pd.to_datetime(SALES_DATA['date'])
    print(f"✓ Loaded sales data: {len(SALES_DATA)} records")
except Exception as e:
    print(f"⚠ Could not load train2.csv: {e}")
    SALES_DATA = None

# Initialize the Flask app
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
TEMPLATE_MAIN = 'index.html'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# --- Electronic Devices Reference Configuration ---
# NOTE: Product categories not provided, using Electronic Devices as reference
ELECTRONIC_DEVICES_CONFIG = {
    'product_category': 'Electronic Devices',
    'typical_price_range': '$100-$800',
    'seasonal_patterns': {
        'high_season': [11, 12, 6, 7, 8],  # Holiday & Summer
        'moderate_season': [3, 4, 9, 10],   # Spring & Fall
        'low_season': [1, 2, 5]             # Winter & Late Spring
    },
    'performance_benchmarks': {
        'excellent_uplift': 0.7,    # 70%+ improvement
        'good_uplift': 0.4,         # 40%+ improvement  
        'moderate_uplift': 0.2,     # 20%+ improvement
        'high_conversion': 0.8,     # 80%+ conversion
        'moderate_conversion': 0.5,  # 50%+ conversion
        'low_conversion': 0.3       # 30%+ conversion
    },
    'recommended_channels': {
        'premium_digital': {'cost': 25.0, 'best_for': 'High-value electronics'},
        'targeted_email': {'cost': 5.0, 'best_for': 'Repeat customers'},
        'social_media': {'cost': 8.0, 'best_for': 'Young demographics'},
        'personalized_offers': {'cost': 15.0, 'best_for': 'Custom recommendations'},
        'standard_email': {'cost': 2.0, 'best_for': 'General promotions'},
        'display_ads': {'cost': 10.0, 'best_for': 'Brand awareness'},
        'retention_program': {'cost': 20.0, 'best_for': 'Customer loyalty'},
        'newsletter': {'cost': 1.5, 'best_for': 'Regular updates'},
        'broad_awareness': {'cost': 12.0, 'best_for': 'Mass market'}
    }
}

# --- Campaign Analysis Functions ---

def calculate_uplift_metrics(predicted_sales, store_id, item_id, base_sales_avg=None):
    """
    Calculate predicted uplift based on sales predictions and store/item characteristics
    """
    # Use historical average if provided, otherwise estimate based on store/item factors
    if base_sales_avg is None:
        # Estimate baseline using store and item factors
        store_factor = min(1.0 + (store_id % 10) * 0.1, 2.0)  # Store performance factor
        item_factor = min(1.0 + (item_id % 20) * 0.05, 1.5)   # Item popularity factor
        base_sales_avg = predicted_sales / (store_factor * item_factor)
    
    # Calculate uplift percentage
    if base_sales_avg > 0:
        uplift_pct = (predicted_sales - base_sales_avg) / base_sales_avg
    else:
        uplift_pct = 0.0
    
    # Normalize to 0-1 scale for classification
    uplift_score = max(0, min(1, (uplift_pct + 0.5) / 1.5))  # Assuming uplift range of -50% to +100%
    
    return {
        'uplift_percentage': uplift_pct,
        'uplift_score': uplift_score,
        'baseline_sales': base_sales_avg
    }

def calculate_conversion_probability(predicted_sales, store_id, item_id, day_of_week, month):
    """
    Calculate conversion probability based on sales prediction and contextual factors
    """
    # Base conversion from sales prediction (higher sales = higher conversion likelihood)
    base_conversion = min(0.95, predicted_sales / 100.0)  # Normalize based on typical sales range
    
    # Seasonal adjustments
    seasonal_boost = 1.0
    if month in [11, 12]:  # Holiday season
        seasonal_boost = 1.3
    elif month in [6, 7, 8]:  # Summer season
        seasonal_boost = 1.1
    
    # Day of week adjustments (weekends typically have different patterns)
    dow_factor = 1.0
    if day_of_week in [5, 6]:  # Weekend
        dow_factor = 1.2
    elif day_of_week == 0:  # Monday
        dow_factor = 0.9
    
    # Store performance factor (simulate store quality impact)
    store_performance = min(1.5, 0.8 + (store_id % 7) * 0.1)
    
    # Item category factor (simulate item appeal)
    item_appeal = min(1.4, 0.9 + (item_id % 15) * 0.033)
    
    # Combine all factors
    conversion_prob = base_conversion * seasonal_boost * dow_factor * store_performance * item_appeal
    
    # Ensure it stays within 0-1 bounds
    return max(0.05, min(0.95, conversion_prob))

def determine_campaign_channel(uplift_score, conversion_prob):
    """
    Determine the best campaign channel based on uplift and conversion scores
    """
    # Classify uplift and conversion into categories
    uplift_cat = 'high' if uplift_score >= CAMPAIGN_CONFIG['uplift_thresholds']['high'] else \
                 'moderate' if uplift_score >= CAMPAIGN_CONFIG['uplift_thresholds']['low'] else 'low'
    
    conversion_cat = 'high' if conversion_prob >= CAMPAIGN_CONFIG['conversion_thresholds']['high'] else \
                     'moderate' if conversion_prob >= CAMPAIGN_CONFIG['conversion_thresholds']['low'] else 'low'
    
    # Get recommended channel
    channel_key = f"{uplift_cat}_uplift_{conversion_cat}_conversion"
    return CAMPAIGN_CONFIG['channel_mapping'].get(channel_key, 'standard_email')

def calculate_segment_uplift_score(uplift_score, conversion_prob, predicted_sales):
    """
    Calculate a composite segment uplift score for customer segmentation
    """
    # Weight the different factors
    uplift_weight = 0.4
    conversion_weight = 0.3
    sales_weight = 0.3
    
    # Normalize sales (assuming typical range 0-200)
    normalized_sales = min(1.0, predicted_sales / 200.0)
    
    # Calculate composite score
    segment_score = (
        uplift_score * uplift_weight +
        conversion_prob * conversion_weight +
        normalized_sales * sales_weight
    )
    
    return min(1.0, segment_score)

def calculate_risk_cost_ratio(predicted_sales, conversion_prob, channel):
    """
    Calculate risk/cost ratio for campaign decision making
    """
    # Expected revenue (sales * conversion probability)
    expected_revenue = predicted_sales * conversion_prob
    
    # Campaign cost based on channel
    campaign_cost = CAMPAIGN_CONFIG['base_cost_per_channel'].get(channel, 10.0)
    
    # Risk factors (lower conversion = higher risk)
    risk_factor = 1.0 - conversion_prob + 0.1  # Add base risk
    
    # Calculate risk-adjusted cost
    risk_adjusted_cost = campaign_cost * risk_factor
    
    # Risk/Cost ratio (higher is better)
    if risk_adjusted_cost > 0:
        ratio = expected_revenue / risk_adjusted_cost
    else:
        ratio = 0.0
    
    return {
        'ratio': ratio,
        'expected_revenue': expected_revenue,
        'campaign_cost': campaign_cost,
        'risk_factor': risk_factor
    }

def determine_recommended_action(risk_cost_ratio, uplift_score, conversion_prob):
    """
    Provide recommended action based on all calculated metrics
    """
    ratio = risk_cost_ratio['ratio']
    
    if ratio >= 3.0 and uplift_score >= 0.6:
        return "High Priority - Launch Premium Campaign"
    elif ratio >= 2.0 and conversion_prob >= 0.5:
        return "Medium Priority - Standard Campaign"
    elif ratio >= 1.5:
        return "Low Priority - Basic Campaign"
    elif conversion_prob >= 0.7:
        return "Consider Retention Campaign"
    elif uplift_score >= 0.4:
        return "Test Small Campaign"
    else:
        return "Hold - Monitor Performance"

def enrich_predictions_with_campaign_data(df, predictions):
    """
    Add all campaign analysis columns to the predictions DataFrame
    """
    enriched_df = df.copy()
    enriched_df['predicted_sales'] = np.round(predictions).astype(int)
    
    # Calculate all campaign metrics
    campaign_data = []
    for idx, row in enriched_df.iterrows():
        predicted_sales = enriched_df.loc[idx, 'predicted_sales']
        store_id = row['store']
        item_id = row['item']
        
        # Extract date features
        date_obj = pd.to_datetime(row['date'])
        day_of_week = date_obj.dayofweek
        month = date_obj.month
        
        # Calculate uplift metrics
        uplift_data = calculate_uplift_metrics(predicted_sales, store_id, item_id)
        
        # Calculate conversion probability
        conversion_prob = calculate_conversion_probability(
            predicted_sales, store_id, item_id, day_of_week, month
        )
        
        # Determine best channel
        best_channel = determine_campaign_channel(uplift_data['uplift_score'], conversion_prob)
        
        # Calculate segment score
        segment_score = calculate_segment_uplift_score(
            uplift_data['uplift_score'], conversion_prob, predicted_sales
        )
        
        # Calculate risk/cost ratio
        risk_cost_data = calculate_risk_cost_ratio(predicted_sales, conversion_prob, best_channel)
        
        # Get recommended action
        recommended_action = determine_recommended_action(
            risk_cost_data, uplift_data['uplift_score'], conversion_prob
        )
        
        campaign_data.append({
            'predicted_uplift': round(uplift_data['uplift_percentage'], 3),
            'predicted_conversion': round(conversion_prob, 3),
            'predicted_best_campaign_channel': best_channel,
            'segment_uplift_score': round(segment_score, 3),
            'risk_cost_ratio': round(risk_cost_data['ratio'], 2),
            'recommended_action': recommended_action
        })
    
    # Add all new columns to DataFrame
    for key in campaign_data[0].keys():
        enriched_df[key] = [item[key] for item in campaign_data]
    
    return enriched_df

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
        predicted_sales = int(np.round(np.ravel(prediction)[0]))
        model_name = model_choice.replace('_', ' ').title()
        
        # Calculate campaign metrics for single prediction
        store_id = int(form_data['store'])
        item_id = int(form_data['item'])
        
        # Calculate uplift metrics
        uplift_data = calculate_uplift_metrics(predicted_sales, store_id, item_id)
        
        # Calculate conversion probability
        conversion_prob = calculate_conversion_probability(
            predicted_sales, store_id, item_id, date.dayofweek, month
        )
        
        # Determine best channel
        best_channel = determine_campaign_channel(uplift_data['uplift_score'], conversion_prob)
        
        # Calculate segment score
        segment_score = calculate_segment_uplift_score(
            uplift_data['uplift_score'], conversion_prob, predicted_sales
        )
        
        # Calculate risk/cost ratio
        risk_cost_data = calculate_risk_cost_ratio(predicted_sales, conversion_prob, best_channel)
        
        # Get recommended action
        recommended_action = determine_recommended_action(
            risk_cost_data, uplift_data['uplift_score'], conversion_prob
        )
        
        # Create professional point-wise prediction text
        prediction_text = f'''
        <div class="prediction-results">
            <div class="result-header">
                <h5><i class="fas fa-chart-line me-2"></i>Campaign Intelligence Analysis ({model_name})</h5>
            </div>
            
            <div class="result-section">
                <h6><i class="fas fa-shopping-cart text-primary"></i> Sales Prediction</h6>
                <ul class="result-list">
                    <li><strong>Predicted Sales Units:</strong> <span class="highlight-value">{predicted_sales}</span></li>
                    <li><strong>Baseline Comparison:</strong> <span class="text-muted">{uplift_data['baseline_sales']:.1f} units (baseline)</span></li>
                    <li><strong>Performance Indicator:</strong> <span class="badge bg-{"success" if predicted_sales > uplift_data['baseline_sales'] else "warning"}"">{"Above" if predicted_sales > uplift_data['baseline_sales'] else "Below"} Baseline</span></li>
                </ul>
            </div>

            <div class="result-section">
                <h6><i class="fas fa-rocket text-success"></i> Uplift Analysis</h6>
                <ul class="result-list">
                    <li><strong>Predicted Uplift:</strong> <span class="highlight-value text-{"success" if uplift_data['uplift_percentage'] > 0 else "danger"}">{uplift_data['uplift_percentage']:.1%}</span></li>
                    <li><strong>Uplift Score:</strong> <span class="text-muted">{uplift_data['uplift_score']:.3f} / 1.000</span></li>
                    <li><strong>Uplift Category:</strong> <span class="badge bg-{"success" if uplift_data['uplift_score'] >= 0.7 else "warning" if uplift_data['uplift_score'] >= 0.4 else "secondary"}">{"High" if uplift_data['uplift_score'] >= 0.7 else "Moderate" if uplift_data['uplift_score'] >= 0.4 else "Low"} Potential</span></li>
                </ul>
            </div>

            <div class="result-section">
                <h6><i class="fas fa-target text-info"></i> Conversion Analysis</h6>
                <ul class="result-list">
                    <li><strong>Conversion Probability:</strong> <span class="highlight-value text-{"success" if conversion_prob >= 0.6 else "warning" if conversion_prob >= 0.4 else "danger"}">{conversion_prob:.1%}</span></li>
                    <li><strong>Confidence Level:</strong> <span class="text-muted">{"High" if conversion_prob >= 0.7 else "Moderate" if conversion_prob >= 0.5 else "Low"} Confidence</span></li>
                    <li><strong>Expected Revenue:</strong> <span class="text-success">${risk_cost_data['expected_revenue']:.2f}</span></li>
                </ul>
            </div>

            <div class="result-section">
                <h6><i class="fas fa-bullhorn text-warning"></i> Campaign Strategy</h6>
                <ul class="result-list">
                    <li><strong>Recommended Channel:</strong> <span class="channel-badge">{best_channel.replace('_', ' ').title()}</span></li>
                    <li><strong>Campaign Cost:</strong> <span class="text-muted">${risk_cost_data['campaign_cost']:.2f}</span></li>
                    <li><strong>Segment Score:</strong> <span class="highlight-value">{segment_score:.3f}</span> <span class="text-muted">/ 1.000</span></li>
                </ul>
            </div>

            <div class="result-section">
                <h6><i class="fas fa-calculator text-danger"></i> Financial Analysis</h6>
                <ul class="result-list">
                    <li><strong>Risk/Cost Ratio:</strong> <span class="highlight-value text-{"success" if risk_cost_data['ratio'] >= 2.0 else "warning" if risk_cost_data['ratio'] >= 1.0 else "danger"}">{risk_cost_data['ratio']:.2f}</span></li>
                    <li><strong>Risk Factor:</strong> <span class="text-muted">{risk_cost_data['risk_factor']:.2f}</span></li>
                    <li><strong>ROI Indicator:</strong> <span class="badge bg-{"success" if risk_cost_data['ratio'] >= 2.0 else "warning" if risk_cost_data['ratio'] >= 1.0 else "danger"}">{"Excellent" if risk_cost_data['ratio'] >= 3.0 else "Good" if risk_cost_data['ratio'] >= 2.0 else "Moderate" if risk_cost_data['ratio'] >= 1.0 else "Poor"} ROI</span></li>
                </ul>
            </div>

            <div class="result-section recommendation-section">
                <h6><i class="fas fa-lightbulb text-warning"></i> Strategic Recommendation</h6>
                <div class="recommendation-box">
                    <div class="recommendation-text">{recommended_action}</div>
                    <div class="recommendation-details">
                        <small class="text-muted">
                            Based on predictive analysis of store {store_id}, item {item_id} for {date.strftime('%B %d, %Y')}
                        </small>
                    </div>
                </div>
            </div>
        </div>
        '''
        
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

            # Enrich with campaign analysis data
            enriched_df = enrich_predictions_with_campaign_data(test_df, predictions)
            
            # Add metadata columns
            model_name = 'catboost' if 'catboost' in models else list(models.keys())[0] if models else 'none'
            enriched_df['model_used'] = model_name
            enriched_df['processing_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save enriched results with enhanced filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_filename = f'enriched_campaign_analysis_{timestamp}_{file.filename}'
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            enriched_df.to_csv(result_filepath, index=False)

            # Calculate summary statistics
            summary_stats = {
                'total_records': int(len(enriched_df)),
                'average_predicted_sales': round(float(enriched_df['predicted_sales'].mean()), 2),
                'high_uplift_count': int((enriched_df['predicted_uplift'] >= 0.3).sum()),
                'high_conversion_count': int((enriched_df['predicted_conversion'] >= 0.6).sum()),
                'top_channel': enriched_df['predicted_best_campaign_channel'].mode().iloc[0] if len(enriched_df) > 0 else 'unknown',
                'avg_risk_cost_ratio': round(float(enriched_df['risk_cost_ratio'].mean()), 2),
                'model_used': model_name
            }

            return jsonify({
                'success': True, 
                'download_url': f'/download/{result_filename}',
                'filename': result_filename,
                'summary': summary_stats
            })

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    return jsonify({'success': False, 'error': 'Invalid file type, please upload a .csv file'})


@app.route('/download/<filename>')
def download(filename):
    """Provides the results file for download."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/analyze_store_performance', methods=['POST'])
def analyze_store_performance():
    """Analyze store performance using actual sales data from train2.csv"""
    try:
        # Check if it's a file upload or form submission
        if 'file' in request.files and request.files['file'].filename:
            # Handle batch CSV upload
            file = request.files['file']
            
            if not file.filename.endswith('.csv'):
                return jsonify({'success': False, 'error': 'File must be CSV format'})
            
            # Save uploaded file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'batch_input_{timestamp}_{file.filename}'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read the CSV file
            input_df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['date', 'item', 'store']
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            if missing_cols:
                # Try alternative column names
                alt_mapping = {'item': 'item_number', 'store': 'store_id'}
                for missing in missing_cols[:]:
                    for alt_name in input_df.columns:
                        if alt_name.lower() in [missing, alt_mapping.get(missing, missing)]:
                            input_df = input_df.rename(columns={alt_name: missing})
                            missing_cols.remove(missing)
                            break
                
                if missing_cols:
                    return jsonify({
                        'success': False, 
                        'error': f'Missing required columns: {", ".join(missing_cols)}. Required: date, store, item'
                    })
            
            # Clean and validate data
            input_df = input_df.dropna(subset=['date', 'store', 'item'])
            input_df['item'] = pd.to_numeric(input_df['item'], errors='coerce')
            input_df['store'] = pd.to_numeric(input_df['store'], errors='coerce')
            input_df = input_df.dropna(subset=['item', 'store'])
            
            if len(input_df) == 0:
                return jsonify({'success': False, 'error': 'No valid data rows found in CSV'})
            
            # Limit batch size
            if len(input_df) > 1000:
                input_df = input_df.head(1000)
                
        else:
            # Handle individual form submission
            date = request.form.get('date')
            item_number = request.form.get('item_number') 
            store_ids = request.form.get('store_ids', '').strip()
            
            # Parse store IDs
            if ',' in store_ids:
                store_list = [int(s.strip()) for s in store_ids.split(',') if s.strip().isdigit()]
            elif '-' in store_ids:
                start, end = store_ids.split('-')
                store_list = list(range(int(start.strip()), int(end.strip()) + 1))
            else:
                store_list = [int(store_ids)] if store_ids.isdigit() else [1]
            
            # Limit to reasonable number of stores
            store_list = store_list[:50]
            
            # Create input data
            input_data = []
            for store_id in store_list:
                input_data.append({
                    'date': date,
                    'item': int(item_number),
                    'store': store_id
                })
            
            input_df = pd.DataFrame(input_data)
        
        # Analyze store performance using actual sales data
        results = []
        
        for _, row in input_df.iterrows():
            store_id = int(row['store'])
            item_id = int(row['item'])
            date_str = str(row['date'])
            
            # Get actual sales data from train2.csv
            actual_sales = get_actual_sales_from_csv(store_id, item_id, date_str)
            baseline_sales = get_baseline_sales_from_csv(store_id, item_id, date_str)
            
            # Calculate performance metrics
            if actual_sales is not None and baseline_sales is not None:
                uplift_amount = actual_sales - baseline_sales
                uplift_percentage = (uplift_amount / baseline_sales * 100) if baseline_sales > 0 else 0
                
                # Categorize performance
                if uplift_percentage >= 25:  # 25%+ uplift = top performer
                    is_top_performer = True
                    is_underperformer = False
                    performance_category = 'Top Performer'
                    performance_by_amount = f"+${uplift_amount:.2f} ({uplift_percentage:.1f}% above baseline)"
                elif uplift_percentage <= -10:  # -10% or more = underperformer
                    is_top_performer = False
                    is_underperformer = True
                    performance_category = 'Underperformer'
                    performance_by_amount = f"-${abs(uplift_amount):.2f} ({abs(uplift_percentage):.1f}% below baseline)"
                else:
                    is_top_performer = False
                    is_underperformer = False
                    performance_category = 'Average Performer'
                    performance_by_amount = f"${uplift_amount:+.2f} ({uplift_percentage:+.1f}% from baseline)"
                
                has_actual_data = True
            else:
                # Generate synthetic data if no actual data available
                baseline_sales = 50 + (store_id % 10) * 5 + (item_id % 5) * 3
                actual_sales = baseline_sales * (0.8 + (store_id + item_id) % 50 / 100)
                uplift_amount = actual_sales - baseline_sales
                uplift_percentage = (uplift_amount / baseline_sales * 100) if baseline_sales > 0 else 0
                
                is_top_performer = uplift_percentage >= 25
                is_underperformer = uplift_percentage <= -10
                performance_category = 'Top Performer' if is_top_performer else 'Underperformer' if is_underperformer else 'Average Performer'
                performance_by_amount = f"${uplift_amount:+.2f} ({uplift_percentage:+.1f}% synthetic)"
                has_actual_data = False
            
            results.append({
                'store': int(store_id),
                'item': int(item_id),
                'date': date_str,
                'current_sales': float(actual_sales) if actual_sales is not None else 0.0,
                'baseline_sales': float(baseline_sales) if baseline_sales is not None else 0.0,
                'uplift_pct': round(float(uplift_percentage), 2),
                'uplift_amount': round(float(uplift_amount), 2),
                'performance_category': performance_category,
                'performance_by_amount': performance_by_amount,
                'is_top_performer': is_top_performer,
                'is_underperformer': is_underperformer,
                'has_actual_data': has_actual_data
            })
        
        results_df = pd.DataFrame(results)
        
        # Generate detailed analysis
        top_performers_detailed = []
        underperformers_detailed = []
        
        for _, store in results_df.iterrows():
            if store['is_top_performer']:
                top_performers_detailed.append({
                    'store_id': int(store['store']),
                    'item_number': int(store['item']),
                    'current_sales': float(store['current_sales']),
                    'baseline_sales': float(store['baseline_sales']),
                    'performance_by_amount': store['performance_by_amount'],
                    'has_actual_data': bool(store['has_actual_data']),
                    'is_top_performer': 'YES',
                    'performance_level': 'Top Performer'
                })
            elif store['is_underperformer']:
                underperformers_detailed.append({
                    'store_id': int(store['store']),
                    'item_number': int(store['item']),
                    'current_sales': float(store['current_sales']),
                    'baseline_sales': float(store['baseline_sales']),
                    'performance_by_amount': store['performance_by_amount'],
                    'has_actual_data': bool(store['has_actual_data']),
                    'is_underperformer': 'YES',
                    'performance_level': 'Underperformer',
                    'underperforming_by': f"${abs(float(store['uplift_amount'])):.2f}"
                })
        
        # Calculate campaign performance metrics
        total_stores = len(results_df)
        successful_stores = len(results_df[results_df['uplift_pct'] > 0])
        avg_roi = 45.0 + (successful_stores / total_stores) * 30  # Synthetic ROI calculation
        total_revenue_increase = results_df['uplift_amount'].sum()
        
        campaign_performance = {
            'success_rate': round(float(successful_stores / total_stores * 100), 1) if total_stores > 0 else 0.0,
            'avg_roi': round(float(avg_roi), 1),
            'total_revenue_increase': round(float(total_revenue_increase), 2),
            'campaign_effectiveness': 'high' if avg_roi > 50 else 'medium' if avg_roi > 0 else 'low'
        }
        
        # Generate summary statistics
        summary_stats = {
            'total_stores': int(total_stores),
            'avg_uplift_pct': round(float(results_df['uplift_pct'].mean()), 2),
            'total_uplift_amount': round(float(results_df['uplift_amount'].sum()), 2),
            'stores_with_actual_data': int(results_df['has_actual_data'].sum()),
            'top_performers_count': int(len(top_performers_detailed)),
            'underperformers_count': int(len(underperformers_detailed)),
        }
        
        # Save results to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'store_performance_analysis_{timestamp}.csv'
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        results_df.to_csv(output_filepath, index=False)
        
        return jsonify({
            'success': True,
            'store_results': results_df.to_dict('records'),
            'top_performers_detailed': top_performers_detailed,
            'underperformers_detailed': underperformers_detailed,
            'campaign_performance': campaign_performance,
            'summary_stats': summary_stats,
            'download_url': f'/download/{output_filename}',
            'analysis_info': {
                'date_analyzed': str(input_df['date'].iloc[0]) if len(input_df) > 0 else 'N/A',
                'item_analyzed': int(input_df['item'].iloc[0]) if len(input_df) > 0 else 'N/A',
                'stores_analyzed': int(len(results_df))
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def get_actual_sales_from_csv(store_id, item_id, date_str):
    """Get actual sales data from train2.csv"""
    if SALES_DATA is None:
        return None
        
    try:
        target_date = pd.to_datetime(date_str)
        
        # Filter for exact match
        filtered_data = SALES_DATA[
            (SALES_DATA['date'] == target_date) &
            (SALES_DATA['store'] == store_id) &
            (SALES_DATA['item'] == item_id)
        ]
        
        if not filtered_data.empty:
            return float(filtered_data.iloc[0]['sales'])
        else:
            # Try to find nearby data for same store/item
            store_item_data = SALES_DATA[
                (SALES_DATA['store'] == store_id) &
                (SALES_DATA['item'] == item_id)
            ]
            
            if not store_item_data.empty:
                return float(store_item_data['sales'].mean())
            else:
                return None
                
    except Exception as e:
        print(f"Error getting sales data: {e}")
        return None


def get_baseline_sales_from_csv(store_id, item_id, date_str, days_back=7):
    """Get baseline sales data from previous period"""
    if SALES_DATA is None:
        return None
        
    try:
        target_date = pd.to_datetime(date_str)
        baseline_date = target_date - timedelta(days=days_back)
        
        # Get baseline data
        baseline_data = SALES_DATA[
            (SALES_DATA['date'] == baseline_date) &
            (SALES_DATA['store'] == store_id) &
            (SALES_DATA['item'] == item_id)
        ]
        
        if not baseline_data.empty:
            return float(baseline_data.iloc[0]['sales'])
        else:
            # Fallback to average sales for this store/item
            store_item_data = SALES_DATA[
                (SALES_DATA['store'] == store_id) &
                (SALES_DATA['item'] == item_id)
            ]
            
            if not store_item_data.empty:
                return float(store_item_data['sales'].mean() * 0.9)  # Assume 10% lower baseline
            else:
                return None
                
    except Exception as e:
        print(f"Error getting baseline data: {e}")
        return None


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5002"))
    print("🚀 Samsung Worklet 8 – Enhanced Campaign Analysis Pipeline")
    print(f"🌐 Starting server on http://localhost:{port}")
    print("✨ New Features: Uplift Prediction, Conversion Analysis, Channel Recommendations")
    app.run(host="0.0.0.0", port=port, debug=True)