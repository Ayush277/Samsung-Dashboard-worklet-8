import os
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Add project paths to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dashboard'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Loan delinquency risk'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Campaign performance (marketing)'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Sell-out performance forecasting (sales uplift)'))

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'templates'))

# Import individual app modules
try:
    # Import loan delinquency app
    loan_dir = os.path.join(os.path.dirname(__file__), '..', 'Loan delinquency risk')
    sys.path.insert(0, loan_dir)
    import loan_risk_inference_demo as loan_app
    
    # Import campaign app
    campaign_dir = os.path.join(os.path.dirname(__file__), '..', 'Campaign performance (marketing)')
    sys.path.insert(0, campaign_dir)
    import campaign_performance_inference_demo as campaign_app
    
    # Import sales app
    sales_dir = os.path.join(os.path.dirname(__file__), '..', 'Sell-out performance forecasting (sales uplift)')
    sys.path.insert(0, sales_dir)
    import sales_uplift_inference_demo as sales_app
    
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

@app.route('/')
def index():
    """Main dashboard"""
    status = {
        "loan": {
            "name": "Loan Delinquency Risk",
            "port": None,  # Not applicable in serverless
            "running": True,
        },
        "campaign": {
            "name": "Campaign Performance (Marketing)",
            "port": None,
            "running": True,
        },
        "sales": {
            "name": "Sell-out Performance Forecasting",
            "port": None,
            "running": True,
        }
    }
    return render_template("index.html", status=status)

@app.route('/open/<app_id>')
def open_app(app_id: str):
    """Redirect to specific app functionality"""
    if app_id == 'loan':
        return redirect(url_for('loan_interface'))
    elif app_id == 'campaign':
        return redirect(url_for('campaign_interface'))
    elif app_id == 'sales':
        return redirect(url_for('sales_interface'))
    else:
        return redirect(url_for('index'))

# Loan Delinquency Risk Routes
@app.route('/loan')
def loan_interface():
    """Loan risk assessment interface"""
    try:
        # Load loan risk template
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'Loan delinquency risk', 'templates')
        return render_template('index.html', base_url='/loan')
    except Exception as e:
        return f"Loan app error: {str(e)}"

@app.route('/loan/predict', methods=['POST'])
def loan_predict():
    """Handle loan risk prediction"""
    try:
        # Get form data and make prediction using loan_app
        if hasattr(loan_app, 'predict_risk'):
            result = loan_app.predict_risk(request.form.to_dict())
            return jsonify(result)
        else:
            return jsonify({"error": "Prediction function not available"})
    except Exception as e:
        return jsonify({"error": str(e)})

# Campaign Performance Routes
@app.route('/campaign')
def campaign_interface():
    """Campaign performance interface"""
    try:
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'Campaign performance (marketing)', 'templates')
        return render_template('index.html', base_url='/campaign')
    except Exception as e:
        return f"Campaign app error: {str(e)}"

@app.route('/campaign/predict', methods=['POST'])
def campaign_predict():
    """Handle campaign performance prediction"""
    try:
        if hasattr(campaign_app, 'predict_performance'):
            result = campaign_app.predict_performance(request.form.to_dict())
            return jsonify(result)
        else:
            return jsonify({"error": "Prediction function not available"})
    except Exception as e:
        return jsonify({"error": str(e)})

# Sales Forecasting Routes
@app.route('/sales')
def sales_interface():
    """Sales forecasting interface"""
    try:
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'Sell-out performance forecasting (sales uplift)', 'templates')
        return render_template('index.html', base_url='/sales')
    except Exception as e:
        return f"Sales app error: {str(e)}"

@app.route('/sales/predict', methods=['POST'])
def sales_predict():
    """Handle sales forecasting prediction"""
    try:
        if hasattr(sales_app, 'predict_sales'):
            result = sales_app.predict_sales(request.form.to_dict())
            return jsonify(result)
        else:
            return jsonify({"error": "Prediction function not available"})
    except Exception as e:
        return jsonify({"error": str(e)})

# API Routes for AJAX calls
@app.route('/api/start/<app_id>', methods=['POST'])
def api_start(app_id: str):
    """API endpoint to 'start' an app (always returns success in serverless)"""
    apps = {
        'loan': 'Loan Delinquency Risk',
        'campaign': 'Campaign Performance',
        'sales': 'Sales Forecasting'
    }
    
    if app_id not in apps:
        return jsonify({"ok": False, "error": "unknown app"}), 404
    
    return jsonify({
        "ok": True, 
        "message": f"{apps[app_id]} is ready",
        "port": None
    })

@app.route('/api/stop/<app_id>', methods=['POST'])
def api_stop(app_id: str):
    """API endpoint to 'stop' an app (not applicable in serverless)"""
    return jsonify({"ok": True, "message": "stopped"})

# Vercel entry point
def handler(request):
    return app(request.environ, lambda status, headers: None)

if __name__ == '__main__':
    app.run(debug=True)