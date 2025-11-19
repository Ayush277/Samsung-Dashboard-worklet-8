import os
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set up Flask app with correct template folder
template_dir = os.path.join(PROJECT_ROOT, 'dashboard', 'templates')
app = Flask(__name__, template_folder=template_dir)

# Simple module placeholders (will be loaded dynamically when needed)
loan_app = None
campaign_app = None
sales_app = None

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
    try:
        return render_template("index.html", status=status)
    except Exception as e:
        return f"Dashboard Error: {str(e)}", 500

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
        return "<h1>Loan Risk Assessment</h1><p>Coming soon - Loan delinquency risk prediction interface</p>"
    except Exception as e:
        return f"Loan app error: {str(e)}", 500

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
        return "<h1>Campaign Performance</h1><p>Coming soon - Marketing campaign analysis interface</p>"
    except Exception as e:
        return f"Campaign app error: {str(e)}", 500

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
        return "<h1>Sales Forecasting</h1><p>Coming soon - Sales uplift prediction interface</p>"
    except Exception as e:
        return f"Sales app error: {str(e)}", 500

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

# Health check endpoint
@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy", "message": "Samsung Dashboard Worklet 8 is running"})

# Vercel entry point
app.wsgi_app = app.wsgi_app

if __name__ == '__main__':
    app.run(debug=True, port=5000)