from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <html>
    <head>
        <title>Samsung Dashboard Worklet 8</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #1428A0; text-align: center; }
            .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .app-link { display: block; background: #1428A0; color: white; padding: 12px 20px; margin: 10px 0; text-decoration: none; border-radius: 5px; text-align: center; }
            .app-link:hover { background: #0f1e7a; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏆 PRISM Worklet 8</h1>
            <h2>Harnessing TabFM and AI to Drive Smarter Financing, Campaign, and Sales</h2>
            <div class="status">
                ✅ <strong>Deployment Status:</strong> Successfully deployed on Vercel!
            </div>
            <p><strong>Available Applications:</strong></p>
            <a href="/loan" class="app-link">🏦 Loan Delinquency Risk Assessment</a>
            <a href="/campaign" class="app-link">📈 Campaign Performance Analysis</a>
            <a href="/sales" class="app-link">📊 Sales Forecasting & Uplift</a>
            <a href="/api/health" class="app-link">🔍 System Health Check</a>
        </div>
    </body>
    </html>
    '''

@app.route('/loan')
def loan():
    return '''
    <h1>🏦 Loan Delinquency Risk Assessment</h1>
    <p>Advanced ML-powered risk assessment for loan delinquency prediction.</p>
    <p><strong>Status:</strong> Application module ready for integration.</p>
    <a href="/">← Back to Dashboard</a>
    '''

@app.route('/campaign')
def campaign():
    return '''
    <h1>📈 Campaign Performance Analysis</h1>
    <p>Store performance analysis and marketing campaign optimization.</p>
    <p><strong>Status:</strong> Application module ready for integration.</p>
    <a href="/">← Back to Dashboard</a>
    '''

@app.route('/sales')
def sales():
    return '''
    <h1>📊 Sales Forecasting & Uplift</h1>
    <p>AI-driven sales forecasting with uplift prediction capabilities.</p>
    <p><strong>Status:</strong> Application module ready for integration.</p>
    <a href="/">← Back to Dashboard</a>
    '''

@app.route('/api/health')
def health():
    return {
        "status": "healthy", 
        "message": "Samsung Dashboard Worklet 8 is running successfully on Vercel",
        "applications": ["loan", "campaign", "sales"],
        "version": "1.0.0"
    }

# For Vercel
app.debug = False

if __name__ == '__main__':
    app.run(debug=True, port=5000)