print("=" * 50)
print("STARTING FLASK APPLICATION DEBUG")
print("=" * 50)

try:
    print("1. Importing Flask modules...")
    from flask import Flask, request, render_template, jsonify, send_file
    print("   ✓ Flask imported successfully")
    
    print("2. Importing os...")
    import os
    print("   ✓ os imported successfully")
    
    print("3. Current working directory:", os.getcwd())
    print("4. Files in current directory:", os.listdir('.'))
    
    print("5. Importing Config...")
    from config import Config
    print("   ✓ Config imported successfully")
    
    print("6. Checking Config attributes...")
    print("   - UPLOAD_FOLDER:", getattr(Config, 'UPLOAD_FOLDER', 'NOT FOUND'))
    print("   - EXPECTED_FEATURES:", len(getattr(Config, 'EXPECTED_FEATURES', [])), "features")
    
    print("7. Importing ModelLoader...")
    from utils.model_loader import ModelLoader
    print("   ✓ ModelLoader imported successfully")
    
    print("8. Importing DataProcessor...")
    from utils.data_processor import DataProcessor
    print("   ✓ DataProcessor imported successfully")
    
    print("9. Importing FileHandler...")
    from utils.file_handler import FileHandler
    print("   ✓ FileHandler imported successfully")
    
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 50)
print("CREATING FLASK APP")
print("=" * 50)

try:
    app = Flask(__name__)
    print("✓ Flask app created")
    
    app.config.from_object(Config)
    print("✓ Config loaded into app")
    
except Exception as e:
    print(f"FLASK APP CREATION ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 50)
print("INITIALIZING COMPONENTS")
print("=" * 50)

try:
    print("Loading ModelLoader...")
    model_loader = ModelLoader()
    print("✓ ModelLoader initialized")
    
    print("Loading DataProcessor...")
    data_processor = DataProcessor(model_loader.scaler, model_loader.encoder)
    print("✓ DataProcessor initialized")
    
    print("Loading FileHandler...")
    file_handler = FileHandler(app.config['UPLOAD_FOLDER'])
    print("✓ FileHandler initialized")
    
except Exception as e:
    print(f"COMPONENT INITIALIZATION ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 50)
print("DEFINING ROUTES")
print("=" * 50)

@app.route('/')
def home():
    print("Home route accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route accessed")
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be CSV format'})
        
        result = file_handler.handle_single_csv_upload(
            file, model_loader.model, data_processor
        )
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': f'File not found: {str(e)}'})

print("✓ Routes defined")

print("\n" + "=" * 50)
print("STARTING FLASK SERVER")
print("=" * 50)

if __name__ == '__main__':
    try:
        Config.create_upload_folder()
        print("✓ Upload folder created")
        
        print("Starting Flask development server...")
        print("Access the app at: http://localhost:5000")
        print("=" * 50)
        
        import os as _os
        _port = int(_os.getenv('PORT', '5000'))
        app.run(debug=True, host='0.0.0.0', port=_port)
        
    except Exception as e:
        print(f"SERVER START ERROR: {e}")
        import traceback
        traceback.print_exc()
