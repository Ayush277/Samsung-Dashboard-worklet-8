#!/usr/bin/env python3
"""
Loan Delinquency Risk Assessment - Inference Demo Script

This script demonstrates how to use the loan risk assessment system
for both individual predictions and batch processing.

Author: Samsung PRISM Worklet 8 Team
Date: November 2024
"""

import requests
import pandas as pd
import json
import time
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://127.0.0.1:5001"
DEMO_DATA_PATH = Path("demo_data")

def check_service_health():
    """Check if the loan risk service is running"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Loan Risk Assessment service is running")
            return True
        else:
            print(f"❌ Service returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to service. Make sure it's running on port 5001")
        return False
    except Exception as e:
        print(f"❌ Error checking service: {e}")
        return False

def demo_individual_prediction():
    """Demonstrate individual loan risk prediction"""
    print("\n🔍 Individual Loan Risk Assessment Demo")
    print("=" * 50)
    
    # Sample loan application data
    sample_applications = [
        {
            "age": 35,
            "income": 75000,
            "employment_type": "Permanent",
            "credit_score": 720,
            "loan_amount": 25000,
            "loan_purpose": "Home Improvement",
            "existing_debt": 5000,
            "education_level": "Bachelor's Degree",
            "marital_status": "Married",
            "dependents": 2
        },
        {
            "age": 28,
            "income": 45000,
            "employment_type": "Contract",
            "credit_score": 650,
            "loan_amount": 15000,
            "loan_purpose": "Personal",
            "existing_debt": 8000,
            "education_level": "High School",
            "marital_status": "Single",
            "dependents": 0
        },
        {
            "age": 42,
            "income": 95000,
            "employment_type": "Permanent",
            "credit_score": 780,
            "loan_amount": 50000,
            "loan_purpose": "Business",
            "existing_debt": 2000,
            "education_level": "Master's Degree",
            "marital_status": "Married",
            "dependents": 1
        }
    ]
    
    for i, application in enumerate(sample_applications, 1):
        print(f"\n📋 Application {i}:")
        print(f"   Age: {application['age']}, Income: ${application['income']:,}")
        print(f"   Credit Score: {application['credit_score']}, Loan Amount: ${application['loan_amount']:,}")
        print(f"   Purpose: {application['loan_purpose']}, Employment: {application['employment_type']}")
        
        try:
            response = requests.post(f"{BASE_URL}/predict", data=application)
            if response.status_code == 200:
                result = response.json()
                risk_score = result.get('risk_score', 'N/A')
                risk_level = result.get('risk_level', 'Unknown')
                decision = result.get('decision', 'Unknown')
                confidence = result.get('confidence', 'N/A')
                
                print(f"   🎯 Risk Score: {risk_score}")
                print(f"   📊 Risk Level: {risk_level}")
                print(f"   ✅ Decision: {decision}")
                print(f"   🔒 Confidence: {confidence}")
                
                # Color coding for risk levels
                if isinstance(risk_score, (int, float)):
                    if risk_score < 0.3:
                        print("   💚 LOW RISK - Approve with standard terms")
                    elif risk_score < 0.7:
                        print("   🟡 MEDIUM RISK - Manual review recommended")
                    else:
                        print("   🔴 HIGH RISK - Reject or require additional security")
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
        
        time.sleep(1)  # Brief pause between requests

def create_sample_batch_data():
    """Create sample CSV data for batch processing demo"""
    DEMO_DATA_PATH.mkdir(exist_ok=True)
    
    sample_data = [
        {
            "applicant_id": "APP_001",
            "age": 32,
            "income": 65000,
            "employment_type": "Permanent",
            "credit_score": 710,
            "loan_amount": 30000,
            "loan_purpose": "Home",
            "existing_debt": 4000,
            "education_level": "Bachelor's",
            "marital_status": "Single",
            "dependents": 0
        },
        {
            "applicant_id": "APP_002", 
            "age": 45,
            "income": 85000,
            "employment_type": "Self-Employed",
            "credit_score": 680,
            "loan_amount": 40000,
            "loan_purpose": "Business",
            "existing_debt": 12000,
            "education_level": "Master's",
            "marital_status": "Married",
            "dependents": 3
        },
        {
            "applicant_id": "APP_003",
            "age": 29,
            "income": 55000,
            "employment_type": "Contract",
            "credit_score": 640,
            "loan_amount": 20000,
            "loan_purpose": "Personal",
            "existing_debt": 7000,
            "education_level": "High School",
            "marital_status": "Single", 
            "dependents": 1
        },
        {
            "applicant_id": "APP_004",
            "age": 38,
            "income": 120000,
            "employment_type": "Permanent",
            "credit_score": 800,
            "loan_amount": 75000,
            "loan_purpose": "Investment",
            "existing_debt": 1000,
            "education_level": "PhD",
            "marital_status": "Married",
            "dependents": 2
        },
        {
            "applicant_id": "APP_005",
            "age": 25,
            "income": 40000,
            "employment_type": "Part-time",
            "credit_score": 590,
            "loan_amount": 10000,
            "loan_purpose": "Education",
            "existing_debt": 15000,
            "education_level": "College",
            "marital_status": "Single",
            "dependents": 0
        }
    ]
    
    df = pd.DataFrame(sample_data)
    batch_file = DEMO_DATA_PATH / "loan_applications_batch.csv"
    df.to_csv(batch_file, index=False)
    print(f"✅ Created sample batch file: {batch_file}")
    return batch_file

def demo_batch_prediction():
    """Demonstrate batch loan risk prediction"""
    print("\n📊 Batch Loan Risk Assessment Demo")
    print("=" * 50)
    
    # Create sample data
    batch_file = create_sample_batch_data()
    
    print(f"\n📤 Uploading batch file: {batch_file.name}")
    print(f"   File size: {batch_file.stat().st_size} bytes")
    
    try:
        with open(batch_file, 'rb') as f:
            files = {'file': (batch_file.name, f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/batch-predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch processing completed successfully!")
            print(f"   Records processed: {result.get('records_processed', 'N/A')}")
            print(f"   Download URL: {result.get('download_url', 'N/A')}")
            
            # Download results if available
            if 'download_url' in result:
                download_response = requests.get(f"{BASE_URL}{result['download_url']}")
                if download_response.status_code == 200:
                    results_file = DEMO_DATA_PATH / "loan_risk_results.csv"
                    with open(results_file, 'wb') as f:
                        f.write(download_response.content)
                    print(f"📥 Results downloaded: {results_file}")
                    
                    # Display sample results
                    results_df = pd.read_csv(results_file)
                    print(f"\n📋 Sample Results (first 3 rows):")
                    print(results_df.head(3).to_string(index=False))
        else:
            print(f"❌ Batch processing failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Batch processing error: {e}")

def demo_api_endpoints():
    """Demonstrate all available API endpoints"""
    print("\n🔌 API Endpoints Demo")
    print("=" * 50)
    
    endpoints = [
        ("GET", "/", "Main interface"),
        ("POST", "/predict", "Individual prediction"),
        ("POST", "/batch-predict", "Batch processing"),
        ("GET", "/health", "Health check (if available)")
    ]
    
    print("Available endpoints:")
    for method, endpoint, description in endpoints:
        print(f"   {method:4} {endpoint:15} - {description}")

def display_sample_curl_commands():
    """Show sample cURL commands for API testing"""
    print("\n🖥️  Sample cURL Commands")
    print("=" * 50)
    
    curl_commands = [
        {
            "description": "Individual prediction",
            "command": """curl -X POST http://127.0.0.1:5001/predict \\
  -H 'Content-Type: application/x-www-form-urlencoded' \\
  --data-urlencode 'age=35' \\
  --data-urlencode 'income=75000' \\
  --data-urlencode 'employment_type=Permanent' \\
  --data-urlencode 'credit_score=720' \\
  --data-urlencode 'loan_amount=25000' \\
  --data-urlencode 'loan_purpose=Home' \\
  --data-urlencode 'existing_debt=5000'"""
        },
        {
            "description": "Batch prediction",
            "command": """curl -X POST http://127.0.0.1:5001/batch-predict \\
  -F 'file=@demo_data/loan_applications_batch.csv'"""
        }
    ]
    
    for cmd in curl_commands:
        print(f"\n{cmd['description']}:")
        print(cmd['command'])

def main():
    """Main demo function"""
    print("🏦 Loan Delinquency Risk Assessment - Inference Demo")
    print("🎯 Samsung PRISM Worklet 8")
    print("=" * 60)
    
    # Check if service is running
    if not check_service_health():
        print("\n🚀 To start the service, run:")
        print("   cd 'Loan delinquency risk'")
        print("   python app.py")
        sys.exit(1)
    
    try:
        # Run demos
        demo_individual_prediction()
        demo_batch_prediction()
        demo_api_endpoints()
        display_sample_curl_commands()
        
        print("\n✅ Demo completed successfully!")
        print("\n📋 Summary:")
        print("   - Individual predictions: Real-time risk assessment")
        print("   - Batch processing: Bulk application analysis")
        print("   - API integration: Ready for system integration")
        print("   - Professional output: Business-ready results")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    
    print("\n🎯 For more information, see the complete documentation")
    print("   Visual Documentation & Schematics/index.html")

if __name__ == "__main__":
    main()
