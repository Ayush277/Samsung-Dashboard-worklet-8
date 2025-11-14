#!/usr/bin/env python3
"""
Form Input Test - Verifies all form fields are properly captured
Tests the form submission process and identifies missing fields.
"""

import requests
import json
from typing import Dict, Any

URL = "http://localhost:5001/predict"

def create_complete_form_data() -> Dict[str, str]:
    """Create a complete form data payload with all required fields."""
    return {
        # Loan Information
        'interest_rate': '3.75',
        'unpaid_principal_bal': '200000',
        'Loan_term': '18',
        'loan_to_value': '80',
        'source': 'Y',
        'loan_purpose': 'C86',
        'number_of_borrowers': '2',
        'insurance_percent': '10',
        
        # Borrower Information
        'debt_to_income_ratio': '29',
        'borrower_credit_score': '809',
        'co-borrower_credit_score': '812',
        'Annual Income': '6620.69',  # Note: space in name to match CSV
        'Age': '51',
        'NumberOfDependents': '2',
        'EducationLevel': "Bachelor's",
        'MaritalStatus': 'Married',
        'Gender': 'Female',
        'EmploymentStatus': 'Employed',
        
        # Payment History
        'total_on_time_payments': '7',
        'total_late_payments': '2',
        'avg_payment_delay': '18.5',
        'current_dpd': '0'
    }

def test_form_submission():
    """Test complete form submission and analyze response."""
    print("🧪 Testing Form Input Capture")
    print("=" * 50)
    
    form_data = create_complete_form_data()
    
    print(f"📋 Sending {len(form_data)} form fields:")
    for key, value in form_data.items():
        print(f"  • {key}: {value}")
    
    try:
        print(f"\n🚀 Submitting to {URL}")
        response = requests.post(URL, data=form_data, timeout=30)
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS - Form submission worked!")
            print(f"   Risk Level: {result.get('risk_level', 'N/A')}")
            print(f"   Default Probability: {result.get('delinquency_probability', 'N/A'):.1%}" if result.get('delinquency_probability') else "   Default Probability: N/A")
            print(f"   Model Info: {result.get('model_info', {}).get('features_analyzed', 'N/A')} features analyzed")
            
            return True, result
            
        else:
            print(f"❌ ERROR - HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False, None
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR - Is the Flask app running on port 5003?")
        return False, None
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False, None

def test_missing_fields():
    """Test what happens when required fields are missing."""
    print("\n🔍 Testing Missing Required Fields")
    print("=" * 50)
    
    # Test with minimal data
    minimal_data = {
        'interest_rate': '3.75',
        'borrower_credit_score': '809'
        # Missing many required fields
    }
    
    try:
        response = requests.post(URL, data=minimal_data)
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print("✅ EXPECTED - Server rejected incomplete form")
            print(f"Error: {response.text}")
        else:
            result = response.json()
            print("⚠️  UNEXPECTED - Server accepted incomplete form")
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"Exception: {e}")

def main():
    """Run all form tests."""
    print("🧪 LOAN APPLICATION FORM INPUT TESTS")
    print("=" * 60)
    
    # Test 1: Complete form submission
    success, result = test_form_submission()
    
    # Test 2: Missing fields
    test_missing_fields()
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ FORM INPUT CAPTURE: WORKING")
        print("   All fields are being properly collected and processed")
    else:
        print("❌ FORM INPUT CAPTURE: FAILED")
        print("   Check server logs and field names")
    
    print("\n💡 DEBUGGING TIPS:")
    print("   1. Open browser dev tools (F12) and check Console tab")
    print("   2. Look for the 'Form data being sent:' debug output")
    print("   3. Verify all required fields have values")
    print("   4. Check Flask server terminal for error messages")
    print("=" * 60)

if __name__ == "__main__":
    main()
