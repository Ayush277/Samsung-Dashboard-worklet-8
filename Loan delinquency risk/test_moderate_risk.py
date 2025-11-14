#!/usr/bin/env python3
"""
Test moderate risk scenario to debug the risk classification issue.
"""

import requests
import json

URL = "http://localhost:5002/predict"

# Moderate Risk Profile (should have ~30-40% default probability)
moderate_risk_data = {
    'interest_rate': '5.5',
    'unpaid_principal_bal': '200000',  
    'Loan_term': '25',
    'loan_to_value': '85',
    'number_of_borrowers': '1',
    'debt_to_income_ratio': '35',
    'borrower_credit_score': '700',
    'insurance_percent': '15',
    'co-borrower_credit_score': '0', 
    'Age': '40',
    'NumberOfDependents': '2',
    'Annual_Income': '6500',
    'total_on_time_payments': '8',
    'total_late_payments': '4',
    'avg_payment_delay': '8',
    'current_dpd': '5',
    'source': 'Y',
    'loan_purpose': 'B12',
    'EducationLevel': "Master's",
    'MaritalStatus': 'Married',
    'Gender': 'Female',
    'EmploymentStatus': 'Employed'
}

def test_moderate_risk():
    """Test moderate risk prediction."""
    try:
        response = requests.post(URL, data=moderate_risk_data)
        if response.status_code == 200:
            result = response.json()
            print("=== MODERATE RISK TEST ===")
            print(f"Default Probability: {result['delinquency_probability']:.1%}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Binary Prediction: {result['binary_prediction']} ({'Default' if result['binary_prediction'] else 'Good'})")
            
            # Debug the thresholds
            prob = result['delinquency_probability']
            print(f"\nDebug Risk Classification:")
            print(f"Probability: {prob}")
            print(f"< 0.25 (Low): {prob < 0.25}")
            print(f"< 0.50 (Moderate): {prob < 0.50}")
            print(f"< 0.75 (High): {prob < 0.75}")
            print(f">= 0.75 (Critical): {prob >= 0.75}")
            
            return result
        else:
            print(f"ERROR: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

if __name__ == "__main__":
    test_moderate_risk()
