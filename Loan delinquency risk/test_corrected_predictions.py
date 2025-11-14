#!/usr/bin/env python3
"""
Test script to verify that the corrected prediction logic works properly.
This tests both low-risk and high-risk scenarios.
"""

import requests
import json

# Test URL
URL = "http://localhost:5002/predict"

# Test Case 1: Low Risk Profile (should have low default probability ~15-25%)
low_risk_data = {
    'interest_rate': '3.5',
    'unpaid_principal_bal': '250000',  
    'Loan_term': '30',
    'loan_to_value': '75',
    'number_of_borrowers': '2',
    'debt_to_income_ratio': '25',
    'borrower_credit_score': '780',
    'insurance_percent': '0',
    'co-borrower_credit_score': '770', 
    'Age': '35',
    'NumberOfDependents': '2',
    'Annual_Income': '8500',
    'total_on_time_payments': '15',
    'total_late_payments': '1',
    'avg_payment_delay': '2',
    'current_dpd': '0',
    'source': 'X',
    'loan_purpose': 'A23',
    'EducationLevel': "Bachelor's",
    'MaritalStatus': 'Married',
    'Gender': 'Male',
    'EmploymentStatus': 'Employed'
}

# Test Case 2: High Risk Profile (should have high default probability ~60-80%)
high_risk_data = {
    'interest_rate': '8.5',
    'unpaid_principal_bal': '180000',
    'Loan_term': '15', 
    'loan_to_value': '95',
    'number_of_borrowers': '1',
    'debt_to_income_ratio': '45',
    'borrower_credit_score': '620',
    'insurance_percent': '30',
    'co-borrower_credit_score': '0',
    'Age': '55',
    'NumberOfDependents': '3',
    'Annual_Income': '4500',
    'total_on_time_payments': '3',
    'total_late_payments': '12',
    'avg_payment_delay': '25',
    'current_dpd': '30',
    'source': 'Z',
    'loan_purpose': 'C86',
    'EducationLevel': 'High School',
    'MaritalStatus': 'Divorced',
    'Gender': 'Female', 
    'EmploymentStatus': 'Unemployed'
}

def test_prediction(data, case_name):
    """Test a prediction and print results."""
    try:
        response = requests.post(URL, data=data)
        if response.status_code == 200:
            result = response.json()
            print(f"\n=== {case_name} ===")
            print(f"Default Probability: {result['delinquency_probability']:.1%}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Binary Prediction: {result['binary_prediction']} ({'Default' if result['binary_prediction'] else 'Good'})")
            print(f"Confidence: {result['confidence_score']:.3f}")
            
            if result.get('top_drivers'):
                print("Top Risk Drivers:")
                for i, driver in enumerate(result['top_drivers'][:3], 1):
                    print(f"  {i}. {driver['feature']}: {driver['value']} (median: {driver['median']}) - {driver['direction']}")
            
            return result['delinquency_probability']
        else:
            print(f"\n=== {case_name} FAILED ===")
            print(f"Status: {response.status_code}")
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"\n=== {case_name} ERROR ===")
        print(f"Exception: {e}")
        return None

if __name__ == "__main__":
    print("Testing Corrected Loan Risk Predictions")
    print("=" * 50)
    
    # Test both scenarios
    low_risk_prob = test_prediction(low_risk_data, "LOW RISK PROFILE")
    high_risk_prob = test_prediction(high_risk_data, "HIGH RISK PROFILE")
    
    # Validate results
    print(f"\n{'='*50}")
    print("VALIDATION RESULTS:")
    
    if low_risk_prob is not None and high_risk_prob is not None:
        print(f"✓ Low Risk Probability:  {low_risk_prob:.1%}")
        print(f"✓ High Risk Probability: {high_risk_prob:.1%}")
        
        # Check if the ordering makes sense
        if high_risk_prob > low_risk_prob:
            print("✅ PASS: High risk profile has higher default probability than low risk")
            
            # Check reasonable ranges
            if low_risk_prob < 0.4:  # Should be under 40%
                print("✅ PASS: Low risk profile has reasonable default probability")
            else:
                print(f"⚠️  WARNING: Low risk probability still seems high ({low_risk_prob:.1%})")
                
        else:
            print("❌ FAIL: Risk ordering is still incorrect")
    else:
        print("❌ FAIL: Could not get predictions for both test cases")
    
    print(f"{'='*50}")
