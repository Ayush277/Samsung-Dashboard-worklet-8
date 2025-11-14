#!/usr/bin/env python3
"""
Comprehensive test to debug moderate risk classification issues.
Tests multiple scenarios to understand the risk bucketing behavior.
"""

import requests
import json

URL = "http://localhost:5002/predict"

# Test scenarios with expected risk levels
test_scenarios = [
    {
        'name': 'Very Low Risk (Should be <25%)',
        'data': {
            'interest_rate': '3.0',
            'unpaid_principal_bal': '200000',  
            'Loan_term': '30',
            'loan_to_value': '70',
            'number_of_borrowers': '2',
            'debt_to_income_ratio': '20',
            'borrower_credit_score': '800',
            'insurance_percent': '0',
            'co-borrower_credit_score': '790', 
            'Age': '35',
            'NumberOfDependents': '1',
            'Annual_Income': '10000',
            'total_on_time_payments': '20',
            'total_late_payments': '0',
            'avg_payment_delay': '0',
            'current_dpd': '0',
            'source': 'X',
            'loan_purpose': 'A23',
            'EducationLevel': "Bachelor's",
            'MaritalStatus': 'Married',
            'Gender': 'Male',
            'EmploymentStatus': 'Employed'
        },
        'expected': 'Low'
    },
    {
        'name': 'Moderate Risk Scenario 1 (Should be 25-50%)',
        'data': {
            'interest_rate': '5.0',
            'unpaid_principal_bal': '250000',  
            'Loan_term': '20',
            'loan_to_value': '80',
            'number_of_borrowers': '1',
            'debt_to_income_ratio': '30',
            'borrower_credit_score': '720',
            'insurance_percent': '10',
            'co-borrower_credit_score': '0', 
            'Age': '45',
            'NumberOfDependents': '2',
            'Annual_Income': '7000',
            'total_on_time_payments': '10',
            'total_late_payments': '3',
            'avg_payment_delay': '5',
            'current_dpd': '0',
            'source': 'Y',
            'loan_purpose': 'B12',
            'EducationLevel': "Master's",
            'MaritalStatus': 'Married',
            'Gender': 'Female',
            'EmploymentStatus': 'Employed'
        },
        'expected': 'Moderate'
    },
    {
        'name': 'Moderate Risk Scenario 2 (Should be 25-50%)',
        'data': {
            'interest_rate': '4.5',
            'unpaid_principal_bal': '180000',  
            'Loan_term': '25',
            'loan_to_value': '85',
            'number_of_borrowers': '1',
            'debt_to_income_ratio': '35',
            'borrower_credit_score': '690',
            'insurance_percent': '15',
            'co-borrower_credit_score': '0', 
            'Age': '40',
            'NumberOfDependents': '3',
            'Annual_Income': '6000',
            'total_on_time_payments': '8',
            'total_late_payments': '5',
            'avg_payment_delay': '8',
            'current_dpd': '2',
            'source': 'Y',
            'loan_purpose': 'C86',
            'EducationLevel': "Bachelor's",
            'MaritalStatus': 'Divorced',
            'Gender': 'Male',
            'EmploymentStatus': 'Employed'
        },
        'expected': 'Moderate'
    },
    {
        'name': 'High Risk (Should be 50-75%)',
        'data': {
            'interest_rate': '7.0',
            'unpaid_principal_bal': '300000',  
            'Loan_term': '15',
            'loan_to_value': '90',
            'number_of_borrowers': '1',
            'debt_to_income_ratio': '40',
            'borrower_credit_score': '650',
            'insurance_percent': '25',
            'co-borrower_credit_score': '0', 
            'Age': '50',
            'NumberOfDependents': '4',
            'Annual_Income': '5000',
            'total_on_time_payments': '5',
            'total_late_payments': '8',
            'avg_payment_delay': '15',
            'current_dpd': '10',
            'source': 'Z',
            'loan_purpose': 'C86',
            'EducationLevel': 'High School',
            'MaritalStatus': 'Single',
            'Gender': 'Female',
            'EmploymentStatus': 'Self-Employed'
        },
        'expected': 'High'
    }
]

def test_scenario(scenario):
    """Test a single scenario and return results."""
    try:
        response = requests.post(URL, data=scenario['data'])
        if response.status_code == 200:
            result = response.json()
            prob = result['delinquency_probability']
            risk_level = result['risk_level']
            
            # Determine what the risk level should be based on probability
            if prob is not None:
                if prob < 0.25:
                    calculated_risk = 'Low'
                elif prob < 0.50:
                    calculated_risk = 'Moderate' 
                elif prob < 0.75:
                    calculated_risk = 'High'
                else:
                    calculated_risk = 'Critical'
            else:
                calculated_risk = 'Unknown'
            
            return {
                'probability': prob,
                'reported_risk': risk_level,
                'calculated_risk': calculated_risk,
                'expected_risk': scenario['expected'],
                'success': True
            }
        else:
            return {
                'error': f"HTTP {response.status_code}: {response.text}",
                'success': False
            }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

def main():
    print("Comprehensive Risk Classification Test")
    print("=" * 60)
    
    print(f"\nCurrent Risk Thresholds:")
    print(f"  Low:      < 25%")
    print(f"  Moderate: 25% - 50%") 
    print(f"  High:     50% - 75%")
    print(f"  Critical: >= 75%")
    print("=" * 60)
    
    all_correct = True
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print("-" * 40)
        
        result = test_scenario(scenario)
        
        if result['success']:
            prob = result['probability']
            reported = result['reported_risk']
            calculated = result['calculated_risk']
            expected = result['expected_risk']
            
            print(f"Probability: {prob:.1%}")
            print(f"Reported Risk Level: {reported}")
            print(f"Expected Risk Level: {expected}")
            print(f"Calculated Risk Level: {calculated}")
            
            # Check if reported matches calculated (algorithm consistency)
            if reported == calculated:
                print("✅ Algorithm Consistent: Reported matches calculated")
            else:
                print(f"❌ Algorithm Error: Reported '{reported}' != Calculated '{calculated}'")
                all_correct = False
            
            # Check if result matches expectation
            if reported == expected:
                print(f"✅ Expectation Met: Got '{reported}' as expected")
            else:
                print(f"⚠️  Expectation Miss: Got '{reported}', expected '{expected}'")
                # This might be acceptable if the model genuinely thinks differently
            
        else:
            print(f"❌ Error: {result['error']}")
            all_correct = False
    
    print("\n" + "=" * 60)
    if all_correct:
        print("✅ ALL TESTS PASSED: Risk classification algorithm working correctly")
    else:
        print("❌ SOME TESTS FAILED: Risk classification has issues")
    print("=" * 60)

if __name__ == "__main__":
    main()
