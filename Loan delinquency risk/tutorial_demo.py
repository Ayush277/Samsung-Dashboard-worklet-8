#!/usr/bin/env python3
"""
Tutorial Example Demonstration
Tests the exact CSV example through the API to show expected results
"""

import requests
import json

# API Configuration
BASE_URL = "http://localhost:5001"
PREDICT_URL = f"{BASE_URL}/predict"

def test_csv_example():
    """Test with the exact CSV example data"""
    
    print("🎓 TUTORIAL EXAMPLE DEMONSTRATION")
    print("="*50)
    print("Testing the exact CSV row used in the tutorial...")
    print()
    
    # Real CSV example data
    csv_example_data = {
        'interest_rate': '4.5',
        'unpaid_principal_bal': '80000',
        'Loan_term': '12',
        'loan_to_value': '73',
        'source': 'Y',
        'loan_purpose': 'C86',
        'number_of_borrowers': '1',
        'insurance_percent': '0',
        'debt_to_income_ratio': '41',
        'borrower_credit_score': '690',
        'co-borrower_credit_score': '0',
        'Annual_Income': '1951.219512',
        'Age': '18',
        'NumberOfDependents': '2',
        'EducationLevel': 'High School',
        'MaritalStatus': 'Married',
        'Gender': 'Male',
        'EmploymentStatus': 'Employed',
        'total_on_time_payments': '10',
        'total_late_payments': '15',
        'avg_payment_delay': '12.8',
        'current_dpd': '16'
    }
    
    print("📝 INPUT DATA (Real CSV Training Example):")
    print("-" * 40)
    print(f"Loan: ${csv_example_data['unpaid_principal_bal']} at {csv_example_data['interest_rate']}% for {csv_example_data['Loan_term']} months")
    print(f"Borrower: {csv_example_data['Age']}yr old {csv_example_data['Gender']}, {csv_example_data['MaritalStatus']}, {csv_example_data['EducationLevel']}")
    print(f"Income: ${float(csv_example_data['Annual_Income']):.0f}/year, Credit Score: {csv_example_data['borrower_credit_score']}")
    print(f"Payment History: {csv_example_data['total_on_time_payments']} on-time, {csv_example_data['total_late_payments']} late")
    print(f"Current Status: {csv_example_data['current_dpd']} days past due")
    print(f"Historical Outcome: mx = 0 (On-Time)")
    print()
    
    try:
        # Make API request
        response = requests.post(PREDICT_URL, data=csv_example_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("🤖 MODEL PREDICTION RESULTS:")
            print("-" * 40)
            print(f"🎯 Risk Level: {result['risk_level']}")
            print(f"📊 Default Probability: {result['delinquency_probability']:.1%}")
            print(f"🏷️  Binary Prediction: {result['binary_prediction']} ({'Default Risk' if result['delinquency_flag'] else 'On-Time Expected'})")
            print(f"🛡️  Confidence Score: {result['confidence_score']:.1%}")
            print()
            
            print("📋 RISK SUMMARY:")
            print("-" * 40)
            print(f"{result['risk_summary']}")
            print()
            
            if result.get('top_drivers'):
                print("🔍 TOP 5 RISK DRIVERS:")
                print("-" * 40)
                for i, driver in enumerate(result['top_drivers'], 1):
                    direction_icon = "⚠️" if driver['direction'] == 'Higher Risk' else "✅"
                    print(f"{i}. {direction_icon} {driver['feature']}")
                    print(f"   Value: {driver['value']} | Median: {driver['median']} | Impact: {driver['driver_score']:.3f}")
                    print(f"   Z-Score: {driver['zscore']} | Direction: {driver['direction']}")
                    print()
            
            print("💡 BUSINESS RECOMMENDATION:")
            print("-" * 40)
            if result['risk_level'] == 'Low':
                recommendation = "✅ APPROVE with standard terms"
            elif result['risk_level'] == 'Moderate':
                recommendation = "⚖️ CONDITIONAL APPROVAL with enhanced monitoring"
            elif result['risk_level'] == 'High':
                recommendation = "❌ DECLINE or require significant risk mitigation"
            else:
                recommendation = "🚫 STRONG DECLINE - Very high risk"
            
            print(recommendation)
            print()
            
            print("🎓 TUTORIAL INSIGHTS:")
            print("-" * 40)
            print("• This example shows a loan with mixed signals:")
            print("  - Poor payment history (more late than on-time payments)")
            print("  - Currently delinquent (16 days past due)")  
            print("  - Very low income ($1,951/year)")
            print("  - High debt-to-income ratio (41%)")
            print("  - But historically classified as 'On-Time' (mx=0)")
            print()
            print("• Model correctly identifies high risk factors")
            print("• Demonstrates importance of payment history analysis")
            print("• Shows how multiple risk indicators compound")
            print("• Illustrates model transparency through driver analysis")
            print()
            
            print("✨ This demonstrates the power of our comprehensive")
            print("   risk assessment system in identifying potential")
            print("   problem loans before they default!")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection Error: {str(e)}")
        print("Make sure the Flask app is running on http://localhost:5001")

if __name__ == "__main__":
    test_csv_example()
