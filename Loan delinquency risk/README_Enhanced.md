# Loan Risk Predictor - Enhanced Structure Documentation

## Overview
This enhanced loan risk prediction system analyzes comprehensive borrower and loan data to predict default probability using machine learning. The system has been restructured to match the actual training dataset structure and provide more detailed risk assessment.

## Data Structure Analysis

### Training Dataset Features (31 columns total):

#### Loan Information (8 features):
- `loan_id`: Unique loan identifier
- `source`: Loan origination channel (X, Y, Z)
- `financial_institution`: Lending institution name
- `interest_rate`: Annual interest rate percentage
- `unpaid_principal_bal`: Current outstanding balance
- `Loan_term`: Loan term in months
- `loan_to_value`: Loan-to-value ratio percentage
- `number_of_borrowers`: Number of borrowers (1 or 2)
- `insurance_percent`: Insurance coverage percentage
- `loan_purpose`: Purpose code (A23, B12, C86)

#### Borrower Demographics (7 features):
- `borrower_credit_score`: Primary borrower FICO score (300-850)
- `co-borrower_credit_score`: Co-borrower FICO score (if applicable)
- `Age`: Borrower age
- `Gender`: Borrower gender (Male, Female, Other)
- `EducationLevel`: Education level (High School, Bachelor's, Master's, PhD, Doctorate)
- `MaritalStatus`: Marital status (Single, Married, Divorced)
- `EmploymentStatus`: Employment status (Employed, Self-Employed, Unemployed)

#### Financial Information (4 features):
- `debt_to_income_ratio`: Monthly debt payments / monthly income
- `Annual_Income`: Gross annual income
- `NumberOfDependents`: Number of dependents
- `insurance_type`: Insurance type indicator

#### Payment History (6 features):
- `current_month`: Current loan month
- `total_on_time_payments`: Count of on-time payments
- `total_late_payments`: Count of late payments
- `avg_payment_delay`: Average delay in days for late payments
- `current_dpd`: Current days past due
- `mx`: Target variable (0=on-time, 1=delinquent)

#### Additional Fields (6 features):
- `origination_date`: Loan origination date
- `first_payment_date`: First payment due date
- `Occupation`: Borrower occupation
- `Borrower_State`: Borrower state

## Input Structure

### Required Fields (*):
```json
{
  "loan_information": {
    "interest_rate": "float (0-50)",
    "unpaid_principal_bal": "integer (positive)",
    "Loan_term": "integer (months)",
    "loan_to_value": "float (0-100)",
    "source": "string (X|Y|Z)",
    "loan_purpose": "string (A23|B12|C86)",
    "number_of_borrowers": "integer (1|2)"
  },
  "borrower_information": {
    "debt_to_income_ratio": "float (percentage)",
    "borrower_credit_score": "integer (300-850)",
    "Annual_Income": "float (positive)",
    "Age": "integer (18-100)",
    "NumberOfDependents": "integer (0-4+)",
    "EducationLevel": "string (High School|Bachelor's|Master's|PhD|Doctorate)",
    "MaritalStatus": "string (Single|Married|Divorced)",
    "Gender": "string (Male|Female|Other)",
    "EmploymentStatus": "string (Employed|Self-Employed|Unemployed)"
  },
  "optional_fields": {
    "co-borrower_credit_score": "integer (300-850)",
    "insurance_percent": "integer (0-100)",
    "total_on_time_payments": "integer (0+)",
    "total_late_payments": "integer (0+)",
    "avg_payment_delay": "float (0+)",
    "current_dpd": "integer (0+)"
  }
}
```

## Output Structure

### Comprehensive Risk Assessment:
```json
{
  "primary_indicators": {
    "binary_prediction": "integer (0|1)",
    "delinquency_probability": "float (0.0-1.0)",
    "risk_level": "string (Low|Moderate|High|Critical)",
    "delinquency_flag": "boolean",
    "confidence_score": "float (0.0-1.0)"
  },
  "risk_analysis": {
    "risk_summary": "string (detailed explanation)",
    "top_drivers": [
      {
        "feature": "string (feature name)",
        "value": "float (actual value)",
        "median": "float (portfolio median)",
        "zscore": "float (standard deviations from median)",
        "importance": "float (model feature importance)",
        "driver_score": "float (risk contribution score)",
        "direction": "string (Higher Risk|Lower Risk)"
      }
    ]
  },
  "loan_summary": {
    "loan_amount": "float",
    "interest_rate": "float",
    "loan_term": "integer",
    "ltv_ratio": "float",
    "borrower_age": "integer",
    "credit_score": "integer",
    "annual_income": "float",
    "debt_to_income": "float"
  },
  "model_info": {
    "features_analyzed": "integer (total features)",
    "numeric_features": "integer (count)",
    "categorical_features": "integer (count)",
    "model_type": "string"
  },
  "metadata": {
    "explanation": "string (model description)",
    "timestamp": "string (ISO format)"
  }
}
```

## Risk Level Thresholds

- **Low Risk**: Probability < 0.25 (Green)
- **Moderate Risk**: 0.25 ≤ Probability < 0.50 (Yellow) 
- **High Risk**: 0.50 ≤ Probability < 0.75 (Red)
- **Critical Risk**: Probability ≥ 0.75 (Purple)

## Workflow Example

### Input Example:
```json
{
  "interest_rate": 4.5,
  "unpaid_principal_bal": 250000,
  "Loan_term": 36,
  "loan_to_value": 80,
  "source": "X",
  "loan_purpose": "A23",
  "number_of_borrowers": 1,
  "debt_to_income_ratio": 35,
  "borrower_credit_score": 720,
  "Annual_Income": 75000,
  "Age": 35,
  "NumberOfDependents": 2,
  "EducationLevel": "Bachelor's",
  "MaritalStatus": "Married",
  "Gender": "Female",
  "EmploymentStatus": "Employed",
  "total_on_time_payments": 12,
  "total_late_payments": 1,
  "avg_payment_delay": 5.0,
  "current_dpd": 0
}
```

### Output Example:
```json
{
  "binary_prediction": 0,
  "delinquency_probability": 0.18,
  "risk_level": "Low",
  "delinquency_flag": false,
  "confidence_score": 0.64,
  "risk_summary": "Loan default probability: 18.0% (Low Risk). Model analyzed 16 numeric features and 6 categorical features. Risk assessment based on borrower profile, loan characteristics, and payment history patterns.",
  "top_drivers": [
    {
      "feature": "Borrower Credit Score",
      "value": 720,
      "median": 695,
      "zscore": 1.2,
      "importance": 0.15,
      "driver_score": 0.18,
      "direction": "Lower Risk"
    }
  ],
  "loan_summary": {
    "loan_amount": 250000,
    "interest_rate": 4.5,
    "loan_term": 36,
    "ltv_ratio": 80,
    "borrower_age": 35,
    "credit_score": 720,
    "annual_income": 75000,
    "debt_to_income": 35
  },
  "model_info": {
    "features_analyzed": 22,
    "numeric_features": 16,
    "categorical_features": 6,
    "model_type": "RandomForest Enhanced"
  }
}
```

## Key Improvements

1. **Comprehensive Feature Coverage**: All 31 CSV columns mapped to input fields
2. **Enhanced Validation**: Input validation with reasonable ranges
3. **Detailed Risk Drivers**: Top 5 risk factors with explanations
4. **Professional UI**: Modern, responsive interface with clear sections
5. **Rich Output**: Multiple risk indicators and detailed analysis
6. **Real-time Feedback**: Immediate comprehensive risk assessment

## Usage Instructions

1. **Access the Application**:
   - Standard version: `http://localhost:5000/`
   - Enhanced version: `http://localhost:5000/comprehensive`

2. **Fill Required Fields**: Complete all fields marked with (*)

3. **Optional Enhancement**: Add payment history for more accurate predictions

4. **Analyze Risk**: Click "Analyze Risk Profile" for comprehensive results

5. **Interpret Results**: 
   - Risk Level: Overall assessment (Low/Moderate/High/Critical)
   - Probability: Numerical likelihood of default
   - Drivers: Key factors influencing the prediction
   - Recommendation: Suggested action based on risk level

## Technical Details

- **Model**: RandomForest Classifier with 160 estimators
- **Features**: 16 numeric + 6 categorical = 22 total features
- **Training Data**: 116,058 historical loan records
- **Target Distribution**: 69% defaults (class 1), 31% on-time (class 0)
- **Preprocessing**: Standardization, median imputation, one-hot encoding
- **Risk Calibration**: Probability thresholds based on business requirements
