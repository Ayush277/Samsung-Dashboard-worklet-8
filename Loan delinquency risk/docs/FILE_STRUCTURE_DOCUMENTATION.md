# Loan Application - File Structure Documentation

## Directory Structure Overview

```
loan_app/
├── Core Application Files
│   ├── app.py                          # Main Flask application
│   └── approach_train.csv              # Training dataset (116K+ records)
│
├── Model Artifacts (models/)
│   ├── tabpfn.pkl                     # Trained RandomForest classifier
│   ├── scaler.pkl                     # StandardScaler for numeric features  
│   ├── dummy_columns.pkl              # Column order for one-hot encoded features
│   └── medians.json                   # Median values for missing data imputation
│
├── Web Interface (templates/)
│   ├── index.html                     # Main UI - comprehensive loan risk form
│   └── index_old.html                 # Backup of original simple form
│
├── Testing & Validation (tests/)
│   ├── comprehensive_risk_test.py     # Full risk classification testing
│   ├── test_corrected_predictions.py  # Tests for prediction accuracy fix
│   ├── test_moderate_risk.py          # Specific moderate risk scenario tests
│   └── api_demo.py                    # API endpoint demonstration
│
├── Documentation (docs/)
│   ├── README_Enhanced.md             # Technical implementation guide
│   ├── Loan_Risk_Predictor_User_Guide.html  # User manual with screenshots
│   ├── workflow_diagram.html          # Visual ML pipeline explanation
│   ├── RESTRUCTURING_SUMMARY.md       # Major changes documentation
│   ├── TUTORIAL_GUIDE.md             # Tutorial system documentation
│   ├── TUTORIAL_IMPLEMENTATION.md    # Tutorial technical details
│   └── TUTORIAL_FORMATTING_COMPLETE.md # Tutorial completion notes
│
├── Demo & Examples (examples/)
│   └── tutorial_demo.py              # Interactive tutorial demonstrations
│
└── Build Artifacts (__pycache__/)
    └── app.cpython-313.pyc           # Compiled Python bytecode
```

---

## Detailed File Descriptions

### 🔧 **Core Application Files**

#### **`app.py`** - Main Flask Application
- **Purpose**: Primary web server handling loan risk predictions
- **Key Features**:
  - Flask routes: `/` (form), `/predict` (risk calculation)
  - ML pipeline: preprocessing → model prediction → risk classification
  - Auto-training fallback if model artifacts missing
  - Enhanced driver analysis with feature importance
  - Risk bucketing: Low (<25%), Moderate (25-50%), High (50-75%), Critical (≥75%)

- **Critical Functions**:
  ```python
  load_artifacts()      # Loads ML model and preprocessing artifacts
  preprocess()          # Handles missing data, validation, scaling
  predict()            # Main prediction endpoint with enhanced output
  _risk_bucket()       # Converts probability to risk level
  ```

#### **`approach_train.csv`** - Training Dataset  
- **Purpose**: Historical loan performance data for model training
- **Size**: 116,058 records × 31 features
- **Target Variable**: `mx` (0=default, 1=good loan) - **INVERTED FROM CONVENTION**
- **Key Features**: Credit scores, income, payment history, loan characteristics
- **Default Rate**: 69% (unusually high, indicates synthetic/research data)

---

### 🤖 **Model Artifacts (`models/`)**

#### **`tabpfn.pkl`** - Trained ML Model
- **Type**: RandomForestClassifier (160 trees, max_depth=12)
- **Training**: Balanced class weights, handles 69% default rate
- **Input**: 22 features (16 numeric + 6 categorical one-hot encoded)
- **Output**: Binary prediction + probability scores

#### **`scaler.pkl`** - Feature Standardization
- **Type**: StandardScaler from scikit-learn
- **Purpose**: Normalizes numeric features (mean=0, std=1)
- **Features**: All 16 numeric columns after one-hot encoding

#### **`dummy_columns.pkl`** - Feature Schema
- **Purpose**: Maintains exact column order for one-hot encoded features
- **Critical**: Ensures prediction-time features match training schema
- **Contains**: List of all feature names in correct order

#### **`medians.json`** - Missing Data Strategy
- **Purpose**: Median values for imputing missing numeric features
- **Structure**: `{"medians": {...}, "stds": {...}}`
- **Usage**: Fallback when users don't provide all input values

---

### 🌐 **Web Interface (`templates/`)**

#### **`index.html`** - Enhanced UI (Current)
- **Purpose**: Comprehensive loan application form
- **Features**:
  - 22 input fields matching CSV structure
  - Real-time validation with CSV data ranges
  - Professional two-column layout
  - Interactive tutorial system with 4 demo scenarios
  - Enhanced results display with risk drivers
  - Bootstrap 5 styling with gradient theme

- **Sections**:
  1. Loan Information (interest rate, amount, term, etc.)
  2. Borrower Profile (demographics, income, credit)
  3. Payment History (on-time/late payments, current status)

#### **`index_old.html`** - Simple UI (Backup)
- **Purpose**: Original basic form before enhancement
- **Usage**: Fallback/reference for minimal implementation

---

### 🧪 **Testing & Validation (`tests/`)**

#### **`comprehensive_risk_test.py`** - Full Test Suite
- **Purpose**: Validates risk classification across all scenarios
- **Tests**: Low (1.7%), Moderate (49.8%), High (56.2%), Critical scenarios
- **Validates**: Algorithm consistency, threshold accuracy
- **Usage**: `python3 comprehensive_risk_test.py`

#### **`test_corrected_predictions.py`** - Prediction Accuracy
- **Purpose**: Tests the critical fix for inverted target variable
- **Scenarios**: Low-risk (19.3%) vs High-risk (51.6%) profiles
- **Validates**: Model predictions align with expected financial logic

#### **`test_moderate_risk.py`** - Specific Risk Level
- **Purpose**: Debug moderate risk classification edge cases
- **Focus**: Scenarios that should yield 25-50% default probability

#### **`api_demo.py`** - API Documentation
- **Purpose**: Demonstrates programmatic API usage
- **Shows**: POST requests, response parsing, integration examples

---

### 📚 **Documentation (`docs/`)**

#### **`README_Enhanced.md`** - Technical Guide
- **Audience**: Developers and data scientists
- **Content**: Architecture, model details, API specifications
- **Includes**: Installation, configuration, troubleshooting

#### **`Loan_Risk_Predictor_User_Guide.html`** - User Manual
- **Audience**: End users and business stakeholders  
- **Content**: Step-by-step usage, screenshots, business interpretation
- **Features**: Visual workflow, risk level explanations

#### **`workflow_diagram.html`** - Visual Pipeline
- **Purpose**: Interactive diagram of ML pipeline
- **Shows**: Data flow from input → preprocessing → prediction → output

#### **`RESTRUCTURING_SUMMARY.md`** - Change Log
- **Purpose**: Documents major architectural changes
- **Content**: Migration from simple to comprehensive system
- **Includes**: Before/after comparisons, breaking changes

#### **Tutorial Documentation**:
- **`TUTORIAL_GUIDE.md`**: Overview of tutorial system
- **`TUTORIAL_IMPLEMENTATION.md`**: Technical implementation details  
- **`TUTORIAL_FORMATTING_COMPLETE.md`**: Completion status

---

### 🎯 **Demo & Examples (`examples/`)**

#### **`tutorial_demo.py`** - Interactive Demos
- **Purpose**: Programmatic generation of tutorial scenarios
- **Content**: Low/Moderate/High risk examples with explanations
- **Usage**: Can be integrated into web interface or run standalone

---

## Data Flow Architecture

```mermaid
graph TB
    A[User Input] --> B[app.py Flask Server]
    B --> C[preprocess() Function]
    C --> D[Load medians.json]
    C --> E[Apply scaler.pkl]  
    C --> F[Align dummy_columns.pkl]
    E --> G[tabpfn.pkl Model]
    F --> G
    G --> H[Risk Classification]
    H --> I[Enhanced Output + Drivers]
    I --> J[Web Interface Display]
    
    K[approach_train.csv] --> L[Auto-training]
    L --> M[Generate Artifacts]
    M --> D
    M --> E
    M --> F
    M --> G
```

---

## File Dependencies

### **Critical Path** (Required for operation):
1. `app.py` → Main application
2. `approach_train.csv` → Training data (if artifacts missing)
3. `models/tabpfn.pkl` → ML model
4. `models/scaler.pkl` → Feature scaling
5. `models/dummy_columns.pkl` → Feature schema
6. `models/medians.json` → Missing data handling
7. `templates/index.html` → User interface

### **Supporting Files** (Enhancement/Documentation):
- All files in `docs/` → Documentation and guides
- All files in `tests/` → Quality assurance
- `examples/tutorial_demo.py` → Interactive demonstrations
- `templates/index_old.html` → Legacy backup

---

## Recommended Reorganization

To improve maintainability, consider this structure:

```bash
# Create organized directories
mkdir -p loan_app/{tests,docs,examples}

# Move files to appropriate directories  
mv loan_app/*test*.py loan_app/tests/
mv loan_app/api_demo.py loan_app/tests/
mv loan_app/*.md loan_app/docs/
mv loan_app/*.html loan_app/docs/ (except templates/)
mv loan_app/tutorial_demo.py loan_app/examples/

# Keep in root: app.py, approach_train.csv, models/, templates/
```

This organization separates concerns and makes the project more professional and maintainable.
