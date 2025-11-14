# 🏗️ PRISM Worklet 8 - Complete System Architecture

```mermaid
graph TB
    %% === USER INTERFACE LAYER ===
    subgraph "👥 User Interface Layer"
        WEB["`🌐 **Web Browser**
        Chrome/Firefox/Safari`"]
        
        MOBILE["`📱 **Mobile Interface**
        Responsive Design`"]
        
        API_CLIENT["`🔗 **API Clients**
        cURL/Postman/Python`"]
    end
    
    %% === DASHBOARD ORCHESTRATION ===
    subgraph "🎛️ Dashboard Orchestration (Port 5050)"
        DASH_APP["`📊 **Main Dashboard**
        Flask Application
        • Unified Control Panel
        • Application Management
        • Process Orchestration`"]
        
        DASH_ROUTES["`🔀 **Dashboard Routes**
        / → Main Interface
        /open/<app> → Launch Apps
        /api/start → Start Services
        /api/stop → Stop Services`"]
        
        PROC_MGR["`⚙️ **Process Manager**
        • Subprocess Control
        • Port Management
        • Health Monitoring
        • Log Aggregation`"]
        
        DASH_LOGS["`📝 **Log Manager**
        logs/loan.log
        logs/campaign.log
        logs/sales.log`"]
    end
    
    %% === CORE AI APPLICATIONS ===
    subgraph "🤖 Core AI Applications"
        %% LOAN RISK APPLICATION
        subgraph "🏦 Loan Delinquency Risk (Port 7001)"
            LOAN_APP["`💰 **Flask App**
            Risk Assessment Engine`"]
            
            LOAN_MODELS["`🧠 **ML Models**
            • TabPFN Foundation Model
            • Random Forest Ensemble
            • XGBoost Classifier
            • SVM Risk Scorer`"]
            
            LOAN_API["`🔌 **API Endpoints**
            / → Risk Form
            /predict → Individual Assessment
            /batch → Bulk Processing`"]
            
            LOAN_DATA["`📊 **Data Layer**
            • approach_train.csv (116K records)
            • models/ (TabPFN, Scaler, Encoder)
            • Feature Engineering Pipeline`"]
        end
        
        %% CAMPAIGN PERFORMANCE APPLICATION  
        subgraph "📈 Campaign Performance (Port 7002)"
            CAMP_APP["`📊 **Flask App**
            Marketing Analytics Engine`"]
            
            CAMP_MODELS["`🧠 **ML Models**
            • CatBoost Regressor
            • LightGBM Ensemble
            • Ridge Regression
            • TabPFN Integration`"]
            
            CAMP_API["`🔌 **API Endpoints**
            / → Performance Dashboard
            /predict → Store Analysis
            /upload → Batch Analysis`"]
            
            CAMP_DATA["`📊 **Data Layer**
            • train2.csv (Store Performance)
            • test_store_performance.csv
            • Model Artifacts (CatBoost, LGBM, Ridge)`"]
        end
        
        %% SALES FORECASTING APPLICATION
        subgraph "📊 Sales Uplift Forecasting (Port 7003)"
            SALES_APP["`📈 **Flask App**
            Sales Prediction Engine`"]
            
            SALES_MODELS["`🧠 **ML Models**
            • XGBoost Forecaster
            • Time Series Analysis
            • Feature Engineering
            • Uplift Calculation`"]
            
            SALES_API["`🔌 **API Endpoints**
            / → Forecast Interface
            /predict → Sales Prediction
            /upload → Batch Forecasting`"]
            
            SALES_DATA["`📊 **Data Layer**
            • train.csv (Historical Sales)
            • store.csv (Store Metadata)
            • Encoder/Scaler Pipeline`"]
        end
    end
    
    %% === FOUNDATION LAYER ===
    subgraph "🔧 Foundation & Infrastructure"
        subgraph "🐍 Python Runtime Environment"
            PYTHON["`🐍 **Python 3.10+**
            Virtual Environment
            • Flask Framework
            • Pandas/NumPy
            • Scikit-learn
            • XGBoost/LightGBM/CatBoost`"]
            
            TABPFN["`🚀 **TabPFN Foundation**
            Advanced Tabular ML
            • Pre-trained Models
            • Zero-shot Learning
            • Feature Automation`"]
        end
        
        subgraph "💾 Data Storage"
            CSV_DATA["`📄 **CSV Datasets**
            • Training Data
            • Test Data  
            • Sample Data`"]
            
            MODEL_STORE["`🗃️ **Model Artifacts**
            • .pkl Model Files
            • Scalers & Encoders
            • Feature Schemas
            • Metadata`"]
            
            UPLOADS["`📤 **Upload Storage**
            • User CSV Files
            • Batch Processing
            • Temporary Data`"]
        end
        
        subgraph "📊 Analytics & Logging"
            LOGS["`📝 **Application Logs**
            • Debug Information
            • Performance Metrics
            • Error Tracking`"]
            
            DEMO_LOGS["`📋 **Demo Scripts**
            • Inference Testing
            • Health Monitoring
            • Integration Validation`"]
        end
    end
    
    %% === DOCUMENTATION & VISUALIZATION ===
    subgraph "📚 Documentation & Visual Assets"
        VIS_DOCS["`🎨 **Visual Documentation**
        Visual Documentation & Schematics/
        • Interactive Diagrams
        • System Flows
        • API Sequences
        • Architecture Guides`"]
        
        README_DOCS["`📖 **README Documentation**
        • Main Project Guide
        • Individual App READMEs
        • Samsung Mentors Guide
        • Setup Instructions`"]
        
        DEMO_SCRIPTS["`🔬 **Inference Scripts**
        • loan_risk_inference_demo.py
        • campaign_performance_inference_demo.py
        • sales_uplift_inference_demo.py`"]
    end
    
    %% === EXTERNAL INTEGRATIONS ===
    subgraph "🌐 External Integration Points"
        SAMSUNG_API["`🏢 **Samsung Systems**
        Future Integration Points
        • Enterprise APIs
        • Data Warehouses
        • Business Intelligence`"]
        
        CLOUD_DEPLOY["`☁️ **Cloud Deployment**
        Production Ready
        • AWS/Azure Integration
        • Docker Containerization
        • Kubernetes Orchestration`"]
    end
    
    %% === CONNECTION FLOWS ===
    
    %% User Interface Connections
    WEB --> DASH_APP
    MOBILE --> DASH_APP
    API_CLIENT --> LOAN_API
    API_CLIENT --> CAMP_API
    API_CLIENT --> SALES_API
    
    %% Dashboard Orchestration Flows
    DASH_APP --> DASH_ROUTES
    DASH_ROUTES --> PROC_MGR
    PROC_MGR --> LOAN_APP
    PROC_MGR --> CAMP_APP
    PROC_MGR --> SALES_APP
    PROC_MGR --> DASH_LOGS
    
    %% Application Internal Flows
    LOAN_APP --> LOAN_API
    LOAN_API --> LOAN_MODELS
    LOAN_MODELS --> LOAN_DATA
    
    CAMP_APP --> CAMP_API
    CAMP_API --> CAMP_MODELS
    CAMP_MODELS --> CAMP_DATA
    
    SALES_APP --> SALES_API
    SALES_API --> SALES_MODELS
    SALES_MODELS --> SALES_DATA
    
    %% Foundation Connections
    LOAN_APP --> PYTHON
    CAMP_APP --> PYTHON
    SALES_APP --> PYTHON
    
    LOAN_MODELS --> TABPFN
    CAMP_MODELS --> TABPFN
    
    LOAN_DATA --> CSV_DATA
    LOAN_DATA --> MODEL_STORE
    CAMP_DATA --> CSV_DATA
    CAMP_DATA --> MODEL_STORE
    SALES_DATA --> CSV_DATA
    SALES_DATA --> MODEL_STORE
    
    %% File Upload Flows
    LOAN_API --> UPLOADS
    CAMP_API --> UPLOADS
    SALES_API --> UPLOADS
    
    %% Logging Flows
    DASH_APP --> LOGS
    LOAN_APP --> LOGS
    CAMP_APP --> LOGS
    SALES_APP --> LOGS
    
    %% Demo & Documentation Connections
    DEMO_SCRIPTS --> LOAN_APP
    DEMO_SCRIPTS --> CAMP_APP
    DEMO_SCRIPTS --> SALES_APP
    DEMO_SCRIPTS --> DEMO_LOGS
    
    %% Future Integration
    DASH_APP -.-> SAMSUNG_API
    PYTHON -.-> CLOUD_DEPLOY
    
    %% === STYLING ===
    classDef userLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef dashboardLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef appLayer fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef foundationLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef docLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef externalLayer fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    
    class WEB,MOBILE,API_CLIENT userLayer
    class DASH_APP,DASH_ROUTES,PROC_MGR,DASH_LOGS dashboardLayer
    class LOAN_APP,LOAN_MODELS,LOAN_API,LOAN_DATA,CAMP_APP,CAMP_MODELS,CAMP_API,CAMP_DATA,SALES_APP,SALES_MODELS,SALES_API,SALES_DATA appLayer
    class PYTHON,TABPFN,CSV_DATA,MODEL_STORE,UPLOADS,LOGS,DEMO_LOGS foundationLayer
    class VIS_DOCS,README_DOCS,DEMO_SCRIPTS docLayer
    class SAMSUNG_API,CLOUD_DEPLOY externalLayer
```

## 🔄 Data Flow Architecture

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant D as 📊 Dashboard
    participant L as 🏦 Loan App
    participant C as 📈 Campaign App  
    participant S as 📊 Sales App
    participant M as 🧠 ML Models
    participant DB as 💾 Data Store
    
    %% Dashboard Initialization
    Note over U,DB: Dashboard Startup & Application Orchestration
    U->>D: Access http://127.0.0.1:5050
    D->>D: Load Application Status
    D->>L: Check Port 7001 Status
    D->>C: Check Port 7002 Status
    D->>S: Check Port 7003 Status
    D-->>U: Display Unified Interface
    
    %% Application Launch Flow
    Note over U,DB: Individual Application Launch
    U->>D: Click "Launch Loan Risk App"
    D->>L: Start Process (Port 7001)
    L->>M: Load TabPFN Models
    L->>DB: Load Training Data
    L-->>D: Application Ready
    D-->>U: Redirect to Loan App
    
    %% Prediction Flow - Individual
    Note over U,DB: Individual Prediction Workflow
    U->>L: Submit Loan Application
    L->>L: Validate Input Data
    L->>M: Process with TabPFN/RF/XGB
    M->>M: Calculate Risk Score
    M-->>L: Return Prediction + Confidence
    L->>L: Generate Risk Classification
    L-->>U: Risk Assessment Results
    
    %% Batch Processing Flow
    Note over U,DB: Batch Processing Workflow
    U->>C: Upload CSV File (Store Data)
    C->>DB: Save to uploads/
    C->>C: Validate CSV Schema
    loop For Each Record
        C->>M: Process with CatBoost/LGBM
        M-->>C: Performance Score
    end
    C->>DB: Save Results
    C-->>U: Download Results CSV
    
    %% Cross-Application Integration
    Note over U,DB: Multi-Application Workflow
    U->>D: Access Unified Dashboard
    D->>L: Get Loan Risk Metrics
    D->>C: Get Campaign Performance
    D->>S: Get Sales Forecasts
    L-->>D: Risk Portfolio Stats
    C-->>D: Marketing ROI Data
    S-->>D: Revenue Predictions
    D-->>U: Consolidated Business Intelligence
```

## 🏗️ Technical Stack Architecture

```mermaid
graph LR
    %% Frontend Layer
    subgraph "🎨 Frontend Layer"
        HTML["`📄 **HTML5**
        Semantic Structure`"]
        CSS["`🎨 **CSS3/Bootstrap**
        Samsung UI Theme`"]
        JS["`⚡ **JavaScript**
        Interactive Elements`"]
    end
    
    %% Application Layer
    subgraph "🔧 Application Layer"
        FLASK["`🌶️ **Flask Framework**
        • Routing & Templates
        • Request Handling
        • Session Management`"]
        
        JINJA["`📝 **Jinja2 Templates**
        • Dynamic Content
        • Component Reuse
        • Data Binding`"]
        
        WERKZEUG["`⚙️ **Werkzeug WSGI**
        • HTTP Protocol
        • Request/Response
        • File Uploads`"]
    end
    
    %% ML & Data Layer
    subgraph "🤖 ML & Data Processing"
        PANDAS["`🐼 **Pandas**
        Data Manipulation`"]
        
        NUMPY["`🔢 **NumPy**
        Numerical Computing`"]
        
        SKLEARN["`📊 **Scikit-learn**
        ML Pipeline`"]
        
        XGBOOST["`🚀 **XGBoost**
        Gradient Boosting`"]
        
        TABPFN["`🧠 **TabPFN**
        Foundation Model`"]
        
        CATBOOST["`🐱 **CatBoost**
        Categorical Features`"]
        
        LIGHTGBM["`💡 **LightGBM**
        Fast Training`"]
    end
    
    %% Infrastructure Layer
    subgraph "🏗️ Infrastructure"
        PYTHON["`🐍 **Python 3.10+**
        Runtime Environment`"]
        
        VENV["`📦 **Virtual Environment**
        Dependency Isolation`"]
        
        PIP["`📥 **Pip Package Manager**
        Dependency Management`"]
        
        OS["`💻 **macOS/Linux**
        Operating System`"]
    end
    
    %% Connections
    HTML --> FLASK
    CSS --> FLASK
    JS --> FLASK
    
    FLASK --> JINJA
    FLASK --> WERKZEUG
    
    FLASK --> PANDAS
    PANDAS --> NUMPY
    PANDAS --> SKLEARN
    
    SKLEARN --> XGBOOST
    SKLEARN --> TABPFN
    SKLEARN --> CATBOOST
    SKLEARN --> LIGHTGBM
    
    FLASK --> PYTHON
    PANDAS --> PYTHON
    SKLEARN --> PYTHON
    
    PYTHON --> VENV
    VENV --> PIP
    PYTHON --> OS
    
    %% Styling
    classDef frontend fill:#e3f2fd,stroke:#0277bd
    classDef application fill:#f1f8e9,stroke:#388e3c
    classDef ml fill:#fff8e1,stroke:#f57c00
    classDef infrastructure fill:#fce4ec,stroke:#c2185b
    
    class HTML,CSS,JS frontend
    class FLASK,JINJA,WERKZEUG application
    class PANDAS,NUMPY,SKLEARN,XGBOOST,TABPFN,CATBOOST,LIGHTGBM ml
    class PYTHON,VENV,PIP,OS infrastructure
```

## 📊 Port & Service Architecture

```mermaid
graph TB
    subgraph "🌐 Network Layer (127.0.0.1)"
        PORT_5050["`🎛️ **Port 5050**
        Main Dashboard
        • Process Management
        • Unified Interface
        • Health Monitoring`"]
        
        PORT_7001["`🏦 **Port 7001**
        Loan Risk Assessment
        • Individual Predictions
        • Batch Processing
        • Risk Classification`"]
        
        PORT_7002["`📈 **Port 7002**
        Campaign Performance
        • Store Analysis
        • Marketing ROI
        • Performance Benchmarks`"]
        
        PORT_7003["`📊 **Port 7003**
        Sales Forecasting
        • Uplift Predictions
        • Revenue Projections
        • Time Series Analysis`"]
    end
    
    subgraph "🔄 Service Dependencies"
        HEALTH["`💚 **Health Checks**
        • Port Availability
        • Service Status
        • Response Validation`"]
        
        LOGS["`📝 **Centralized Logging**
        • Application Logs
        • Error Tracking
        • Performance Metrics`"]
        
        PROC["`⚙️ **Process Control**
        • Start/Stop Services
        • Resource Management
        • Dependency Handling`"]
    end
    
    PORT_5050 --> PORT_7001
    PORT_5050 --> PORT_7002
    PORT_5050 --> PORT_7003
    
    PORT_5050 --> HEALTH
    PORT_5050 --> LOGS
    PORT_5050 --> PROC
    
    HEALTH --> PORT_7001
    HEALTH --> PORT_7002
    HEALTH --> PORT_7003
```

---

**🏆 PRISM Worklet 8 - Advanced AI Platform Architecture**  
*Preparing and Inspiring Student Minds*

This comprehensive architecture diagram shows the complete system structure, data flows, technical stack, and service dependencies for the Samsung PRISM Worklet 8 project.
