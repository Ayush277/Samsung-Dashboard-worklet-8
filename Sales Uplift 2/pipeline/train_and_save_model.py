import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from utils.data_processor import DataProcessor

def train_and_save_model():
    """
    Train the model and save all components
    """
    # Load your data (replace with your actual data loading)
    train_df = pd.read_csv('train.csv')
    store_df = pd.read_csv('store.csv')
    
    # Initialize data processor with dummy scaler
    temp_scaler = StandardScaler()
    data_processor = DataProcessor(temp_scaler)
    
    # Process the data
    processed_df = data_processor.preprocess_data(train_df, store_df)
    
    # Prepare features and target
    processed_df['Date'] = pd.to_datetime(processed_df['Date'])
    
    # Split by date
    X = processed_df.drop(['Sales', 'Customers'], axis=1)
    y = processed_df['Sales']
    
    X_train = X[processed_df['Date'] < '2015-06-19']
    y_train = y[processed_df['Date'] < '2015-06-19']
    X_test = X[processed_df['Date'] >= '2015-06-19']
    y_test = y[processed_df['Date'] >= '2015-06-19']
    
    # Remove Date from features
    X_train = X_train.drop('Date', axis=1)
    X_test = X_test.drop('Date', axis=1)
    
    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    xgboost = xgb.XGBRegressor(objective='reg:squarederror', verbosity=0)
    parameters = {
        'max_depth': [2, 5, 10],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_child_weight': [1, 2, 5],
        'gamma': [0, 0.1, 0.3],
        'colsample_bytree': [0.3, 0.5, 0.7]
    }
    
    xg_reg = RandomizedSearchCV(
        estimator=xgboost, 
        param_distributions=parameters, 
        n_iter=10, 
        cv=3, 
        random_state=42
    )
    xg_reg.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xg_reg, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model and scaler saved successfully!")
    print(f"Best parameters: {xg_reg.best_params_}")
    
    # Test the model
    y_pred = xg_reg.predict(X_test_scaled)
    from sklearn.metrics import r2_score
    print(f"Test R2 Score: {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    train_and_save_model()
