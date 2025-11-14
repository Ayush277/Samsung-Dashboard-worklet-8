"""
Train TabPFN model for sales prediction with electronic devices assumption
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
import pickle
import os

# Load training data
print("Loading training data...")
train_df = pd.read_csv('train2.csv', parse_dates=['date'])
print(f"Training data shape: {train_df.shape}")

# Create features
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['dayofweek'] = train_df['date'].dt.dayofweek
train_df['dayofyear'] = train_df['date'].dt.dayofyear
train_df['weekofyear'] = train_df['date'].dt.isocalendar().week.astype(int)

# Feature columns
feature_cols = ['store', 'item', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear']
X = train_df[feature_cols]
y = train_df['sales']

# Take a sample for TabPFN (it has limitations on dataset size)
print("Sampling data for TabPFN training...")
sample_size = min(10000, len(X))  # TabPFN works better with smaller datasets
sample_idx = np.random.choice(len(X), sample_size, replace=False)
X_sample = X.iloc[sample_idx]
y_sample = y.iloc[sample_idx]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# Train TabPFN model
print("Training TabPFN model...")
tabpfn_model = TabPFNRegressor(device='cpu')
tabpfn_model.fit(X_scaled, y_sample)

# Save models
print("Saving TabPFN model and scaler...")
with open('tabpfn_sales_model.pkl', 'wb') as f:
    pickle.dump(tabpfn_model, f)

with open('tabpfn_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ TabPFN model training complete!")
print("Files saved: tabpfn_sales_model.pkl, tabpfn_scaler.pkl")

# Test the model
test_features = [[1, 1, 6, 15, 0, 166, 24]]  # Sample test
test_scaled = scaler.transform(test_features)
prediction = tabpfn_model.predict(test_scaled)
print(f"Sample prediction: {prediction[0]:.2f} units")
