# create_encoder.py
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

def create_and_save_encoder():
    """
    Create and save the OneHotEncoder for the Rossmann sales prediction pipeline
    """
    print("Creating OneHotEncoder for Rossmann Sales Prediction...")
    
    # Initialize OneHotEncoder with same parameters as in training
    encoder = OneHotEncoder(
        sparse_output=False, 
        dtype=int, 
        handle_unknown='ignore'  # Handle unseen categories gracefully
    )
    
    # FIXED: Create sample data with SAME LENGTH arrays
    sample_categorical_data = pd.DataFrame({
        'StoreType': ['a', 'b', 'c', 'd'],    # 4 categories
        'Assortment': ['a', 'b', 'c', 'a']    # 4 values (repeated 'a' to match length)
    })
    
    print("Sample categorical data:")
    print(sample_categorical_data)
    
    # Fit the encoder on the sample data
    encoder.fit(sample_categorical_data)
    
    # Display encoder information
    print("\nEncoder fitted successfully!")
    print(f"StoreType categories: {encoder.categories_[0]}")
    print(f"Assortment categories: {encoder.categories_[1]}")
    print(f"Feature names: {list(encoder.get_feature_names_out(['StoreType', 'Assortment']))}")
    
    # Save the encoder
    encoder_path = 'encoder.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    
    print(f"\n✓ Encoder saved successfully as '{encoder_path}'")
    
    # Verify the saved encoder
    try:
        with open(encoder_path, 'rb') as f:
            loaded_encoder = pickle.load(f)
        
        # Test transformation
        test_transform = loaded_encoder.transform(sample_categorical_data)
        print(f"✓ Encoder verification successful - Output shape: {test_transform.shape}")
        print(f"Output columns: {test_transform.shape[1]} (should be 7)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying saved encoder: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("ROSSMANN SALES PREDICTION - ENCODER GENERATOR")
    print("="*60)
    
    success = create_and_save_encoder()
    
    if success:
        print("\n" + "="*60)
        print("ENCODER CREATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Files created:")
        print("- encoder.pkl (OneHotEncoder for StoreType and Assortment)")
        print("\nExpected output features:")
        print("- StoreType_a, StoreType_b, StoreType_c, StoreType_d")
        print("- Assortment_a, Assortment_b, Assortment_c")
        print("- Total: 7 features")
        print("\nYou can now run your Flask application:")
        print("python app.py")
    else:
        print("\n✗ Encoder creation failed.")
